"""
MILP model for territory optimization.

Orchestrates clustering, feature engineering, and solving into
a unified optimization model.
"""
import logging
from typing import Dict, Any
import numpy as np

from optimization.clustering import SpatialClusterer
from optimization.constraints import OptimizationConstraints
from optimization.solver import TerritorySolver

logger = logging.getLogger(__name__)


class TerritoryModel:
    """Builds and manages the MILP optimization model for territory rebalancing."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize territory optimization model.

        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.clusterer = SpatialClusterer(config.get('clustering', {}))
        self.constraints = OptimizationConstraints(config.get('constraints', {}))
        self.solver = TerritorySolver(config.get('solver', {}))
        self.model_result = None

    def build_and_solve(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build and solve the complete territory optimization model.

        Args:
            features: Dictionary containing all processed features:
                - assignment_matrix: Current assignment [D x F]
                - workload: Dealer workload array [D]
                - capacity: FTC capacity array [F]
                - compatibility: Compatibility matrix [D x F]
                - distance: Normalized distance matrix [D x F]
                - distance_km: Raw distance in km [D x F]
                - dealers: Dealers dataframe
                - ftcs: FTCs dataframe

        Returns:
            Solution dictionary with assignments and metrics
        """
        logger.info("Building and solving territory optimization model")

        assignment_matrix = features['assignment_matrix']
        workload = features.get('workload', np.array([]))
        capacity = features.get('capacity', np.array([]))
        compatibility = features.get('compatibility', np.zeros_like(assignment_matrix, dtype=float))
        distance = features.get('distance', np.zeros_like(assignment_matrix, dtype=float))

        cluster_labels = features.get('cluster_labels', None)

        num_dealers, num_ftcs = assignment_matrix.shape
        logger.info(f"Problem size: {num_dealers} dealers x {num_ftcs} FTCs")

        # Validate dimensions
        self._validate_dimensions(num_dealers, num_ftcs, workload, capacity, compatibility, distance)

        # Compute feasibility mask
        distance_km = features.get('distance_km', None)
        if distance_km is not None:
            feasibility_mask = self.constraints.compute_feasibility_mask(distance_km)
        else:
            feasibility_mask = np.ones((num_dealers, num_ftcs), dtype=int)

        # Apply product compatibility mask
        if 'dealers' in features and 'ftcs' in features:
            product_mask = self.constraints.compute_product_compatibility_mask(
                features['dealers'], features['ftcs']
            )
            feasibility_mask = feasibility_mask & product_mask

        # Ensure every dealer still has at least one feasible option after all masks are combined.
        # Without this repair step, the model can leave dealers unconstrained, which makes the
        # solver spend excessive time in search and can produce invalid solutions.
        for i in range(num_dealers):
            if feasibility_mask[i].sum() == 0:
                fallback_ftc = int(np.argmax(compatibility[i] - distance[i]))
                feasibility_mask[i, fallback_ftc] = 1

        # Save pre-expansion mask for post-solve enforcement (cluster cohesion
        # can give individual dealers FTCs that only their cluster-mates can reach).
        pre_enforce_mask = feasibility_mask.copy()

        # Expand feasibility mask for cluster cohesion: within a compact micro-location
        # cluster, if any dealer can reach an FTC, allow all dealers in that cluster to
        # reach it too.  This guarantees the cluster cohesion constraint (added in the
        # solver) is always satisfiable — without this, the intersection of feasible FTCs
        # across cluster members is often empty due to the tight 50km radius.
        if cluster_labels is not None:
            unique_clusters = np.unique(cluster_labels)
            for c in unique_clusters:
                cluster_indices = np.where(cluster_labels == c)[0]
                cluster_union = np.zeros(num_ftcs, dtype=int)
                for i in cluster_indices:
                    cluster_union |= feasibility_mask[i]
                for i in cluster_indices:
                    feasibility_mask[i] = cluster_union
            logger.info(f"Expanded feasibility mask for {len(unique_clusters)} clusters")

        # Enforce static dealers stay at their current FTC.
        # A static dealer cannot be reassigned to a different FTC; only the
        # FTC they are currently assigned to is feasible.
        dealers_df = features.get('dealers')
        if dealers_df is not None and 'dealer_type' in dealers_df.columns:
            static_mask = dealers_df['dealer_type'].str.lower() == 'static'
            static_count = static_mask.sum()
            if static_count > 0:
                frozen = 0
                for i in range(num_dealers):
                    if not static_mask.iloc[i]:
                        continue
                    current_ftc = np.where(assignment_matrix[i] == 1)[0]
                    if len(current_ftc) > 0:
                        feasibility_mask[i, :] = 0
                        feasibility_mask[i, current_ftc[0]] = 1
                        pre_enforce_mask[i, :] = 0
                        pre_enforce_mask[i, current_ftc[0]] = 1
                        frozen += 1
                logger.info(f"Frozen {frozen}/{static_count} static dealers to their current FTC")

        # Get optimization parameters
        opt_config = self.config.get('optimization', {})
        alpha_1 = opt_config.get('alpha_1', 0.5)
        alpha_2 = opt_config.get('alpha_2', 0.3)
        lambda_penalty = opt_config.get('lambda', 1.0)

        bounds = self.constraints.get_assignment_bounds()
        min_d = bounds['min_dealers']
        max_d = bounds['max_dealers']
        
        # Relax constraints for toy datasets to avoid infeasibility
        # 1000 dealers spread across all of India is too sparse to guarantee 3 per FTC
        if num_dealers < 1000:
            min_d = 1
        if num_dealers < max_d:
            max_d = num_dealers

        # Solve
        result = self.solver.solve(
            compatibility=compatibility,
            distance=distance,
            current_assignment=assignment_matrix,
            feasibility_mask=feasibility_mask,
            workload=workload,
            capacity=capacity,
            distance_km=distance_km,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_penalty=lambda_penalty,
            min_dealers=min_d,
            max_dealers=max_d,
            cluster_labels=cluster_labels
        )

        # Enforce individual dealer feasibility against the pre-expansion mask.
        # The cluster cohesion expansion lets clusters share FTCs across members,
        # which can give individual dealers assignments that their own feasibility
        # mask would have blocked.  This pass fixes that.
        if result['status'] in ('OPTIMAL', 'FEASIBLE'):
            if distance_km is not None:
                assign = result['assignments']
                ftc_counts = assign.sum(axis=0).astype(int)
                any_fixed = False
                violations_before = 0
                for i in range(num_dealers):
                    j = np.argmax(assign[i])
                    if pre_enforce_mask[i, j] != 1:
                        violations_before += 1
                if violations_before:
                    logger.info(f"  Enforcement: {violations_before} dealers in infeasible FTCs")
                for i in range(num_dealers):
                    j = np.argmax(assign[i])
                    if pre_enforce_mask[i, j] == 1:
                        continue
                    # Move to nearest truly feasible FTC
                    feasible_ftcs = np.where(pre_enforce_mask[i] == 1)[0]
                    if len(feasible_ftcs) == 0:
                        continue
                    order = feasible_ftcs[np.argsort(distance_km[i][feasible_ftcs])]
                    for candidate in order:
                        if ftc_counts[candidate] < max_d:
                            assign[i, :] = 0
                            assign[i, candidate] = 1
                            ftc_counts[candidate] += 1
                            ftc_counts[j] -= 1
                            any_fixed = True
                            break
                # Second pass: relax capacity for any remaining violations
                for i in range(num_dealers):
                    j = np.argmax(assign[i])
                    if pre_enforce_mask[i, j] == 1:
                        continue
                    feasible_ftcs = np.where(pre_enforce_mask[i] == 1)[0]
                    if len(feasible_ftcs) == 0:
                        continue
                    order = feasible_ftcs[np.argsort(distance_km[i][feasible_ftcs])]
                    candidate = order[0]
                    assign[i, :] = 0
                    assign[i, candidate] = 1
                    ftc_counts[candidate] += 1
                    ftc_counts[j] -= 1
                    any_fixed = True
                if any_fixed:
                    logger.info(f"  Fixed {violations_before} dealer-level feasibility violations")
                    result['assignments'] = assign
                    # Recompute stale metrics from corrected assignment
                    cur = result.get('current_assignment')
                    if cur is None or cur.shape != assign.shape:
                        cur = np.zeros_like(assign)
                    diff = np.abs(assign - cur)
                    result['total_changes'] = int(np.sum(diff))
                    result['changed_dealers'] = int(np.sum(np.any(diff > 0, axis=1)))
                    result['change_rate'] = result['changed_dealers'] / num_dealers
                    result['ftcs_used'] = int(np.sum(assign.sum(axis=0) > 0))
                    if distance_km is not None:
                        new_dists = [distance_km[i, np.argmax(assign[i])] for i in range(num_dealers)]
                        result['mean_distance_km'] = float(np.mean(new_dists))
                        result['median_distance_km'] = float(np.median(new_dists))
                    logger.info(f"  Post-fix: {result['changed_dealers']} changed, "
                                 f"{result['ftcs_used']} active, "
                                 f"mean={result['mean_distance_km']:.2f}km")

            # Validate final solution
            validation = self.constraints.validate_solution(
                result['assignments'],
                distance_km
            )
            result['validation'] = validation
            if validation['violations']:
                logger.warning(f"Solution has {len(validation['violations'])} constraint violations")
                for v in validation['violations'][:5]:
                    logger.warning(f"  - {v}")

        self.model_result = result
        return result

    def _validate_dimensions(self, num_dealers, num_ftcs, workload, capacity, compatibility, distance):
        """Validate feature array dimensions are consistent."""
        if len(workload) > 0 and len(workload) != num_dealers:
            raise ValueError(f"Workload size {len(workload)} != {num_dealers} dealers")
        if len(capacity) > 0 and len(capacity) != num_ftcs:
            raise ValueError(f"Capacity size {len(capacity)} != {num_ftcs} FTCs")
        if compatibility.size > 0 and compatibility.shape != (num_dealers, num_ftcs):
            raise ValueError(f"Compatibility shape {compatibility.shape} != ({num_dealers}, {num_ftcs})")
        if distance.size > 0 and distance.shape != (num_dealers, num_ftcs):
            raise ValueError(f"Distance shape {distance.shape} != ({num_dealers}, {num_ftcs})")

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary information about the last optimization run."""
        if self.model_result is None:
            return {'status': 'not_run'}
        return {
            'status': self.model_result['status'],
            'solve_time': self.model_result['solve_time'],
            'total_changes': self.model_result['total_changes'],
            'changed_dealers': self.model_result['changed_dealers'],
            'change_rate': self.model_result['change_rate'],
            'ftcs_used': self.model_result['ftcs_used'],
            'objective_value': self.model_result['objective_value'],
        }
