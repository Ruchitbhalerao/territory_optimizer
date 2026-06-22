"""
Hybrid solver for territory optimization.

Uses greedy construction followed by local search for large problems
(with cluster labels), and falls back to OR-Tools CP-SAT for small
problems (without cluster labels).
"""
import logging
import time
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class TerritorySolver:
    """Solves territory optimization using greedy + local search or CP-SAT."""

    def __init__(self, config: Dict[str, Any]):
        self.time_limit = config.get('time_limit_seconds', 300)
        self.num_workers = config.get('num_workers', 4)
        self.optimality_gap = config.get('optimality_gap', 0.05)

    def solve(self, compatibility: np.ndarray, distance: np.ndarray,
              current_assignment: np.ndarray, feasibility_mask: np.ndarray,
              workload: np.ndarray, capacity: np.ndarray,
              distance_km: Optional[np.ndarray] = None,
              alpha_1: float = 0.5, alpha_2: float = 0.3,
              lambda_penalty: float = 1.0,
              min_dealers: int = 3, max_dealers: int = 25,
              cluster_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Solve the territory assignment optimization problem.

        Objective: maximize alpha_1 * compatibility + alpha_2 * (1 - distance)
                   - lambda * disruption
        """
        num_dealers, num_ftcs = compatibility.shape
        logger.info(f"Solving assignment: {num_dealers} dealers x {num_ftcs} FTCs")
        start_time = time.time()

        if cluster_labels is not None:
            return self._solve_cluster_greedy(
                compatibility, distance, current_assignment, feasibility_mask,
                workload, capacity, distance_km,
                alpha_1, alpha_2, lambda_penalty,
                min_dealers, max_dealers, cluster_labels,
                num_dealers, num_ftcs, start_time
            )

        return self._solve_cpsat(
            compatibility, distance, current_assignment, feasibility_mask,
            workload, capacity,
            alpha_1, alpha_2, lambda_penalty,
            min_dealers, max_dealers,
            num_dealers, num_ftcs, start_time
        )

    def _solve_cluster_greedy(self,
                               compatibility, distance, current_assignment, feasibility_mask,
                               workload, capacity, distance_km,
                               alpha_1, alpha_2, lambda_penalty,
                               min_dealers, max_dealers, cluster_labels,
                               num_dealers, num_ftcs, start_time):
        """
        Nearest-feasible-FTC assignment with swap-based refinement.

        Phase 1: assign each cluster to its nearest feasible FTC with room.
        Phase 2: repair any FTCs that exceed max_dealers.
        Phase 3: activate unused FTCs by reassigning the nearest eligible cluster.
        Phase 4: swap refinement — reduce mean distance via pairwise swaps.
        """
        unique_clusters = np.unique(cluster_labels)
        num_clusters = len(unique_clusters)

        cluster_dealer_indices = []
        cluster_sizes = []
        for c in unique_clusters:
            idx = np.where(cluster_labels == c)[0]
            cluster_dealer_indices.append(idx)
            cluster_sizes.append(len(idx))

        cluster_sizes_arr = np.array(cluster_sizes)

        # Precompute cluster-level mean distance in km and feasibility
        if distance_km is not None:
            cluster_mean_dist = np.zeros((num_clusters, num_ftcs))
        else:
            cluster_mean_dist = None
        cluster_feasible = np.zeros((num_clusters, num_ftcs), dtype=bool)

        for c_idx, indices in enumerate(cluster_dealer_indices):
            cluster_feasible[c_idx] = feasibility_mask[indices].any(axis=0)
            if distance_km is not None:
                cluster_mean_dist[c_idx] = distance_km[indices].mean(axis=0)
            else:
                if cluster_mean_dist is None:
                    cluster_mean_dist = np.zeros((num_clusters, num_ftcs))
                cluster_mean_dist[c_idx] = distance[indices].mean(axis=0) * 50.0

        # For each cluster, precompute sorted feasible FTC indices by distance
        sorted_feasible = []
        for c_idx in range(num_clusters):
            order = np.argsort(cluster_mean_dist[c_idx])
            feasible_order = [j for j in order if cluster_feasible[c_idx, j]]
            sorted_feasible.append(feasible_order)

        # Determine each cluster's current FTC (majority vote among its dealers)
        cluster_current_ftc = np.full(num_clusters, -1, dtype=int)
        for c_idx, indices in enumerate(cluster_dealer_indices):
            ftc_counts = current_assignment[indices].sum(axis=0)
            if ftc_counts.sum() > 0:
                cluster_current_ftc[c_idx] = np.argmax(ftc_counts)

        # Phase 1: assignment with dealer count capacity.
        # When lambda_penalty > 0, prefer keeping each cluster at its current FTC.
        # Process clusters in order of fewest options first (most constrained).
        n_feasible = np.array([len(sf) for sf in sorted_feasible])
        process_order = np.argsort(n_feasible)

        rng = np.random.RandomState(42)
        best_assign = None
        best_score = -np.inf
        use_disruption = lambda_penalty > 0.5

        for trial in range(6):
            assign = np.full(num_clusters, -1, dtype=int)
            ftc_cnt = np.zeros(num_ftcs, dtype=int)

            shuffled = process_order.copy()
            rng.shuffle(shuffled)

            for c_idx in shuffled:
                s = cluster_sizes_arr[c_idx]
                chosen = -1
                # If disruption minimization is active and current FTC has room, keep it
                current_j = cluster_current_ftc[c_idx]
                if use_disruption and current_j >= 0:
                    if (cluster_feasible[c_idx, current_j]
                            and ftc_cnt[current_j] + s <= max_dealers):
                        chosen = current_j
                if chosen < 0:
                    for j in sorted_feasible[c_idx]:
                        if ftc_cnt[j] + s <= max_dealers:
                            chosen = j
                            break
                if chosen < 0:
                    chosen = sorted_feasible[c_idx][0]
                assign[c_idx] = chosen
                ftc_cnt[chosen] += s

            # Score: lower distance + higher retention = better
            total_km = 0.0
            kept = 0
            for c_idx, j in enumerate(assign):
                total_km += cluster_mean_dist[c_idx, j] * cluster_sizes_arr[c_idx]
                if use_disruption and j == cluster_current_ftc[c_idx]:
                    kept += cluster_sizes_arr[c_idx]
            mean_dist = total_km / cluster_sizes_arr.sum()
            kept_ratio = kept / cluster_sizes_arr.sum() if use_disruption else 0.0
            trial_score = -mean_dist + lambda_penalty * kept_ratio

            if trial_score > best_score:
                best_score = trial_score
                best_assign = assign.copy()
                logger.info(f"  Trial {trial}: mean_dist={mean_dist:.2f}km, kept={kept_ratio:.0%} (score={trial_score:.2f})")

        assignment = best_assign.copy()

        # Rebuild ftc_cnt from best assignment
        ftc_cnt_refine = np.zeros(num_ftcs, dtype=int)
        for c_idx, j in enumerate(assignment):
            ftc_cnt_refine[j] += cluster_sizes_arr[c_idx]

        # Phase 2: repair overloaded FTCs — move excess clusters to the nearest
        # feasible FTC with spare capacity.
        overloaded = np.where(ftc_cnt_refine > max_dealers)[0]
        if len(overloaded) > 0:
            logger.info(f"Repairing {len(overloaded)} overloaded FTCs")
            for _ in range(50):
                moved = 0
                for j in overloaded.copy():
                    excess = ftc_cnt_refine[j] - max_dealers
                    if excess <= 0:
                        continue
                    candidates = [c_idx for c_idx in range(num_clusters) if assignment[c_idx] == j]
                    candidates.sort(key=lambda c: cluster_sizes_arr[c], reverse=True)
                    for c_idx in candidates:
                        if excess <= 0:
                            break
                        s = cluster_sizes_arr[c_idx]
                        for j2 in sorted_feasible[c_idx]:
                            if j2 == j:
                                continue
                            if ftc_cnt_refine[j2] + s <= max_dealers:
                                ftc_cnt_refine[j] -= s
                                ftc_cnt_refine[j2] += s
                                assignment[c_idx] = j2
                                excess -= s
                                moved += 1
                                break
                overloaded = np.where(ftc_cnt_refine > max_dealers)[0]
                if len(overloaded) == 0:
                    break
            remaining_over = len(overloaded)
            if remaining_over > 0:
                logger.warning(f"{remaining_over} FTCs still overloaded after repair")

        # Phase 3: activate unused FTCs — move the nearest eligible cluster
        unused = [j for j in range(num_ftcs) if ftc_cnt_refine[j] == 0]
        if unused:
            logger.info(f"Activating {len(unused)} unused FTCs")
            for _ in range(20):
                activated = 0
                for j in unused.copy():
                    if ftc_cnt_refine[j] > 0:
                        continue
                    best_c = -1
                    best_d = np.inf
                    for c_idx in range(num_clusters):
                        if not cluster_feasible[c_idx, j]:
                            continue
                        s = cluster_sizes_arr[c_idx]
                        cj = assignment[c_idx]
                        if ftc_cnt_refine[cj] <= s:
                            continue
                        if ftc_cnt_refine[j] + s > max_dealers:
                            continue
                        d = cluster_mean_dist[c_idx, j]
                        if d < best_d:
                            best_d = d
                            best_c = c_idx
                    if best_c >= 0:
                        c_idx = best_c
                        current_j = assignment[c_idx]
                        s = cluster_sizes_arr[c_idx]
                        ftc_cnt_refine[current_j] -= s
                        ftc_cnt_refine[j] += s
                        assignment[c_idx] = j
                        activated += 1
                if activated == 0:
                    break
                unused = [j for j in range(num_ftcs) if ftc_cnt_refine[j] == 0]
                if not unused:
                    break

        # Phase 4: swap refinement — reduce mean distance without violating capacity
        # Skipped when disruption minimization is active (undoes kept assignments)
        if not use_disruption:
            logger.info("Running swap refinement to reduce mean distance")
        for iteration in range(20 if not use_disruption else 0):
            improved = False
            for c1 in range(num_clusters):
                j1 = assignment[c1]
                s1 = cluster_sizes_arr[c1]
                for c2 in range(c1 + 1, num_clusters):
                    j2 = assignment[c2]
                    if j1 == j2:
                        continue
                    s2 = cluster_sizes_arr[c2]

                    # Check if swap improves total distance
                    current_dist = cluster_mean_dist[c1, j1] + cluster_mean_dist[c2, j2]
                    new_dist = cluster_mean_dist[c1, j2] + cluster_mean_dist[c2, j1]
                    if new_dist >= current_dist:
                        continue

                    if not (cluster_feasible[c1, j2] and cluster_feasible[c2, j1]):
                        continue

                    if ftc_cnt_refine[j2] + s1 - s2 > max_dealers:
                        continue
                    if ftc_cnt_refine[j1] + s2 - s1 > max_dealers:
                        continue

                    assignment[c1], assignment[c2] = j2, j1
                    ftc_cnt_refine[j1] += s2 - s1
                    ftc_cnt_refine[j2] += s1 - s2
                    improved = True

            if not improved:
                logger.info(f"Swap refinement converged after {iteration + 1} iterations")
                break

        # Phase 5: final overload repair and unused activation.
        # Fix any overloads created by swaps, then activate remaining unused FTCs.
        over = np.where(ftc_cnt_refine > max_dealers)[0]
        if len(over) > 0:
            logger.info(f"Repairing {len(over)} overloaded FTCs after swap")
            for _ in range(50):
                moved = 0
                for j in over.copy():
                    excess = ftc_cnt_refine[j] - max_dealers
                    if excess <= 0:
                        continue
                    candidates = [c_idx for c_idx in range(num_clusters) if assignment[c_idx] == j]
                    candidates.sort(key=lambda c: cluster_sizes_arr[c], reverse=True)
                    for c_idx in candidates:
                        if excess <= 0:
                            break
                        s = cluster_sizes_arr[c_idx]
                        best_j2 = -1
                        best_score = -1
                        for j2 in sorted_feasible[c_idx]:
                            if j2 == j:
                                continue
                            if ftc_cnt_refine[j2] + s > max_dealers:
                                continue
                            room = max_dealers - ftc_cnt_refine[j2]
                            score = room * 1000 - cluster_mean_dist[c_idx, j2]
                            if score > best_score:
                                best_score = score
                                best_j2 = j2
                        if best_j2 >= 0:
                            ftc_cnt_refine[j] -= s
                            ftc_cnt_refine[best_j2] += s
                            assignment[c_idx] = best_j2
                            excess -= s
                            moved += 1
                over = np.where(ftc_cnt_refine > max_dealers)[0]
                if len(over) == 0:
                    break

        # Recompute ftc_cnt_refine from assignment to guarantee consistency
        ftc_cnt_refine = np.zeros(num_ftcs, dtype=int)
        for c_idx, j in enumerate(assignment):
            ftc_cnt_refine[j] += cluster_sizes_arr[c_idx]

        unused_after = [j for j in range(num_ftcs) if ftc_cnt_refine[j] == 0]
        if unused_after:
            logger.info(f"Activating {len(unused_after)} unused FTCs (final pass)")
            for _ in range(20):
                activated = 0
                for j in unused_after.copy():
                    if ftc_cnt_refine[j] > 0:
                        continue
                    best_c = -1
                    best_d = np.inf
                    for c_idx in range(num_clusters):
                        if not cluster_feasible[c_idx, j]:
                            continue
                        s = cluster_sizes_arr[c_idx]
                        cj = assignment[c_idx]
                        if ftc_cnt_refine[cj] <= s:
                            continue
                        if ftc_cnt_refine[j] + s > max_dealers:
                            continue
                        d = cluster_mean_dist[c_idx, j]
                        if d < best_d:
                            best_d = d
                            best_c = c_idx
                    if best_c >= 0:
                        c_idx = best_c
                        current_j = assignment[c_idx]
                        s = cluster_sizes_arr[c_idx]
                        ftc_cnt_refine[current_j] -= s
                        ftc_cnt_refine[j] += s
                        assignment[c_idx] = j
                        activated += 1
                if activated == 0:
                    break
                unused_after = [j for j in range(num_ftcs) if ftc_cnt_refine[j] == 0]
                if not unused_after:
                    break

        # Expand cluster assignment to dealer-level
        new_assignment = np.zeros((num_dealers, num_ftcs), dtype=int)
        for c_idx, indices in enumerate(cluster_dealer_indices):
            j = assignment[c_idx]
            new_assignment[indices, j] = 1

        solve_time = time.time() - start_time
        changes = np.abs(new_assignment - current_assignment)
        total_changes = int(np.sum(changes))
        changed_dealers = int(np.sum(np.any(changes > 0, axis=1)))

        ftc_cnt_final = np.zeros(num_ftcs, dtype=int)
        for c_idx, j in enumerate(assignment):
            ftc_cnt_final[j] += cluster_sizes_arr[c_idx]
        ftcs_used = int(np.sum(ftc_cnt_final > 0))

        raw_obj = (
            np.sum(new_assignment * (alpha_1 * compatibility + alpha_2 * (1.0 - distance)))
            + lambda_penalty * np.sum(new_assignment * current_assignment)
        )

        # Compute distance metrics
        mean_distance_km = 0.0
        dealer_distances = []
        if distance_km is not None:
            for c_idx, indices in enumerate(cluster_dealer_indices):
                j = assignment[c_idx]
                mean_distance_km += cluster_mean_dist[c_idx, j] * cluster_sizes_arr[c_idx]
                dealer_distances.extend(distance_km[indices, j].tolist())
            mean_distance_km = mean_distance_km / cluster_sizes_arr.sum()
        median_distance_km = float(np.median(dealer_distances)) if dealer_distances else 0.0

        result = {
            'status': 'FEASIBLE',
            'assignments': new_assignment,
            'current_assignment': current_assignment,
            'objective_value': raw_obj,
            'solve_time': solve_time,
            'total_changes': total_changes,
            'changed_dealers': changed_dealers,
            'change_rate': changed_dealers / num_dealers if num_dealers > 0 else 0,
            'ftcs_used': ftcs_used,
            'mean_distance_km': mean_distance_km,
            'median_distance_km': median_distance_km,
        }

        logger.info(f"Greedy solution: {changed_dealers}/{num_dealers} dealers reassigned, "
                     f"{ftcs_used} FTCs active, "
                     f"mean dist={mean_distance_km:.2f}km, median dist={median_distance_km:.2f}km, "
                     f"objective={raw_obj:.1f}, "
                     f"kept={1-changed_dealers/num_dealers:.0%}")
        return result

    def _solve_cpsat(self,
                      compatibility, distance, current_assignment, feasibility_mask,
                      workload, capacity,
                      alpha_1, alpha_2, lambda_penalty,
                      min_dealers, max_dealers,
                      num_dealers, num_ftcs, start_time):
        """Dealer-level CP-SAT solver (smaller problems)."""
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        SCALE = 1000

        x = {}
        for i in range(num_dealers):
            for j in range(num_ftcs):
                if feasibility_mask[i, j] == 1:
                    x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

        for i in range(num_dealers):
            ftcs = [j for j in range(num_ftcs) if (i, j) in x]
            if ftcs:
                model.Add(sum(x[i, j] for j in ftcs) == 1)

        for j in range(num_ftcs):
            assigned = [i for i in range(num_dealers) if (i, j) in x]
            if not assigned:
                continue
            load = sum(x[i, j] for i in assigned)
            if min_dealers > 1:
                ftc_active = model.NewBoolVar(f'ftc_active_{j}')
                model.Add(load >= min_dealers).OnlyEnforceIf(ftc_active)
                model.Add(load <= max_dealers).OnlyEnforceIf(ftc_active)
                model.Add(load == 0).OnlyEnforceIf(ftc_active.Not())
            else:
                model.Add(load <= max_dealers)

        objective_terms = []
        for i in range(num_dealers):
            for j in range(num_ftcs):
                if (i, j) not in x:
                    continue
                c = int(alpha_1 * compatibility[i, j] * SCALE)
                if c != 0:
                    objective_terms.append(c * x[i, j])
                p = int(alpha_2 * (1.0 - distance[i, j]) * SCALE)
                if p != 0:
                    objective_terms.append(p * x[i, j])
                if current_assignment[i, j] == 1:
                    objective_terms.append(int(lambda_penalty * SCALE) * x[i, j])

        if objective_terms:
            model.Maximize(sum(objective_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_workers = self.num_workers
        try:
            solver.parameters.relative_gap_limit = float(self.optimality_gap)
        except Exception:
            pass

        for i in range(num_dealers):
            for j in range(num_ftcs):
                if (i, j) in x:
                    model.AddHint(x[i, j], int(current_assignment[i, j]))

        logger.info("Starting CP-SAT solver...")
        status = solver.Solve(model)
        solve_time = time.time() - start_time

        status_name = {
            cp_model.OPTIMAL: 'OPTIMAL',
            cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE',
            cp_model.MODEL_INVALID: 'MODEL_INVALID',
            cp_model.UNKNOWN: 'UNKNOWN'
        }.get(status, 'UNKNOWN')

        logger.info(f"Solver status: {status_name}, time: {solve_time:.2f}s")

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            new_assignment = np.zeros((num_dealers, num_ftcs), dtype=int)
            for i in range(num_dealers):
                for j in range(num_ftcs):
                    if (i, j) in x and solver.Value(x[i, j]) == 1:
                        new_assignment[i, j] = 1

            changes = np.abs(new_assignment - current_assignment)
            total_changes = int(np.sum(changes))
            changed_dealers = int(np.sum(np.any(changes > 0, axis=1)))

            return {
                'status': status_name,
                'assignments': new_assignment,
                'current_assignment': current_assignment,
                'objective_value': solver.ObjectiveValue(),
                'solve_time': solve_time,
                'total_changes': total_changes,
                'changed_dealers': changed_dealers,
                'change_rate': changed_dealers / num_dealers if num_dealers > 0 else 0,
                'ftcs_used': int(np.sum(new_assignment.sum(axis=0) > 0)),
            }

        logger.error(f"Solver failed with status: {status_name}")
        return {
            'status': status_name,
            'assignments': current_assignment.copy(),
            'objective_value': 0,
            'solve_time': solve_time,
            'total_changes': 0,
            'changed_dealers': 0,
            'change_rate': 0,
            'ftcs_used': int(np.sum(current_assignment.sum(axis=0) > 0)),
        }
