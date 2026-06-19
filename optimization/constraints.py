"""
Constraint definitions for territory optimization MILP model.
"""
import logging
from typing import Dict, Any, List
import numpy as np

logger = logging.getLogger(__name__)


class OptimizationConstraints:
    """Defines and manages constraints for territory optimization."""

    def __init__(self, config: Dict[str, Any]):
        self.max_dealers_per_ftc = config.get('max_dealers_per_ftc', 25)
        self.min_dealers_per_ftc = config.get('min_dealers_per_ftc', 3)
        self.max_travel_radius_km = config.get('max_travel_radius_km', 50)
        self.workload_balance_threshold = config.get('workload_balance_threshold', 0.3)

    def get_assignment_bounds(self) -> Dict[str, int]:
        """Get min/max dealers per FTC bounds."""
        return {
            'min_dealers': self.min_dealers_per_ftc,
            'max_dealers': self.max_dealers_per_ftc
        }

    def compute_feasibility_mask(self, distance_matrix: np.ndarray) -> np.ndarray:
        """
        Compute binary mask of feasible dealer-FTC assignments based on distance.

        Args:
            distance_matrix: Distance in km between each dealer and FTC [D x F]

        Returns:
            Binary mask where 1 = feasible assignment
        """
        # Allow assignment only if within max travel radius
        # distance_matrix values are in km (or normalized — caller should ensure km)
        mask = (distance_matrix <= self.max_travel_radius_km).astype(int)

        # Ensure every dealer has at least one feasible FTC
        for i in range(mask.shape[0]):
            if mask[i].sum() == 0:
                # Assign to closest FTC regardless of distance
                closest = np.argmin(distance_matrix[i])
                mask[i, closest] = 1

        logger.info(f"Feasibility mask: {mask.sum()} / {mask.size} assignments feasible "
                     f"({100 * mask.sum() / mask.size:.1f}%)")
        return mask

    def compute_product_compatibility_mask(self, dealers_df, ftcs_df) -> np.ndarray:
        """
        Compute binary mask of product-compatible dealer-FTC pairs.

        Args:
            dealers_df: Dealers dataframe with product_group column
            ftcs_df: FTCs dataframe with product_group column

        Returns:
            Binary mask where 1 = product compatible
        """
        num_dealers = len(dealers_df)
        num_ftcs = len(ftcs_df)
        mask = np.zeros((num_dealers, num_ftcs), dtype=int)

        dealer_products = [set(p.split(',')) for p in dealers_df['product_group'].values]
        ftc_products = [set(p.split(',')) for p in ftcs_df['product_group'].values]

        for i, dp in enumerate(dealer_products):
            for j, fp in enumerate(ftc_products):
                if dp & fp:  # Any product overlap
                    mask[i, j] = 1

        # Ensure every dealer has at least one compatible FTC
        for i in range(num_dealers):
            if mask[i].sum() == 0:
                mask[i, :] = 1  # Allow all if no match found

        logger.info(f"Product compatibility: {mask.sum()} / {mask.size} pairs compatible "
                     f"({100 * mask.sum() / mask.size:.1f}%)")
        return mask

    def validate_solution(self, assignment_matrix: np.ndarray,
                          distance_matrix_km: np.ndarray = None) -> Dict[str, Any]:
        """
        Validate a solution against all constraints.

        Returns:
            Dictionary with validation results
        """
        results = {'valid': True, 'violations': []}

        # Check each dealer assigned to exactly one FTC
        dealer_counts = assignment_matrix.sum(axis=1)
        unassigned = (dealer_counts == 0).sum()
        multi = (dealer_counts > 1).sum()
        if unassigned > 0:
            results['violations'].append(f"{unassigned} dealers unassigned")
            results['valid'] = False
        if multi > 0:
            results['violations'].append(f"{multi} dealers multi-assigned")
            results['valid'] = False

        # Check FTC load bounds
        ftc_loads = assignment_matrix.sum(axis=0)
        active_ftcs = ftc_loads[ftc_loads > 0]
        overloaded = (active_ftcs > self.max_dealers_per_ftc).sum()
        underloaded = (active_ftcs < self.min_dealers_per_ftc).sum()
        if overloaded > 0:
            results['violations'].append(f"{overloaded} FTCs overloaded (>{self.max_dealers_per_ftc})")
        if underloaded > 0:
            results['violations'].append(f"{underloaded} FTCs underloaded (<{self.min_dealers_per_ftc})")

        # Check distance constraint
        if distance_matrix_km is not None:
            for i in range(assignment_matrix.shape[0]):
                assigned_ftcs = np.where(assignment_matrix[i] == 1)[0]
                for j in assigned_ftcs:
                    if distance_matrix_km[i, j] > self.max_travel_radius_km:
                        results['violations'].append(
                            f"Dealer {i} -> FTC {j}: {distance_matrix_km[i, j]:.1f}km > {self.max_travel_radius_km}km"
                        )

        if results['violations']:
            results['valid'] = False

        return results
