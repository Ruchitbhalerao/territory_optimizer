"""
Workload feature computation for territory optimization.
"""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class WorkloadEngineer:
    """Computes dealer workload features for optimization."""
    
    def __init__(self):
        """Initialize workload engineer."""
        self.scaler = MinMaxScaler()
    
    def compute_workload(self, dealers_df: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized workload for each dealer.
        
        Workload = 0.7 * normalized(average_cases_per_day) + 0.3 * normalized(count_bfl_disbursement)
        
        Args:
            dealers_df: Dealers dataframe with required columns
            
        Returns:
            Array of normalized workload values for each dealer
        """
        logger.info("Computing dealer workload features")
        
        try:
            # Extract required columns
            cases = dealers_df['average_cases_per_day'].values.reshape(-1, 1)
            disbursements = dealers_df['count_bfl_disbursement'].values.reshape(-1, 1)
            
            # Normalize features
            normalized_cases = self.scaler.fit_transform(cases).flatten()
            normalized_disbursements = self.scaler.fit_transform(disbursements).flatten()
            
            # Compute weighted workload
            workload = 0.7 * normalized_cases + 0.3 * normalized_disbursements
            
            logger.info(f"Workload computation completed for {len(workload)} dealers")
            return workload
            
        except Exception as e:
            logger.error(f"Workload computation failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance weights for workload components.
        
        Returns:
            Dictionary with feature names and their weights
        """
        return {
            'average_cases_per_day': 0.7,
            'count_bfl_disbursement': 0.3
        }
