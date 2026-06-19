"""
Capacity feature computation for territory optimization.
"""
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class CapacityEngineer:
    """Computes FTC capacity features for optimization."""
    
    def __init__(self):
        """Initialize capacity engineer."""
        self.scaler = MinMaxScaler()
    
    def compute_capacity(self, ftcs_df: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized capacity for each FTC.
        
        Capacity = 0.4 * normalized(per_sum_mob) + 0.3 * normalized(ntb_share) 
                 + 0.2 * normalized(cross_sell) + 0.1 * normalized(ftc_vintage)
        
        Args:
            ftcs_df: FTCs dataframe with required columns
            
        Returns:
            Array of normalized capacity values for each FTC
        """
        logger.info("Computing FTC capacity features")
        
        try:
            # Extract required columns
            per_sum_mob = ftcs_df['per_sum_mob'].values.reshape(-1, 1)
            ntb_share = ftcs_df['ntb_share'].values.reshape(-1, 1)
            cross_sell = ftcs_df['cross_sell'].values.reshape(-1, 1)
            ftc_vintage = ftcs_df['ftc_vintage'].values.reshape(-1, 1)
            
            # Normalize features
            normalized_per_sum_mob = self.scaler.fit_transform(per_sum_mob).flatten()
            normalized_ntb_share = self.scaler.fit_transform(ntb_share).flatten()
            normalized_cross_sell = self.scaler.fit_transform(cross_sell).flatten()
            normalized_ftc_vintage = self.scaler.fit_transform(ftc_vintage).flatten()
            
            # Compute weighted capacity
            capacity = (0.4 * normalized_per_sum_mob + 
                       0.3 * normalized_ntb_share + 
                       0.2 * normalized_cross_sell + 
                       0.1 * normalized_ftc_vintage)
            
            logger.info(f"Capacity computation completed for {len(capacity)} FTCs")
            return capacity
            
        except Exception as e:
            logger.error(f"Capacity computation failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance weights for capacity components.
        
        Returns:
            Dictionary with feature names and their weights
        """
        return {
            'per_sum_mob': 0.4,
            'ntb_share': 0.3,
            'cross_sell': 0.2,
            'ftc_vintage': 0.1
        }
