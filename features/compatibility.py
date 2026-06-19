"""
Compatibility feature computation for territory optimization.
"""
import logging
from typing import Dict, Any, Set
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class CompatibilityEngineer:
    """Computes dealer-FTC compatibility features for optimization."""
    
    def __init__(self):
        """Initialize compatibility engineer."""
        self.scaler = MinMaxScaler()
    
    def compute_compatibility(self, dealers_df: pd.DataFrame, ftcs_df: pd.DataFrame, 
                            relationships_df: pd.DataFrame) -> np.ndarray:
        """
        Compute compatibility matrix between dealers and FTCs.
        
        Compatibility = 0.6 * product_match + 0.3 * historical_performance + 0.1 * dealer_type_match
        
        Args:
            dealers_df: Dealers dataframe
            ftcs_df: FTCs dataframe
            relationships_df: Relationships dataframe with historical performance
            
        Returns:
            2D array of compatibility scores [dealers x ftcs]
        """
        logger.info("Computing dealer-FTC compatibility features")
        
        try:
            num_dealers = len(dealers_df)
            num_ftcs = len(ftcs_df)
            compatibility_matrix = np.zeros((num_dealers, num_ftcs))
            
            # Create mappings for quick lookup
            dealer_product_map = dict(zip(dealers_df['dealer_id'], dealers_df['product_group']))
            ftc_product_map = dict(zip(ftcs_df['ftc_id'], ftcs_df['product_group']))
            dealer_type_map = dict(zip(dealers_df['dealer_id'], dealers_df['dealer_type']))
            
            # Create historical performance lookup
            historical_performance = {}
            for _, row in relationships_df.iterrows():
                key = (row['dealer_id'], row['ftc_id'])
                historical_performance[key] = row['avg_cases_per_day']
            
            # Normalize historical performance values
            if historical_performance:
                perf_values = list(historical_performance.values())
                perf_array = np.array(perf_values).reshape(-1, 1)
                normalized_perf = self.scaler.fit_transform(perf_array).flatten()
                normalized_performance = dict(zip(historical_performance.keys(), normalized_perf))
            else:
                normalized_performance = {}
            
            # Compute compatibility for each dealer-FTC pair
            for i, dealer_row in dealers_df.iterrows():
                dealer_id = dealer_row['dealer_id']
                dealer_products = set(dealer_row['product_group'].split(','))
                dealer_type = dealer_row['dealer_type']
                
                for j, ftc_row in ftcs_df.iterrows():
                    ftc_id = ftc_row['ftc_id']
                    ftc_products = set(ftc_row['product_group'].split(','))
                    
                    # Product match (0.6 weight)
                    product_match = 1.0 if dealer_products & ftc_products else 0.0
                    
                    # Historical performance (0.3 weight)
                    perf_key = (dealer_id, ftc_id)
                    historical_perf = normalized_performance.get(perf_key, 0.0)
                    
                    # Dealer type match (0.1 weight) - assuming static/mobile expertise
                    type_match = 1.0 if dealer_type == 'static' or ftc_products else 0.5
                    
                    # Compute weighted compatibility
                    compatibility = (0.6 * product_match + 
                                  0.3 * historical_perf + 
                                  0.1 * type_match)
                    
                    compatibility_matrix[i, j] = compatibility
            
            logger.info(f"Compatibility matrix computed: {compatibility_matrix.shape}")
            return compatibility_matrix
            
        except Exception as e:
            logger.error(f"Compatibility computation failed: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get the importance weights for compatibility components.
        
        Returns:
            Dictionary with feature names and their weights
        """
        return {
            'product_match': 0.6,
            'historical_performance': 0.3,
            'dealer_type_match': 0.1
        }
