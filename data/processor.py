"""
Data processing module for territory optimization system.
"""
import logging
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """Processes raw data into optimization-ready features and structures."""
    
    def __init__(self):
        """Initialize data processor."""
        self.dealer_index_map: Dict[str, int] = {}
        self.ftc_index_map: Dict[str, int] = {}
        self.dealer_ids: List[str] = []
        self.ftc_ids: List[str] = []
    
    def process_all_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Process all data into optimization-ready format.
        
        Args:
            data: Dictionary containing all raw dataframes
            
        Returns:
            Dictionary with processed arrays and mappings
        """
        logger.info("Processing all data for optimization")
        
        try:
            # Create index mappings
            self._create_index_mappings(data['dealers'], data['ftcs'])
            
            # Normalize product groups
            normalized_data = self._normalize_product_groups(data)
            
            # Create assignment matrix
            assignment_matrix = self._create_assignment_matrix(
                normalized_data['relationships'], 
                len(self.dealer_ids), 
                len(self.ftc_ids)
            )
            
            # Process features
            processed_data = {
                'dealer_index_map': self.dealer_index_map,
                'ftc_index_map': self.ftc_index_map,
                'dealer_ids': self.dealer_ids,
                'ftc_ids': self.ftc_ids,
                'assignment_matrix': assignment_matrix,
                'dealers': normalized_data['dealers'],
                'ftcs': normalized_data['ftcs'],
                'relationships': normalized_data['relationships'],
                'proximity': normalized_data['proximity']
            }
            
            logger.info("Data processing completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise
    
    def _create_index_mappings(self, dealers_df: pd.DataFrame, ftcs_df: pd.DataFrame) -> None:
        """
        Create index mappings for dealers and FTCs.
        
        Args:
            dealers_df: Dealers dataframe
            ftcs_df: FTCs dataframe
        """
        logger.info("Creating dealer and FTC index mappings")
        
        # Create dealer index mapping
        self.dealer_ids = sorted(dealers_df['dealer_id'].unique())
        self.dealer_index_map = {dealer_id: idx for idx, dealer_id in enumerate(self.dealer_ids)}
        
        # Create FTC index mapping
        self.ftc_ids = sorted(ftcs_df['ftc_id'].unique())
        self.ftc_index_map = {ftc_id: idx for idx, ftc_id in enumerate(self.ftc_ids)}
        
        logger.info(f"Created mappings for {len(self.dealer_ids)} dealers and {len(self.ftc_ids)} FTCs")
    
    def _normalize_product_groups(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Normalize product group representations across all datasets.
        
        Args:
            data: Dictionary containing all dataframes
            
        Returns:
            Dictionary with normalized dataframes
        """
        logger.info("Normalizing product group representations")
        
        normalized_data = {}
        
        # Normalize case for all product group fields
        for key, df in data.items():
            normalized_df = df.copy()
            
            # Normalize product group columns if they exist
            product_group_columns = [col for col in df.columns if 'product_group' in col.lower()]
            for col in product_group_columns:
                if col in normalized_df.columns:
                    normalized_df[col] = normalized_df[col].str.lower().str.strip()
            
            normalized_data[key] = normalized_df
        
        return normalized_data
    
    def _create_assignment_matrix(self, relationships_df: pd.DataFrame, 
                                num_dealers: int, num_ftcs: int) -> np.ndarray:
        """
        Create current assignment matrix x0 from relationships data.
        
        Args:
            relationships_df: Relationships dataframe
            num_dealers: Number of dealers
            num_ftcs: Number of FTCs
            
        Returns:
            Binary assignment matrix where x0[i,j] = 1 if dealer i assigned to FTC j
        """
        logger.info("Creating current assignment matrix")
        
        # Initialize zero matrix
        assignment_matrix = np.zeros((num_dealers, num_ftcs), dtype=int)
        
        # Fill in current assignments
        for _, row in relationships_df.iterrows():
            dealer_id = row['dealer_id']
            ftc_id = row['ftc_id']
            
            # Get indices from mappings
            dealer_idx = self.dealer_index_map.get(dealer_id)
            ftc_idx = self.ftc_index_map.get(ftc_id)
            
            # Only process if both IDs exist in our mappings
            if dealer_idx is not None and ftc_idx is not None:
                assignment_matrix[dealer_idx, ftc_idx] = 1
            else:
                logger.warning(f"Skipping assignment for dealer {dealer_id} and FTC {ftc_id} - not in mappings")
        
        # Validate that each dealer has exactly one assignment
        dealer_assignments = assignment_matrix.sum(axis=1)
        unassigned_dealers = np.sum(dealer_assignments == 0)
        multi_assigned_dealers = np.sum(dealer_assignments > 1)
        
        if unassigned_dealers > 0:
            logger.warning(f"{unassigned_dealers} dealers have no current assignment")
        
        if multi_assigned_dealers > 0:
            logger.warning(f"{multi_assigned_dealers} dealers have multiple assignments")
        
        logger.info(f"Assignment matrix created: {assignment_matrix.shape}")
        return assignment_matrix
    
    def get_dealer_index(self, dealer_id: str) -> int:
        """
        Get index for a dealer ID.
        
        Args:
            dealer_id: Dealer identifier
            
        Returns:
            Index of dealer in optimization model
            
        Raises:
            KeyError: If dealer ID not found
        """
        return self.dealer_index_map[dealer_id]
    
    def get_ftc_index(self, ftc_id: str) -> int:
        """
        Get index for an FTC ID.
        
        Args:
            ftc_id: FTC identifier
            
        Returns:
            Index of FTC in optimization model
            
        Raises:
            KeyError: If FTC ID not found
        """
        return self.ftc_index_map[ftc_id]
    
    def get_dealer_id(self, dealer_idx: int) -> str:
        """
        Get dealer ID for an index.
        
        Args:
            dealer_idx: Dealer index
            
        Returns:
            Dealer identifier
            
        Raises:
            IndexError: If index out of range
        """
        if dealer_idx < 0 or dealer_idx >= len(self.dealer_ids):
            raise IndexError(f"Dealer index {dealer_idx} out of range")
        return self.dealer_ids[dealer_idx]
    
    def get_ftc_id(self, ftc_idx: int) -> str:
        """
        Get FTC ID for an index.
        
        Args:
            ftc_idx: FTC index
            
        Returns:
            FTC identifier
            
        Raises:
            IndexError: If index out of range
        """
        if ftc_idx < 0 or ftc_idx >= len(self.ftc_ids):
            raise IndexError(f"FTC index {ftc_idx} out of range")
        return self.ftc_ids[ftc_idx]
