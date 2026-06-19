"""
Data loading module for territory optimization system.
"""
import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from config.settings import config_manager

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading of all required data sources for territory optimization."""
    
    def __init__(self):
        """Initialize data loader with configuration."""
        self.config = config_manager
    
    def load_dealers(self) -> pd.DataFrame:
        """
        Load dealers data from configured source.
        
        Returns:
            DataFrame with dealers data
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data format is invalid
        """
        file_path = self.config.get_data_path("dealers")
        logger.info(f"Loading dealers data from {file_path}")
        
        try:
            df = self._load_dataframe(file_path)
            
            # Ensure required columns exist
            required_columns = [
                'dealer_id', 'sm_id', 'dealer_type', 'product_group',
                'count_bfl_disbursement', 'average_cases_per_day',
                'dealer_latitude', 'dealer_longitude'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in dealers data: {missing_columns}")
            
            logger.info(f"Successfully loaded {len(df)} dealers")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load dealers data: {e}")
            raise
    
    def load_ftcs(self) -> pd.DataFrame:
        """
        Load FTCs data from configured source.
        
        Returns:
            DataFrame with FTCs data
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data format is invalid
        """
        file_path = self.config.get_data_path("ftcs")
        logger.info(f"Loading FTCs data from {file_path}")
        
        try:
            df = self._load_dataframe(file_path)
            
            # Ensure required columns exist
            required_columns = [
                'ftc_id', 'sm_id', 'product_group', 'ftc_vintage',
                'count_bfl_disbursement', 'average_cases_per_day',
                'per_sum_mob', 'ntb_share', 'cross_sell'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in FTCs data: {missing_columns}")
            
            logger.info(f"Successfully loaded {len(df)} FTCs")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load FTCs data: {e}")
            raise
    
    def load_relationships(self) -> pd.DataFrame:
        """
        Load FTC-dealer relationships data from configured source.
        
        Returns:
            DataFrame with relationships data
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data format is invalid
        """
        file_path = self.config.get_data_path("relationships")
        logger.info(f"Loading relationships data from {file_path}")
        
        try:
            df = self._load_dataframe(file_path)
            
            # Ensure required columns exist
            required_columns = [
                'dealer_id', 'ftc_id', 'product_category', 'avg_cases_per_day'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in relationships data: {missing_columns}")
            
            logger.info(f"Successfully loaded {len(df)} relationships")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load relationships data: {e}")
            raise
    
    def load_proximity(self) -> pd.DataFrame:
        """
        Load dealer-to-dealer proximity data from configured source.
        
        Returns:
            DataFrame with proximity data
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data format is invalid
        """
        file_path = self.config.get_data_path("proximity")
        logger.info(f"Loading proximity data from {file_path}")
        
        try:
            df = self._load_dataframe(file_path)
            
            # Ensure required columns exist
            required_columns = [
                'dealer_id', 'related_dealer_id', 'product_group',
                'latitude', 'longitude', 'spatial_distance'
            ]
            
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in proximity data: {missing_columns}")
            
            logger.info(f"Successfully loaded {len(df)} proximity records")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load proximity data: {e}")
            raise
    
    def _load_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Load DataFrame from file, supporting both CSV and Parquet formats.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file not found
            ValueError: If file format not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        if path.suffix.lower() == '.parquet':
            return pd.read_parquet(path)
        elif path.suffix.lower() == '.csv':
            return pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Supported formats: .csv, .parquet")


# Convenience functions
def load_all_data() -> dict:
    """
    Load all required data sources.
    
    Returns:
        Dictionary with all loaded dataframes
    """
    loader = DataLoader()
    return {
        'dealers': loader.load_dealers(),
        'ftcs': loader.load_ftcs(),
        'relationships': loader.load_relationships(),
        'proximity': loader.load_proximity()
    }
