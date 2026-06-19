"""
Data validation module for territory optimization system.
"""
import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from config.settings import config_manager

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data quality and integrity for territory optimization."""
    
    def __init__(self):
        """Initialize data validator with configuration."""
        self.config = config_manager
        self.validation_errors: List[str] = []
    
    def validate_all_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate all data sources for completeness and integrity.
        
        Args:
            data: Dictionary containing all dataframes
            
        Returns:
            True if all validations pass, False otherwise
        """
        self.validation_errors = []
        
        try:
            # Validate individual datasets
            self.validate_dealers(data['dealers'])
            self.validate_ftcs(data['ftcs'])
            self.validate_relationships(data['relationships'])
            self.validate_proximity(data['proximity'])
            
            # Validate cross-dataset relationships
            self.validate_foreign_keys(data)
            self.validate_assignment_completeness(data)
            
            if self.validation_errors:
                logger.error(f"Validation failed with {len(self.validation_errors)} errors:")
                for error in self.validation_errors:
                    logger.error(f"  - {error}")
                return False
            
            logger.info("All data validation checks passed")
            return True
            
        except Exception as e:
            logger.error(f"Validation process failed: {e}")
            return False
    
    def validate_dealers(self, dealers_df: pd.DataFrame) -> None:
        """
        Validate dealers dataframe schema and data quality.
        
        Args:
            dealers_df: Dealers dataframe to validate
        """
        logger.info("Validating dealers data")
        
        # Check for null values
        null_counts = dealers_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        if not null_columns.empty:
            for col, count in null_columns.items():
                self.validation_errors.append(f"Dealers: {count} null values in column '{col}'")
        
        # Check for duplicates
        duplicate_dealers = dealers_df.duplicated(subset=['dealer_id']).sum()
        if duplicate_dealers > 0:
            self.validation_errors.append(f"Dealers: {duplicate_dealers} duplicate dealer IDs")
        
        # Validate dealer types
        valid_types = set(self.config.get("validation.dealer_type_values", ["static", "mobile"]))
        invalid_types = set(dealers_df['dealer_type'].unique()) - valid_types
        if invalid_types:
            self.validation_errors.append(f"Dealers: Invalid dealer types {invalid_types}")
        
        # Validate geographic coordinates
        lat_range = self.config.get("validation.latitude_range", [-90, 90])
        lon_range = self.config.get("validation.longitude_range", [-180, 180])
        
        invalid_lat = dealers_df[
            (dealers_df['dealer_latitude'] < lat_range[0]) | 
            (dealers_df['dealer_latitude'] > lat_range[1])
        ]
        if not invalid_lat.empty:
            self.validation_errors.append(f"Dealers: {len(invalid_lat)} invalid latitude values")
        
        invalid_lon = dealers_df[
            (dealers_df['dealer_longitude'] < lon_range[0]) | 
            (dealers_df['dealer_longitude'] > lon_range[1])
        ]
        if not invalid_lon.empty:
            self.validation_errors.append(f"Dealers: {len(invalid_lon)} invalid longitude values")
        
        # Validate numeric fields
        numeric_fields = ['count_bfl_disbursement', 'average_cases_per_day']
        for field in numeric_fields:
            if (dealers_df[field] < 0).any():
                negative_count = (dealers_df[field] < 0).sum()
                self.validation_errors.append(f"Dealers: {negative_count} negative values in '{field}'")
    
    def validate_ftcs(self, ftcs_df: pd.DataFrame) -> None:
        """
        Validate FTCs dataframe schema and data quality.
        
        Args:
            ftcs_df: FTCs dataframe to validate
        """
        logger.info("Validating FTCs data")
        
        # Check for null values
        null_counts = ftcs_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        if not null_columns.empty:
            for col, count in null_columns.items():
                self.validation_errors.append(f"FTCs: {count} null values in column '{col}'")
        
        # Check for duplicates
        duplicate_ftcs = ftcs_df.duplicated(subset=['ftc_id']).sum()
        if duplicate_ftcs > 0:
            self.validation_errors.append(f"FTCs: {duplicate_ftcs} duplicate FTC IDs")
        
        # Validate numeric ranges
        ntb_range = self.config.get("validation.ntb_share_range", [0, 1])
        invalid_ntb = ftcs_df[
            (ftcs_df['ntb_share'] < ntb_range[0]) | 
            (ftcs_df['ntb_share'] > ntb_range[1])
        ]
        if not invalid_ntb.empty:
            self.validation_errors.append(f"FTCs: {len(invalid_ntb)} invalid NTB share values")
        
        # Validate non-negative fields
        non_negative_fields = [
            'ftc_vintage', 'count_bfl_disbursement', 
            'average_cases_per_day', 'per_sum_mob', 'cross_sell'
        ]
        for field in non_negative_fields:
            if (ftcs_df[field] < 0).any():
                negative_count = (ftcs_df[field] < 0).sum()
                self.validation_errors.append(f"FTCs: {negative_count} negative values in '{field}'")
    
    def validate_relationships(self, relationships_df: pd.DataFrame) -> None:
        """
        Validate relationships dataframe schema and data quality.
        
        Args:
            relationships_df: Relationships dataframe to validate
        """
        logger.info("Validating relationships data")
        
        # Check for null values
        null_counts = relationships_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        if not null_columns.empty:
            for col, count in null_columns.items():
                self.validation_errors.append(f"Relationships: {count} null values in column '{col}'")
        
        # Check for duplicates
        duplicate_relationships = relationships_df.duplicated(subset=['dealer_id', 'ftc_id']).sum()
        if duplicate_relationships > 0:
            self.validation_errors.append(f"Relationships: {duplicate_relationships} duplicate relationships")
        
        # Validate non-negative fields
        if (relationships_df['avg_cases_per_day'] < 0).any():
            negative_count = (relationships_df['avg_cases_per_day'] < 0).sum()
            self.validation_errors.append(f"Relationships: {negative_count} negative avg_cases_per_day values")
    
    def validate_proximity(self, proximity_df: pd.DataFrame) -> None:
        """
        Validate proximity dataframe schema and data quality.
        
        Args:
            proximity_df: Proximity dataframe to validate
        """
        logger.info("Validating proximity data")
        
        # Check for null values
        null_counts = proximity_df.isnull().sum()
        null_columns = null_counts[null_counts > 0]
        if not null_columns.empty:
            for col, count in null_columns.items():
                self.validation_errors.append(f"Proximity: {count} null values in column '{col}'")
        
        # Check for duplicates
        duplicate_proximity = proximity_df.duplicated(subset=['dealer_id', 'related_dealer_id']).sum()
        if duplicate_proximity > 0:
            self.validation_errors.append(f"Proximity: {duplicate_proximity} duplicate proximity records")
        
        # Validate geographic coordinates
        lat_range = self.config.get("validation.latitude_range", [-90, 90])
        lon_range = self.config.get("validation.longitude_range", [-180, 180])
        
        invalid_lat = proximity_df[
            (proximity_df['latitude'] < lat_range[0]) | 
            (proximity_df['latitude'] > lat_range[1])
        ]
        if not invalid_lat.empty:
            self.validation_errors.append(f"Proximity: {len(invalid_lat)} invalid latitude values")
        
        invalid_lon = proximity_df[
            (proximity_df['longitude'] < lon_range[0]) | 
            (proximity_df['longitude'] > lon_range[1])
        ]
        if not invalid_lon.empty:
            self.validation_errors.append(f"Proximity: {len(invalid_lon)} invalid longitude values")
        
        # Validate non-negative distance
        if (proximity_df['spatial_distance'] < 0).any():
            negative_count = (proximity_df['spatial_distance'] < 0).sum()
            self.validation_errors.append(f"Proximity: {negative_count} negative spatial_distance values")
    
    def validate_foreign_keys(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate foreign key relationships between datasets.
        
        Args:
            data: Dictionary containing all dataframes
        """
        logger.info("Validating foreign key relationships")
        
        dealers_df = data['dealers']
        ftcs_df = data['ftcs']
        relationships_df = data['relationships']
        
        # Validate dealer IDs in relationships
        dealer_ids = set(dealers_df['dealer_id'].unique())
        relationship_dealer_ids = set(relationships_df['dealer_id'].unique())
        missing_dealers = relationship_dealer_ids - dealer_ids
        if missing_dealers:
            self.validation_errors.append(f"Relationships: {len(missing_dealers)} dealer IDs not found in dealers data")
        
        # Validate FTC IDs in relationships
        ftc_ids = set(ftcs_df['ftc_id'].unique())
        relationship_ftc_ids = set(relationships_df['ftc_id'].unique())
        missing_ftcs = relationship_ftc_ids - ftc_ids
        if missing_ftcs:
            self.validation_errors.append(f"Relationships: {len(missing_ftcs)} FTC IDs not found in FTCs data")
        
        # Validate SM IDs consistency
        dealer_sm_ids = set(dealers_df['sm_id'].unique())
        ftc_sm_ids = set(ftcs_df['sm_id'].unique())
        # Note: This may not be an error - just logging for awareness
        logger.info(f"Dealer SM IDs: {len(dealer_sm_ids)}, FTC SM IDs: {len(ftc_sm_ids)}")
    
    def validate_assignment_completeness(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Validate that all dealers have current assignments.
        
        Args:
            data: Dictionary containing all dataframes
        """
        logger.info("Validating assignment completeness")
        
        dealers_df = data['dealers']
        relationships_df = data['relationships']
        
        # Check if all dealers have assignments
        assigned_dealers = set(relationships_df['dealer_id'].unique())
        all_dealers = set(dealers_df['dealer_id'].unique())
        unassigned_dealers = all_dealers - assigned_dealers
        
        if unassigned_dealers:
            self.validation_errors.append(f"Assignment completeness: {len(unassigned_dealers)} dealers have no current assignment")
        
        # Check for dealers with multiple assignments
        dealer_assignment_counts = relationships_df['dealer_id'].value_counts()
        multiple_assignments = dealer_assignment_counts[dealer_assignment_counts > 1]
        
        if not multiple_assignments.empty:
            self.validation_errors.append(f"Assignment completeness: {len(multiple_assignments)} dealers have multiple assignments")
