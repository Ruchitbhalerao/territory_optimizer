"""
Configuration management for territory optimization system.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages application configuration loading and validation."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file. Uses default if None.
        """
        self.config_path = config_path or "config/parameters.json"
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from JSON file."""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self.config = self._get_default_config()
                return

            with open(config_file, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "data_paths": {
                "dealers": "data/dealers.parquet",
                "ftcs": "data/ftcs.parquet",
                "relationships": "data/relationships.parquet",
                "proximity": "data/proximity.parquet"
            },
            "optimization": {
                "alpha_1": 0.5,
                "alpha_2": 0.3,
                "lambda": 1.0,
                "time_limit": 3600,
                "optimality_gap": 0.01
            },
            "clustering": {
                "method": "kmeans",
                "min_cluster_size": 5,
                "max_cluster_size": 25,
                "cluster_ratio": 1.0
            },
            "solver": {
                "time_limit_seconds": 300,
                "num_workers": 4,
                "optimality_gap": 0.05
            },
            "scheduler": {
                "enabled": True,
                "cron_hour": 2,
                "cron_minute": 0,
                "max_concurrent_jobs": 1
            },
            "constraints": {
                "max_dealers_per_ftc": 25,
                "min_dealers_per_ftc": 3,
                "max_travel_radius_km": 50,
                "workload_balance_threshold": 0.3
            },
            "validation": {
                "dealer_type_values": ["static", "mobile"],
                "latitude_range": [-90, 90],
                "longitude_range": [-180, 180],
                "ntb_share_range": [0, 1]
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notation key.

        Args:
            key: Configuration key (e.g., 'optimization.alpha_1')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_data_path(self, dataset: str) -> str:
        """Get path for specific dataset."""
        return self.get(f"data_paths.{dataset}", f"data/{dataset}.parquet")

    def get_optimization_param(self, param: str) -> float:
        """Get optimization parameter value."""
        return self.get(f"optimization.{param}", 0.0)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.get(section, {})


# Global configuration instance
config_manager = ConfigManager()
