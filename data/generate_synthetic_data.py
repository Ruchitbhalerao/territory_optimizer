"""
Synthetic data generator for territory optimization system.

Generates realistic dealer, FTC, relationship, and proximity datasets
for development and testing purposes.
"""
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Indian metro area centers (lat, lon) with approximate spread radius
METRO_AREAS = {
    'pune': {'center': (18.5204, 73.8567), 'radius': 0.15, 'weight': 0.15},
    'mumbai': {'center': (19.0760, 72.8777), 'radius': 0.20, 'weight': 0.20},
    'bangalore': {'center': (12.9716, 77.5946), 'radius': 0.18, 'weight': 0.15},
    'delhi': {'center': (28.7041, 77.1025), 'radius': 0.20, 'weight': 0.15},
    'hyderabad': {'center': (17.3850, 78.4867), 'radius': 0.15, 'weight': 0.10},
    'chennai': {'center': (13.0827, 80.2707), 'radius': 0.15, 'weight': 0.08},
    'kolkata': {'center': (22.5726, 88.3639), 'radius': 0.15, 'weight': 0.07},
    'ahmedabad': {'center': (23.0225, 72.5714), 'radius': 0.12, 'weight': 0.05},
    'jaipur': {'center': (26.9124, 75.7873), 'radius': 0.10, 'weight': 0.03},
    'lucknow': {'center': (26.8467, 80.9462), 'radius': 0.10, 'weight': 0.02},
}

PRODUCT_GROUPS = [
    'personal_loan', 'business_loan', 'consumer_durable',
    'home_loan', 'two_wheeler', 'sme_loan'
]

DEALER_TYPES = ['static', 'mobile']


class SyntheticDataGenerator:
    """Generates realistic synthetic datasets for territory optimization."""

    def __init__(self, num_dealers: int = 5000, num_ftcs: int = 500, seed: int = 42):
        """
        Initialize synthetic data generator.

        Args:
            num_dealers: Number of dealers to generate
            num_ftcs: Number of FTCs to generate
            seed: Random seed for reproducibility
        """
        self.num_dealers = num_dealers
        self.num_ftcs = num_ftcs
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def generate_all(self, output_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Generate all synthetic datasets and save to disk.

        Args:
            output_dir: Directory to save parquet files

        Returns:
            Dictionary with all generated dataframes
        """
        logger.info(f"Generating synthetic data: {self.num_dealers} dealers, {self.num_ftcs} FTCs")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate datasets
        dealers_df = self._generate_dealers()
        ftcs_df = self._generate_ftcs(dealers_df)
        relationships_df = self._generate_relationships(dealers_df, ftcs_df)
        proximity_df = self._generate_proximity(dealers_df)

        # Save to parquet
        dealers_df.to_parquet(output_path / "dealers.parquet", index=False)
        ftcs_df.to_parquet(output_path / "ftcs.parquet", index=False)
        relationships_df.to_parquet(output_path / "relationships.parquet", index=False)
        proximity_df.to_parquet(output_path / "proximity.parquet", index=False)

        logger.info(f"Synthetic data saved to {output_path}")
        logger.info(f"  Dealers: {len(dealers_df)}")
        logger.info(f"  FTCs: {len(ftcs_df)}")
        logger.info(f"  Relationships: {len(relationships_df)}")
        logger.info(f"  Proximity records: {len(proximity_df)}")

        return {
            'dealers': dealers_df,
            'ftcs': ftcs_df,
            'relationships': relationships_df,
            'proximity': proximity_df
        }

    def _generate_dealers(self) -> pd.DataFrame:
        """Generate synthetic dealer data distributed across metro areas."""
        logger.info(f"Generating {self.num_dealers} dealers")

        dealers = []
        dealer_id_counter = 1

        for city, info in METRO_AREAS.items():
            city_count = int(self.num_dealers * info['weight'])
            center_lat, center_lon = info['center']
            radius = info['radius']

            for _ in range(city_count):
                # Generate coordinates with gaussian clustering around city center
                # Create sub-clusters within each city to simulate neighborhoods
                num_sub_clusters = np.random.randint(3, 8)
                sub_cluster_center_lat = center_lat + np.random.uniform(-radius, radius)
                sub_cluster_center_lon = center_lon + np.random.uniform(-radius, radius)

                lat = sub_cluster_center_lat + np.random.normal(0, radius * 0.15)
                lon = sub_cluster_center_lon + np.random.normal(0, radius * 0.15)

                # Assign SM (Sales Manager) - each city has ~5-10 SMs
                sm_id = f"SM_{city[:3].upper()}_{np.random.randint(1, 8):03d}"

                # Randomly assign product groups (1-3 per dealer)
                num_products = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                product_group = ','.join(random.sample(PRODUCT_GROUPS, num_products))

                dealer = {
                    'dealer_id': f"D{dealer_id_counter:06d}",
                    'sm_id': sm_id,
                    'dealer_type': np.random.choice(DEALER_TYPES, p=[0.7, 0.3]),
                    'product_group': product_group,
                    'count_bfl_disbursement': max(0, int(np.random.lognormal(3, 1.2))),
                    'average_cases_per_day': round(max(0.1, np.random.lognormal(0.5, 0.8)), 2),
                    'dealer_latitude': round(lat, 6),
                    'dealer_longitude': round(lon, 6),
                    'city': city
                }
                dealers.append(dealer)
                dealer_id_counter += 1

        # Fill remaining dealers to hit exact count
        while len(dealers) < self.num_dealers:
            city = random.choice(list(METRO_AREAS.keys()))
            info = METRO_AREAS[city]
            center_lat, center_lon = info['center']
            radius = info['radius']

            lat = center_lat + np.random.normal(0, radius * 0.5)
            lon = center_lon + np.random.normal(0, radius * 0.5)
            sm_id = f"SM_{city[:3].upper()}_{np.random.randint(1, 8):03d}"
            num_products = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            product_group = ','.join(random.sample(PRODUCT_GROUPS, num_products))

            dealer = {
                'dealer_id': f"D{dealer_id_counter:06d}",
                'sm_id': sm_id,
                'dealer_type': np.random.choice(DEALER_TYPES, p=[0.7, 0.3]),
                'product_group': product_group,
                'count_bfl_disbursement': max(0, int(np.random.lognormal(3, 1.2))),
                'average_cases_per_day': round(max(0.1, np.random.lognormal(0.5, 0.8)), 2),
                'dealer_latitude': round(lat, 6),
                'dealer_longitude': round(lon, 6),
                'city': city
            }
            dealers.append(dealer)
            dealer_id_counter += 1

        df = pd.DataFrame(dealers[:self.num_dealers])
        logger.info(f"Generated {len(df)} dealers across {len(METRO_AREAS)} cities")
        return df

    def _generate_ftcs(self, dealers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate FTCs anchored on dealer sub-clusters within each city.

        Places each FTC near a randomly chosen dealer from the same city with
        small jitter (~500m), so every FTC has dealers within walking distance
        and dense dealer areas get multiple nearby FTCs.
        """
        logger.info(f"Generating {self.num_ftcs} FTCs near dealer clusters")

        # Group dealers by city
        dealer_by_city = {}
        for _, d in dealers_df.iterrows():
            dealer_by_city.setdefault(d['city'], []).append(d)

        ftcs = []
        ftc_id_counter = 1

        for city, info in METRO_AREAS.items():
            city_count = max(1, int(self.num_ftcs * info['weight']))
            city_dealers = dealer_by_city.get(city, [])

            for _ in range(city_count):
                sm_id = f"SM_{city[:3].upper()}_{np.random.randint(1, 8):03d}"
                num_products = np.random.choice([1, 2], p=[0.6, 0.4])
                product_group = ','.join(random.sample(PRODUCT_GROUPS, num_products))

                # Anchor FTC on a random dealer from this city, jitter ~500m
                if city_dealers:
                    anchor = random.choice(city_dealers)
                    lat = anchor['dealer_latitude'] + np.random.normal(0, 0.005)
                    lon = anchor['dealer_longitude'] + np.random.normal(0, 0.005)
                else:
                    center_lat, center_lon = info['center']
                    radius = info['radius']
                    lat = center_lat + np.random.normal(0, radius * 0.2)
                    lon = center_lon + np.random.normal(0, radius * 0.2)

                ftc = {
                    'ftc_id': f"F{ftc_id_counter:05d}",
                    'sm_id': sm_id,
                    'product_group': product_group,
                    'ftc_vintage': round(max(0.1, np.random.exponential(2.5)), 1),
                    'count_bfl_disbursement': max(0, int(np.random.lognormal(4, 1.0))),
                    'average_cases_per_day': round(max(0.5, np.random.lognormal(1.0, 0.6)), 2),
                    'per_sum_mob': round(np.random.uniform(0.3, 0.95), 3),
                    'ntb_share': round(np.random.uniform(0.05, 0.85), 3),
                    'cross_sell': round(np.random.uniform(0.0, 0.6), 3),
                    'ftc_latitude': round(lat, 6),
                    'ftc_longitude': round(lon, 6),
                    'city': city
                }
                ftcs.append(ftc)
                ftc_id_counter += 1

        # Fill remaining
        while len(ftcs) < self.num_ftcs:
            city = random.choice(list(METRO_AREAS.keys()))
            city_dealers = dealer_by_city.get(city, [])
            if city_dealers:
                anchor = random.choice(city_dealers)
                lat = anchor['dealer_latitude'] + np.random.normal(0, 0.005)
                lon = anchor['dealer_longitude'] + np.random.normal(0, 0.005)
            else:
                info = METRO_AREAS[city]
                center_lat, center_lon = info['center']
                radius = info['radius']
                lat = center_lat + np.random.normal(0, radius * 0.2)
                lon = center_lon + np.random.normal(0, radius * 0.2)
            sm_id = f"SM_{city[:3].upper()}_{np.random.randint(1, 8):03d}"
            num_products = np.random.choice([1, 2], p=[0.6, 0.4])
            product_group = ','.join(random.sample(PRODUCT_GROUPS, num_products))

            ftc = {
                'ftc_id': f"F{ftc_id_counter:05d}",
                'sm_id': sm_id,
                'product_group': product_group,
                'ftc_vintage': round(max(0.1, np.random.exponential(2.5)), 1),
                'count_bfl_disbursement': max(0, int(np.random.lognormal(4, 1.0))),
                'average_cases_per_day': round(max(0.5, np.random.lognormal(1.0, 0.6)), 2),
                'per_sum_mob': round(np.random.uniform(0.3, 0.95), 3),
                'ntb_share': round(np.random.uniform(0.05, 0.85), 3),
                'cross_sell': round(np.random.uniform(0.0, 0.6), 3),
                'ftc_latitude': round(lat, 6),
                'ftc_longitude': round(lon, 6),
                'city': city
            }
            ftcs.append(ftc)
            ftc_id_counter += 1

        df = pd.DataFrame(ftcs[:self.num_ftcs])
        logger.info(f"Generated {len(df)} FTCs")
        return df

    def _generate_relationships(self, dealers_df: pd.DataFrame,
                                ftcs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate current dealer-FTC assignment relationships.

        Simulates a suboptimal manual assignment where dealers are assigned
        to FTCs somewhat randomly within the same city, with some cross-city
        misassignments to represent real-world inefficiency.
        """
        logger.info("Generating dealer-FTC relationships")

        relationships = []

        # Group FTCs by city for assignment
        ftc_by_city = {}
        for _, ftc in ftcs_df.iterrows():
            city = ftc['city']
            if city not in ftc_by_city:
                ftc_by_city[city] = []
            ftc_by_city[city].append(ftc)

        all_ftc_ids = ftcs_df['ftc_id'].tolist()

        for _, dealer in dealers_df.iterrows():
            city = dealer['city']
            dealer_products = set(dealer['product_group'].split(','))

            # 85% chance assigned to FTC in same city, 15% cross-city (inefficiency)
            if np.random.random() < 0.85 and city in ftc_by_city and len(ftc_by_city[city]) > 0:
                city_ftcs = ftc_by_city[city]
                # Prefer product-matching FTCs but don't guarantee it
                matching_ftcs = [
                    f for f in city_ftcs
                    if set(f['product_group'].split(',')) & dealer_products
                ]
                if matching_ftcs and np.random.random() < 0.7:
                    assigned_ftc = random.choice(matching_ftcs)
                else:
                    assigned_ftc = random.choice(city_ftcs)
            else:
                # Cross-city misassignment
                assigned_ftc_id = random.choice(all_ftc_ids)
                assigned_ftc = ftcs_df[ftcs_df['ftc_id'] == assigned_ftc_id].iloc[0]

            # Generate product category from dealer's products
            product_category = random.choice(list(dealer_products))

            relationship = {
                'dealer_id': dealer['dealer_id'],
                'ftc_id': assigned_ftc['ftc_id'],
                'product_category': product_category,
                'avg_cases_per_day': round(max(0.1, np.random.lognormal(0.3, 0.7)), 2)
            }
            relationships.append(relationship)

        df = pd.DataFrame(relationships)
        logger.info(f"Generated {len(df)} relationships")
        return df

    def _generate_proximity(self, dealers_df: pd.DataFrame,
                            max_distance_km: float = 50.0,
                            max_neighbors: int = 20) -> pd.DataFrame:
        """
        Generate proximity data between nearby dealers.

        Only computes pairwise distances for dealers within max_distance_km
        to avoid N² explosion.
        """
        logger.info("Generating proximity data")

        from math import radians, cos, sin, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            return 6371 * c

        proximity_records = []
        coords = dealers_df[['dealer_id', 'dealer_latitude', 'dealer_longitude', 'product_group']].values

        # For efficiency, group by approximate location grid
        # and only compute distances within same/neighboring grid cells
        grid_size = 0.5  # ~55km grid cells
        grid_map: Dict[Tuple[int, int], List] = {}

        for row in coords:
            grid_key = (int(float(row[1]) / grid_size), int(float(row[2]) / grid_size))
            if grid_key not in grid_map:
                grid_map[grid_key] = []
            grid_map[grid_key].append(row)

        for grid_key, dealers_in_cell in grid_map.items():
            # Get dealers in neighboring cells too
            neighbor_dealers = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    neighbor_key = (grid_key[0] + di, grid_key[1] + dj)
                    if neighbor_key in grid_map:
                        neighbor_dealers.extend(grid_map[neighbor_key])

            for dealer in dealers_in_cell:
                distances = []
                for neighbor in neighbor_dealers:
                    if dealer[0] == neighbor[0]:
                        continue
                    dist = haversine(
                        float(dealer[1]), float(dealer[2]),
                        float(neighbor[1]), float(neighbor[2])
                    )
                    if dist <= max_distance_km:
                        distances.append((neighbor, dist))

                # Keep only closest neighbors
                distances.sort(key=lambda x: x[1])
                for neighbor, dist in distances[:max_neighbors]:
                    proximity_records.append({
                        'dealer_id': dealer[0],
                        'related_dealer_id': neighbor[0],
                        'product_group': dealer[3],
                        'latitude': float(dealer[1]),
                        'longitude': float(dealer[2]),
                        'spatial_distance': round(dist, 3)
                    })

        df = pd.DataFrame(proximity_records)
        logger.info(f"Generated {len(df)} proximity records")
        return df


def generate_synthetic_data(num_dealers: int = 5000, num_ftcs: int = 500,
                            output_dir: str = "data", seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate synthetic data.

    Args:
        num_dealers: Number of dealers
        num_ftcs: Number of FTCs
        output_dir: Output directory
        seed: Random seed

    Returns:
        Dictionary with all generated dataframes
    """
    generator = SyntheticDataGenerator(num_dealers, num_ftcs, seed)
    return generator.generate_all(output_dir)
