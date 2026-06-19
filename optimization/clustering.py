"""
Spatial clustering module for territory optimization.

Creates micro-location clusters from dealer geographic coordinates
using constrained K-Means clustering.
"""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SpatialClusterer:
    """Creates optimal micro-location clusters from dealer locations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.min_cluster_size = self.config.get('min_cluster_size', 5)
        self.max_cluster_size = self.config.get('max_cluster_size', 25)
        self.cluster_ratio = self.config.get('cluster_ratio', 1.0)
        self.labels_ = None
        self.centroids_ = None
        self.cluster_stats_ = None

    def fit_predict(self, dealers_df: pd.DataFrame, num_ftcs: int) -> np.ndarray:
        """Cluster dealers into micro-locations."""
        logger.info(f"Clustering {len(dealers_df)} dealers into micro-locations")
        n_clusters = max(1, min(int(num_ftcs * self.cluster_ratio), len(dealers_df)))
        coords = dealers_df[['dealer_latitude', 'dealer_longitude']].values
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)

        if len(dealers_df) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024, n_init=5)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)

        labels = kmeans.fit_predict(coords_scaled)
        labels = self._enforce_size_constraints(coords_scaled, labels, n_clusters)
        self.labels_ = labels
        self.centroids_ = self._compute_centroids(coords, labels)
        self.cluster_stats_ = self._compute_cluster_stats(dealers_df, labels)

        if 1 < len(set(labels)) < len(labels):
            try:
                sample_n = min(5000, len(labels))
                idx = np.random.choice(len(labels), sample_n, replace=False)
                score = silhouette_score(coords_scaled[idx], labels[idx])
                logger.info(f"Silhouette score: {score:.3f}")
            except Exception:
                pass

        logger.info(f"Created {len(set(labels))} micro-location clusters")
        return labels

    def _enforce_size_constraints(self, coords, labels, n_clusters):
        """Merge tiny clusters, split oversized ones."""
        labels = labels.copy()
        for _ in range(10):
            sizes = np.bincount(labels, minlength=n_clusters)
            changes = 0
            # Merge tiny
            for c in np.where((sizes > 0) & (sizes < self.min_cluster_size))[0]:
                for idx in np.where(labels == c)[0]:
                    best_c, best_d = -1, float('inf')
                    for c2 in range(n_clusters):
                        if c2 == c or not (labels == c2).any():
                            continue
                        d = np.linalg.norm(coords[idx] - coords[labels == c2].mean(axis=0))
                        if d < best_d:
                            best_d, best_c = d, c2
                    if best_c >= 0:
                        labels[idx] = best_c
                        changes += 1
            # Split oversized
            sizes = np.bincount(labels, minlength=n_clusters)
            for c in np.where(sizes > self.max_cluster_size)[0]:
                idxs = np.where(labels == c)[0]
                empty = np.where(sizes == 0)[0]
                if len(empty) == 0 or len(idxs) <= self.max_cluster_size:
                    continue
                n_sub = min(len(empty) + 1, max(2, len(idxs) // self.max_cluster_size + 1))
                sub_labels = KMeans(n_clusters=n_sub, random_state=42, n_init=3).fit_predict(coords[idxs])
                for si in range(1, min(n_sub, len(empty) + 1)):
                    labels[idxs[sub_labels == si]] = empty[si - 1]
                    changes += (sub_labels == si).sum()
                sizes = np.bincount(labels, minlength=n_clusters)
            if changes == 0:
                break
        return labels

    def _compute_centroids(self, coords, labels):
        unique = np.unique(labels)
        centroids = np.zeros((len(unique), 2))
        for i, l in enumerate(unique):
            centroids[i] = coords[labels == l].mean(axis=0)
        return centroids

    def _compute_cluster_stats(self, dealers_df, labels):
        df = dealers_df.copy()
        df['cluster'] = labels
        return df.groupby('cluster').agg(
            num_dealers=('dealer_id', 'count'),
            avg_lat=('dealer_latitude', 'mean'),
            avg_lon=('dealer_longitude', 'mean'),
            total_disbursements=('count_bfl_disbursement', 'sum'),
            avg_cases=('average_cases_per_day', 'mean'),
        ).reset_index()

    def get_cluster_assignments(self):
        return self.labels_

    def get_centroids(self):
        return self.centroids_

    def get_cluster_stats(self):
        return self.cluster_stats_
