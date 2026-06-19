"""
Tests for clustering and optimization modules.
"""
import pytest
import numpy as np
import pandas as pd

from optimization.clustering import SpatialClusterer
from optimization.solver import TerritorySolver

def test_spatial_clustering():
    dealers = pd.DataFrame({
        'dealer_id': [f"D{i}" for i in range(20)],
        'dealer_latitude': np.random.uniform(18.0, 19.0, 20),
        'dealer_longitude': np.random.uniform(72.0, 73.0, 20),
        'count_bfl_disbursement': np.random.randint(1, 10, 20),
        'average_cases_per_day': np.random.uniform(0.5, 5.0, 20)
    })
    
    config = {
        'min_cluster_size': 2,
        'max_cluster_size': 10,
        'cluster_ratio': 1.0
    }
    
    clusterer = SpatialClusterer(config)
    num_ftcs = 4
    labels = clusterer.fit_predict(dealers, num_ftcs)
    
    assert len(labels) == 20
    
    # Check max cluster size
    sizes = np.bincount(labels)
    assert (sizes <= config['max_cluster_size']).all()


def test_territory_solver():
    config = {
        'time_limit_seconds': 5,
        'num_workers': 1,
        'optimality_gap': 0.05
    }
    
    num_dealers = 10
    num_ftcs = 3
    
    compatibility = np.random.rand(num_dealers, num_ftcs)
    distance = np.random.rand(num_dealers, num_ftcs)
    current_assignment = np.zeros((num_dealers, num_ftcs), dtype=int)
    for i in range(num_dealers):
        current_assignment[i, np.random.randint(num_ftcs)] = 1
        
    feasibility_mask = np.ones((num_dealers, num_ftcs), dtype=int)
    workload = np.random.rand(num_dealers)
    capacity = np.random.rand(num_ftcs) * 5
    
    solver = TerritorySolver(config)
    result = solver.solve(
        compatibility=compatibility,
        distance=distance,
        current_assignment=current_assignment,
        feasibility_mask=feasibility_mask,
        workload=workload,
        capacity=capacity,
        min_dealers=1,
        max_dealers=5
    )
    
    assert result['status'] in ('OPTIMAL', 'FEASIBLE')
    assert result['assignments'].shape == (num_dealers, num_ftcs)
    
    # Check assignment constraints: each dealer assigned to exactly one FTC
    assert (result['assignments'].sum(axis=1) == 1).all()
