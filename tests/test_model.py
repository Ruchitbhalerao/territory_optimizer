"""
Tests for full model integration.
"""
import pytest
import numpy as np

from optimization.model import TerritoryModel

def test_territory_model_integration():
    config = {
        'optimization': {
            'alpha_1': 0.5,
            'alpha_2': 0.3,
            'lambda': 1.0
        },
        'solver': {
            'time_limit_seconds': 5,
            'num_workers': 1
        },
        'constraints': {
            'min_dealers_per_ftc': 1,
            'max_dealers_per_ftc': 5,
            'max_travel_radius_km': 100
        }
    }
    
    num_dealers = 10
    num_ftcs = 3
    
    # Create fake current assignment
    current_assignment = np.zeros((num_dealers, num_ftcs), dtype=int)
    for i in range(num_dealers):
        current_assignment[i, np.random.randint(num_ftcs)] = 1
        
    features = {
        'assignment_matrix': current_assignment,
        'workload': np.random.rand(num_dealers),
        'capacity': np.random.rand(num_ftcs),
        'compatibility': np.random.rand(num_dealers, num_ftcs),
        'distance': np.random.rand(num_dealers, num_ftcs),
        'distance_km': np.random.rand(num_dealers, num_ftcs) * 50  # all within 100km
    }
    
    model = TerritoryModel(config)
    result = model.build_and_solve(features)
    
    assert result['status'] in ('OPTIMAL', 'FEASIBLE')
    assert 'validation' in result
    assert result['validation']['valid'] is True


def test_model_preserves_feasible_assignment_options_after_combining_masks():
    import numpy as np
    import pandas as pd
    from optimization.model import TerritoryModel

    config = {
        'optimization': {'alpha_1': 0.5, 'alpha_2': 0.3, 'lambda': 1.0},
        'solver': {'time_limit_seconds': 5, 'num_workers': 1, 'optimality_gap': 0.05},
        'constraints': {'min_dealers_per_ftc': 1, 'max_dealers_per_ftc': 2, 'max_travel_radius_km': 50}
    }

    model = TerritoryModel(config)

    assignment_matrix = np.array([[1, 0], [0, 1]])
    compatibility = np.array([[0.9, 0.1], [0.8, 0.2]])
    distance = np.array([[0.1, 0.9], [0.2, 0.8]])
    distance_km = np.array([[10.0, 100.0], [10.0, 100.0]])

    dealers = pd.DataFrame({
        'dealer_id': ['D1', 'D2'],
        'product_group': ['a', 'b'],
        'dealer_type': ['static', 'static']
    })
    ftcs = pd.DataFrame({
        'ftc_id': ['F1', 'F2'],
        'product_group': ['b', 'a']
    })

    result = model.build_and_solve({
        'assignment_matrix': assignment_matrix,
        'workload': np.array([1.0, 1.0]),
        'capacity': np.array([2.0, 2.0]),
        'compatibility': compatibility,
        'distance': distance,
        'distance_km': distance_km,
        'dealers': dealers,
        'ftcs': ftcs
    })

    assert result['status'] in ('OPTIMAL', 'FEASIBLE')
    assert (result['assignments'].sum(axis=1) == 1).all()
