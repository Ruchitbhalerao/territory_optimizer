"""
Tests for feature engineering modules.
"""
import pytest
import pandas as pd
import numpy as np

from features.workload import WorkloadEngineer
from features.capacity import CapacityEngineer
from features.compatibility import CompatibilityEngineer

def test_workload_engineer():
    dealers = pd.DataFrame({
        'dealer_id': ['D1', 'D2'],
        'average_cases_per_day': [2.0, 5.0],
        'count_bfl_disbursement': [10, 50]
    })
    
    engineer = WorkloadEngineer()
    workload = engineer.compute_workload(dealers)
    
    assert len(workload) == 2
    assert isinstance(workload, np.ndarray)
    assert (workload >= 0).all() and (workload <= 1).all()


def test_capacity_engineer():
    ftcs = pd.DataFrame({
        'ftc_id': ['F1', 'F2'],
        'per_sum_mob': [0.8, 0.9],
        'ntb_share': [0.5, 0.6],
        'cross_sell': [0.2, 0.3],
        'ftc_vintage': [2.0, 3.0]
    })
    
    engineer = CapacityEngineer()
    capacity = engineer.compute_capacity(ftcs)
    
    assert len(capacity) == 2
    assert isinstance(capacity, np.ndarray)
    assert (capacity >= 0).all() and (capacity <= 1).all()


def test_compatibility_engineer():
    dealers = pd.DataFrame({
        'dealer_id': ['D1', 'D2'],
        'product_group': ['personal_loan', 'business_loan'],
        'dealer_type': ['static', 'mobile']
    })
    
    ftcs = pd.DataFrame({
        'ftc_id': ['F1', 'F2'],
        'product_group': ['personal_loan', 'personal_loan,business_loan']
    })
    
    relationships = pd.DataFrame({
        'dealer_id': ['D1', 'D2'],
        'ftc_id': ['F1', 'F2'],
        'avg_cases_per_day': [2.0, 5.0]
    })
    
    engineer = CompatibilityEngineer()
    compatibility = engineer.compute_compatibility(dealers, ftcs, relationships)
    
    assert compatibility.shape == (2, 2)
    assert isinstance(compatibility, np.ndarray)
    assert (compatibility >= 0).all() and (compatibility <= 1).all()
    # D2 has business loan, F1 does not, should have lower compatibility than D2 and F2
    assert compatibility[1, 0] <= compatibility[1, 1]
