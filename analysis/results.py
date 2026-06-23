"""
Results analysis for territory optimization system.
"""
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ResultAnalyzer:
    """Analyzes optimization results and computes business impact metrics."""
    
    def __init__(self):
        """Initialize result analyzer."""
        pass
    
    def analyze_business_impact(self, solution: Dict[str, Any], features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze business impact of the optimization solution.
        
        Args:
            solution: Optimization solution dictionary
            features: Processed features dictionary
            
        Returns:
            Dictionary with business impact metrics
        """
        logger.info("Analyzing business impact of optimization solution")
        
        try:
            # Extract solution components
            new_assignments = solution.get('assignments', np.array([]))
            current_assignments = features.get('assignment_matrix', np.array([]))
            compatibility = features.get('compatibility', np.array([]))
            distance = features.get('distance', np.array([]))
            workload = features.get('workload', np.array([]))
            capacity = features.get('capacity', np.array([]))
            
            if new_assignments.size == 0 or current_assignments.size == 0:
                raise ValueError("Missing required assignment matrices")
            
            num_dealers, num_ftcs = new_assignments.shape
            
            # Compute business impact metrics
            impact_metrics = {
                'compatibility_improvement': self._compute_compatibility_improvement(
                    current_assignments, new_assignments, compatibility
                ),
                'distance_reduction': self._compute_distance_reduction(
                    current_assignments, new_assignments, distance
                ),
                'workload_balance': self._compute_workload_balance_improvement(
                    current_assignments, new_assignments, workload, capacity
                ),
                'disruption_metrics': self._compute_disruption_metrics(
                    current_assignments, new_assignments
                )
            }
            
            logger.info("Business impact analysis completed")
            return impact_metrics
            
        except Exception as e:
            logger.error(f"Business impact analysis failed: {e}")
            raise
    
    def _compute_compatibility_improvement(self, current: np.ndarray, new: np.ndarray, 
                                         compatibility: np.ndarray) -> float:
        """
        Compute improvement in compatibility scores.
        
        Args:
            current: Current assignment matrix
            new: New assignment matrix
            compatibility: Compatibility matrix
            
        Returns:
            Compatibility improvement ratio
        """
        if compatibility.size == 0:
            return 0.0
            
        current_score = np.sum(current * compatibility)
        new_score = np.sum(new * compatibility)
        
        if current_score == 0:
            return float('inf') if new_score > 0 else 0.0
            
        return (new_score - current_score) / current_score
    
    def _compute_distance_reduction(self, current: np.ndarray, new: np.ndarray, 
                                  distance: np.ndarray) -> float:
        """
        Compute reduction in travel distance.
        
        Args:
            current: Current assignment matrix
            new: New assignment matrix
            distance: Distance matrix
            
        Returns:
            Distance reduction ratio
        """
        if distance.size == 0:
            return 0.0
            
        current_distance = np.sum(current * distance)
        new_distance = np.sum(new * distance)
        
        if current_distance == 0:
            return 0.0
            
        return (current_distance - new_distance) / current_distance
    
    def _compute_workload_balance_improvement(self, current: np.ndarray, new: np.ndarray,
                                            workload: np.ndarray, capacity: np.ndarray) -> float:
        """
        Compute improvement in workload balance.
        
        Args:
            current: Current assignment matrix
            new: New assignment matrix
            workload: Dealer workload array
            capacity: FTC capacity array
            
        Returns:
            Workload balance improvement score
        """
        if workload.size == 0 or capacity.size == 0:
            return 0.0
            
        # Compute current workload distribution
        current_workload = np.sum(current.T * workload, axis=1)
        current_utilization = current_workload / np.maximum(capacity, 1e-8)
        current_balance = np.std(current_utilization)
        
        # Compute new workload distribution
        new_workload = np.sum(new.T * workload, axis=1)
        new_utilization = new_workload / np.maximum(capacity, 1e-8)
        new_balance = np.std(new_utilization)
        
        if current_balance == 0:
            return 0.0
            
        return (current_balance - new_balance) / current_balance
    
    def _compute_disruption_metrics(self, current: np.ndarray, new: np.ndarray) -> Dict[str, Any]:
        """
        Compute disruption metrics from assignment changes.
        
        Args:
            current: Current assignment matrix
            new: New assignment matrix
            
        Returns:
            Dictionary with disruption metrics
        """
        # Compute change matrix
        changes = np.abs(current - new)
        total_changes = np.sum(changes)
        
        return {
            'total_changes': int(total_changes),
            'change_rate': float(total_changes / current.shape[0]),
            'changed_dealers': int(np.sum(np.sum(changes, axis=1) > 0))
        }
