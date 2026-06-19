"""
End-to-end pipeline orchestrator for territory optimization.

Coordinates: Load → Validate → Process → Cluster → Features → Solve → Analyze → Store
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from config.settings import ConfigManager
from data.loader import DataLoader
from data.validator import DataValidator
from data.processor import DataProcessor
from features.workload import WorkloadEngineer
from features.capacity import CapacityEngineer
from features.compatibility import CompatibilityEngineer
from features.distance import DistanceEngineer
from optimization.clustering import SpatialClusterer
from optimization.model import TerritoryModel
from analysis.results import ResultAnalyzer
from analysis.reporting import ReportGenerator
from database.connection import DatabaseConnection
from database.models import OptimizationJob, Solution, DealerChange
from database.models import OptimizationJobModel, SolutionModel, DealerChangeModel

logger = logging.getLogger(__name__)


class OptimizationPipeline:
    """Orchestrates the full territory optimization pipeline."""

    def __init__(self, config: Optional[ConfigManager] = None):
        """
        Initialize pipeline with configuration.

        Args:
            config: ConfigManager instance. Uses default if None.
        """
        self.config = config or ConfigManager()
        self.db = DatabaseConnection()
        self.db.initialize_schema()

    def run(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the full optimization pipeline.

        Args:
            parameters: Optional override parameters for this run

        Returns:
            Dictionary with complete pipeline results
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        start_time = time.time()

        logger.info(f"Starting optimization pipeline - Job {job_id}")

        # Merge parameters with defaults
        run_config = self._build_run_config(parameters)

        # Create job record
        job_model = OptimizationJobModel(self.db)
        job = OptimizationJob(
            job_id=job_id,
            status='running',
            parameters=run_config.get('optimization', {})
        )
        job_model.create_job(job)

        try:
            # Step 1: Load data
            logger.info("[1/7] Loading data...")
            loader = DataLoader()
            data = {
                'dealers': loader.load_dealers(),
                'ftcs': loader.load_ftcs(),
                'relationships': loader.load_relationships(),
                'proximity': loader.load_proximity()
            }

            # Step 2: Validate data
            logger.info("[2/7] Validating data...")
            validator = DataValidator()
            # Log warnings but don't stop pipeline for validation issues
            validation_result = validator.validate_all_data(data)
            if not validation_result:
                logger.warning("Data validation has warnings - continuing with available data")

            # Step 3: Process data
            logger.info("[3/7] Processing data...")
            processor = DataProcessor()
            processed = processor.process_all_data(data)

            # Step 4: Cluster dealers into micro-locations
            logger.info("[4/7] Clustering dealers into micro-locations...")
            clusterer = SpatialClusterer(run_config.get('clustering', {}))
            cluster_labels = clusterer.fit_predict(
                processed['dealers'],
                len(processed['ftc_ids'])
            )
            processed['cluster_labels'] = cluster_labels
            processed['cluster_stats'] = clusterer.get_cluster_stats()

            # Step 5: Engineer features
            logger.info("[5/7] Engineering features...")
            features = self._engineer_features(processed)

            # Step 6: Optimize
            logger.info("[6/7] Running optimization solver...")
            model = TerritoryModel(run_config)
            solution = model.build_and_solve(features)

            # Step 7: Analyze results
            logger.info("[7/7] Analyzing results...")
            analyzer = ResultAnalyzer()
            if solution['status'] in ('OPTIMAL', 'FEASIBLE'):
                impact = analyzer.analyze_business_impact(solution, features)
            else:
                impact = {'error': f"Solver returned {solution['status']}"}

            # Store results
            self._store_results(job_id, solution, impact, processor)

            # Update job status
            elapsed = time.time() - start_time
            job_model.update_job_status(
                job_id, 'completed',
                datetime.now().isoformat()
            )

            result = {
                'job_id': job_id,
                'status': solution['status'],
                'solve_time': solution['solve_time'],
                'total_pipeline_time': elapsed,
                'total_changes': solution['total_changes'],
                'changed_dealers': solution['changed_dealers'],
                'change_rate': solution['change_rate'],
                'ftcs_used': solution['ftcs_used'],
                'num_dealers': len(processed['dealer_ids']),
                'num_ftcs': len(processed['ftc_ids']),
                'num_clusters': len(set(cluster_labels)),
                'mean_distance_km': solution.get('mean_distance_km', 0),
                'median_distance_km': solution.get('median_distance_km', 0),
                'business_impact': impact,
                'model_summary': model.get_model_summary(),
                'cluster_stats': clusterer.get_cluster_stats().to_dict() if clusterer.get_cluster_stats() is not None else {},
            }

            logger.info(f"Pipeline completed in {elapsed:.2f}s - {solution['status']}")
            logger.info(f"  Changes: {solution['changed_dealers']}/{len(processed['dealer_ids'])} dealers")
            logger.info(f"  Active FTCs: {solution['ftcs_used']}/{len(processed['ftc_ids'])}")

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            job_model.update_job_status(job_id, 'failed', datetime.now().isoformat())
            return {
                'job_id': job_id,
                'status': 'FAILED',
                'error': str(e),
                'total_pipeline_time': time.time() - start_time
            }

    def _build_run_config(self, parameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Build run configuration by merging defaults with overrides."""
        config = {
            'optimization': {
                'alpha_1': self.config.get('optimization.alpha_1', 0.5),
                'alpha_2': self.config.get('optimization.alpha_2', 0.3),
                'lambda': self.config.get('optimization.lambda', 1.0),
            },
            'clustering': {
                'min_cluster_size': self.config.get('clustering.min_cluster_size', 5),
                'max_cluster_size': self.config.get('clustering.max_cluster_size', 25),
                'cluster_ratio': self.config.get('clustering.cluster_ratio', 1.0),
            },
            'solver': {
                'time_limit_seconds': self.config.get('solver.time_limit_seconds', 300),
                'num_workers': self.config.get('solver.num_workers', 4),
                'optimality_gap': self.config.get('solver.optimality_gap', 0.05),
            },
            'constraints': {
                'max_dealers_per_ftc': self.config.get('constraints.max_dealers_per_ftc', 25),
                'min_dealers_per_ftc': self.config.get('constraints.min_dealers_per_ftc', 3),
                'max_travel_radius_km': self.config.get('constraints.max_travel_radius_km', 50),
                'workload_balance_threshold': self.config.get('constraints.workload_balance_threshold', 0.3),
            }
        }

        if parameters:
            for key, value in parameters.items():
                if key in config:
                    config[key].update(value)
                elif '.' in key:
                    section, param = key.split('.', 1)
                    if section in config:
                        config[section][param] = value

        return config

    def _engineer_features(self, processed: Dict) -> Dict[str, Any]:
        """Run all feature engineering on processed data."""
        dealers_df = processed['dealers']
        ftcs_df = processed['ftcs']
        relationships_df = processed['relationships']
        proximity_df = processed['proximity']

        # Workload
        workload_eng = WorkloadEngineer()
        workload = workload_eng.compute_workload(dealers_df)

        # Capacity
        capacity_eng = CapacityEngineer()
        capacity = capacity_eng.compute_capacity(ftcs_df)

        # Compatibility
        compat_eng = CompatibilityEngineer()
        compatibility = compat_eng.compute_compatibility(dealers_df, ftcs_df, relationships_df)

        # Distance
        dist_eng = DistanceEngineer()
        distance = dist_eng.compute_distance(dealers_df, ftcs_df, proximity_df)

        # Keep both normalized and raw distance values. The solver objective uses the
        # normalized matrix, but feasibility filtering must use real kilometer values.
        distance_km = dist_eng.get_raw_distance_matrix()

        features = {
            'assignment_matrix': processed['assignment_matrix'],
            'workload': workload,
            'capacity': capacity,
            'compatibility': compatibility,
            'distance': distance,
            'distance_km': distance_km,
            'dealers': dealers_df,
            'ftcs': ftcs_df,
            'dealer_ids': processed['dealer_ids'],
            'ftc_ids': processed['ftc_ids'],
            'cluster_labels': processed['cluster_labels'],
        }

        return features

    def _store_results(self, job_id: str, solution: Dict, impact: Dict,
                       processor: DataProcessor):
        """Store optimization results in database."""
        try:
            solution_id = f"sol_{uuid.uuid4().hex[:12]}"
            sol_model = SolutionModel(self.db)
            sol = Solution(
                solution_id=solution_id,
                job_id=job_id,
                business_impact=impact if isinstance(impact, dict) else {},
                disruption_metrics={
                    'total_changes': solution.get('total_changes', 0),
                    'changed_dealers': solution.get('changed_dealers', 0),
                    'change_rate': solution.get('change_rate', 0),
                }
            )
            sol_model.create_solution(sol)

            # Store individual dealer changes
            if solution['status'] in ('OPTIMAL', 'FEASIBLE'):
                assignments = solution['assignments']
                current = solution.get('current_assignment',
                                       np.zeros_like(assignments))

                changes = []
                for i in range(assignments.shape[0]):
                    old_ftc = np.where(current[i] == 1)[0]
                    new_ftc = np.where(assignments[i] == 1)[0]

                    if len(old_ftc) > 0 and len(new_ftc) > 0:
                        if old_ftc[0] != new_ftc[0]:
                            dealer_id = processor.get_dealer_id(i) if i < len(processor.dealer_ids) else f"D{i}"
                            from_ftc = processor.get_ftc_id(old_ftc[0]) if old_ftc[0] < len(processor.ftc_ids) else f"F{old_ftc[0]}"
                            to_ftc = processor.get_ftc_id(new_ftc[0]) if new_ftc[0] < len(processor.ftc_ids) else f"F{new_ftc[0]}"

                            changes.append(DealerChange(
                                solution_id=solution_id,
                                dealer_id=dealer_id,
                                from_ftc_id=from_ftc,
                                to_ftc_id=to_ftc,
                                impact_score=0.0
                            ))

                if changes:
                    change_model = DealerChangeModel(self.db)
                    change_model.create_changes(changes)

            logger.info(f"Results stored: solution {solution_id}")

        except Exception as e:
            logger.error(f"Failed to store results: {e}")
