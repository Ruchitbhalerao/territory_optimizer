"""
Database models for territory optimization system.
"""
import logging
from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass, asdict
from database.connection import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class OptimizationJob:
    """Optimization job model."""
    job_id: str
    status: str
    parameters: Dict[str, Any]
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class Solution:
    """Optimization solution model."""
    solution_id: str
    job_id: str
    business_impact: Dict[str, Any]
    disruption_metrics: Dict[str, Any]
    created_at: Optional[str] = None


@dataclass
class DealerChange:
    """Dealer change recommendation model."""
    solution_id: str
    dealer_id: str
    from_ftc_id: str
    to_ftc_id: str
    impact_score: float


class OptimizationJobModel:
    """Model for optimization job operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def create_job(self, job: OptimizationJob) -> bool:
        """Create new optimization job."""
        try:
            query = """
                INSERT INTO optimization_jobs 
                (job_id, status, parameters) 
                VALUES (?, ?, ?)
            """
            params = (
                job.job_id,
                job.status,
                json.dumps(job.parameters)
            )
            
            self.db.execute_update(query, params)
            logger.info(f"Created optimization job: {job.job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create job {job.job_id}: {e}")
            return False
    
    def update_job_status(self, job_id: str, status: str, completed_at: Optional[str] = None) -> bool:
        """Update job status."""
        try:
            if completed_at:
                query = """
                    UPDATE optimization_jobs 
                    SET status = ?, completed_at = ? 
                    WHERE job_id = ?
                """
                params = (status, completed_at, job_id)
            else:
                query = """
                    UPDATE optimization_jobs 
                    SET status = ? 
                    WHERE job_id = ?
                """
                params = (status, job_id)
            
            rows_affected = self.db.execute_update(query, params)
            logger.info(f"Updated job {job_id} status to {status}")
            return rows_affected > 0
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            return False
    
    def get_job(self, job_id: str) -> Optional[OptimizationJob]:
        """Get job by ID."""
        try:
            query = "SELECT * FROM optimization_jobs WHERE job_id = ?"
            results = self.db.execute_query(query, (job_id,))
            
            if not results:
                return None
            
            row = results[0]
            return OptimizationJob(
                job_id=row['job_id'],
                status=row['status'],
                parameters=json.loads(row['parameters']),
                created_at=row['created_at'],
                completed_at=row['completed_at']
            )
            
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None


class SolutionModel:
    """Model for solution operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def create_solution(self, solution: Solution) -> bool:
        """Create new solution."""
        try:
            query = """
                INSERT INTO solutions 
                (solution_id, job_id, business_impact, disruption_metrics) 
                VALUES (?, ?, ?, ?)
            """
            params = (
                solution.solution_id,
                solution.job_id,
                json.dumps(solution.business_impact),
                json.dumps(solution.disruption_metrics)
            )
            
            self.db.execute_update(query, params)
            logger.info(f"Created solution: {solution.solution_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create solution {solution.solution_id}: {e}")
            return False
    
    def get_solution(self, solution_id: str) -> Optional[Solution]:
        """Get solution by ID."""
        try:
            query = "SELECT * FROM solutions WHERE solution_id = ?"
            results = self.db.execute_query(query, (solution_id,))
            
            if not results:
                return None
            
            row = results[0]
            return Solution(
                solution_id=row['solution_id'],
                job_id=row['job_id'],
                business_impact=json.loads(row['business_impact']),
                disruption_metrics=json.loads(row['disruption_metrics']),
                created_at=row['created_at']
            )
            
        except Exception as e:
            logger.error(f"Failed to get solution {solution_id}: {e}")
            return None


class DealerChangeModel:
    """Model for dealer change operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
    
    def create_changes(self, changes: List[DealerChange]) -> bool:
        """Create multiple dealer changes."""
        try:
            query = """
                INSERT INTO dealer_changes 
                (solution_id, dealer_id, from_ftc_id, to_ftc_id, impact_score) 
                VALUES (?, ?, ?, ?, ?)
            """
            
            for change in changes:
                params = (
                    change.solution_id,
                    change.dealer_id,
                    change.from_ftc_id,
                    change.to_ftc_id,
                    change.impact_score
                )
                self.db.execute_update(query, params)
            
            logger.info(f"Created {len(changes)} dealer changes")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create dealer changes: {e}")
            return False
    
    def get_changes_by_solution(self, solution_id: str) -> List[DealerChange]:
        """Get all changes for a solution."""
        try:
            query = """
                SELECT solution_id, dealer_id, from_ftc_id, to_ftc_id, impact_score 
                FROM dealer_changes 
                WHERE solution_id = ?
                ORDER BY impact_score DESC
            """
            results = self.db.execute_query(query, (solution_id,))
            
            changes = []
            for row in results:
                changes.append(DealerChange(
                    solution_id=row['solution_id'],
                    dealer_id=row['dealer_id'],
                    from_ftc_id=row['from_ftc_id'],
                    to_ftc_id=row['to_ftc_id'],
                    impact_score=row['impact_score']
                ))
            
            logger.info(f"Retrieved {len(changes)} changes for solution {solution_id}")
            return changes
            
        except Exception as e:
            logger.error(f"Failed to get changes for solution {solution_id}: {e}")
            return []
