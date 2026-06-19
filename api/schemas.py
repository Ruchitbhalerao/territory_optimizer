"""
API schemas for territory optimization system.
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class OptimizationParameters(BaseModel):
    """Optimization parameter schema."""
    alpha_1: float = Field(default=0.5, gt=0, description="Compatibility weight")
    alpha_2: float = Field(default=0.3, gt=0, description="Distance weight")
    lambda_param: float = Field(default=1.0, alias="lambda", gt=0, description="Disruption penalty")
    time_limit: int = Field(default=3600, gt=0, description="Solver time limit in seconds")
    optimality_gap: float = Field(default=0.01, ge=0, le=1, description="Solver optimality gap")

class OptimizationFilters(BaseModel):
    """Optimization filter schema."""
    region: Optional[str] = Field(default=None, description="Region filter")
    supervisor: Optional[str] = Field(default=None, description="Supervisor filter")
    dealer_type: Optional[str] = Field(default=None, description="Dealer type filter")

class OptimizationRequest(BaseModel):
    """Optimization request schema."""
    parameters: OptimizationParameters = Field(default_factory=OptimizationParameters)
    filters: OptimizationFilters = Field(default_factory=OptimizationFilters)

class BusinessImpact(BaseModel):
    """Business impact metrics schema."""
    compatibility_improvement: float
    distance_reduction: float
    workload_balance: float

class DisruptionMetrics(BaseModel):
    """Disruption metrics schema."""
    total_changes: int
    change_rate: float
    changed_dealers: int

class OptimizationResponse(BaseModel):
    """Optimization response schema."""
    status: str
    solution_id: str
    business_impact: BusinessImpact
    disruption_metrics: DisruptionMetrics
    execution_time: float
    timestamp: str

class DealerChange(BaseModel):
    """Dealer change recommendation schema."""
    dealer_id: str
    ftc_id: str
    change_indicator: bool
    impact_score: float

class SolutionDetails(BaseModel):
    """Solution details schema."""
    solution_id: str
    dealer_assignments: List[DealerChange]
    summary: Dict[str, Any]
