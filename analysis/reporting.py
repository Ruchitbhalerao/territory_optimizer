"""
Report generation for territory optimization results.

Generates summary reports with key metrics, territory breakdowns,
and change recommendations.
"""
import logging
from typing import Dict, Any, List
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates optimization result reports."""

    def generate_summary_report(self, pipeline_result: Dict[str, Any]) -> str:
        """
        Generate a text summary report from pipeline results.

        Args:
            pipeline_result: Result dictionary from OptimizationPipeline.run()

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("TERRITORY OPTIMIZATION REPORT")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append("")

        # Overview
        lines.append("OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"  Job ID:           {pipeline_result.get('job_id', 'N/A')}")
        lines.append(f"  Status:           {pipeline_result.get('status', 'N/A')}")
        lines.append(f"  Total Dealers:    {pipeline_result.get('num_dealers', 'N/A')}")
        lines.append(f"  Total FTCs:       {pipeline_result.get('num_ftcs', 'N/A')}")
        lines.append(f"  Clusters Created: {pipeline_result.get('num_clusters', 'N/A')}")
        lines.append(f"  Solve Time:       {pipeline_result.get('solve_time', 0):.2f}s")
        lines.append(f"  Pipeline Time:    {pipeline_result.get('total_pipeline_time', 0):.2f}s")
        lines.append("")

        # Changes
        lines.append("REASSIGNMENT SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Dealers Reassigned: {pipeline_result.get('changed_dealers', 0)}")
        lines.append(f"  Total Changes:      {pipeline_result.get('total_changes', 0)}")
        lines.append(f"  Change Rate:        {pipeline_result.get('change_rate', 0):.1%}")
        lines.append(f"  Active FTCs:        {pipeline_result.get('ftcs_used', 0)}")
        lines.append("")
        lines.append("DISTANCE METRICS")
        lines.append("-" * 40)
        lines.append(f"  Mean Distance:      {pipeline_result.get('mean_distance_km', 0):.2f} km")
        lines.append(f"  Median Distance:    {pipeline_result.get('median_distance_km', 0):.2f} km")
        lines.append("")

        # Business Impact
        impact = pipeline_result.get('business_impact', {})
        if impact and 'error' not in impact:
            lines.append("BUSINESS IMPACT")
            lines.append("-" * 40)
            lines.append(f"  Compatibility Improvement: {impact.get('compatibility_improvement', 0):.2%}")
            lines.append(f"  Distance Reduction:        {impact.get('distance_reduction', 0):.2%}")
            lines.append(f"  Workload Balance Improvement: {impact.get('workload_balance', 0):.2%}")

            disruption = impact.get('disruption_metrics', {})
            if disruption:
                lines.append(f"  Disruption Change Rate:    {disruption.get('change_rate', 0):.2%}")
            lines.append("")

        lines.append("=" * 70)

        report = "\n".join(lines)
        logger.info("Summary report generated")
        return report

    def generate_change_list(self, pipeline_result: Dict[str, Any],
                              top_n: int = 20) -> pd.DataFrame:
        """
        Generate a ranked list of dealer change recommendations.

        Args:
            pipeline_result: Pipeline result dictionary
            top_n: Number of top changes to include

        Returns:
            DataFrame with change recommendations
        """
        # This would be populated from stored DealerChange records
        # For now, return empty structure
        return pd.DataFrame(columns=[
            'dealer_id', 'from_ftc', 'to_ftc', 'impact_score'
        ])
