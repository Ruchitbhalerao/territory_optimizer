"""
Main entry point for territory optimization system.

Provides CLI commands for data generation, optimization runs,
and starting the API server.
"""
import logging
import argparse
import sys
from pathlib import Path

from config.settings import config_manager
from data.generate_synthetic_data import generate_synthetic_data
from pipeline import OptimizationPipeline
from analysis.reporting import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_generate(args):
    """Run synthetic data generation."""
    logger.info(f"Generating synthetic data: {args.dealers} dealers, {args.ftcs} FTCs")
    generate_synthetic_data(
        num_dealers=args.dealers,
        num_ftcs=args.ftcs,
        output_dir=args.output,
        seed=args.seed
    )
    logger.info("Data generation complete.")
    return True


def run_optimize(args):
    """Run one-shot optimization pipeline."""
    logger.info("Starting one-shot optimization run")
    
    pipeline = OptimizationPipeline(config_manager)
    
    # Allow overriding parameters from command line
    params = {}
    if args.time_limit:
        params['solver.time_limit_seconds'] = args.time_limit
        
    result = pipeline.run(parameters=params)
    
    if result['status'] in ('OPTIMAL', 'FEASIBLE'):
        logger.info(f"Optimization successful! Job ID: {result['job_id']}")
        
        # Optionally generate report
        if args.report:
            report_gen = ReportGenerator()
            report = report_gen.generate_summary_report(result)
            
            report_path = Path("optimization_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {report_path}")
            
        return True
    else:
        logger.error(f"Optimization failed with status: {result['status']}")
        if 'error' in result:
            logger.error(f"Error details: {result['error']}")
        return False


def run_server(args):
    """Start API server and scheduler."""
    logger.info("Starting territory optimizer server")
    
    # Start scheduler if enabled
    from scheduler import OptimizationScheduler
    scheduler_config = config_manager.get_section('scheduler')
    
    pipeline = OptimizationPipeline(config_manager)
    scheduler = OptimizationScheduler(scheduler_config, pipeline_runner=pipeline.run)
    
    if scheduler_config.get('enabled', True):
        scheduler.start()
        
    # Start Flask API
    from api.server import run_server as start_flask
    
    try:
        start_flask(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("Server stopping...")
    finally:
        scheduler.stop()
        
    return True


def main():
    """Parse arguments and execute command."""
    parser = argparse.ArgumentParser(description="Territory Optimizer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Generate command
    parser_gen = subparsers.add_parser("generate", help="Generate synthetic data")
    parser_gen.add_argument("--dealers", type=int, default=5000, help="Number of dealers")
    parser_gen.add_argument("--ftcs", type=int, default=500, help="Number of FTCs")
    parser_gen.add_argument("--output", type=str, default="data", help="Output directory")
    parser_gen.add_argument("--seed", type=int, default=42, help="Random seed")
    parser_gen.set_defaults(func=run_generate)
    
    # Optimize command
    parser_opt = subparsers.add_parser("optimize", help="Run one-shot optimization")
    parser_opt.add_argument("--time-limit", type=int, help="Solver time limit in seconds")
    parser_opt.add_argument("--report", action="store_true", help="Generate text report")
    parser_opt.set_defaults(func=run_optimize)
    
    # Server command
    parser_srv = subparsers.add_parser("serve", help="Start API server and scheduler")
    parser_srv.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser_srv.add_argument("--port", type=int, default=5000, help="Bind port")
    parser_srv.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser_srv.set_defaults(func=run_server)
    
    args = parser.parse_args()
    success = args.func(args)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
