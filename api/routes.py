"""
API routes for territory optimization system.
"""
import logging
from typing import Dict, Any
from flask import Blueprint, request, jsonify
from functools import wraps

from config.settings import config_manager
from pipeline import OptimizationPipeline
from database.connection import DatabaseConnection
from database.models import OptimizationJobModel, SolutionModel, DealerChangeModel
from scheduler import OptimizationScheduler
from data.generate_synthetic_data import generate_synthetic_data
from data.loader import DataLoader
from data.uploader import ExcelUploader

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize components
db = DatabaseConnection()
pipeline = OptimizationPipeline(config_manager)
scheduler = OptimizationScheduler(config_manager.get_section('scheduler'), pipeline.run)

# We will start the scheduler from the main entry point, but we need the instance here.

def require_db(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # We ensure db schema is initialized in the pipeline, but good practice to check
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Endpoint error: {e}")
            return jsonify({'error': str(e)}), 500
    return decorated_function


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'territory-optimizer'
    })


@api_bp.route('/optimize', methods=['POST'])
@require_db
def optimize_territories():
    """
    Run territory optimization.
    
    Expected JSON body:
    {
        "parameters": {
            "optimization.alpha_1": 0.5,
            "optimization.alpha_2": 0.3,
            "optimization.lambda": 1.0,
            "solver.time_limit_seconds": 300
        }
    }
    """
    try:
        data = request.get_json() or {}
        parameters = data.get('parameters', {})
        
        # Trigger an immediate run using the scheduler (this can run synchronously or background depending on implementation)
        # For a production API, this should queue a task (e.g. Celery). Here we run it directly or trigger the scheduler.
        # Since this could take time, we might want to return 202 Accepted and run it in a thread.
        # For simplicity, we'll run it synchronously here and return the result.
        logger.info(f"Optimization request received with parameters: {parameters}")
        
        # In a real async scenario, we'd spawn a thread and return job ID.
        result = pipeline.run(parameters)
        
        if result['status'] == 'FAILED':
            return jsonify({
                'status': 'error',
                'message': result.get('error', 'Optimization failed'),
                'job_id': result.get('job_id')
            }), 400
            
        return jsonify({
            'status': 'success',
            'message': 'Optimization completed',
            'job_id': result['job_id'],
            'result': {
                'status': result['status'],
                'solve_time': result['solve_time'],
                'changes': result['total_changes'],
                'business_impact': result['business_impact']
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Optimization request failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@api_bp.route('/solution/<job_id>', methods=['GET'])
@require_db
def get_solution(job_id):
    """Get optimization solution by job ID."""
    try:
        sol_model = SolutionModel(db)
        # We need to query solutions by job_id. We'll fetch the first one.
        query = "SELECT solution_id FROM solutions WHERE job_id = ?"
        results = db.execute_query(query, (job_id,))
        
        if not results:
             return jsonify({'status': 'error', 'message': f'No solution found for job {job_id}'}), 404
             
        solution_id = results[0]['solution_id']
        solution = sol_model.get_solution(solution_id)
        
        if not solution:
            return jsonify({'status': 'error', 'message': 'Solution data missing'}), 404
            
        response = {
            'job_id': job_id,
            'solution_id': solution.solution_id,
            'status': 'completed',
            'created_at': solution.created_at,
            'business_impact': solution.business_impact,
            'disruption_metrics': solution.disruption_metrics
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Solution request failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/solution/<job_id>/changes', methods=['GET'])
@require_db
def get_changes(job_id):
    """Get detailed change recommendations for a solution."""
    try:
        query = "SELECT solution_id FROM solutions WHERE job_id = ?"
        results = db.execute_query(query, (job_id,))
        
        if not results:
             return jsonify({'status': 'error', 'message': f'No solution found for job {job_id}'}), 404
             
        solution_id = results[0]['solution_id']
        change_model = DealerChangeModel(db)
        changes = change_model.get_changes_by_solution(solution_id)
        
        change_list = [
            {
                'dealer_id': c.dealer_id,
                'from_ftc': c.from_ftc_id,
                'to_ftc': c.to_ftc_id,
                'impact_score': c.impact_score
            } for c in changes
        ]
        
        response = {
            'job_id': job_id,
            'solution_id': solution_id,
            'total_changes': len(change_list),
            'changes': change_list
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Changes request failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/scheduler/status', methods=['GET'])
def get_scheduler_status():
    """Get the current status of the optimization scheduler."""
    return jsonify(scheduler.get_status())

@api_bp.route('/upload', methods=['POST'])
@require_db
def upload_data():
    """Upload Excel file with 3 sheets (Dealers, FTCs, F2D relationships)."""
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400

        import tempfile
        import os
        suffix = os.path.splitext(file.filename)[1] or '.xlsx'
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            file.save(tmp.name)
            output_dir = config_manager.get_section('data').get('output_dir', 'data')
            uploader = ExcelUploader(output_dir=output_dir)
            stats = uploader.upload(tmp.name)
        finally:
            os.unlink(tmp.name)

        return jsonify({
            'status': 'success',
            'message': 'Data uploaded and processed successfully',
            'stats': stats
        }), 200

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@api_bp.route('/generate', methods=['POST'])
def generate_data():
    """Generate synthetic data."""
    try:
        data = request.get_json() or {}
        dealers = int(data.get('dealers', 1000))
        ftcs = int(data.get('ftcs', 100))
        output_dir = config_manager.get_section('data').get('output_dir', 'data')
        generate_synthetic_data(num_dealers=dealers, num_ftcs=ftcs, output_dir=output_dir)
        return jsonify({'status': 'success', 'message': 'Data generated successfully'}), 200
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/data/dealers', methods=['GET'])
def get_dealers():
    """Get generated dealers data."""
    try:
        loader = DataLoader()
        df = loader.load_dealers()
        df = df.where(df.notnull(), None)
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        logger.error(f"Failed to fetch dealers: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/data/ftcs', methods=['GET'])
def get_ftcs():
    """Get generated FTCs data."""
    try:
        loader = DataLoader()
        df = loader.load_ftcs()
        df = df.where(df.notnull(), None)
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        logger.error(f"Failed to fetch FTCs: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@api_bp.route('/data/relationships', methods=['GET'])
def get_relationships():
    """Get original relationships."""
    try:
        loader = DataLoader()
        df = loader.load_relationships()
        df = df.where(df.notnull(), None)
        return jsonify(df.to_dict(orient='records')), 200
    except Exception as e:
        logger.error(f"Failed to fetch relationships: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

