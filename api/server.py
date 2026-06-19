"""
API server for territory optimization system.
"""
import logging
from flask import Flask
from flask_cors import CORS
from api.routes import api_bp

logger = logging.getLogger(__name__)

import os

def create_app():
    """Create and configure Flask application."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    app = Flask(__name__, static_folder=static_dir, static_url_path='')
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api/v1')
    
    # Add error handlers
    @app.errorhandler(404)
    def not_found(error):
        return {'error': 'Not found'}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return {'error': 'Internal server error'}, 500
    
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
    logger.info("Flask application created")
    return app

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the API server."""
    try:
        app = create_app()
        app.run(host=host, port=port, debug=debug)
        logger.info(f"Server running on http://{host}:{port}")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise

if __name__ == '__main__':
    run_server(debug=True)
