import os
import sys
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from backend.routes.motion_routes import motion_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__, 
            static_folder='../frontend/dist',
            static_url_path='')

# Enable CORS for all routes and origins
CORS(app, supports_credentials=True)

@app.route('/check')
def check():
    return jsonify({"status": "ok", "message": "Server is running"})

# Serve frontend static files
@app.route('/')
def serve_frontend():
    return app.send_static_file('index.html')

# Handle frontend routes by serving index.html
@app.route('/<path:path>')
def catch_all(path):
    if path.startswith('api/'):
        return jsonify({"error": "Not found"}), 404
    return app.send_static_file('index.html')

# Register blueprints
app.register_blueprint(motion_bp, url_prefix='/api/motion')

if __name__ == '__main__':
    app.run(debug=True)