import os
import torch

# Path Configuration
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(ROOT_DIR, 'pretrained_models/humanml_enc_512_50steps/model000750000.pt')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')

# API Configuration
API_HOST = '127.0.0.1'
API_PORT = 3000

# Output Configuration
SUPPORTED_FORMATS = {
    'mp4': 'generate_video',
    'npy': 'generate_npy',
    'json': 'generate_json'
}

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)