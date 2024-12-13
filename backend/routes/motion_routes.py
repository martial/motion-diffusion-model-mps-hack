from flask import Blueprint, request, jsonify, send_from_directory
import os
import logging
import shutil
import time
from typing import Dict, Any
from flask_cors import cross_origin
from argparse import Namespace
from backend.config import (
    OUTPUT_DIR, 
    SUPPORTED_FORMATS, 
    MODEL_PATH, 

)
from backend.services.motion_generator import MotionGenerator
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate

from sample.generate import main

import json
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)
motion_bp = Blueprint('motion', __name__)
motion_bp.motion_service = None

DEFAULT_PARAMS = {
    "prompt": "",
    "num_samples": 1,
    "num_repetitions": 1,
    "motion_length": 6.0,
    "guidance_param": 2.5,
    "fps": 20,
    "temperature": 1.0,
    "seed": 42,
    "output_format": ["mp4"]
}

import sys
from argparse import ArgumentParser

CLAUDE_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

@motion_bp.route('/generate', methods=['POST'])
@cross_origin()
def generate_motion():
    """Generate motion from text prompt"""
    try:
        # Get request data
        data = request.get_json()
        
        # Create output directory
        timestamp = int(time.time())
        out_path = os.path.join(OUTPUT_DIR, f'motion_{timestamp}')
        
        # Extract parameters with defaults
        prompt = data.get('prompt', DEFAULT_PARAMS['prompt'])
        num_samples = data.get('num_samples', DEFAULT_PARAMS['num_samples'])
        num_repetitions = data.get('num_repetitions', DEFAULT_PARAMS['num_repetitions'])
        motion_length = data.get('motion_length', DEFAULT_PARAMS['motion_length'])
        guidance_param = data.get('guidance_param', DEFAULT_PARAMS['guidance_param'])
        seed = data.get('seed', DEFAULT_PARAMS['seed'])
        
        # Initialize generator if not already done
        #if motion_bp.motion_service is None:
        motion_bp.motion_service = MotionGenerator(model_path=MODEL_PATH)
        
        # Generate motion
        generated_files = motion_bp.motion_service.generate(
            prompt=prompt,
            output_dir=out_path,
            num_samples=num_samples,
            num_repetitions=num_repetitions,
            motion_length=motion_length,
            guidance_param=guidance_param,
            seed=seed
        )
        
        
      # Return success response
        return jsonify({
            'status': 'success',
            'output_dir': f'motion_{timestamp}',
            'files': {
                'visualizations': [f'motion_{timestamp}/{file}' for file in generated_files],
                'data': [
                    {
                        'sample_id': sample_i,
                        'repetition_id': rep_i,
                        'motion_data': f'motion_{timestamp}/sample{sample_i:02d}_rep{rep_i:02d}.npy',
                        'parameters': f'motion_{timestamp}/sample{sample_i:02d}_rep{rep_i:02d}_params.json',
                        'text_prompt': f'motion_{timestamp}/sample{sample_i:02d}_rep{rep_i:02d}.txt',
                        'motion_length': f'motion_{timestamp}/sample{sample_i:02d}_rep{rep_i:02d}_len.txt',
                        'visualization': f'motion_{timestamp}/sample{sample_i:02d}_rep{rep_i:02d}.mp4'
                    }
                    for sample_i in range(num_samples)
                    for rep_i in range(num_repetitions)
                ]
            },
            'parameters': {
                'prompt': prompt,
                'num_samples': num_samples,
                'num_repetitions': num_repetitions,
                'motion_length': motion_length,
                'guidance_param': guidance_param,
                'seed': seed
            }
        })

    except Exception as e:
        logger.error(f"Error generating motion: {str(e)}")
        # Clean up output directory if it exists
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@motion_bp.route('/list', methods=['GET'])
@cross_origin()
def list_motions():
    """List all generated motions from the output directory"""
    try:
        # Get all directories in OUTPUT_DIR
        motion_dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d)) and d.startswith('motion_')]
        
        results = []
        for motion_dir in motion_dirs:
            dir_path = os.path.join(OUTPUT_DIR, motion_dir)
            
            # Get timestamp from directory name
            timestamp = int(motion_dir.replace('motion_', ''))
            
            # Find all files in the directory
            files = os.listdir(dir_path)
            
            # Group files by sample and repetition
            motion_data = []
            params_file = None
            
            # Find all sample files
            sample_files = [f for f in files if f.startswith('sample') and f.endswith('.npy')]
            
            for sample_file in sample_files:
                # Extract sample and repetition numbers
                base_name = sample_file.replace('.npy', '')
                sample_num = int(base_name[6:8])  # extract XX from sampleXX
                rep_num = int(base_name[-2:])     # extract YY from repYY
                
                # Get associated files
                params_path = os.path.join(dir_path, f'{base_name}_params.json')
                if os.path.exists(params_path):
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                else:
                    params = {}
                
                motion_data.append({
                    'sample_id': sample_num,
                    'repetition_id': rep_num,
                    'motion_data': f'{motion_dir}/{sample_file}',
                    'parameters': f'{motion_dir}/{base_name}_params.json',
                    'text_prompt': f'{motion_dir}/{base_name}.txt',
                    'motion_length': f'{motion_dir}/{base_name}_len.txt',
                    'visualization': f'{motion_dir}/{base_name}.mp4',
                    'generation_params': params
                })
            
            results.append({
                'id': motion_dir,
                'timestamp': timestamp,
                'files': {
                    'visualizations': [f'{motion_dir}/{f}' for f in files if f.endswith('.mp4')],
                    'data': motion_data
                }
            })
        
        # Sort by timestamp descending (newest first)
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'motions': results
        })
        
    except Exception as e:
        logger.error(f"Error listing motions: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    
@motion_bp.route('/formats', methods=['GET'])
@cross_origin()
def get_available_formats():
    """Return list of available output formats"""
    return jsonify({
        "status": "success",
        "formats": list(SUPPORTED_FORMATS.keys()),
        "default": "mp4"
    })

@motion_bp.route('/parameters', methods=['GET'])
@cross_origin()
def get_parameters():
    """Return list of available parameters and their default values"""
    return jsonify({
        "status": "success",
        "parameters": DEFAULT_PARAMS
    })

@motion_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint"""
    status = motion_bp.motion_service.get_status()
    return jsonify({
        "status": "healthy" if status["model_loaded"] else "initializing",
        **status,
        "version": "1.0.0"
    })

@motion_bp.route('/outputs/<path:filename>')
@cross_origin()
def serve_output(filename):
    """Serve files from the output directory"""
    try:
        # Split the path into directory and file
        if '/' in filename:
            directory, file = filename.split('/', 1)
        else:
            directory = filename
            file = ''
            
        logger.info(f"Attempting to serve: directory={directory}, file={file}")  # Changed to info level
        
        # Construct the full path
        full_path = os.path.join(OUTPUT_DIR, directory)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        return send_from_directory(full_path, file)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 404

@motion_bp.route('/ai-batch', methods=['POST'])
@cross_origin()
def generate_ai_batch():
    """Generate multiple motion prompts using Claude AI"""
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        
        if not input_text:
            return jsonify({
                'status': 'error',
                'message': 'Input text is required'
            }), 400

        # Initialize Anthropic client
        anthropic = Anthropic(api_key=CLAUDE_API_KEY)
        
        # Prompt for Claude
        system_prompt = """Given the input text, generate 3 different motion prompts that describe human movements. 
        Return only a JSON object with an array called 'prompts' containing 3 strings. Each prompt should be detailed 
        and focus on physical movement. Unless specified, the prompts should start by "a person". Format: {"prompts": ["prompt1", "prompt2", "prompt3"]}"""
        
        # Get response from Claude
        message = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=300,
            temperature=0.7,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": input_text
            }]
        )
        
        # Parse the response
        try:
            response_content = message.content[0].text
            prompts_data = json.loads(response_content)
            
            return jsonify({
                'status': 'success',
                'prompts': prompts_data['prompts'],
                'results': []  # Initialize empty results array
            })
            
        except (json.JSONDecodeError, KeyError) as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to parse AI response: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in AI batch generation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

