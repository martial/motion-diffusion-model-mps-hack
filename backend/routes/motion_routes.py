from flask import Blueprint, request, jsonify, send_from_directory, send_file
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
from visualize import vis_utils

import json
from anthropic import Anthropic
from dotenv import load_dotenv
import re
import subprocess
import platform
import zipfile
from io import BytesIO

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
    "output_format": ["mp4"],
    "sampling_method": "ddim",
    "ddim_eta": 0.5,
    "plms_order": 2
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
        fps = data.get('fps', DEFAULT_PARAMS['fps'])
        sampling_method = data.get('sampling_method', DEFAULT_PARAMS['sampling_method'])
        ddim_eta = data.get('ddim_eta', DEFAULT_PARAMS['ddim_eta'])
        plms_order = data.get('plms_order', DEFAULT_PARAMS['plms_order'])
        
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
            seed=seed,
            fps=fps,
            sampling_method=sampling_method,
            ddim_eta=ddim_eta,
            plms_order=plms_order
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
                'seed': seed,
                'sampling_method': sampling_method,
                'ddim_eta': ddim_eta,
                'plms_order': plms_order
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
            try:
                dir_path = os.path.join(OUTPUT_DIR, motion_dir)
                
                # Get timestamp from directory name more safely
                try:
                    timestamp = int(motion_dir.replace('motion_', ''))
                except ValueError:
                    logger.warning(f"Skipping directory with invalid timestamp: {motion_dir}")
                    continue
                
                # Find all files in the directory
                files = os.listdir(dir_path)
                
                # Group files by sample and repetition
                motion_data = []
                
                # Find all sample files, excluding smpl files
                sample_files = [f for f in files if f.startswith('sample') and 
                              f.endswith('.npy') and '_smpl' not in f]
                
                for sample_file in sample_files:
                    # Extract sample and repetition numbers
                    base_name = sample_file.replace('.npy', '')
                    print(base_name)
                    sample_num = int(base_name[6:8])  # extract XX from sampleXX
                    rep_num = int(base_name[-2:])     # extract YY from repYY
                    
                    # Get associated files
                    params_path = os.path.join(dir_path, f'{base_name}_params.json')
                    if os.path.exists(params_path):
                        with open(params_path, 'r') as f:
                            params = json.load(f)
                    else:
                        params = {}
                    
                    # Check for SMPL file with both naming conventions
                    npy_smpl_path = None
                    smpl_variants = [
                        f'sample{sample_num:02d}_rep{rep_num:02d}_smpl.npy',  # with leading zeros
                        f'sample{sample_num}_rep{rep_num}_smpl.npy'           # without leading zeros
                    ]
                    
                    for variant in smpl_variants:
                        temp_path = os.path.join(dir_path, variant)
                        if os.path.exists(temp_path):
                            npy_smpl_path = temp_path
                            npy_smpl_path = npy_smpl_path + ".pkl"
                            break

                    motion_data.append({
                        'sample_id': sample_num,
                        'repetition_id': rep_num,
                        'motion_data': f'{motion_dir}/{sample_file}',
                        'parameters': f'{motion_dir}/{base_name}_params.json',
                        'text_prompt': f'{motion_dir}/{base_name}.txt',
                        'motion_length': f'{motion_dir}/{base_name}_len.txt',
                        'visualization': f'{motion_dir}/{base_name}.mp4',
                        'generation_params': params,
                        'smpl_data': npy_smpl_path
                    })
                
                results.append({
                    'id': motion_dir,
                    'timestamp': timestamp,
                    'files': {
                        'visualizations': [f'{motion_dir}/{f}' for f in files if f.endswith('.mp4')],
                        'data': motion_data
                    }
                })
            
            except Exception as e:
                logger.error(f"Error listing motions: {str(e)}", exc_info=True)
                exit()
                continue
        
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

@motion_bp.route('/export-smpl', methods=['POST'])
@cross_origin()
def export_smpl():
    """Export SMPL data for a given motion file"""
    try:
        data = request.get_json()
        motion_path = data.get('motion_path', '')
        
        if not motion_path:
            return jsonify({
                'status': 'error',
                'message': 'Motion path is required'
            }), 400

        # Split the path into directory and file
        if '/' in motion_path:
            directory, file = motion_path.split('/', 1)
        else:
            return jsonify({
                'status': 'error',
                'message': 'Invalid motion path format'
            }), 400

        # Construct the full paths
        dir_path = os.path.join(OUTPUT_DIR, directory)
        input_path = os.path.join(dir_path, file)

        if not os.path.exists(input_path):
            return jsonify({
                'status': 'error',
                'message': 'Motion file not found'
            }), 404

        # Extract sample and repetition numbers from filename
        # Assuming filename format: sampleXX_repYY.npy
        filename = os.path.basename(file)
        base_name = filename.replace('.npy', '')
        
        try:
            # Extract numbers more safely
            sample_match = re.search(r'sample(\d+)', base_name)
            rep_match = re.search(r'rep(\d+)', base_name)
            
            if not sample_match or not rep_match:
                raise ValueError("Invalid filename format")
                
            sample_num = int(sample_match.group(1))
            rep_num = int(rep_match.group(1))
        except (ValueError, AttributeError) as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid filename format: {str(e)}'
            }), 400

        # Generate SMPL npy file
        npy_smpl_path = os.path.join(dir_path, f'sample{sample_num:02d}_rep{rep_num:02d}_smpl.npy')
        #print(npy_smpl_path)

        try:
            print(f"Generating SMPL npy file for sample {sample_num} repetition {rep_num}")
            npy2obj = vis_utils.npy2obj(input_path, 0, 0, device="mps", cuda=False)  # Always use index 0
            npy2obj.save_npy(npy_smpl_path)
            print("savec to ", npy_smpl_path)
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to generate SMPL data: {str(e)}'
            }), 500

        return jsonify({
            'status': 'success',
            'smpl_path': f'{directory}/sample{sample_num:02d}_rep{rep_num:02d}_smpl.npy.pkl'
        })

    except Exception as e:
        logger.error(f"Error exporting SMPL data: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

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

@motion_bp.route('/favorites', methods=['GET', 'POST', 'DELETE'])
@cross_origin()
def handle_favorites():
    """Handle favorite motions"""
    try:
        # Store favorites in a JSON file
        favorites_path = os.path.join(OUTPUT_DIR, 'favorites.json')
        
        # GET: Retrieve favorites
        if request.method == 'GET':
            if os.path.exists(favorites_path):
                with open(favorites_path, 'r') as f:
                    return jsonify({
                        'status': 'success',
                        'favorites': json.load(f)
                    })
            return jsonify({'status': 'success', 'favorites': []})
        
        # POST: Add to favorites
        elif request.method == 'POST':
            data = request.get_json()
            video_id = data.get('videoId')
            
            if not video_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Video ID is required'
                }), 400
                
            favorites = []
            if os.path.exists(favorites_path):
                with open(favorites_path, 'r') as f:
                    favorites = json.load(f)
                    
            if video_id not in favorites:
                favorites.append(video_id)
                
            with open(favorites_path, 'w') as f:
                json.dump(favorites, f)
                
            return jsonify({
                'status': 'success',
                'favorites': favorites
            })
            
        # DELETE: Remove from favorites
        elif request.method == 'DELETE':
            data = request.get_json()
            video_id = data.get('videoId')
            
            if not video_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Video ID is required'
                }), 400
                
            if os.path.exists(favorites_path):
                with open(favorites_path, 'r') as f:
                    favorites = json.load(f)
                    
                if video_id in favorites:
                    favorites.remove(video_id)
                    
                with open(favorites_path, 'w') as f:
                    json.dump(favorites, f)
                    
            return jsonify({
                'status': 'success',
                'favorites': favorites
            })
            
    except Exception as e:
        logger.error(f"Error handling favorites: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@motion_bp.route('/hidden', methods=['GET', 'POST', 'DELETE'])
@cross_origin()
def handle_hidden():
    """Handle hidden motions"""
    try:
        # Store hidden videos in a JSON file
        hidden_path = os.path.join(OUTPUT_DIR, 'hidden.json')
        
        # GET: Retrieve hidden videos
        if request.method == 'GET':
            if os.path.exists(hidden_path):
                with open(hidden_path, 'r') as f:
                    return jsonify({
                        'status': 'success',
                        'hidden': json.load(f)
                    })
            return jsonify({'status': 'success', 'hidden': []})
        
        # POST: Add to hidden
        elif request.method == 'POST':
            data = request.get_json()
            video_id = data.get('videoId')
            
            if not video_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Video ID is required'
                }), 400
                
            hidden = []
            if os.path.exists(hidden_path):
                with open(hidden_path, 'r') as f:
                    hidden = json.load(f)
                    
            if video_id not in hidden:
                hidden.append(video_id)
                
            with open(hidden_path, 'w') as f:
                json.dump(hidden, f)
                
            return jsonify({
                'status': 'success',
                'hidden': hidden
            })
            
        # DELETE: Remove from hidden (unhide)
        elif request.method == 'DELETE':
            data = request.get_json()
            video_id = data.get('videoId')
            
            if not video_id:
                return jsonify({
                    'status': 'error',
                    'message': 'Video ID is required'
                }), 400
                
            if os.path.exists(hidden_path):
                with open(hidden_path, 'r') as f:
                    hidden = json.load(f)
                    
                if video_id in hidden:
                    hidden.remove(video_id)
                    
                with open(hidden_path, 'w') as f:
                    json.dump(hidden, f)
                    
            return jsonify({
                'status': 'success',
                'hidden': hidden
            })
            
    except Exception as e:
        logger.error(f"Error handling hidden videos: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@motion_bp.route('/open-folder', methods=['POST'])
@cross_origin()
def open_folder():
    """Open the folder containing motion files in the system's file explorer"""
    try:
        data = request.get_json()
        motion_id = data.get('motionId')
        
        if not motion_id:
            return jsonify({
                'status': 'error',
                'message': 'Motion ID is required'
            }), 400
            
        folder_path = os.path.join(OUTPUT_DIR, motion_id)
        
        if not os.path.exists(folder_path):
            return jsonify({
                'status': 'error',
                'message': 'Folder not found'
            }), 404
            
        # Different commands for different operating systems
        system = platform.system()
        try:
            if system == 'Darwin':  # macOS
                subprocess.run(['open', folder_path])
            elif system == 'Windows':
                subprocess.run(['explorer', folder_path])
            else:  # Linux and other Unix-like
                subprocess.run(['xdg-open', folder_path])
                
            return jsonify({
                'status': 'success',
                'message': 'Folder opened successfully'
            })
            
        except subprocess.SubprocessError as e:
            logger.error(f"Error opening folder: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': f'Failed to open folder: {str(e)}'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in open_folder route: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@motion_bp.route('/hidden/bulk', methods=['POST'])
@cross_origin()
def handle_hidden_bulk():
    """Handle bulk hiding of motions"""
    try:
        data = request.get_json()
        video_ids = data.get('videoIds', [])
        
        if not video_ids or not isinstance(video_ids, list):
            return jsonify({
                'status': 'error',
                'message': 'Video IDs array is required'
            }), 400
            
        hidden_path = os.path.join(OUTPUT_DIR, 'hidden.json')
        
        # Load existing hidden videos
        hidden = []
        if os.path.exists(hidden_path):
            with open(hidden_path, 'r') as f:
                hidden = json.load(f)
                
        # Add new videos to hidden list
        hidden.extend([vid for vid in video_ids if vid not in hidden])
                
        # Save updated hidden list
        with open(hidden_path, 'w') as f:
            json.dump(hidden, f)
            
        return jsonify({
            'status': 'success',
            'hidden': hidden
        })
            
    except Exception as e:
        logger.error(f"Error handling bulk hidden videos: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@motion_bp.route('/favorites/download', methods=['GET'])
@cross_origin()
def download_favorites():
    """Download all favorite motions as a zip file"""
    try:
        # Get favorites list
        favorites_path = os.path.join(OUTPUT_DIR, 'favorites.json')
        if not os.path.exists(favorites_path):
            return jsonify({
                'status': 'error',
                'message': 'No favorites found'
            }), 404

        with open(favorites_path, 'r') as f:
            favorites = json.load(f)

        # Get hidden videos list
        hidden_path = os.path.join(OUTPUT_DIR, 'hidden.json')
        hidden_videos = set()
        if os.path.exists(hidden_path):
            with open(hidden_path, 'r') as f:
                hidden_videos = set(json.load(f))

        # Filter out hidden videos from favorites
        favorites = [f for f in favorites if f not in hidden_videos]

        if not favorites:
            return jsonify({
                'status': 'error',
                'message': 'No visible favorites found'
            }), 404

        # Create a BytesIO object to store the zip file
        memory_file = BytesIO()
        
        # Create the zip file
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            # For each favorite
            for favorite in favorites:
                # Parse the video ID to get motion_id, sample_id, and rep_id
                motion_id, sample_id, rep_id = favorite.split('-')
                base_path = os.path.join(OUTPUT_DIR, motion_id)
                
                if os.path.exists(base_path):
                    # Try both naming formats
                    file_patterns = [
                        # Format with leading zeros
                        f'sample{int(sample_id):02d}_rep{int(rep_id):02d}',
                        # Format without leading zeros
                        f'sample{int(sample_id)}_rep{int(rep_id)}'
                    ]
                    
                    # Find all files in the directory
                    for file in os.listdir(base_path):
                        # Check if file matches either pattern
                        for pattern in file_patterns:
                            if file.startswith(pattern):
                                file_path = os.path.join(base_path, file)
                                arc_name = os.path.join(motion_id, file)
                                zf.write(file_path, arc_name)
                                break

        # Seek to the beginning of the BytesIO object
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='favorite_motions.zip'
        )

    except Exception as e:
        logger.error(f"Error downloading favorites: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

