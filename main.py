from flask import Flask, request, jsonify
import os
import torch
from argparse import Namespace
import numpy as np
import time
import shutil
import json
from typing import List, Dict, Any, Optional
import logging

from utils.fixseed import fixseed
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from model.cfg_sampler import ClassifierFreeSampleModel
from utils import dist_util
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.get_data import get_dataset_loader
from visualize.motions2hik import motions2hik
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
diffusion = None
data = None
args = None

# Supported output formats and their handlers
OUTPUT_FORMATS = {
    'mp4': 'generate_video',
    'npy': 'generate_npy',
    'json': 'generate_json',
    'fbx': 'generate_fbx',
    'bvh': 'generate_bvh'
}

class MotionGenerator:
    def __init__(self, args):
        self.args = args
        self.output_handlers = {
            'mp4': self.generate_video,
            'npy': self.generate_npy,
            'json': self.generate_json,
            'fbx': self.generate_fbx,
            'bvh': self.generate_bvh
        }

    def generate_video(self, motion_data: Dict[str, Any], out_path: str) -> List[str]:
        """Generate MP4 visualization of the motion"""
        skeleton = paramUtil.kit_kinematic_chain if self.args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        video_files = []

        for sample_i in range(self.args.num_samples):
            for rep_i in range(self.args.num_repetitions):
                caption = motion_data['text'][rep_i * self.args.batch_size + sample_i]
                length = motion_data['lengths'][rep_i * self.args.batch_size + sample_i]
                motion = motion_data['motion'][rep_i * self.args.batch_size + sample_i].transpose(2, 0, 1)[:length]
                
                save_file = f'sample{sample_i:02d}_rep{rep_i:02d}.mp4'
                animation_save_path = os.path.join(out_path, save_file)
                
                plot_3d_motion(
                    animation_save_path, 
                    skeleton, 
                    motion, 
                    dataset=self.args.dataset, 
                    title=caption, 
                    fps=self.args.fps
                )
                video_files.append(animation_save_path)

        return video_files

    def generate_npy(self, motion_data: Dict[str, Any], out_path: str) -> str:
        """Save motion data as NPY file"""
        npy_path = os.path.join(out_path, 'results.npy')
        np.save(npy_path, motion_data)
        return npy_path

    def generate_json(self, motion_data: Dict[str, Any], out_path: str) -> str:
        """Convert motion data to JSON format"""
        json_path = os.path.join(out_path, 'results.json')
        json_data = {
            'text': motion_data['text'],
            'motion': motion_data['motion'].tolist(),
            'lengths': motion_data['lengths'].tolist(),
            'num_samples': motion_data['num_samples'],
            'num_repetitions': motion_data['num_repetitions']
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f)
        return json_path

    def generate_fbx(self, motion_data: Dict[str, Any], out_path: str) -> str:
        """Generate FBX file from motion data"""
        # Implement FBX conversion here
        fbx_path = os.path.join(out_path, 'results.fbx')
        # TODO: Implement actual FBX conversion
        return fbx_path

    def generate_bvh(self, motion_data: Dict[str, Any], out_path: str) -> str:
        """Generate BVH file from motion data"""
        # Implement BVH conversion here
        bvh_path = os.path.join(out_path, 'results.bvh')
        # TODO: Implement actual BVH conversion
        return bvh_path

def get_args():
    args = Namespace()
    
    # Basic parameters
    args.fps = 20
    args.model_path = './pretrained_models/humanml_enc_512_50steps/model000750000.pt'
    args.guidance_param = 2.5
    args.unconstrained = False
    args.dataset = 'humanml'
    args.motion_length = 6.0
    args.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # Diffusion parameters
    args.diffusion_steps = 1000
    args.noise_schedule = 'cosine'
    args.sigma_small = True
    args.lambda_vel = 0.0
    args.lambda_rcxyz = 0.0
    args.lambda_fc = 0.0
    
    # Model architecture parameters
    args.cond_mask_prob = 1
    args.emb_trans_dec = False
    args.latent_dim = 512
    args.layers = 8
    args.arch = 'trans_enc'
    args.njoints = 263  # Adding njoints parameter
    args.nfeats = 1
    args.data_rep = 'hml_vec'
    args.nclasses = 1
    
    # Training parameters (needed for model initialization)
    args.dropout = 0.1
    args.emb_trans_dec = False
    args.clip_dim = 512
    args.train_platform_type = None
    
    # Generation parameters
    args.seed = 42
    args.num_repetitions = 1
    args.temperature = 1.0
    args.top_k = None
    args.top_p = None
    args.skip_timesteps = 0
    args.const_noise = False
    args.subset = None
    args.batch_size = 1
    args.num_samples = 1
    
    # Output parameters
    args.output_format = ['mp4']
    args.visualization_type = '3d'
    args.save_intermediates = False
    args.render_resolution = 'high'
    
    # Motion blending parameters
    args.interpolation_weight = None
    args.blend_mode = 'linear'
    
    return args

def initialize_model():
    global model, diffusion, data, args
    
    try:
        args = get_args()
        dist_util.setup_dist(args.device)
        
        logger.info("Loading dataset...")
        data = get_dataset_loader(
            name=args.dataset,
            batch_size=1,
            num_frames=196,
            split='test',
            hml_mode='text_only'
        )

        logger.info("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, data)

        logger.info(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)
        model.to(dist_util.dev())
        logger.info(f"Loaded model to device: {str(dist_util.dev())}")
        model.eval()

        return model, diffusion, data, args
        
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise

@app.route('/generate', methods=['POST'])
def generate_motion():
    global model, diffusion, data, args
    
    try:
        # 1. Parameter Extraction and Validation
        params = request.get_json()
        
        # Basic parameters
        args.text_prompt = params.get('prompt', "the person walked forward")
        args.seed = params.get('seed', 42)
        args.num_repetitions = params.get('num_repetitions', 1)
        args.motion_length = params.get('motion_length', 6.0)
        args.guidance_param = params.get('guidance_param', 2.5)
        args.fps = params.get('fps', 20)
        args.num_samples = params.get('num_samples', 1)
        args.unconstrained = params.get('unconstrained', False)
        
        # Advanced generation parameters
        args.temperature = params.get('temperature', 1.0)
        args.top_k = params.get('top_k', None)
        args.top_p = params.get('top_p', None)
        args.skip_timesteps = params.get('skip_timesteps', 0)
        args.const_noise = params.get('const_noise', False)
        args.subset = params.get('subset', None)
        
        # Output parameters
        output_format = params.get('output_format', ['mp4'])
        args.visualization_type = params.get('visualization_type', '3d')
        args.save_intermediates = params.get('save_intermediates', False)
        args.render_resolution = params.get('render_resolution', 'high')
        
        # Motion blending parameters
        args.interpolation_weight = params.get('interpolation_weight', None)
        args.blend_mode = params.get('blend_mode', 'linear')
        
        # Validate and accumulate output formats
        args.output_format = []
        for fmt in output_format:
            if fmt in OUTPUT_FORMATS:
                if fmt not in args.output_format:
                    args.output_format.append(fmt)
            else:
                return jsonify({
                    "status": "error",
                    "message": f"Unsupported output format: {fmt}"
                }), 400
        
        # 2. Setup
        fixseed(args.seed)
        
        max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
        fps = 12.5 if args.dataset == 'kit' else args.fps
        n_frames = min(max_frames, int(args.motion_length * fps))
        
        timestamp = int(time.time())
        out_path = f'outputs/motion_{timestamp}'
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)
        
        # 3. Motion Generation
        texts = [args.text_prompt]
        args.batch_size = args.num_samples
        
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
        collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
        _, model_kwargs = collate(collate_args)
        
        all_motions = []
        all_lengths = []
        all_text = []
        
        for rep_i in range(args.num_repetitions):
            logger.info(f'### Sampling [repetitions #{rep_i}]')
            
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
            
            # Apply temperature scaling if specified
            noise = None
            if args.temperature != 1.0:
                noise = torch.randn(
                    (args.batch_size, model.njoints, model.nfeats, max_frames),
                    device=dist_util.dev()
                ) * args.temperature
            
            sample = diffusion.p_sample_loop(
                model,
                (args.batch_size, model.njoints, model.nfeats, max_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=args.skip_timesteps,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=noise,
                const_noise=args.const_noise
            )
            
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
            
            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
            
            sample = model.rot2xyz(
                x=sample,
                mask=rot2xyz_mask,
                pose_rep=rot2xyz_pose_rep,
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False
            )
            
            text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
            all_text += model_kwargs['y'][text_key]
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        
        # 4. Process Results
        all_motions = np.concatenate(all_motions, axis=0)
        all_lengths = np.concatenate(all_lengths, axis=0)
        
        motion_data = {
            'motion': all_motions,
            'text': all_text,
            'lengths': all_lengths,
            'num_samples': args.num_samples,
            'num_repetitions': args.num_repetitions
        }
        
        # 5. Save Parameters
        # Prepare parameters for saving
        parameters_to_save = {
            'request': {
                'prompt': args.text_prompt,
                'seed': args.seed,
                'num_repetitions': args.num_repetitions,
                'motion_length': args.motion_length,
                'guidance_param': args.guidance_param,
                'fps': args.fps,
                'num_samples': args.num_samples,
                'temperature': args.temperature,
                'output_format': args.output_format,
                'visualization_type': args.visualization_type,
                'save_intermediates': args.save_intermediates,
            },
            'model': {
                'model_path': args.model_path,
                'dataset': args.dataset,
                'diffusion_steps': args.diffusion_steps,
                'noise_schedule': args.noise_schedule,
                'lambda_vel': args.lambda_vel,
                'lambda_rcxyz': args.lambda_rcxyz,
                'lambda_fc': args.lambda_fc,
                'cond_mask_prob': args.cond_mask_prob,
                'njoints': args.njoints,
                'nfeats': args.nfeats,
                'data_rep': args.data_rep,
            },
            'generation_info': {
                'timestamp': timestamp,
                'device': str(dist_util.dev()),
                'output_path': out_path,
            }
        }

        # Save as JSON
        params_file = os.path.join(out_path, 'parameters.json')
        with open(params_file, 'w') as f:
            json.dump(parameters_to_save, f, indent=2, default=str)
        logger.info(f"Saved parameters to {params_file}")

        # Save as separate NPY file
        params_npy_file = os.path.join(out_path, 'parameters.npy')
        np.save(params_npy_file, parameters_to_save)
        logger.info(f"Saved parameters to {params_npy_file}")
        
        # Include in motion data
        motion_data['parameters'] = parameters_to_save
        
        # 6. Generate Outputs
        generator = MotionGenerator(args)
        output_files = {
            'parameters_json': params_file
        }
        
        for fmt in args.output_format:
            try:
                handler = getattr(generator, OUTPUT_FORMATS[fmt])
                result = handler(motion_data, out_path)
                output_files[fmt] = result
            except Exception as e:
                logger.error(f"Error generating {fmt} output: {str(e)}")
                output_files[fmt] = None
        
        # Add parameter files to output files
        output_files['parameters_json'] = params_file
        output_files['parameters_npy'] = params_npy_file
        
        return jsonify({
            "status": "success",
            "output_path": out_path,
            "outputs": output_files
        })
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/formats', methods=['GET'])
def get_available_formats():
    """Return list of available output formats"""
    return jsonify({
        "status": "success",
        "formats": list(OUTPUT_FORMATS.keys()),
        "default": "mp4"
    })

@app.route('/parameters', methods=['GET'])
def get_parameters():
    """Return list of available parameters and their default values"""
    return jsonify({
        "status": "success",
        "parameters": {
            "basic": {
                "prompt": "the person walked forward",
                "seed": 42,
                "num_repetitions": 1,
                "motion_length": 6.0,
                "guidance_param": 2.5,
                "fps": 20,
                "num_samples": 1,
                "unconstrained": False
            },
            "advanced": {
                "temperature": 1.0,
                "top_k": None,
                "top_p": None,
                "skip_timesteps": 0,
                "const_noise": False,
                "subset": None
            },
            "output": {
                "output_format": ["mp4"],
                "visualization_type": "3d",
                "save_intermediates": False,
                "render_resolution": "high"
            },
            "blending": {
                "interpolation_weight": None,
                "blend_mode": "linear"
            }
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(dist_util.dev()),
        "version": "1.0.0"
    })

if __name__ == '__main__':
    print("Initializing model...")
    model, diffusion, data, args = initialize_model()
    print("Starting server...")
    app.run(host='0.0.0.0', port=6000, debug=True)