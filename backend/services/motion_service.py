import os
import sys
import torch
import numpy as np
import json
import logging
import time
import shutil
from typing import Dict, Any, List, Tuple
from argparse import Namespace

# Add parent directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.tensors import collate
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import data_loaders.humanml.utils.paramUtil as paramUtil
from model.cfg_sampler import ClassifierFreeSampleModel

logger = logging.getLogger(__name__)

class MotionService:
    def __init__(self):
        self.model = None
        self.diffusion = None
        self.data = None
        self.args = None
        self._initialized = False

    def initialize(self, args):
        """Initialize model and diffusion"""
        try:
            self.args = args
            
            # Set all random seeds explicitly
            logger.info(f"Setting seed: {args.seed}")
            fixseed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            np.random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # Then do device setup
            dist_util.setup_dist(args.device)
            
            # Log parameters at initialization
            logger.info('='*50)
            logger.info('Running generation with parameters:')
            logger.info(f'Seed: {args.seed}')
            logger.info(f'Model path: {args.model_path}')
            logger.info(f'Guidance param: {args.guidance_param}')
            logger.info(f'Motion length: {args.motion_length}')
            if args.text_prompt:
                logger.info(f'Text prompt: {args.text_prompt}')
            logger.info('='*50)
            
            # Initialize dataset first
            logger.info("Loading dataset...")
            self.data = get_dataset_loader(
                name=args.dataset,
                batch_size=1,
                num_frames=196,
                split='test',
                hml_mode='text_only'
            )
            
            logger.info("Creating model and diffusion...")
            # Now create model with dataset
            model, diffusion = create_model_and_diffusion(args, self.data)
            self.model = model
            self.diffusion = diffusion

            logger.info(f"Loading checkpoints from [{args.model_path}]...")
            state_dict = torch.load(args.model_path, map_location='cpu')
            load_model_wo_clip(self.model, state_dict)

            if args.guidance_param != 1:
                self.model = ClassifierFreeSampleModel(self.model)
            self.model.to(dist_util.dev())
            logger.info(f"Loaded model to device: {str(dist_util.dev())}")
            self.model.eval()

            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def generate_motion(self, args: Namespace, out_path: str) -> Tuple[Dict[str, Any], List[str]]:
        """Generate motion based on provided arguments"""
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")

        try:
            # Reset seed at the start of each generation
            logger.info(f"Resetting seed for generation: {args.seed}")
            fixseed(args.seed)

            # Calculate frames and FPS
            max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
            fps = 12.5 if args.dataset == 'kit' else args.fps
            n_frames = min(max_frames, int(args.motion_length * fps))
            
            # Always using direct input for this service version
            is_using_data = False

            # Validate and prepare text inputs
            if args.text_prompt:
                texts = [args.text_prompt]
                args.num_samples = 1

            # Validate batch size
            if args.num_samples > args.batch_size:
                raise ValueError(f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})')
            args.batch_size = args.num_samples

            # Prepare model kwargs
            if is_using_data:
                iterator = iter(self.data)
                _, model_kwargs = next(iterator)
            else:
                collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
                if args.text_prompt:
                    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
                else:
                    action = self.data.dataset.action_name_to_action(action_text)
                    collate_args = [dict(arg, action=one_action, action_text=one_action_text) 
                                  for arg, one_action, one_action_text in zip(collate_args, action, action_text)]
                _, model_kwargs = collate(collate_args)

            all_motions = []
            all_lengths = []
            all_text = []

            for rep_i in range(args.num_repetitions):
                logger.info(f'### Sampling [repetitions #{rep_i}]')
                
                # Reset seed for each repetition to ensure consistency
                fixseed(args.seed + rep_i)
                
                # Clear CUDA cache between repetitions
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Add CFG scale to batch
                if args.guidance_param != 1:
                    model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

                sample_fn = self.diffusion.p_sample_loop

                # Generate sample
                # Generate unique noise for each repetition
                torch.manual_seed(args.seed + rep_i)  # Set torch seed explicitly
                torch.cuda.manual_seed(args.seed + rep_i)  # Set CUDA seed explicitly
                noise = torch.randn(
                    (args.batch_size, self.model.njoints, self.model.nfeats, max_frames),
                    device=dist_util.dev()
                )
                
                sample = sample_fn(
                    self.model,
                    (args.batch_size, self.model.njoints, self.model.nfeats, max_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=noise,  # Pass in our explicitly generated noise
                    const_noise=False
                )

                # Process HumanML3D vectors if needed
                if self.model.data_rep == 'hml_vec':
                    n_joints = 22 if sample.shape[1] == 263 else 21
                    sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                    sample = recover_from_ric(sample, n_joints)
                    sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

                # Convert rotations to xyz coordinates
                rot2xyz_pose_rep = 'xyz' if self.model.data_rep in ['xyz', 'hml_vec'] else self.model.data_rep
                rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, n_frames).bool()
                
                sample = self.model.rot2xyz(
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

                # Store results
                if args.unconstrained:
                    all_text += ['unconstrained'] * args.num_samples
                else:
                    text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                    all_text += model_kwargs['y'][text_key]

                all_motions.append(sample.cpu().numpy())
                all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            # Process final results
            total_num_samples = args.num_samples * args.num_repetitions
            all_motions = np.concatenate(all_motions, axis=0)
            all_motions = all_motions[:total_num_samples]
            all_text = all_text[:total_num_samples]
            all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

            motion_data = {
                'motion': all_motions,
                'text': all_text,
                'lengths': all_lengths,
                'num_samples': args.num_samples,
                'num_repetitions': args.num_repetitions,
                'parameters': self._get_generation_parameters(args)
            }

            return motion_data, self._save_outputs(motion_data, args, out_path)

        except Exception as e:
            logger.error(f"Error generating motion: {str(e)}")
            raise

    def _get_generation_parameters(self, args: Namespace) -> Dict[str, Any]:
        """Prepare parameters for saving"""
        return {
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
            },
            'model': {
                'model_path': args.model_path,
                'dataset': args.dataset,
                'device': str(dist_util.dev()),
                'njoints': args.njoints,
                'data_rep': args.data_rep,
            },
            'generation_info': {
                'timestamp': int(time.time()),
                'max_frames': 196 if args.dataset in ['kit', 'humanml'] else 60,
                'actual_fps': 12.5 if args.dataset == 'kit' else args.fps,
            }
        }

    def _save_outputs(self, motion_data: Dict[str, Any], args: Namespace, out_path: str) -> List[str]:
        """Save motion data in specified formats"""
        # Create output directory if it doesn't exist
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path)

        skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
        generated_files = []
        fps = 12.5 if args.dataset == 'kit' else args.fps

        logger.info(f"Saving outputs with shape: {motion_data['motion'].shape}")
        logger.info(f"Number of texts: {len(motion_data['text'])}")
        logger.info(f"Number of lengths: {len(motion_data['lengths'])}")
        logger.info(f"Expected total animations: {args.num_samples * args.num_repetitions}")

        # Save NPY
        if 'npy' in args.output_format:
            npy_path = os.path.join(out_path, 'results.npy')
            np.save(npy_path, motion_data)
            generated_files.append(npy_path)
            logger.info(f"Saved NPY file: {npy_path}")

        # Save JSON
        if 'json' in args.output_format:
            json_path = os.path.join(out_path, 'results.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'text': motion_data['text'],
                    'lengths': motion_data['lengths'].tolist(),
                    'parameters': motion_data['parameters']
                }, f, indent=2)
            generated_files.append(json_path)
            logger.info(f"Saved JSON file: {json_path}")

        # Save MP4
        if 'mp4' in args.output_format:
            sample_files = []
            num_samples_in_out_file = 7

            for sample_i in range(args.num_samples):
                rep_files = []
                for rep_i in range(args.num_repetitions):
                    idx = (rep_i * args.num_samples) + sample_i
                    logger.info(f"Processing animation {idx} (sample {sample_i}, rep {rep_i})")
                    
                    caption = motion_data['text'][idx]
                    length = motion_data['lengths'][idx]
                    motion = motion_data['motion'][idx].transpose(2, 0, 1)[:length]
                    
                    save_file = f'sample{sample_i:02d}_rep{rep_i:02d}.mp4'
                    animation_path = os.path.join(out_path, save_file)
                    
                    logger.info(f"Saving animation to: {animation_path}")
                    logger.info(f"Motion shape: {motion.shape}")
                    logger.info(f"Caption: {caption}")
                    logger.info(f"Length: {length}")
                    logger.info(f"Index used: {idx}")
                    
                    plot_3d_motion(
                        animation_path,
                        skeleton,
                        motion,
                        dataset=args.dataset,
                        title=caption,
                        fps=args.fps
                    )
                    rep_files.append(animation_path)

                # Combine repetitions horizontally
                all_rep_save_file = f'sample{sample_i:02d}.mp4'
                all_rep_save_path = os.path.join(out_path, all_rep_save_file)
                if args.num_repetitions > 1:
                    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
                    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}'
                    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
                    os.system(ffmpeg_rep_cmd)
                    generated_files.append(all_rep_save_path)
                    sample_files.append(all_rep_save_path)

                # Combine samples vertically when threshold reached
                if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
                    if len(sample_files) > 1:
                        all_sample_save_file = f'samples_{sample_i - len(sample_files) + 1:02d}_to_{sample_i:02d}.mp4'
                        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
                        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
                        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}'
                        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
                        os.system(ffmpeg_rep_cmd)
                        generated_files.append(all_sample_save_path)
                    sample_files = []

        # Save parameters
        params_path = os.path.join(out_path, 'parameters.json')
        with open(params_path, 'w') as f:
            json.dump(motion_data['parameters'], f, indent=2)
        generated_files.append(params_path)
        logger.info(f"Saved parameters to: {params_path}")

        return generated_files

    def get_status(self) -> Dict[str, Any]:
        """Get current service status"""
        return {
            "model_loaded": self.model is not None,
            "device": str(dist_util.dev()) if self.model else None,
            "dataset": self.args.dataset if self.args else None,
            "initialized": self._initialized
        }

    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter settings"""
        if not self._initialized:
            return {
                "model": DEFAULT_MODEL_CONFIG,
                "generation": DEFAULT_GEN_CONFIG
            }
        
        model_params = {
            "model_path": self.args.model_path,
            "njoints": self.args.njoints,
            "nfeats": getattr(self.args, 'nfeats', DEFAULT_MODEL_CONFIG['nfeats']),
            "data_rep": self.args.data_rep,
            "dataset": self.args.dataset
        }
        
        gen_params = {
            "num_samples": self.args.num_samples,
            "fps": self.args.fps,
            "motion_length": self.args.motion_length,
            "input_text": getattr(self.args, 'input_text', ''),
            "text_prompt": getattr(self.args, 'text_prompt', ''),
            "num_repetitions": self.args.num_repetitions,
            "seed": self.args.seed,
            "device": self.args.device
        }
        
        return {
            "model": model_params,
            "generation": gen_params
        }

    def update_parameters(self, params: Dict[str, Any]) -> None:
        """Update service parameters
        
        Args:
            params: Dictionary containing model and/or generation parameters to update
        """
        if not self._initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
            
        if "model" in params:
            model_params = params["model"]
            for key, value in model_params.items():
                if hasattr(self.args, key):
                    setattr(self.args, key, value)
                    
        if "generation" in params:
            gen_params = params["generation"]
            for key, value in gen_params.items():
                if hasattr(self.args, key):
                    setattr(self.args, key, value)

# Make sure to export the class
__all__ = ['MotionService']