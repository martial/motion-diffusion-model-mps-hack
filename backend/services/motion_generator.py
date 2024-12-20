import os
import json
import torch
import numpy as np
from argparse import Namespace
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
import data_loaders.humanml.utils.paramUtil as paramUtil
from sample.generate import fixseed

class MotionGenerator:
    def __init__(self, model_path, device=0):
        """
        Initialize the motion generator
        
        Args:
            model_path (str): Path to the model checkpoint
            device (int): GPU device ID
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.diffusion = None
        self.data = None  # Store dataset reference
        
        # Create default args
        self.args = Namespace(
            # Required base arguments
            model_path=model_path,
            cuda=torch.cuda.is_available(),
            device=device,
            seed=10,
            batch_size=64,
            
            # Dataset arguments
            dataset='humanml',
            data_dir="",
            
            # Model architecture arguments
            arch='trans_enc',
            emb_trans_dec=False,
            layers=8,
            latent_dim=512,
            cond_mask_prob=0.1,
            lambda_rcxyz=0.0,
            lambda_vel=0.0,
            lambda_fc=0.0,
            unconstrained=False,
            
            # Diffusion arguments
            noise_schedule='cosine',
            diffusion_steps=1000,
            sigma_small=True,
            
            # Will be overwritten by model's args.json if it exists
            guidance_param=2.5
        )
        
        # Load args from model checkpoint if available
        self._load_model_args()
        
        # Initialize the model
        self._initialize_model()
        
    def _load_model_args(self):
        """Load arguments from model checkpoint if available"""
        args_path = os.path.join(os.path.dirname(self.model_path), 'args.json')
        if os.path.exists(args_path):
            with open(args_path, 'r') as f:
                model_args = json.load(f)
                # Update args with model's arguments
                for k, v in model_args.items():
                    if hasattr(self.args, k):
                        setattr(self.args, k, v)
        
    def _initialize_model(self):
        """Initialize the model and diffusion"""
        # Setup device
        dist_util.setup_dist(self.device)
        
        print("Creating model and diffusion...")
        self.data = self._load_dataset(max_frames=196, n_frames=196)  # Using max possible frames
        self.model, self.diffusion = create_model_and_diffusion(self.args, self.data)
        
        print(f"Loading checkpoints from [{self.model_path}]...")
        state_dict = torch.load(self.model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)
        
        if self.args.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)
            
        self.model.to(dist_util.dev())
        self.model.eval()  # disable random masking
        
    def _load_dataset(self, max_frames, n_frames):
        """Load the dataset"""
        data = get_dataset_loader(name=self.args.dataset,
                                batch_size=self.args.batch_size,
                                num_frames=max_frames,
                                split='test',
                                hml_mode='text_only')
        
        if self.args.dataset in ['kit', 'humanml']:
            data.dataset.t2m_dataset.fixed_length = n_frames
            
        return data
    
    def _sample_motion(self, num_samples, max_frames, model_kwargs, method='p_sample', 
                      ddim_eta=0.0, plms_order=2, skip_timesteps=0, init_image=None):
        """
        Internal method to sample motion using different sampling methods
        """
        shape = (num_samples, self.model.njoints, self.model.nfeats, max_frames)
        
        if method == 'p_sample':
            return self.diffusion.p_sample_loop(
                self.model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False
            )
        elif method == 'ddim':
            return self.diffusion.ddim_sample_loop(
                self.model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                eta=ddim_eta,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                progress=True
            )
        elif method == 'plms':
            return self.diffusion.plms_sample_loop(
                self.model,
                shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                order=plms_order,
                skip_timesteps=skip_timesteps,
                init_image=init_image,
                progress=True
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
    def generate(self, 
                prompt,
                output_dir,
                num_samples=1,
                num_repetitions=1,
                motion_length=6.0,
                guidance_param=2.5,
                seed=42,
                fps=20,
                sampling_method='ddim',  # Added parameter
                ddim_eta=0.5,           # Added parameter
                plms_order=2):          # Added parameter
        """
        Generate motion from text prompt
        
        Args:
            prompt (str): Text prompt
            output_dir (str): Output directory path
            num_samples (int): Number of samples to generate
            num_repetitions (int): Number of repetitions per sample
            motion_length (float): Motion length in seconds
            guidance_param (float): Guidance scale parameter
            seed (int): Random seed
            fps (int): Frames per second
            sampling_method (str): Sampling method ('p_sample', 'ddim', or 'plms')
            ddim_eta (float): DDIM eta parameter
            plms_order (int): PLMS order parameter
            
        Returns:
            list: List of generated file paths
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        fixseed(seed)
        
        # Calculate frames
        max_frames = 196 if self.args.dataset in ['kit', 'humanml'] else 60
        fps = 12.5 if self.args.dataset == 'kit' else 20
        #fps = 30
        n_frames = min(max_frames, int(motion_length * fps))
        
        print("N_frames: ", n_frames)
        print("motion_length: ", motion_length)
        # Update batch size if needed
        self.args.batch_size = max(num_samples, 1)
        
        # Prepare model kwargs
        collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * num_samples
        collate_args = [dict(arg, text=prompt) for arg in collate_args]
        _, model_kwargs = collate(collate_args)
        
        all_motions = []
        all_lengths = []
        all_text = []

        for rep_i in range(num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            # Add CFG scale to batch
            if guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(num_samples, device=dist_util.dev()) * guidance_param

            # Use the sampling method with parameters
            sample = self._sample_motion(
                num_samples=num_samples,
                max_frames=max_frames,
                model_kwargs=model_kwargs,
                method=sampling_method,
                ddim_eta=ddim_eta,
                plms_order=plms_order
            )

            # Process generated sample
            if self.model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = self.data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            # Convert rotations to xyz coordinates
            rot2xyz_pose_rep = 'xyz' if self.model.data_rep in ['xyz', 'hml_vec'] else self.model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(num_samples, n_frames).bool()
            sample = self.model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                      jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                      get_rotations_back=False)

            # Store results
            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
            all_text.extend([prompt] * num_samples)

        # Save results
        all_motions = np.concatenate(all_motions, axis=0)
        all_lengths = np.concatenate(all_lengths, axis=0)
        
        # Instead of saving one combined file, save individual files per generation
        for sample_i in range(num_samples):
            for rep_i in range(num_repetitions):
                idx = rep_i * num_samples + sample_i
                
                # Create individual npy file for this generation
                npy_path = os.path.join(output_dir, f'sample{sample_i:02d}_rep{rep_i:02d}.npy')
                print(f"Saving results to [{npy_path}]")
                np.save(npy_path, {
                    'motion': all_motions[idx:idx+1],  # Save single motion
                    'text': [all_text[idx]],  # Save single text
                    'lengths': [all_lengths[idx]],  # Save single length
                    'num_samples': 1,
                    'num_repetitions': 1
                })

                # Save parameters for this generation
                params_path = os.path.join(output_dir, f'sample{sample_i:02d}_rep{rep_i:02d}_params.json')
                generation_params = {
                    'prompt': prompt,
                    'motion_length': motion_length,
                    'guidance_param': guidance_param,
                    'seed': seed,
                    'fps': fps,
                    'n_frames': n_frames,
                    'dataset': self.args.dataset,
                    'sampling_method': sampling_method,
                    'ddim_eta': ddim_eta,
                    'plms_order': plms_order
                }
                with open(params_path, 'w') as f:
                    json.dump(generation_params, f, indent=4)

                # Save text prompt
                with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
                    fw.write(all_text[idx])
                    
                # Save length
                with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
                    fw.write(str(all_lengths[idx]))

        # Save visualizations
        generated_files = []
        skeleton = paramUtil.kit_kinematic_chain if self.args.dataset == 'kit' else paramUtil.t2m_kinematic_chain

        for sample_i in range(num_samples):
            for rep_i in range(num_repetitions):
                motion = all_motions[rep_i*num_samples + sample_i].transpose(2, 0, 1)[:all_lengths[rep_i*num_samples + sample_i]]
                save_file = f'sample{sample_i:02d}_rep{rep_i:02d}.mp4'
                save_path = os.path.join(output_dir, save_file)
                plot_3d_motion(save_path, skeleton, motion, dataset=self.args.dataset, title=prompt, fps=fps)
                generated_files.append(save_file)

        return generated_files