"""
Pose and Intrinsics Optimizers as in BARF
https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

from __future__ import annotations
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from barf import pose_analysis
from barf import camera

class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    def __init__(self, num_cameras, device, opt):
        super().__init__()
        self.num_cameras = num_cameras
        self.device = device
        self.opt = opt
        self.pose_noise = None
        self.pose_eye = torch.eye(4, device=device)[None, :3, :4].tile(num_cameras, 1, 1)
        self.annealing = 0.0
         
        if self.opt.noise > 0.0:
            # pre-generate synthetic pose perturbation
            se3_noise_t = (
                torch.randn(self.num_cameras, 3, device=device)
                * self.opt.noise * self.opt.scale
            )
            se3_noise_r = (
                torch.randn(self.num_cameras, 3, device=device)
                * self.opt.noise
            )
            self.pose_noise = camera.lie.se3_to_SE3(torch.cat([se3_noise_t, se3_noise_r], dim=-1))
            
        self.se3_refine = torch.nn.Embedding(self.num_cameras, 6, device=device)
        torch.nn.init.zeros_(self.se3_refine.weight)
        
        self.optimizer = optim.Adam(self.parameters(), lr=opt.c_lr)
        gamma = ((opt.c_lr * 1e-2)/ opt.c_lr) ** (1 / (opt.iters))
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma)
    
    def forward(self, poses, indices):
        return self.provide_refined_poses(poses, indices)
    
    def get_params(self):
        """Get camera optimizer parameters"""
        return list(self.parameters())
    
    def update_annealing(self, new_value):
        self.annealing = new_value
    
    def analyze_pose_optimization(self):
        # GT pose provided in world_to_camera and right, down, forward coordinate convention for PA
        if self.opt.identity:
            poses_gt = pose_analysis.parse_cameras_and_bounds(self)[0].to(self.device)
        elif self.opt.lightstage:
            try:
                print('\n[INFO] Comparing against Lightstage Poses: Flower COLMAP')
                poses_gt = torch.from_numpy(np.load(self.opt.path +'/colmap_poses_flower.npy', allow_pickle=True)).float().to(self.device)
            except:
                if self.opt.noise == None:
                    print('\n[INFO] No GT found, Pose Error not logged')
                    return 0, 0
        else:
            print('\n[INFO] Noise added! Using original Poses as GT for Pose Optimizer')
            poses_gt = pose_analysis.parse_raw_camera(self.opt.poses_gt).to(self.device)
                  
        pose_predictions = self.get_refined_poses(poses_gt)        
        pose_aligned, _ = pose_analysis.prealign_cameras(self.opt, pose_predictions, poses_gt)
        error = pose_analysis.evaluate_camera_alignment(self.opt, pose_aligned, poses_gt)

        rotation_error = np.rad2deg(error.R.mean().detach().cpu().numpy())
        translation_error = error.t.mean().detach().cpu().numpy()
        
        print("\n--------------------------")
        print("rot:   {:8.3f}".format(rotation_error))
        print("trans: {:10.5f}".format(translation_error))
        print("--------------------------")
        
        np.save(self.opt.debug_path + 'pose_predictions.npy', pose_predictions.detach().cpu().numpy())
        np.save(self.opt.debug_path + 'pose_gt.npy', poses_gt.detach().cpu().numpy())
        np.save(self.opt.debug_path + 'pose_predictions_aligned.npy', pose_aligned.detach().cpu().numpy())
        return rotation_error, translation_error
    
    def get_refined_poses(self, poses_gt):
        """Get optimized pose correction matrices"""
        return self(poses_gt, torch.arange(0, self.num_cameras).long())
    
    def provide_refined_poses(self, poses, indices):
        with torch.cuda.amp.autocast(enabled=self.opt.fp16):
            # add the pre-generated pose perturbations
            poses = poses[:,:3,:]
            if self.pose_noise is not None:
                pose_noise = self.pose_noise[indices]
                poses = camera.pose.compose([pose_noise, poses])
            if self.opt.identity:
                poses = self.pose_eye[indices]
                
            # add learnable pose correction
            se3_refine_subset = self.se3_refine.weight[indices]
            pose_refine = camera.lie.se3_to_SE3(se3_refine_subset)
            poses = camera.pose.compose([pose_refine, poses])

        return poses