"""
Pose and Intrinsics Optimizers as in BARF
https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
from easydict import EasyDict as edict
import sys
sys.path.append('./barf/')
import camera

def center_camera_poses(poses):
        # compute average pose
        center = poses[..., 3].mean(dim=0)
        v1 = torch_F.normalize(poses[..., 1].mean(dim=0), dim=0)
        v2 = torch_F.normalize(poses[..., 2].mean(dim=0), dim=0)
        v0 = v1.cross(v2)
        pose_avg = torch.stack([v0, v1, v2, center], dim=-1)[None]  # [1,3,4]
        # apply inverse of averaged pose
        poses = camera.pose.compose([poses, camera.pose.invert(pose_avg)])
        return poses
    
def construct_pose(R=None,t=None):
    # construct a camera pose from the given R and/or t
    assert(R is not None or t is not None)
    if R is None:
        if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
    elif t is None:
        if not isinstance(R,torch.Tensor): R = torch.tensor(R)
        t = torch.zeros(R.shape[:-1],device=R.device)
    else:
        if not isinstance(R,torch.Tensor): R = torch.tensor(R)
        if not isinstance(t,torch.Tensor): t = torch.tensor(t)
    assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
    R = R.float()
    t = t.float()
    pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
    assert(pose.shape[-2:]==(3,4))
    return pose


def get_all_camera_poses(self,opt):
    pose_raw_all = [torch.tensor(f["transform_matrix"],dtype=torch.float32) for f in self.list]
    pose_canon_all = torch.stack([self.parse_raw_camera(opt,p) for p in pose_raw_all],dim=0)
    return pose_canon_all

def get_camera(self,opt,idx):
        intr = torch.tensor([[self.focal,0,self.raw_W/2],
                             [0,self.focal,self.raw_H/2],
                             [0,0,1]]).float()
        pose_raw = torch.tensor(self.list[idx]["transform_matrix"],dtype=torch.float32)
        pose = self.parse_raw_camera(opt,pose_raw)
        return intr,pose

'''def parse_raw_camera(pose_raw):
    pose_flip = camera.pose(R=torch.diag(torch.tensor([1,-1,-1])))
    pose = camera.pose.compose([pose_flip,pose_raw[:3]])
    pose = camera.pose.invert(pose)
    return pose'''

def parse_raw_camera(pose_raw_list):
    """Convert n poses from camera_to_world to world_to_camera and follow the right, down, forward coordinate convention."""
    
    parsed_poses = torch.empty((len(pose_raw_list), 3, 4), dtype=torch.float32, device=pose_raw_list[0].device)
    
    for i, pose_raw in enumerate(pose_raw_list):
        # Construct pose_flip matrix
        pose_flip = construct_pose(R=torch.diag(torch.tensor([1, -1, -1]))).to(pose_raw.device)
        pose = camera.pose.compose([pose_flip, pose_raw[:3]])
        pose = camera.pose.invert(pose)  # world_from_camera --> camera_from_world
        parsed_poses[i] = pose
    return parsed_poses

def parse_cameras_and_bounds(self):
    fname = "{}/poses_bounds.npy".format(self.opt.path)
    data = torch.tensor(np.load(fname), dtype=torch.float32)

    # parse cameras (intrinsics and poses)
    cam_data = data[:, :-2].view([-1, 3, 5])  # [N,3,5]
    poses_raw = cam_data[..., :4]  # [N,3,4]
    poses_raw[..., 0], poses_raw[..., 1] = (
        poses_raw[..., 1],
        -poses_raw[..., 0],
    )

    raw_H, raw_W, self.focal = cam_data[0, :, -1]
    #assert self.raw_H == raw_H and self.raw_W == raw_W
    # parse depth bounds
    bounds = data[:, -2:]  # [N,2]
    poses_raw[..., 3] *= self.opt.scale
    bounds *= self.opt.scale
    poses_raw = center_camera_poses(poses_raw)
 
    return poses_raw, bounds
    
def prealign_cameras(opt, pose, pose_GT):
    # compute 3D similarity transform via Procrustes analysis
    if opt.identity:
        pose = center_camera_poses(pose)
    center = torch.zeros(1, 1, 3, device=opt.device)
    center_pred = camera.cam2world(center, pose)[:, 0]  # [N,3]
    center_GT = camera.cam2world(center, pose_GT)[:, 0]  # [N,3]
    try:
        sim3 = camera.procrustes_analysis(center_GT, center_pred)
    except:
        print("warning: SVD did not converge...")
        sim3 = edict(
            t0=0, t1=0, s0=1, s1=1, R=torch.eye(3, device=opt.device)
        )
    # align the camera poses
    center_aligned = (
        center_pred - sim3.t1
    ) / sim3.s1 @ sim3.R.t() * sim3.s0 + sim3.t0
    R_aligned = pose[..., :3] @ sim3.R.t()
    t_aligned = (-R_aligned @ center_aligned[..., None])[..., 0]
    pose_aligned = camera.pose(R=R_aligned, t=t_aligned) # is this correct?
    return pose_aligned, sim3

def evaluate_camera_alignment(opt, pose_aligned, pose_GT):
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    R_error = camera.rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned - t_GT)[..., 0].norm(dim=-1)
    error = edict(R=R_error, t=t_error)
    return error  