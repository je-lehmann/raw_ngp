"""
Pose and Intrinsics Optimizers as in BARF
https://github.com/chenhsuanlin/bundle-adjusting-NeRF
"""
import torch
import torch.nn.functional as F
import tqdm
from easydict import EasyDict as edict

def to_hom(X):
    """
    get homogeneous coordinates of the input
    
    Args:
        X: tensor of shape [H*W, 2]
    Returns:
        torch tensor of shape [H*W, 3]
    """
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

    
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


def invert_pose(pose,use_inverse=False):
    # invert a camera pose
    R,t = pose[...,:3],pose[...,3:]

    R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
    t_inv = (-R_inv@t)[...,0]
    pose_inv = construct_pose(R=R_inv,t=t_inv)
    return pose_inv


def compose_poses(pose_list):
    # compose a sequence of poses together
    # pose_new(x) = poseN o ... o pose2 o pose1(x)
    pose_new = pose_list[0]
    for pose in pose_list[1:]:
        pose_new = compose_pair(pose_new,pose)
    return pose_new


def compose_pair(pose_a, pose_b):
    # pose_new(x) = pose_b o pose_a(x)
    R_a,t_a = pose_a[...,:3], pose_a[...,3:]
    R_b,t_b = pose_b[...,:3], pose_b[...,3:]
    #print(R_a.shape,R_b.shape)
    R_new = R_b@R_a
    t_new = (R_b@t_a+t_b)[...,0]
    pose_new = construct_pose(R=R_new,t=t_new)
    return pose_new
# basic operations of transforming 3D points between world/camera/image coordinates
def cam2world(X, pose):
    """
    Args:
        X: 3D points in camera space with [H*W, 3]
        pose: camera pose. camera_from_world, or world_to_camera [3, 3]
    Returns:
        transformation in world coordinate system.
    """
    # X of shape 64x3
    # X_hom is of shape 64x4
    X_hom = to_hom(X)
    # pose_inv is world_from_camera pose is of shape 3x4
    pose_inv = invert_pose(pose)
    # world = camera * world_from_camera
    return X_hom@pose_inv.transpose(-1,-2)


def img2cam(X, cam_intr):
    """
    Args:
        X: 3D points in image space.
        cam_intr: camera intrinsics. camera_from_image
    Returns:
        trasnformation in camera coordinate system
    """
    # camera = image * image_from_camera
    return X@cam_intr.inverse().transpose(-1,-2)

def procrustes_analysis(X0,X1): # [N,3]
    """ procruste analysis: statistical shape analysis used to analyse the distribution of a set of shapes.

    Often, these points are selected on the continuous surface of complex objects, such as a human bone,
    and in this case they are called landmark points. The shape of an object can be considered as a member 
    of an equivalence class formed by removing the translational, rotational and uniform scaling components.

    Reference: https://www.youtube.com/watch?v=nnJvBa_ERL0
    Assume the point correspondences are known. subject to RR^T = I, det(R) = 1 for R*, T* where argmin(|Ai-RBi - T|^2) for point i.
    differentiate by translation and set to zero, we have:
    T* = bar(A) - R bar(B), where bar is the average points.
    substitue translation into objective function, we have
    R* = argmin(|Ai - RBi - (bar(A) - R bar(B))|^2) = argmin(|(Ai - bar(A)) - R (Bi - bar(B))|^2)
    here Ai - bar(A) and Bi - bar(B) are centering the point sets.
    After centering, estimate remaining rotation between point sets.
    R* = argmin|Ai - RBi|^ subject to RR^T = I, det(R) = 1 becomes an orthogonal procrustes problem.
    due to invariant to cyclic permutation of matrix products tr(ABC) = tr(CAB) = tr(BCA)
    expand the function square
    R* = argmin[tr(Ai^TAi) + tr((RBi)^T(RBi)) - 2 tr(Ai^TRBi)] where tr(Bi^TR^TRBi) = tr(Bi^TBi)
    = argmax tr(Ai^TRBi)
    = argmax tr(RBiAi^T)
    compute svd of BiAi^T,  
    A = UDV^T, where mxn (orthogonal columns), nxn (non-negative entries, singular values, descending order), nxn (orthogonal matrix)
    R* = argmax tr(RUDV^T) = argmax tr(V^T R UD)
    tr(V^TRUD) = tr(ZD) = sum_i^N zii Dii, matrix Z is orthogonal since it is a product of three orthogonal matrices.
    Diagonal elements of Z are less than or equal to 1. The rotation objective is maximized when Z = I.
    Z = V^T RU = I
    T* = bar(A) - R* bar(B)
    R* = VU^T
    BiAi^T = UDV^T 
    """
    # translation
    t0 = X0.mean(dim=0,keepdim=True)
    t1 = X1.mean(dim=0,keepdim=True)
    X0c = X0-t0 # centering points
    X1c = X1-t1
    # scale
    s0 = (X0c**2).sum(dim=-1).mean().sqrt()
    s1 = (X1c**2).sum(dim=-1).mean().sqrt()
    X0cs = X0c/s0 # scaling points
    X1cs = X1c/s1
    # rotation (use double for SVD, float loses precision)
    U,S,V = (X0cs.t()@X1cs).double().svd(some=True)
    R = (U@V.t()).float()

    # If R.det < 0 --> improper rotations with reflections.
    if R.det()<0: R[2] *= -1
    # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
    sim3 = edict(t0=t0[0],t1=t1[0],s0=s0,s1=s1,R=R)
    return sim3


def rotation_distance(R1,R2,eps=1e-7):
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    R_diff = R1@R2.transpose(-2,-1)
    trace = R_diff[...,0,0]+R_diff[...,1,1]+R_diff[...,2,2]
    angle = ((trace-1)/2).clamp(-1+eps,1-eps).acos_() # numerical stability near -1/+1
    return angle

def prealign_cameras(pred_poses, gt_poses):
    """
    Pre-align predicted camera poses with gt camera poses so that they are adjusted to similar
    coordinate frames. This is helpful for evaluation.
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py#L107

    Args:
        pred_poses: predicted camera transformations. Tensor of shape [N, 3, 4]
        gt_poses: ground truth camera transformations. Tensor of shape [N, 3, 4]
    Returns:
        pose_aligned: aligned predicted camera transformations.
        sim3: 3D similarity transform information. Dictionary with keys 
            ['t0', 't1', 's0', 's1', 'R']. 
    """
    # compute 3D similarity transform via Procrustes analysis
    center = torch.zeros(1, 1, 3,device=pred_poses.device)
    pred_centers = cam2world(center, pred_poses)[:,0]# [N,3]
    gt_centers = cam2world(center, gt_poses)[:,0] # [N,3]
    try:
        # sim3 has keys of ['t0', 't1', 's0', 's1', 'R']
        sim3 = procrustes_analysis(gt_centers, pred_centers)
    except:
        print("warning: SVD did not converge for procrustes_analysis...")
        sim3 = edict(t0=0, t1=0, s0=1, s1=1, R=torch.eye(3,device=pred_poses.device))
    # align the camera poses    
    center_aligned = (pred_centers.double()-sim3.t1.double())/sim3.s1.double()@sim3.R.t().double()*sim3.s0.double()+sim3.t0.double()
    R_aligned = pred_poses[...,:3].double()@sim3.R.t().double()
    t_aligned = (-R_aligned.double()@center_aligned[...,None].double())[...,0]
    pose_aligned = construct_pose(R=R_aligned.double(), t=t_aligned.double())
    return pose_aligned, sim3


def evaluate_camera_alignment(pose_aligned,pose_GT):
    """
    Evaluate camera pose estimation with average Rotation and Translation errors.
    Reference: https://github.com/chenhsuanlin/bundle-adjusting-NeRF/blob/main/model/barf.py#L125
    """
    # measure errors in rotation and translation
    R_aligned, t_aligned = pose_aligned.split([3, 1], dim=-1)
    R_GT, t_GT = pose_GT.split([3, 1], dim=-1)
    R_error = rotation_distance(R_aligned, R_GT)
    t_error = (t_aligned-t_GT)[..., 0].norm(dim=-1)
    error = edict(R=R_error, t=t_error)
    return error