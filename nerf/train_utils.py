import os
import glob
import tqdm
import math
import imageio
import random
import warnings
import tensorboardX

import numpy as np

import time

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from raw import raw_utils
from img import image_utils


try:
    from torchmetrics.functional import structural_similarity_index_measure
except: # old versions
    from torchmetrics.functional import ssim as structural_similarity_index_measure

import trimesh
from rich.console import Console
from torch_ema import ExponentialMovingAverage

from packaging import version as pver
import lpips


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def create_dodecahedron_cameras(radius=1, center=np.array([0, 0, 0])):

    vertices = np.array([
        -0.57735,  -0.57735,  0.57735,
        0.934172,  0.356822,  0,
        0.934172,  -0.356822,  0,
        -0.934172,  0.356822,  0,
        -0.934172,  -0.356822,  0,
        0,  0.934172,  0.356822,
        0,  0.934172,  -0.356822,
        0.356822,  0,  -0.934172,
        -0.356822,  0,  -0.934172,
        0,  -0.934172,  -0.356822,
        0,  -0.934172,  0.356822,
        0.356822,  0,  0.934172,
        -0.356822,  0,  0.934172,
        0.57735,  0.57735,  -0.57735,
        0.57735,  0.57735,  0.57735,
        -0.57735,  0.57735,  -0.57735,
        -0.57735,  0.57735,  0.57735,
        0.57735,  -0.57735,  -0.57735,
        0.57735,  -0.57735,  0.57735,
        -0.57735,  -0.57735,  -0.57735,
        ]).reshape((-1,3), order="C")

    length = np.linalg.norm(vertices, axis=1).reshape((-1, 1))
    vertices = vertices / length * radius + center

    # construct camera poses by lookat
    def normalize(x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    # forward is simple, notice that it is in fact the inversion of camera direction!
    forward_vector = normalize(vertices - center)
    # pick a temp up_vector, usually [0, 1, 0]
    up_vector = np.array([0, 1, 0], dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    # cross(up, forward) --> right
    right_vector = normalize(np.cross(up_vector, forward_vector, axis=-1))
    # rectify up_vector, by cross(forward, right) --> up
    up_vector = normalize(np.cross(forward_vector, right_vector, axis=-1))

    ### construct c2w
    poses = np.eye(4, dtype=np.float32)[None].repeat(forward_vector.shape[0], 0)
    poses[:, :3, :3] = np.stack((right_vector, up_vector, forward_vector), axis=-1)
    poses[:, :3, 3] = vertices

    return poses


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1, coords=None, ldirs= None):
    ''' get rays
    Args:
        poses: [N/1, 4, 4], cam2world N when random image sampling
        intrinsics: [N/1, 4] tensor or [4] ndarray
        H, W, N: int
    Returns:
        rays_o, rays_d: [N, 3]
        i, j: [N]
    '''
    # patch_size = 2
    device = poses.device
    if isinstance(intrinsics, np.ndarray):
        fx, fy, cx, cy = intrinsics
    else:
        fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device)) # float
    i = i.t().contiguous().view(-1) + 0.5
    j = j.t().contiguous().view(-1) + 0.5

    results = {}

    if N > 0:
       
        if coords is not None:
            inds = coords[:, 0] * W + coords[:, 1]

        elif patch_size > 1:

            # random sample left-top cores.
            num_patch = N // (patch_size ** 2)
            inds_x = torch.randint(0, H - patch_size, size=[num_patch], device=device)
            inds_y = torch.randint(0, W - patch_size, size=[num_patch], device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1) # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device), torch.arange(patch_size, device=device))
            offsets = torch.stack([pi.reshape(-1), pj.reshape(-1)], dim=-1) # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0) # [np, p^2, 2]
            inds = inds.view(-1, 2) # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1] # [N], flatten

        else: # random sampling
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['i'] = i.long()
        results['j'] = j.long()

    else:
        inds = torch.arange(H*W, device=device)

    zs = -torch.ones_like(i) # z is flipped
    xs = (i - cx) / fx
    ys = -(j - cy) / fy # y is flipped
    directions = torch.stack((xs, ys, zs), dim=-1) # [N, 3]
    # do not normalize to get actual depth, ref: https://github.com/dunbar12138/DSNeRF/issues/29
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True) 
    
    rays_d = (directions.unsqueeze(1) @ poses[:, :3, :3].transpose(-1, -2)).squeeze(1) # [N, 1, 3] @ [N, 3, 3] --> [N, 1, 3]

    rays_o = poses[:, :3, 3].expand_as(rays_d) # [N, 3]
    rays_ldir = ldirs.expand_as(rays_d) if ldirs != None else None # [N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['rays_ldir'] = rays_ldir

    #print(rays_o.shape, rays_d.shape, rays_ldir.shape, rays_ldir, rays_o, rays_d)

    # visualize_rays(rays_o[0].detach().cpu().numpy(), rays_d[0].detach().cpu().numpy())

    return results


def visualize_rays(rays_o, rays_d):
    
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i in range(0, rays_o.shape[0], 10):
        ro = rays_o[i]
        rd = rays_d[i]

        segs = np.array([[ro, ro + rd * 3]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

        return psnr

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'

class LPIPSMeter:
    def __init__(self, net='vgg', device=None):
        self.V = 0
        self.N = 0
        self.net = net

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs
    
    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [H, W, 3] --> [B, 3, H, W], range in [0, 1]
        v = self.fn(truths, preds, normalize=True).item() # normalize=True: [0, 1] to [-1, 1]
        self.V += v
        self.N += 1

        return v
    
    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, f"LPIPS ({self.net})"), self.measure(), global_step)

    def report(self):
        return f'LPIPS ({self.net}) = {self.measure():.6f}'

class SSIMMeter:
    def __init__(self, device=None):
        self.V = 0
        self.N = 0

        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if len(inp.shape) == 3:
                inp = inp.unsqueeze(0)
            inp = inp.permute(0, 3, 1, 2).contiguous() # [B, 3, H, W]
            inp = inp.to(self.device)
            outputs.append(inp)
        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, H, W, 3] --> [B, 3, H, W], range in [0, 1]

        ssim = structural_similarity_index_measure(preds, truths)
        #cv2.imwrite('./debug/ssim_pred.png', np.array(255. * np.clip(preds.cpu(), 0, 1)).astype(np.uint8))
        #cv2.imwrite('./debug/ssim_truth.png', np.array(255. * np.clip(truths.cpu(), 0, 1)).astype(np.uint8))

        self.V += ssim
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "SSIM"), self.measure(), global_step)

    def report(self):
        return f'SSIM = {self.measure():.6f}'

class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 save_interval=1, # save once every $ epoch (independently from eval)
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ):
        
        self.opt = opt
        self.name = name
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        # try out torch 2.0
        if torch.__version__[0] == '2':
            model = torch.compile(model)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if opt.pose_opt != 'none':
            self.pose_optimizer = self.model.pose_optimizer

        if ema_decay is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=ema_decay)
        else:
            self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.steps_per_epoch = 0
        self.annealing = 0.0
        self.annealing_thres = 0.0
        self.max_epochs = 0
        
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        #self.log(opt)
        self.log(self.model)

        if self.workspace is not None:

            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file

    ### ------------------------------	

    def train_step(self, data):
        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        rays_ldir = data['rays_ldir'] if self.opt.rfield else None # [N, 3]
        index = data['index'] # [1/N]
            
        cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None
        self.annealing = np.clip((self.global_step / (self.max_epochs * self.steps_per_epoch)), 0, 1).astype(np.float16) # devision by 0!
        self.model.update_annealing(self.annealing)

        images = data['images'] # [N, 3/4]
        N, C = images.shape

        if self.opt.background == 'random':
            bg_color = torch.rand(N, 3, device=self.device) # [N, 3], pixel-wise random.
        
        if self.opt.background == 'white' or self.opt.background == 'last_sample':
            bg_color = 1
        
        if self.opt.background == 'black':
            bg_color = 0

        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        shading = 'diffuse' if self.global_step < self.opt.diffuse_step else 'full'
        update_proposal = self.global_step <= 3000 or self.global_step % 5 == 0
        
        outputs = self.model.render(rays_o, rays_d, rays_ldir=rays_ldir, bg_color=bg_color, perturb=True, cam_near_far=cam_near_far, shading=shading, update_proposal=update_proposal)
        pred_rgb = outputs['image']
        if(self.opt.image_mode == 'HDR'):
            # scale prediction by exposure
            exposure = torch.from_numpy(data['exposure']).to(self.device).detach()
            # apply bayer masking when training on mosaiced data
            lossmult = data['lossmult'] if 'lossmult' in data else 1.0
            lossmult = torch.tensor(lossmult).float().to(self.device)
            lossmult_tensor = torch.broadcast_to(torch.tensor(lossmult), gt_rgb[..., :3].shape)
           
            if self.opt.loss_weight == 'gaussian':
                loss_weight = raw_utils.gaussian_weighting(gt_rgb)
            elif self.opt.loss_weight == 'planck':
                loss_weight = raw_utils.planck_taper_weighting(gt_rgb)
            elif self.opt.loss_weight == 'hanning':
                loss_weight = raw_utils.hanning_weighting(gt_rgb)
            else:
                loss_weight = 1.
                
            # use the clipped MSE loss from the raw nerf paper
            rgb_render_clip = torch.minimum(torch.tensor(1.), pred_rgb * exposure.unsqueeze(1))
            resid_sq_clip = (rgb_render_clip - gt_rgb)**2 
            # Scale by gradient of log tonemapping curve.
            scaling_grad = 1. / (1e-3 + rgb_render_clip.detach())

            data_loss = (resid_sq_clip * scaling_grad**2) 
            loss = (data_loss * lossmult_tensor * loss_weight).sum() / lossmult_tensor.sum()

        else:
            # Criterion loss (e.g. MSE)
            loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [N, 3] --> [N]    
            loss = loss.mean()
        # extra loss
        if 'proposal_loss' in outputs and self.opt.lambda_proposal > 0:
            loss = loss + self.opt.lambda_proposal * outputs['proposal_loss']
            
        if 'orientation_loss' in outputs and self.opt.lambda_orientation > 0:
            #print('adding', outputs['orientation_loss'])
            loss = loss + self.opt.lambda_orientation * outputs['orientation_loss']
            
        if 'distort_loss' in outputs and self.opt.lambda_distort > 0:
            loss = loss + outputs['distort_loss'] * self.opt.lambda_distort
            #print('distort loss: ', outputs['distort_loss'])

        if self.opt.lambda_entropy > 0:
            w = outputs['weights_sum'].clamp(1e-5, 1 - 1e-5)
            entropy = - w * torch.log2(w) - (1 - w) * torch.log2(1 - w)
            loss = loss + self.opt.lambda_entropy * (entropy.mean())

        if self.use_tensorboardX:
            self.writer.add_histogram("train/depth_prediction", outputs['depth'], self.global_step)
        
        # adaptive num_rays
        if self.opt.adaptive_num_rays:
            self.opt.num_rays = int(round((self.opt.num_points / outputs['num_points']) * self.opt.num_rays))
            
        # self.opt.corrected_poses[index] = [rays_o.detach().cpu().numpy(), rays_d.detach().cpu().numpy()]

        return pred_rgb, gt_rgb, loss

    def post_train_step(self):

        # unscale grad before modifying it!
        # ref: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        self.scaler.unscale_(self.optimizer)
        # the new inplace TV loss
        if self.opt.lambda_tv > 0:
            self.model.apply_total_variation(self.opt.lambda_tv)
        
        if self.opt.lambda_wd > 0:
            self.model.apply_weight_decay(self.opt.lambda_wd)
                
    def eval_step(self, data):
        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]
        rays_ldir = data['rays_ldir'] if self.opt.rfield else None # [N, 3]
        images = data['images'] # [H, W, 3/4]
        index = data['index'] # [1/N]
        #global_idx = self.opt.val_ids[index[0].cpu().numpy()]

        H, W, C = images.shape
        N = H * W
        batch_size = N // self.opt.eval_batch
        num_batches = (N + batch_size - 1) // batch_size

        pred_rgb_list = []
        pred_depth_list = []
        normals_list = []
        gt_rgb_list = []
        loss_list = []

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            
            rays_o_batch = rays_o[start:end]
            rays_d_batch = rays_d[start:end]
            rays_ldir_batch = rays_ldir[start:end] if self.opt.rfield else None 
            images_batch = images.reshape(N, C)[start:end]

            cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None

            # eval with fixed white background color
            bg_color = 1
            
            if self.opt.background == 'black':
                bg_color = 0

            if C == 4:
                gt_rgb = images_batch[..., :3] * images_batch[..., 3:] + bg_color * (1 - images_batch[..., 3:])
            else:
                gt_rgb = images_batch
                
            outputs = self.model.render(rays_o_batch, rays_d_batch, rays_ldir=rays_ldir_batch, bg_color=bg_color, perturb=True, cam_near_far=cam_near_far)

            pred_depth = outputs['depth']
            normals = outputs['normals'] if 'normals' in outputs else None
            
            if(self.opt.image_mode == 'HDR'):
                exposure = torch.from_numpy(data['exposure']).to(self.device).detach()
                pred_rgb = outputs['image']
                
                if self.opt.loss_weight == 'gaussian':
                    loss_weight = raw_utils.gaussian_weighting(gt_rgb, sigma=0.5, peak_value=0.5)
                elif self.opt.loss_weight == 'planck':
                    loss_weight = raw_utils.planck_taper_weighting(gt_rgb)
                elif self.opt.loss_weight == 'hanning':
                    loss_weight = raw_utils.hanning_weighting(gt_rgb)
                elif self.opt.loss_weight == 'none':
                    loss_weight = 1.
                
                # Clipped MSE loss from the raw nerf paper
                rgb_render_clip = torch.minimum(torch.tensor(1.), pred_rgb * exposure.unsqueeze(1))
                resid_sq_clip = (rgb_render_clip - gt_rgb)**2 
                # Scale by gradient of log tonemapping curve.
                scaling_grad = 1. / (1e-3 + rgb_render_clip.detach())
                # Reweighted L2 loss.
                loss = (resid_sq_clip * loss_weight * scaling_grad**2).mean(-1)
                loss = loss.mean()   
            else:
                # Criterion loss (e.g. MSE)
                pred_rgb = outputs['image'].reshape(N,3)[start:end]
                loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [N, 3] --> [N]    
                loss = loss.mean() 

            pred_rgb_list.append(pred_rgb)
            pred_depth_list.append(pred_depth)
            normals_list.append(normals)
            gt_rgb_list.append(gt_rgb)
            loss_list.append(loss)

        pred_rgb = torch.cat(pred_rgb_list).reshape(H, W, 3)
        pred_depth = torch.cat(pred_depth_list).reshape(H, W)
        normals = torch.cat(normals_list).reshape(H, W, 3) if normals_list[0] is not None else None
        gt_rgb = torch.cat(gt_rgb_list).reshape(H, W, 3)
        loss = torch.stack(loss_list).mean()

        return pred_rgb, pred_depth, normals, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, frameIndex=0, bg_color=None, perturb=False, shading='full'):  

        rays_o = data['rays_o'] # [N, 3]
        rays_d = data['rays_d'] # [N, 3]        
        rays_ldir = data['rays_ldir'][frameIndex] if self.opt.rfield else None # [N, 3]
        H, W = data['H'], data['W']
        N = H * W
        batch_size = N // self.opt.eval_batch
        num_batches = (N + batch_size - 1) // batch_size

        pred_rgb_list = []
        pred_depth_list = []
        normals_list = []

        for b in range(num_batches):
            start = b * batch_size
            end = min((b + 1) * batch_size, N)
            
            rays_o_batch = rays_o[start:end]
            rays_d_batch = rays_d[start:end]

            cam_near_far = data['cam_near_far'] if 'cam_near_far' in data else None # [1/N, 2] or None
            
            if self.opt.background == 'black':
                bg_color = 0
            if self.opt.background == 'white':
                bg_color = 1
            outputs = self.model.render(rays_o_batch, rays_d_batch, rays_ldir=rays_ldir, bg_color=bg_color, perturb=perturb, cam_near_far=cam_near_far, shading=shading)
            normals = outputs['normals'] if 'normals' in outputs else None
            pred_rgb_list.append(outputs['image'])
            pred_depth_list.append(outputs['depth'])
            normals_list.append(normals)

        pred_rgb = torch.cat(pred_rgb_list).reshape(H, W, 3)
        pred_depth = torch.cat(pred_depth_list).reshape(H, W)
        normals = torch.cat(normals_list).reshape(H, W, 3) if normals_list[0] is not None else None

        return pred_rgb, pred_depth, normals

    def save_mesh(self, save_path=None, resolution=128, decimate_target=1e5, dataset=None):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        self.model.export_mesh(save_path, resolution=resolution, decimate_target=decimate_target, dataset=dataset)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):
        self.max_epochs = max_epochs
        
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.opt.mark_untrained:
            self.model.mark_untrained_grid(train_loader._data)

        start_t = time.time()

         # only needed for debugging
        if self.opt.log_poses:
            self.opt.optimized_poses = np.zeros(((self.max_epochs * len(train_loader) * train_loader.batch_size)+1, len(self.opt.poses), 3, 4))
        
        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch
            self.steps_per_epoch = len(train_loader) * train_loader.batch_size
            
            # Freeze the parameters of the pose optimizer after annealing
            if self.opt.pose_opt != 'none' and self.annealing > self.opt.end_annealing:
                for param in self.pose_optimizer.optimizer.param_groups[0]['params']:
                    param.detach_()
                    param.requires_grad = False
                    
                for param_group in self.pose_optimizer.optimizer.param_groups:
                    param_group['lr'] = 0.0

            self.train_one_epoch(train_loader)

            if (self.epoch % self.save_interval == 0 or self.epoch == max_epochs) and self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t)/ 60:.6f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)
        
        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        
        # VIDEO OUTPUT

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normals = []
            all_preds_hdr = []

        with torch.no_grad():

            for i, data in enumerate(loader):
                preds, preds_depth, normals = self.test_step(data, i)
                pred = preds.detach().cpu().numpy()
                
                if(self.opt.image_mode == 'HDR'):
                    processed_pred = raw_utils.postprocess_raw(pred, self.opt.metadict['cam2rgb'][0],
                                                               self.opt.metadict['exposure_levels'][self.opt.exposure_percentile])
                    if self.opt.hdr_merge != 'none':
                        processed_pred_hdr = raw_utils.postprocess_raw_hdr_output(pred, self.opt.metadict['cam2rgb'][0], 
                                                        self.opt.exposure_percentiles, self.opt.hdr_merge, self.opt.hdr_tonemap)
                        
                        hdr_pred = np.clip(processed_pred_hdr * 255, 0, 255).astype(np.uint8)
                    pred = np.clip(processed_pred * 255, 0, 255).astype(np.uint8)
                else:
                    pred = (np.clip(pred, 0, 1) * 255).astype(np.uint8)

                pred_depth = preds_depth.detach().cpu().numpy()
                pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                pred_depth = (pred_depth * 255).astype(np.uint8)
                normal_map = normals.detach().cpu().numpy() if normals is not None else None

                if write_video:
                    all_preds.append(pred)
                    if self.opt.hdr_merge != 'none':
                        all_preds_hdr.append(hdr_pred[..., ::-1])
                    if self.opt.output_depth:
                        all_preds_depth.append(pred_depth)
                    if normal_map is not None:
                        all_preds_normals.append((normal_map[..., ::-1] * 255).astype(np.uint8))
                else:
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), pred)
                    if self.opt.output_depth:
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)
                    if self.opt.hdr_merge != 'none':
                        cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_hdr.png'), hdr_pred)

                pbar.update(loader.batch_size)
        
        if write_video:
            all_preds = np.stack(all_preds, axis=0) # [N, H, W, 3]
            all_preds_depth = np.stack(all_preds_depth, axis=0) if self.opt.output_depth else None # [N, H, W]
            all_preds_normals = np.stack(all_preds_normals, axis=0) if normals is not None else None # [N, H, W, 3]
            all_preds_hdr = np.stack(all_preds_hdr, axis=0) if self.opt.hdr_merge != 'none' else None # [N, H, W, 3]

            # fix ffmpeg not divisible by 2
            all_preds = np.pad(all_preds, ((0, 0), (0, 1 if all_preds.shape[1] % 2 != 0 else 0), (0, 1 if all_preds.shape[2] % 2 != 0 else 0), (0, 0)))
            if all_preds_depth is not None:
                all_preds_depth = np.pad(all_preds_depth, ((0, 0), (0, 1 if all_preds_depth.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_depth.shape[2] % 2 != 0 else 0)))
            if all_preds_normals is not None:
                all_preds_normals = np.pad(all_preds_normals, ((0, 0), (0, 1 if all_preds_normals.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_normals.shape[2] % 2 != 0 else 0), (0, 0)))
            if all_preds_hdr is not None:
                all_preds_hdr = np.pad(all_preds_hdr, ((0, 0), (0, 1 if all_preds_hdr.shape[1] % 2 != 0 else 0), (0, 1 if all_preds_hdr.shape[2] % 2 != 0 else 0), (0, 0)))

            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb_{self.opt.eval_idx}.mp4'), all_preds[...,::-1].astype(np.uint8), fps=24, quality=8, macro_block_size=1)
            if all_preds_depth is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth.astype(np.uint8), fps=24, quality=8, macro_block_size=1)
            if all_preds_normals is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_normals.mp4'), all_preds_normals, fps=24, quality=8, macro_block_size=1)
            if all_preds_hdr is not None:
                imageio.mimwrite(os.path.join(save_path, f'{name}_hdr.mp4'), all_preds_hdr.astype(np.uint8), fps=24, quality=8, macro_block_size=1)
        
        if self.opt.hdr_merge != 'none':
            self.log(f'[INFO] Merge Algorithm: {self.opt.hdr_merge}, Tonemap Algorithm: {self.opt.hdr_tonemap}')
        self.log(f"==> Finished Test.")

    def train_one_epoch(self, loader):
        
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f}, Progress={self.annealing:.6f}...")
        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()
            
        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        self.local_step = 0
        for data in loader:
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                self.model.update_extra_state()
                    
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()
            if self.opt.pose_opt != 'none' and self.annealing < self.opt.end_annealing:
                self.pose_optimizer.optimizer.zero_grad()

            preds, truths, loss_net = self.train_step(data)
            
            loss = loss_net
            self.scaler.scale(loss).backward()
            self.post_train_step() # for TV loss...

            self.scaler.step(self.optimizer)
            if self.opt.pose_opt != 'none' and self.annealing < self.opt.end_annealing:
                self.scaler.step(self.pose_optimizer.optimizer)

            self.scaler.update()
                    
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
                if self.opt.pose_opt != 'none' and self.annealing < self.opt.end_annealing:
                    self.pose_optimizer.lr_scheduler.step()

            loss_val = loss_net.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths, self.opt)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                    # magnitudes??
                    for name, param in self.model.grid_mlp.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            self.writer.add_histogram(name + '_grid', param.grad, self.global_step)
                    
                    for name, param in self.model.view_mlp.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            self.writer.add_histogram(name + '_view', param.grad, self.global_step)
                    
                    '''if self.opt.pose_opt != 'none':
                        self.writer.add_scalar("pose_opt/c_lr", self.pose_optimizer.optimizer.param_groups[0]['lr'], self.global_step)
                        if self.global_step % 200 == 0 or self.global_step == 1:
                            rot_error, trans_error = self.pose_optimizer.analyze_pose_optimization()
                            self.writer.add_scalar("pose_opt/rot_error", rot_error, self.global_step)
                            self.writer.add_scalar("pose_opt/trans_error", trans_error, self.global_step)'''

                    
                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                    if self.opt.pose_opt != 'none':
                        pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f}), lr={self.optimizer.param_groups[0]['lr']:.6f}, c_lr={self.pose_optimizer.optimizer.param_groups[0]['lr']:.6f}")

                else:
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                pbar.update(loader.batch_size)
        
        # pose opt
        if self.opt.pose_opt != 'none' and self.annealing < self.opt.end_annealing:
            self.pose_optimizer.lr_scheduler.step()
        
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        
        self.log(f"==> Finished Epoch {self.epoch}, loss={average_loss:.6f}.")


    def evaluate_one_epoch(self, loader, name=None):
        # debug memory allocation
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        ''' IMAGE OUTPUT '''

        with torch.no_grad():
            self.local_step = 0
            
            '''Estimate additional metadata that is needed for HDR merging beforehand once every evaluation
            This is used to create consistent LDR Video Outputs. However this is deactivated for merged videos
            for now as consistent exposure values can lead to loss in detail for exposure sweeps'''
            if self.opt.image_mode == 'HDR':
                for data in loader:
                    global_idx = self.opt.val_ids[data['index'][0].cpu().numpy()]
                    #print(global_idx, self.opt.metadict['exposure_values'][global_idx])
                    if self.opt.metadict['exposure_values'][global_idx] != 1.0: 
                        continue
                    pred = self.eval_step(data)[0].detach().cpu().numpy()
                    self.opt.metadict['exposure_levels'] = {p: np.percentile(pred, p) for p in self.opt.exposure_percentiles}
                    self.log('[INFO] Exposure Levels for consistent LDR Output are: ', self.opt.metadict['exposure_levels'])
                    break
    
            for data in loader:    
                self.local_step += 1
                preds, preds_depth, normals, truths, loss = self.eval_step(data)
                
                if self.opt.eval:
                    if not os.path.exists(self.opt.workspace + '/eval/'):
                        os.makedirs(self.opt.workspace + '/eval/GT/')
                        os.makedirs(self.opt.workspace + '/eval/pred/')
                    global_idx = self.opt.val_ids[data['index'][0].cpu().numpy()]

                    #print(self.opt.metadict['filename'][global_idx], data['ldir'].detach().cpu().numpy())
                    np.save(self.opt.workspace + '/eval/GT/' + str(global_idx), truths.detach().cpu().numpy(), allow_pickle=True)
                    np.save(self.opt.workspace + '/eval/pred/' + str(global_idx), preds.detach().cpu().numpy(), allow_pickle=True)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                # loss accumulation
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    
                    metric_vals = []
                    for metric in self.metrics:
                        metric_val = metric.update(preds, truths)
                        metric_vals.append(metric_val)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_hdr = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_hdr.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    save_path_error = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_error_{metric_vals[0]:.2f}.png') # metric_vals[0] should be the PSNR
                    save_path_truth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_truth_{metric_vals[0]:.2f}.png') # metric_vals[0] should be the PSNR
                    save_path_processed = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_preprocessed_{metric_vals[0]:.2f}.png') # metric_vals[0] should be the PSNR
                    save_path_normal = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_normal_{metric_vals[0]:.2f}.png')
                    
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds.detach().cpu().numpy()
                    if(self.opt.image_mode == 'HDR'):                        
                        processed_pred = raw_utils.postprocess_raw(pred, self.opt.metadict['cam2rgb'][0],
                                                                   self.opt.metadict['exposure_levels'][self.opt.exposure_percentile])
                        if self.opt.hdr_merge != 'none':
                            processed_pred_hdr = raw_utils.postprocess_raw_hdr_output(pred, self.opt.metadict['cam2rgb'][0], 
                                                            self.opt.exposure_percentiles, self.opt.hdr_merge, self.opt.hdr_tonemap)
                            hdr = np.clip(processed_pred_hdr * 255, 0, 255).astype(np.uint8)

                        pred = np.clip(processed_pred * 255, 0, 255).astype(np.uint8)
                    else:
                        pred = (np.clip(pred, 0, 1) * 255.).astype(np.uint8)

                    pred_depth = preds_depth.detach().cpu().numpy()
                    normal_map = normals.detach().cpu().numpy() if normals is not None else None
                   
                    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-6)
                    pred_depth = (pred_depth * 255).astype(np.uint8)
                    
                    truth = truths.detach().cpu().numpy()
                    if self.opt.image_mode == 'HDR':
                        truth = raw_utils.postprocess_raw(truth, self.opt.metadict['cam2rgb'][0],
                                                          self.opt.metadict['exposure_levels'][self.opt.exposure_percentile])
                    truth = (truth * 255).astype(np.uint8)
                    error = np.abs(truth.astype(np.float32) - pred.astype(np.float32)).mean(-1).astype(np.uint8)
                    
                    if normal_map is not None:
                        cv2.imwrite(save_path_normal, (normal_map[..., ::-1] * 255).astype(np.uint8))
                    
                    if self.opt.output_depth:
                        cv2.imwrite(save_path_depth, pred_depth)
                    if self.opt.output_error:
                        cv2.imwrite(save_path_error, error)
                    if self.opt.output_gt:
                        cv2.imwrite(save_path_truth, truth)
                    if self.opt.bracketing or self.opt.hdr_merge != 'none':
                        cv2.imwrite(save_path_hdr, hdr)
                    
                    cv2.imwrite(save_path, pred)
                  
                    pbar.set_description(f"loss={loss_val:.6f} ({total_loss/self.local_step:.6f})")
                    pbar.update(loader.batch_size)
        
        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()
        
        if self.opt.hdr_merge != 'none':
            self.log(f'[INFO] Merge Algorithm: {self.opt.hdr_merge}, Tonemap Algorithm: {self.opt.hdr_tonemap}')

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'local_step': self.local_step,
            'steps_per_epoch': self.steps_per_epoch,
            'annealing': self.annealing,
            'max_epochs': self.max_epochs,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_density'] = self.model.mean_density
            if self.use_tensorboardX:
                self.writer.add_scalar("train/mean_density", self.model.mean_density, self.global_step)
            
        state['model'] = self.model.state_dict()
        if 'density_grid' in state['model']:
            if self.use_tensorboardX:
                self.writer.add_histogram("train/density_grid", state['model']['density_grid'], self.global_step)
        
        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.opt.pose_opt != 'none':
                state['cam_optimizer'] = self.pose_optimizer.optimizer.state_dict()
                state['cam_lr_scheduler'] = self.pose_optimizer.lr_scheduler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    # we don't consider continued training from the best ckpt, so we discard the unneeded density_grid to save some storage (especially important for dnerf)
                    if 'density_grid' in state['model']:
                        if self.use_tensorboardX:
                            self.writer.add_histogram("train/density_grid", state['model']['density_grid'], self.global_step)

                    #     del state['model']['density_grid']

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):

        if checkpoint is None: # load latest
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth')) 

            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, abort loading latest model.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
    
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.annealing = checkpoint_dict['annealing']
        self.model.update_annealing(self.annealing)
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.opt.pose_opt != 'none' and 'cam_optimizer' in checkpoint_dict:
            try:
                self.pose_optimizer.optimizer.load_state_dict(checkpoint_dict['cam_optimizer'])
                self.log("[INFO] loaded cam_optimizer.")
            except:
                self.log("[WARN] Failed to load camera optimizer.")
        
        if self.opt.pose_opt != 'none' and 'cam_lr_scheduler' in checkpoint_dict:
            try:
                self.pose_optimizer.lr_scheduler.load_state_dict(checkpoint_dict['cam_lr_scheduler'])
                self.log("[INFO] loaded cam_lr_scheduler.")
            except:
                self.log("[WARN] Failed to load camera scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")