import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from barf import camera_optimizers
# from torch.utils.tensorboard import SummaryWriter
import torch

class MLP(nn.Module): # viewMLP and gridMLP
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, opt, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.opt = opt

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if (self.opt.internal_activation == 'relu'):
                    x = F.relu(x, inplace=True)
                if(self.opt.internal_activation == 'softplus'):
                    x = F.softplus(x, beta = self.opt.beta, threshold = 20)
        return x

class NeRFNetwork(NeRFRenderer):
    def __init__(self,opt,):
        super().__init__(opt)
        self.annealing = 0.0

        # pose optimizer
        if(opt.pose_opt != 'none'):
            self.pose_optimizer = camera_optimizers.CameraOptimizer(num_cameras=opt.num_cameras, device=self.opt.device, opt=self.opt)
        
        # grid
        self.level_dim = 2
        self.grid_encoder, self.grid_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=self.level_dim, num_levels=16, log2_hashmap_size=self.opt.hashmap_size, desired_resolution=self.opt.hashgrid_resolution*self.bound) # 19, bound*2048 524288*2
        self.grid_mlp = MLP(self.grid_in_dim, 16, 64, 3, opt, bias=False)

        # view-dependency
        self.view_encoder, self.view_in_dim = get_encoder('sh', input_dim=3, degree=4)

        # view encoder
        ldir_dim = self.view_in_dim if self.opt.rfield else 0
        self.view_mlp = MLP(15 + self.view_in_dim + ldir_dim, 3, 64 + ldir_dim, 3, opt, bias=False)

        # proposal network
        if not self.opt.cuda_ray:
            self.prop_encoders = nn.ModuleList()
            self.prop_mlp = nn.ModuleList()

            # hard coded 2-layer prop network
            prop0_encoder, prop0_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=128)
            prop0_mlp = MLP(prop0_in_dim, 1, 16, 2, opt, bias=False)
            self.prop_encoders.append(prop0_encoder)
            self.prop_mlp.append(prop0_mlp)

            prop1_encoder, prop1_in_dim = get_encoder("hashgrid", input_dim=3, level_dim=2, num_levels=5, log2_hashmap_size=17, desired_resolution=256)
            prop1_mlp = MLP(prop1_in_dim, 1, 16, 2, opt, bias=False)
            self.prop_encoders.append(prop1_encoder)
            self.prop_mlp.append(prop1_mlp)

    def common_forward(self, x):

        f = self.grid_encoder(x, bound=self.bound)
        if(self.opt.pose_opt == 'baangp'): # takes more video memory :/
            L = self.grid_mlp.dim_out - 1 
            start, end = self.opt.start_annealing, self.opt.end_annealing
            k = torch.arange(L, dtype=torch.float32, device=self.opt.device)
            if end == 0:
                end = 1e-12
            alpha = (self.annealing - start)/(end - start) * L
            weight = (1 - (alpha - k).clamp_(min = 0,max = 1).mul_(np.pi).cos_())/2
            weights = torch.cat([torch.ones(self.level_dim, device=self.opt.device), weight.repeat_interleave(self.level_dim)])
            n_features = f.shape[-1]
            assert n_features == len(weights)
            available_features = f[..., weights > 0]
            if len(available_features) <= 0:
                assert False, "no features are selected!"
            coarse_features = available_features[..., -self.level_dim:]
            if not self.opt.cuda_ray:
                coarse_f = coarse_features.repeat(1, 1, L+1)
            else:
                coarse_f = coarse_features.repeat(1, L+1)
            weights[0:2] = 1 # start with 1,1 then anneal
            f = f * weights + coarse_f * (1 - weights)
        
        if(self.opt.pose_opt == 'barf'):
            L = self.grid_mlp.dim_out # self.grid_mlp num_levels 16
            start, end = self.opt.start_annealing, self.opt.end_annealing
            k = torch.arange(L, dtype=torch.float32, device=self.opt.device)
            if end == 0:
                end = 1e-12
            alpha = (self.annealing - start)/(end - start) * L
            weight = (1 - (alpha - k).clamp_(min = 0,max = 1).mul_(np.pi).cos_())/2
            weights = weight.repeat_interleave(self.level_dim)
            weights[0:2] = 1 # start with 1,1 then anneal
            f = f * weights 
        
        f = self.grid_mlp(f)  
        if self.opt.density_activation == 'clamped_exp':
            sigma = trunc_exp(f[..., 0])
        else:
            sigma = F.softplus(f[..., 0], beta = self.opt.beta, threshold = 20) 
        feat = f[..., 1:]
        return sigma, feat

    def forward(self, x, d, ld=None, **kwargs): 
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
                        
        sigma, feat = self.common_forward(x)
        d = self.view_encoder(d) # nan in eval mode for normals?
        if self.opt.rfield:
            ld = self.view_encoder(ld)
            color = self.view_mlp(torch.cat([feat, d, ld], dim=-1))
        else:
            color = self.view_mlp(torch.cat([feat, d], dim=-1))
        
        if(self.opt.color_activation == 'exp'):
            color = torch.exp(color - 5.0)

        if(self.opt.color_activation == 'sigmoid'):
            color = torch.sigmoid(color)

        if(self.opt.color_activation == 'clamped_exp'):
            color = torch.clamp(torch.exp(color - 5.0), max=5.0)
        
        return {
            'sigma': sigma,
            'color': color
        }

    def density(self, x, proposal=-1):

        # proposal network
        if proposal >= 0 and proposal < len(self.prop_encoders):
            sigma = trunc_exp(self.prop_mlp[proposal](self.prop_encoders[proposal](x, bound=self.bound)).squeeze(-1))
        # final NeRF
        else:
            sigma, _ = self.common_forward(x)

        return {
            'sigma': sigma,
        }
    
    def apply_total_variation(self, w):
        self.grid_encoder.grad_total_variation(w)

    def apply_weight_decay(self, w):
        self.grid_encoder.grad_weight_decay(w)

    def update_annealing(self, new_value):
        self.annealing = new_value
    
    # optimizer utils
    def get_params(self, lr):

        params = []

        params.extend([
            {'params': self.grid_encoder.parameters(), 'lr': lr},
            {'params': self.grid_mlp.parameters(), 'lr': lr}, 
            {'params': self.view_mlp.parameters(), 'lr': lr}, 
        ])
    
        if not self.opt.cuda_ray:
            params.extend([
                {'params': self.prop_encoders.parameters(), 'lr': lr},
                {'params': self.prop_mlp.parameters(), 'lr': lr},
            ])

        return params
