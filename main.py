import torch
import argparse
from nerf.train_utils import *

if __name__ == '__main__':
    #torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="recommended settings")
    parser.add_argument('-O2', action='store_true', help="recommended settings")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest') # choose scratch to retrain
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    ### testing options
    parser.add_argument('--save_cnt', type=int, default=50, help="save checkpoints for $ times during training")
    parser.add_argument('--eval_cnt', type=int, default=10, help="perform validation for $ times during training")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--test_no_video', action='store_true', help="test mode: do not save video")
    parser.add_argument('--test_no_mesh', action='store_true', help="test mode: do not save mesh")
    parser.add_argument('--camera_traj', type=str, default='interp', help="interp for interpolation, circle for circular camera")

    ### dataset options
    parser.add_argument('--data_format', type=str, default='colmap', choices=['nerf', 'colmap', 'dtu'])
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'all']) # use trainval for barf?
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    parser.add_argument('--downscale', type=int, default=1, help="downscale training images")
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=-1, help="scale camera location into box[-bound, bound]^3, -1 means automatically determine based on camera poses..")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--enable_cam_near_far', action='store_true', help="colmap mode: use the sparse points to estimate camera near far per view.")
    parser.add_argument('--enable_cam_center', action='store_true', help="use camera center instead of sparse point center (colmap dataset only)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--T_thresh', type=float, default=1e-8, help="minimum transmittance to continue ray marching") #1e-4 before 1e-8 is good

    ### training options
    parser.add_argument('--iters', type=int, default=20000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, nargs='*', default=[256, 96, 48], help="num steps sampled per ray for each proposal level (only valid when NOT using --cuda_ray)")
    parser.add_argument('--contract', action='store_true', help="apply spatial contraction as in mip-nerf 360, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--background', type=str, default='black', choices=['white', 'random', 'last_sample', 'black'], help="training background mode")

    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096 * 4, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--grid_size', type=int, default=128, help="density grid resolution") 
    parser.add_argument('--mark_untrained', action='store_true', help="mark_untrained grid")
    parser.add_argument('--dt_gamma', type=float, default=0, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)") 
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied") 
    
    parser.add_argument('--hashgrid_resolution', type=int, default=2048, help="desired resolution * bound for hash grid encoding")
    parser.add_argument('--hashmap_size', type=int, default=19, help="log2 hash map size")

    # batch size related
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step") 
    parser.add_argument('--adaptive_num_rays', action='store_true', help="adaptive num rays for more efficient training")
    parser.add_argument('--num_points', type=int, default=2 ** 18, help="target num points for each training step, only work with adaptive num_rays") 

    # Regularizations
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale") #0
    parser.add_argument('--lambda_tv', type=float, default=0, help="loss scale")#0 Total Variation
    parser.add_argument('--lambda_wd', type=float, default=0, help="loss scale")#0 Weight Decay 
    parser.add_argument('--lambda_orientation', type=float, default=0, help="loss scale")#0 Ref Nerf
    parser.add_argument('--lambda_proposal', type=float, default=1, help="loss scale (only for non-cuda-ray mode)")
    parser.add_argument('--lambda_distort', type=float, default=0, help="loss scale (only for non-cuda-ray mode)") #0.002 Distortion Loss

    ### mesh options
    parser.add_argument('--mcubes_reso', type=int, default=512, help="resolution for marching cubes")
    parser.add_argument('--env_reso', type=int, default=256, help="max layers (resolution) for env mesh")
    parser.add_argument('--decimate_target', type=int, default=3e5, help="decimate target for number of triangles, <=0 to disable")
    parser.add_argument('--mesh_visibility_culling', action='store_true', help="cull mesh faces based on visibility in training dataset")
    parser.add_argument('--visibility_mask_dilation', type=int, default=5, help="visibility dilation")
    parser.add_argument('--clean_min_f', type=int, default=8, help="mesh clean: min face count for isolated mesh")
    parser.add_argument('--clean_min_d', type=int, default=5, help="mesh clean: min diameter for isolated mesh")
    
    # Validation Image Writing Options
    parser.add_argument('--output_depth', action='store_true', help="")
    parser.add_argument('--output_gt', action='store_true', help="")
    parser.add_argument('--output_error', action='store_true', help="")

    ### RAW Options, only COLMAP
    parser.add_argument('--image_mode', type=str, default='LDR', choices=['LDR', 'HDR'], help="determine image type: LDR for jpg/png or HDR for dng/exr")
    parser.add_argument('--expose', action='store_true', help="expose image before training")
    parser.add_argument('--exposure_range', type=str, default='minimal', choices=['minimal', 'wide'], help="determine how many percentiles are collected for hdr")
    parser.add_argument('--clip', action='store_true', help="clipping hdr lightstage data")
    parser.add_argument('--internal_activation', type=str, default='relu', choices=['relu', 'softplus'])
    parser.add_argument('--color_activation', type=str, default='clamped_exp', choices=['exp', 'sigmoid', 'clamped_exp'])
    parser.add_argument('--density_activation', type=str, default='clamped_exp', choices=['softplus', 'clamped_exp'])
    parser.add_argument('--exposure_percentile', type=float, default=99, help="adjusts the exposure to set the white level to the p-th percentile, default p = 97")
    parser.add_argument('--mosaiced', action='store_true', help="train on mosaiced image")
    parser.add_argument('--hdr_merge', default='none', choices=['robertson', 'debevec', 'none'], help="")
    parser.add_argument('--hdr_tonemap', default='reinhard', choices=['reinhard', 'mantiuk', 'drago'], help="")

    # Lightstage only options
    parser.add_argument('--lightstage', action='store_true', help="preset config for lightstage scenes")
    parser.add_argument('--bracketing', action='store_true', help="multiple exposure training")
    parser.add_argument('--rfield', action='store_true', help="condition on light directions to retrieve reflectance field, only for Prune Sampling")
    parser.add_argument('--masked', action='store_true', help="train on masked images")
    parser.add_argument('--r_mode', default='none', choices=["all", "downsample3", "downsample6", "replace"] , help="...") # 

    # Pose Refinement Options
    parser.add_argument('--pose_opt', default='none', choices=['barf', 'baangp', 'none'], help="use barf like cosine annealing on view encoding")
    parser.add_argument('--num_cameras', type=int, default=-1, help="num cams to optimize, -1 queries this from colmap reconstruction") 
    parser.add_argument('--start_annealing', type=float, default=0.0, help="start annealing")
    parser.add_argument('--end_annealing', type=float, default=0.33, help="anneal only to this far into training process")
    parser.add_argument('--c_lr', type=float, default=1e-3, help="learning rate for pose refinement")
    parser.add_argument('--noise', type=float, default=0.0, help="added noise for debugging pose opt") 
    parser.add_argument('--log_poses', action='store_true', help="logs poses to a file that can be animated later to debug the pose optimization")
    parser.add_argument('--identity', action='store_true', help="init poses with identity matrix")

    # Experimental Options
    parser.add_argument('--gaussian_weighting', action='store_true', help="")
    parser.add_argument('--compute_normals', action='store_true', help="compute normalmaps during evaluation")
    parser.add_argument('--loss_weight', default='none', choices=['gaussian', 'planck', 'hanning', 'none'], help="...")
    parser.add_argument('--reduce_set', action='store_true', help="train on only half of the trainset")
    parser.add_argument('--anneal_lr', action='store_true', help="apply cosine annealing to learning rate")
    parser.add_argument('--beta', type=float, default=2, help="beta used for softplus activation")
    parser.add_argument('--eval_idx', type=int, default=2, help="rfield pose eval")
    parser.add_argument('--eval_batch', type=int, default=1, help="split the eval set into n batches to avoid OOM")
    parser.add_argument('--eval', action='store_true', help="save predictions for later evaluation")
    parser.add_argument('--debug_path', default='/home/lehmann/scratch2/debug/', help="save predictions for later evaluation")

    opt = parser.parse_args()
   
    if opt.lightstage:
        opt.bound = 2
        opt.scale = 2 
        opt.masked = True
        opt.clip = True
        opt.image_mode = 'HDR'
        opt.color_activation = 'clamped_exp'
        opt.data_format = 'colmap'
        opt.camera_traj = 'circle'
        opt.fp16 = True 
        opt.preload = True
        opt.cuda_ray = True
        opt.mark_untrained = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True
  
    if opt.O: # uses CUDA, bounded, prune (selective) sampling points by maintaining a density grid, ngp
        opt.fp16 = True
        opt.preload = True
        opt.cuda_ray = True
        opt.mark_untrained = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True
    
    if opt.O2: # unbounded & non cuda with use proposal network to predict sampling points, nerfstudio nerfacto multinerf like
        opt.fp16 = True
        opt.preload = True 
        opt.contract = True
        opt.adaptive_num_rays = True
        opt.random_image_batch = True
    
    if opt.pose_opt != 'none':   
        #opt.density_activation = 'softplus'     
        opt.random_image_batch = False
        opt.diffuse_step = 0
        opt.train_split = 'trainval'
        if opt.num_cameras == -1:
            paths_to_check = ['/images', '/raw', '/image', '/train']
            for path in paths_to_check:
                if os.path.exists(str(opt.path) + path):
                    opt.num_cameras = len(os.listdir(str(opt.path) + path))
                    if opt.data_format == 'nerf' and opt.train_split == 'trainval':
                        opt.num_cameras += len(os.listdir(str(opt.path) + '/val'))
        print(f'[INFO] Initializing Pose Optimizer with {opt.num_cameras} Cameras')
        
    if opt.contract:
        # mark untrained is not correct in contraction mode...
        opt.mark_untrained = False
    
    # extract valid leds
    if opt.rfield:
        opt.random_image_batch = False
        opt.exposure_percentile = 99.9
        valid_leds = []
        captures = glob.glob(os.path.join(opt.path+'/raw/', "*.exr"))
        for path in captures:
            led = path.split('/')[-1].split('.')[0].split('l')[-1]
            if not int(led) in valid_leds:
                valid_leds.append(int(led))
        opt.valid_leds = valid_leds 
        
    if opt.data_format == 'colmap':
        from nerf.colmap_provider import ColmapDataset as NeRFDataset
    elif opt.data_format == 'dtu':
        from nerf.dtu_provider import NeRFDataset
    else: # nerf
        from nerf.provider import NeRFDataset
  
    opt.metadict = {}
    opt.metadict['filename'] = [] # for debugging
    opt.metadict['ShutterSpeed'] = []
    opt.metadict['cam2rgb'] = []
    opt.metadict['ldirs'] = []

    if opt.exposure_range == 'wide' or opt.bracketing:
        opt.exposure_percentiles = [70, 80, 90, 97, 99, 99.9, 100]
        #opt.exposure_percentiles = [90, 97, 99, 99.9, 100]

        if opt.hdr_merge == 'none':
            opt.hdr_merge = 'robertson'
    else:
        opt.exposure_percentiles = [97, 99, 99.9, 100]
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.device = device
    seed_everything(opt.seed)

    from nerf.network import NeRFNetwork

    model = NeRFNetwork(opt)
    pose_optimizer = model.pose_optimizer if opt.pose_opt != 'none' else None

    #SmoothL1Loss MSELoss
    criterion = torch.nn.MSELoss(reduction='none') if opt.image_mode == 'LDR' else None # replaced by HDR Loss in trainer 
    
    if opt.test:

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if not opt.test_no_video:
            test_loader = NeRFDataset(opt, device=device, ttype='test').dataloader()

            if test_loader.has_gt:
                trainer.metrics = [PSNRMeter(),] # set up metrics SSIMMeter(opt), LPIPSMeter(device=device)
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.

            trainer.test(test_loader, write_video=True) # test and save video
        
        if not opt.test_no_mesh:
            # need train loader to get camera poses for visibility test
            if opt.mesh_visibility_culling:
                train_loader = NeRFDataset(opt, device=device, type=opt.train_split).dataloader()
            trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
    
    else:
        
        optimizer = torch.optim.Adam(model.get_params(opt.lr), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, pose_optimizer=pose_optimizer, ttype=opt.train_split).dataloader()

        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32) # or any number
        save_interval = max(1, max_epoch // max(1, opt.save_cnt)) # save ~50 times during the training
        eval_interval = max(1, max_epoch // max(1, opt.eval_cnt))
        print(f'[INFO] max_epoch {max_epoch}, eval every {eval_interval}, save every {save_interval}.')

        # colmap can estimate a more compact AABB
        if not opt.contract and opt.data_format == 'colmap':
            model.update_aabb(train_loader._data.pts_aabb)

        if opt.anneal_lr:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 6000) #2000
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer,
                          criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler,
                          scheduler_update_every_step=True, use_checkpoint=opt.ckpt, eval_interval=eval_interval,
                          save_interval=save_interval)


        valid_loader = NeRFDataset(opt, device=device, pose_optimizer=pose_optimizer, ttype='val').dataloader()
        trainer.log(opt)
        trainer.metrics = [PSNRMeter()] # SSIMMeter(opt) TODO: For now SSIM And PSNR are evaluated seperately to avoid OOM
        trainer.train(train_loader, valid_loader, max_epoch)
        
        # last validation
        trainer.metrics = [PSNRMeter()] #, SSIMMeter(opt) LPIPSMeter(device=device)
        trainer.evaluate(valid_loader) # evaluate against raw

        # also test
        test_loader = NeRFDataset(opt, device=device, pose_optimizer=pose_optimizer, ttype='test').dataloader()
        
        if test_loader.has_gt:
            trainer.evaluate(test_loader) # blender has gt, so evaluate it.
        
        trainer.test(test_loader, write_video=True) # test and save video 
        trainer.save_mesh(resolution=opt.mcubes_reso, decimate_target=opt.decimate_target, dataset=train_loader._data if opt.mesh_visibility_culling else None)
        