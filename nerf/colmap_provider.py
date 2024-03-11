import os
import cv2
import glob
import json
import tqdm
import random
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import imageio
# import OpenEXR as exr
# from raw import exr_utils
import trimesh
from PIL import Image as pil_image
from img import image_utils
from reflectance import reflectance_utils

# from skimage import io, restoration

import torch
from torch.utils.data import DataLoader

from .train_utils import get_rays, create_dodecahedron_cameras
from .colmap_utils import *

from raw import raw_utils
#from scipy.spatial.transform import Rotation
#from scipy.interpolate import Slerp

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


def center_poses(poses, pts3d=None, enable_cam_center=False):
    
    def normalize(v):
        return v / (np.linalg.norm(v) + 1e-10)

    if pts3d is None or enable_cam_center:
        center = poses[:, :3, 3].mean(0)
    else:
        center = pts3d.mean(0)
        
    
    up = normalize(poses[:, :3, 1].mean(0)) # (3)
    R = rotmat(up, [0, 0, 1])
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1
    
    poses[:, :3, 3] -= center
    poses_centered = R @ poses # (N_images, 4, 4)

    if pts3d is not None:
        pts3d_centered = (pts3d - center) @ R[:3, :3].T
        # pts3d_centered = pts3d @ R[:3, :3].T - center
        return poses_centered, pts3d_centered

    return poses_centered


def visualize_poses(poses, size=0.05, bound=1, points=None):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=[2*bound]*3).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    if bound > 1:
        unit_box = trimesh.primitives.Box(extents=[2]*3).as_outline()
        unit_box.colors = np.array([[128, 128, 128]] * len(unit_box.entities))
        objects.append(unit_box)

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] - size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] - size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    if points is not None:
        print('[visualize points]', points.shape, points.dtype, points.min(0), points.max(0))
        colors = np.zeros((points.shape[0], 4), dtype=np.uint8)
        colors[:, 2] = 255 # blue
        colors[:, 3] = 30 # transparent
        objects.append(trimesh.PointCloud(points, colors))

    scene = trimesh.Scene(objects)
    scene.set_camera(distance=bound, center=[0, 0, 0])
    scene.show()


class ColmapDataset:
    def __init__(self, opt, device, pose_optimizer, ttype='train', n_test=24):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = ttype # train, val, test
        self.downscale = opt.downscale
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        # self.offset = opt.offset # camera offset
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.root_path = opt.path # contains "colmap_sparse"
        self.training = self.type in ['train', 'all', 'trainval']
        self.pose_optimizer = pose_optimizer

        # locate colmap dir
        candidate_paths = [
            os.path.join(self.root_path, "colmap_sparse", "0"),
            os.path.join(self.root_path, "sparse", "0"),
            os.path.join(self.root_path, "colmap"),
        ]
        
        self.colmap_path = None
        for path in candidate_paths:
            if os.path.exists(path):
                self.colmap_path = path
                break

        if self.colmap_path is None:
            raise ValueError(f"Cannot find colmap sparse output under {self.root_path}, please run colmap first!")

        camdata = read_cameras_binary(os.path.join(self.colmap_path, 'cameras.bin'))

        # read image size (assume all images are of the same shape!)
        self.H = int(round(camdata[1].height / self.downscale)) 
        self.W = int(round(camdata[1].width / self.downscale))
        print(f'[INFO] ColmapDataset: image H = {self.H}, W = {self.W}')

        # read image paths
        imdata = read_images_binary(os.path.join(self.colmap_path, "images.bin"))
        unique_img = len(imdata)

        # TODO: Used once for flower colmap GT as this has old image naming
        '''for i in range(unique_img):
            i = i+1
            newData = Image(
                    id=imdata[i].id,
                    qvec=imdata[i].qvec,
                    tvec=imdata[i].tvec,
                    camera_id=imdata[i].camera_id,
                    name=''.join(imdata[i].name.split('_')[1:3]) + '_' + imdata[i].name.split('_')[3] + imdata[i].name.split('_')[4].split('.')[0]+'.exr',
                    xys=imdata[i].xys,
                    point3D_ids=imdata[i].point3D_ids
                )
  
            print(newData.name)
            imdata[i] = newData'''
        # This can be used to add more images (same pose but different exposure) to the dataset
        if(self.opt.bracketing):
            print('[INFO] Multi Exposure Training: We have', unique_img, 'unique poses in the dataset.')
            print('[INFO] Modifying COLMAP database to include different exposures:')
            exposures = [625,2500,10000] #10000
            exp_idx = 0
            for i in range(unique_img):
                i = i+1 #why is this shifted?
                newData = imdata[i]
                #exp = exposures[exp_idx]
                for exp in exposures:
                    print(imdata[i].name, 'was replaced by', imdata[i].name.split('.png')[0].split('_e')[0] +'_e'+str(exp)+'.exr', 'for exposure_idx', exp)
                    newData = Image(
                        id=imdata[i].id,
                        qvec=imdata[i].qvec,
                        tvec=imdata[i].tvec,
                        camera_id=imdata[i].camera_id,
                        name=imdata[i].name.split('.png')[0].split('_e')[0] +'_e'+str(exp),
                        xys=imdata[i].xys,
                        point3D_ids=imdata[i].point3D_ids
                    )                    
                    #imdata[i] = newData # replace image
                    imdata[len(imdata) + 1] = newData   #add data    

                #always include low exposure for highlights if its not added already
                '''if exp_idx != 0:
                    exp = exposures[0]
                    print(imdata[i].name, 'was replaced by', imdata[i].name.split('.png')[0].split('_e')[0] +'_e'+str(exp)+'.exr')
                    newData = Image(
                        id=imdata[i].id,
                        qvec=imdata[i].qvec,
                        tvec=imdata[i].tvec,
                        camera_id=imdata[i].camera_id,
                        name=imdata[i].name.split('.png')[0].split('_e')[0] +'_e'+str(exp),
                        xys=imdata[i].xys,
                        point3D_ids=imdata[i].point3D_ids
                    )                    
                    imdata[len(imdata) + 1] = newData # add image'''
                #exp_idx = (exp_idx + 1) % len(exposures)
        
        # This can be used to add more images (same pose but different light direction) to the dataset
        if(self.opt.rfield):
            print('[INFO] Reflection Field Training: We have', unique_img, 'unique poses in the dataset.')
            print('[INFO] Modifying COLMAP database to include', len(opt.valid_leds), 'light directions:')
            view_idx, led_idx = 0, 0
            self.view_indices = np.zeros(240) #(opt.num_images)
            for i in range(1, unique_img + 1):
                newData = imdata[i]
                leds = opt.valid_leds
                led = leds[led_idx]
                exclude_rotations = ['z18', 'z54', 'z90', 'z126', 'z162', 'z198', 'z234', 'z270', 'z306', 'z342']
                if any(part.split('.')[0] in exclude_rotations for part in imdata[i].name.split('_')):
                    print('Skip ', imdata[i].name)
                    continue
                # replace images
                if opt.r_mode == 'replace':
                    print(imdata[i].name, 'was replaced by', imdata[i].name.split('.')[0]+'_l'+str(led)+'.exr', 'for led_idx', led_idx)
                    newData = Image(
                            id=imdata[i].id,
                            qvec=imdata[i].qvec,
                            tvec=imdata[i].tvec,
                            camera_id=imdata[i].camera_id,
                            name=imdata[i].name.split('.')[0]+'_l'+str(led)+'.exr',
                            xys=imdata[i].xys,
                            point3D_ids=imdata[i].point3D_ids
                        )
                    self.view_indices[view_idx] = view_idx
                    led_idx = (led_idx + 1) % len(opt.valid_leds)
                    imdata[i] = newData
                
                # use led_idx but dont replace entries but add num_leds/3 new entries
                elif opt.r_mode == 'downsample3':
                    shuffled_indices = list(range(8))
                    random.shuffle(shuffled_indices)
                    for j in shuffled_indices:
                        print(imdata[i].name, 'was extended by', imdata[i].name.split('.')[0]+'_l'+str(leds[led_idx+j])+'.exr', 'for led_idx', led_idx+j)
                        newData = Image(
                            id=imdata[i].id,
                            qvec=imdata[i].qvec,
                            tvec=imdata[i].tvec,
                            camera_id=imdata[i].camera_id,
                            name=imdata[i].name.split('.png')[0] +'_l'+str(leds[led_idx+j])+'.exr',
                            xys=imdata[i].xys,
                            point3D_ids=imdata[i].point3D_ids
                        )
                        #self.view_indices[((view_idx*8)+j)] = view_idx
                        imdata[len(imdata) + 1] = newData
                    led_idx = (led_idx + 8) % len(opt.valid_leds)
                
                # use led_idx but dont replace entries but add num_leds/6 new entries
                elif opt.r_mode == 'downsample6':
                    shuffled_indices = list(range(4))
                    random.shuffle(shuffled_indices)
                    for j in shuffled_indices:
                        #print(imdata[i].name, 'was extended by', imdata[i].name.split('.')[0]+'_l'+str(leds[led_idx+j])+'.exr', 'for led_idx', led_idx+j)
                        newData = Image(
                            id=imdata[i].id,
                            qvec=imdata[i].qvec,
                            tvec=imdata[i].tvec,
                            camera_id=imdata[i].camera_id,
                            name=imdata[i].name.split('.png')[0] +'_l'+str(leds[led_idx+j])+'.exr',
                            xys=imdata[i].xys,
                            point3D_ids=imdata[i].point3D_ids
                        )
                        #print(4*(view_idx)+j, imdata[i].name)
                        self.view_indices[((view_idx*4)+j)] = view_idx
                        imdata[len(imdata) + 1] = newData
                    led_idx = (led_idx + 4) % len(opt.valid_leds)

                # include all images with all leds
                elif opt.r_mode == 'all':
                    random.shuffle(opt.valid_leds)
                    for j in opt.valid_leds:
                        print(imdata[i].name, 'was extended by', imdata[i].name.split('.')[0]+'_l'+str(j)+'.exr')
                        newData = Image(
                            id=imdata[i].id,
                            qvec=imdata[i].qvec,
                            tvec=imdata[i].tvec,
                            camera_id=imdata[i].camera_id,
                            name=imdata[i].name.split('.png')[0] +'_l'+str(j)+'.exr',
                            xys=imdata[i].xys,
                            point3D_ids=imdata[i].point3D_ids
                        )
                        #self.view_indices[((view_idx*opt.valid_leds)+j)] = view_idx
                        imdata[len(imdata) + 1] = newData
                view_idx +=1
            
        imkeys = np.array(sorted(imdata.keys()))
        
        if self.opt.reduce_set:
            imkeys = imkeys[1::2]
    
        img_names = [(os.path.basename(imdata[k].name)).rsplit('.', 1)[0] for k in imkeys]

        if(self.opt.image_mode == 'LDR'):
            img_folder = os.path.join(self.root_path, f"images_{self.downscale}")
            if not os.path.exists(img_folder):
                img_folder = os.path.join(self.root_path, "images")
            fileformat = os.listdir(img_folder)[0].split('.')[-1]

        elif(self.opt.image_mode == 'HDR'):
            img_folder = os.path.join(self.root_path, f"raw_{self.downscale}")
            if not os.path.exists(img_folder):
                img_folder = os.path.join(self.root_path, "raw")
            fileformat = os.listdir(img_folder)[0].split('.')[-1]

        img_paths = np.array([os.path.join(img_folder, name) + '.' + fileformat for name in img_names])
        # only keep existing images
        exist_mask = np.array([os.path.exists(f) for f in img_paths])
        # remove images from training set
        print(f'[INFO] {exist_mask.sum()} image exists in all {exist_mask.shape[0]} colmap entries.')
        
        imkeys = imkeys[exist_mask]
        img_paths = img_paths[exist_mask]
        
        # read intrinsics
        intrinsics = []
        for k in imkeys:
            cam = camdata[imdata[k].camera_id]
            if cam.model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
                fl_x = fl_y = cam.params[0] / self.downscale
                cx = cam.params[1] / self.downscale
                cy = cam.params[2] / self.downscale
            elif cam.model in ['PINHOLE', 'OPENCV']:
                fl_x = cam.params[0] / self.downscale
                fl_y = cam.params[1] / self.downscale
                cx = cam.params[2] / self.downscale
                cy = cam.params[3] / self.downscale
            else:
                raise ValueError(f"Unsupported colmap camera model: {cam.model}")
            intrinsics.append(np.array([fl_x, fl_y, cx, cy], dtype=np.float32))
        
        self.intrinsics = torch.from_numpy(np.stack(intrinsics)) # [N, 4]
       
        # read poses
        if(self.opt.rfield):
            self.ldirs = reflectance_utils.load_light_dirs()
            self.num_lights = len(self.ldirs)
        else:
            self.ldirs = None
         
        poses = []
        for k in imkeys:
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = imdata[k].qvec2rotmat()
            P[:3, 3] = imdata[k].tvec
            poses.append(P)
        
        poses = np.linalg.inv(np.stack(poses, axis=0)) # [N, 4, 4]
       
        # read sparse points
        ptsdata = read_points3d_binary(os.path.join(self.colmap_path, "points3D.bin"))
        #print(ptsdata[1])
        ptskeys = np.array(sorted(ptsdata.keys()))
        pts3d = np.array([ptsdata[k].xyz for k in ptskeys]) # [M, 3]
        self.ptserr = np.array([ptsdata[k].error for k in ptskeys]) # [M]
        self.mean_ptserr = np.mean(self.ptserr)

        # center pose
        self.poses, self.pts3d = center_poses(poses, pts3d, self.opt.enable_cam_center)
        np.save(self.opt.debug_path + 'raw_poses_r_field.npy', self.poses)

        print(f'[INFO] ColmapDataset: load poses {self.poses.shape}, points {self.pts3d.shape}')
        print(f'[INFO] ColmapDataset: estimated scale is: {1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean()}')

        # auto-scale
        if self.scale == -1: 
            self.scale = 1 / np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean() # 4
            self.opt.scale = self.scale
            print(f'[INFO] ColmapDataset: auto-scale {self.scale:.4f}')
            
        self.poses[:, :3, 3] *= self.scale
        self.poses = self.poses[:, [1, 0, 2, 3], :]

        # rectify convention...
        self.poses[:, :3, 1:3] *= -1
        self.poses[:, 2] *= -1

        self.pts3d = self.pts3d[:, [1, 0, 2]]
        self.pts3d[:, 2] *= -1
        self.pts3d *= self.scale
        self.num_cameras = len(self.poses)
        
        if self.pose_optimizer != 'none':
            self.opt.poses_gt = self.poses
           
        print('[INFO] ColmapDataset: num_cameras', self.num_cameras)

        # use pts3d to estimate aabb
        # self.pts_aabb = np.concatenate([np.percentile(self.pts3d, 1, axis=0), np.percentile(self.pts3d, 99, axis=0)]) # [6]
        self.pts_aabb = np.concatenate([np.min(self.pts3d, axis=0), np.max(self.pts3d, axis=0)]) # [6]
        
        # set bound automatically
        #if(self.opt.bound == -1):
        # self.opt.bound = np.abs(self.pts_aabb).max()
        print(f'[INFO] ColmapDataset: estimated AABB is: {self.pts_aabb.tolist()}')
        
        if np.abs(self.pts_aabb).max() > self.opt.bound:
            print(f'[WARN] ColmapDataset: estimated AABB {self.pts_aabb.tolist()} exceeds provided bound {self.opt.bound}! Consider improving --bound to make scene included in trainable region.')

        # process pts3d into sparse depth data.

        if self.type != 'test':
        
            self.cam_near_far = [] # always extract this infomation
            
            print(f'[INFO] extracting sparse depth info...')
            # map from colmap points3d dict key to dense array index
            pts_key_to_id = np.ones(ptskeys.max() + 1, dtype=np.int64) * len(ptskeys)
            pts_key_to_id[ptskeys] = np.arange(0, len(ptskeys))
            # loop imgs
            _mean_valid_sparse_depth = 0
            for i, k in enumerate(tqdm.tqdm(imkeys)):
                xys = imdata[k].xys
                xys = np.stack([xys[:, 1], xys[:, 0]], axis=-1) # invert x and y convention...
                pts = imdata[k].point3D_ids

                mask = (pts != -1) & (xys[:, 0] >= 0) & (xys[:, 0] < camdata[1].height) & (xys[:, 1] >= 0) & (xys[:, 1] < camdata[1].width)

                assert mask.any(), 'every image must contain sparse point'
                
                valid_ids = pts_key_to_id[pts[mask]]
                pts = self.pts3d[valid_ids] # points [M, 3]
                err = self.ptserr[valid_ids] # err [M]
                xys = xys[mask] # pixel coord [M, 2], float, original resolution!

                xys = np.round(xys / self.downscale).astype(np.int32) # downscale
                xys[:, 0] = xys[:, 0].clip(0, self.H - 1)
                xys[:, 1] = xys[:, 1].clip(0, self.W - 1)
                
                # calc the depth
                P = self.poses[i]
                depth = (P[:3, 3] - pts) @ P[:3, 2]

                # calc weight
                weight = 2 * np.exp(- (err / self.mean_ptserr) ** 2)

                _mean_valid_sparse_depth += depth.shape[0]

                # camera near far
                # self.cam_near_far.append([np.percentile(depth, 0.1), np.percentile(depth, 99.9)])
                self.cam_near_far.append([np.min(depth), np.max(depth)])

            print(f'[INFO] extracted {_mean_valid_sparse_depth / len(imkeys):.2f} valid sparse depth on average per image')

            self.cam_near_far = torch.from_numpy(np.array(self.cam_near_far, dtype=np.float32)) # [N, 2]

          
        else: # test time: no depth info
            self.cam_near_far = None

        # make split, load cam2rgb here too!
        if self.type == 'test':            
            poses = []
            if self.opt.camera_traj == 'circle':
                print(f'[INFO] use circular camera traj for testing.')
                num_frames = 100
                # circle 360 pose
                radius = np.linalg.norm(self.poses[:, :3, 3], axis=-1).mean(0)
                #radius = 0.1
                theta = np.deg2rad(80)
                for i in range(num_frames):
                    phi = np.deg2rad(i / 100 * 360)
                    center = np.array([
                        radius * np.sin(theta) * np.sin(phi),
                        radius * np.sin(theta) * np.cos(phi),
                        radius * np.cos(theta),
                    ])
                    # look at
                    def normalize(v):
                        return v / (np.linalg.norm(v) + 1e-10)
                    forward_v = normalize(center)
                    up_v = np.array([0, 0, 1])
                    right_v = normalize(np.cross(forward_v, up_v))
                    up_v = normalize(np.cross(right_v, forward_v))
                    # make pose
                    pose = np.eye(4)
                    pose[:3, :3] = np.stack((right_v, up_v, forward_v), axis=-1)
                    pose[:3, 3] = center
                    poses.append(pose)
                
                self.poses = np.stack(poses, axis=0)
            # choose some random poses, and interpolate between.
            else:
                if self.opt.pose_opt != 'none':
                    self.poses = self.pose_optimizer.get_refined_poses(self.poses).detach().cpu().numpy()
                fs = np.random.choice(len(self.poses), 5, replace=False)
                pose0 = self.poses[fs[0]]
                for i in range(1, len(fs)):
                    pose1 = self.poses[fs[i]]
                    rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
                    slerp = Slerp([0, 1], rots)    
                    for i in range(n_test + 1):
                        ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :3] = slerp(ratio).as_matrix()
                        pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                        poses.append(pose)
                    pose0 = pose1
                self.poses = np.stack(poses, axis=0)

            # fix intrinsics for test case logp oses
            self.intrinsics = self.intrinsics[[0]].repeat(self.poses.shape[0], 1)
            self.images = None
            if(self.opt.rfield):
                num_lights = 100
                start = self.opt.metadict['ldirs'][0]
                end = self.opt.metadict['ldirs'][-1]
                t_values = np.linspace(0, 1, num_lights)
                interpolations = (1 - t_values[:, np.newaxis]) * start + t_values[:, np.newaxis] * end
                self.ldir = np.array(interpolations)
                self.poses = np.tile(self.poses[self.opt.eval_idx], (100,1,1))
                print('[INFO] Testing with new Light Directions')
            
        else:
            all_ids = np.arange(len(img_paths))
            val_ids = all_ids[::8] #if self.opt.pose_opt == 'none' else all_ids
            train_ids = np.array([i for i in all_ids if i not in val_ids])

            if self.type == 'train':
                self.poses = self.poses[train_ids]
                self.intrinsics = self.intrinsics[train_ids]
                img_paths = img_paths[train_ids]
                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[train_ids]
            elif self.type == 'val':
                self.poses = self.poses[val_ids]
                self.intrinsics = self.intrinsics[val_ids]
                img_paths = img_paths[val_ids]

                if self.cam_near_far is not None:
                    self.cam_near_far = self.cam_near_far[val_ids]
            # else: trainval use all.
                
            # read images, unindent to use all images as test set
            self.opt.train_ids = train_ids
            self.opt.val_ids = val_ids
            self.images = image_utils.load_images(self.opt, img_paths, self.ldirs, self.H, self.W, self.type, self.root_path, self.num_cameras)
            
        # view all poses.
        if self.opt.vis_pose:
            visualize_poses(self.poses, bound=self.opt.bound, points=self.pts3d)

        self.poses = torch.from_numpy(self.poses.astype(np.float32)) # [N, 4, 4]

        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0).astype(np.float32)) # [N, H, W, C] # before uint8
        
        # perspective projection matrix
        self.near = self.opt.min_near
        self.far = 1000 # infinite
        aspect = self.W / self.H

        projections = []
        for intrinsic in self.intrinsics:
            y = self.H / (2.0 * intrinsic[1].item()) # fl_y
            projections.append(np.array([[1/(y*aspect), 0, 0, 0], 
                                        [0, -1/y, 0, 0],
                                        [0, 0, -(self.far+self.near)/(self.far-self.near), -(2*self.far*self.near)/(self.far-self.near)],
                                        [0, 0, -1, 0]], dtype=np.float32))
        self.projections = torch.from_numpy(np.stack(projections)) # [N, 4, 4]
        self.mvps = self.projections @ torch.inverse(self.poses)
    
        # tmp: dodecahedron_cameras for mesh visibility test
        dodecahedron_poses = create_dodecahedron_cameras()
        # visualize_poses(dodecahedron_poses, bound=self.opt.bound, points=self.pts3d)
        self.dodecahedron_poses = torch.from_numpy(dodecahedron_poses.astype(np.float32)) # [N, 4, 4]
        self.dodecahedron_mvps = self.projections[[0]] @ torch.inverse(self.dodecahedron_poses) # assume the same intrinsic
        
        self.intrinsics = self.intrinsics.to(self.device)
        self.poses = self.poses.to(self.device)
        if self.preload:
            if self.images is not None:
                self.images = self.images.to(self.device)
            if self.cam_near_far is not None:
                self.cam_near_far = self.cam_near_far.to(self.device)
            self.mvps = self.mvps.to(self.device)

    def collate(self, index):
        '''index: refers to the elements in the dataloader (usually train/val/test set),  
        globalIndex: refers to the elements in dictionary (all images)'''
        results = {'H': self.H, 'W': self.W}
        index = np.array(index)
        if self.training:
            num_rays = self.opt.num_rays            
            # randomly sample over images too
            if self.opt.random_image_batch:
                index = np.random.randint(0, len(self.poses), size=(num_rays,))
            global_index = np.array(self.opt.train_ids[index]) if self.type != 'trainval' else index

        elif self.type == 'val':
            global_index = np.array(self.opt.val_ids[index])
            num_rays = -1
        
        else:
            global_index = index
            num_rays = -1
        
        if(self.opt.image_mode == 'HDR' and self.type != 'test'):
            results['exposure'] = self.opt.metadict['exposure_values'][global_index]
        else:
            results['exposure'] = 1.0
        
        index = torch.from_numpy(index).to(self.device).detach()
        poses = self.poses[index].to(self.device) # [1/N, 4, 4]
        if self.opt.pose_opt != 'none' and self.type != 'test':
            if self.opt.rfield or self.opt.bracketing:
                poses = self.pose_optimizer(poses, self.view_indices[global_index])
            else:
                poses = self.pose_optimizer(poses, global_index)

        ldirs = torch.from_numpy(self.opt.metadict['ldirs'][global_index]).float().to(self.device) if (self.opt.rfield) else None
        intrinsics = self.intrinsics[index].to(self.device) # [1/N, 4]
        rays = get_rays(poses, intrinsics, self.H, self.W, num_rays, ldirs=ldirs)
        mvp = self.mvps[index].to(self.device)
        results['mvp'] = mvp

        if self.images is not None: # image shape: [39, 756, 1008, 3]
            if self.training:
                images = self.images[index, rays['j'], rays['i']].float().to(self.device) # [N, 3/4]
                if(self.opt.mosaiced):
                    # Calculate the Bayer mask for the image
                    lossmult = raw_utils.pixels_to_bayer_mask(rays['j'].cpu().numpy(), rays['i'].cpu().numpy())
                    results['lossmult'] = lossmult
            else:
                images = self.images[index].squeeze(0).float().to(self.device) # [H, W, 3/4]
            
            if self.training:
                C = self.images.shape[-1]
                images = images.view(-1, C)
            
            results['images'] = images            
           
        if self.opt.enable_cam_near_far and self.cam_near_far is not None:
            cam_near_far = self.cam_near_far[index].to(self.device) # [1/N, 2]
            results['cam_near_far'] = cam_near_far
        
        results['rays_o'] = rays['rays_o'].requires_grad_(True)
        results['rays_d'] = rays['rays_d'].requires_grad_(True)
        results['index'] = index
        
        if self.opt.rfield:
            if self.type == 'train' or self.type == 'trainval':
                results['rays_ldir'] = rays['rays_ldir']
            elif self.type == 'test':
                results['rays_ldir'] = torch.from_numpy(self.ldir).float().to(self.device)
            else:
                results['rays_ldir'] = ldirs

        return results
        
    def dataloader(self):
        size = len(self.poses)
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader