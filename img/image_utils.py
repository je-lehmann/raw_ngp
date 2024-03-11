import os
import imageio
import numpy as np
from PIL import Image
import cv2
import tqdm
import rawpy
import json
import raw.raw_utils as raw_utils
import struct

def open_file(pth, mode='r'):
  return open(pth, mode=mode)

def file_exists(pth):
  return os.path.exists(pth)

def listdir(pth):
  return os.listdir(pth)

def isdir(pth):
  return os.path.isdir(pth)

def makedirs(pth):
  if not file_exists(pth):
    os.makedirs(pth)

def intPairToDouble(a, b):
    raw = struct.pack('ii', a, b)
    return struct.unpack('d', raw)[0]

_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], # 23
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]])

exifs = []

def load_images(opt, img_paths, ldirs, H, W, ttype, root_path, num_cameras):
    images = []
    ldir = None
    
    def append_meta(dictkey, elem, ttype, pos = None):
        if(ttype == 'train' or ttype == 'trainval'):
            dictkey.append(elem)
        elif(ttype == 'val'):
            #check if elem is already in the list
            if len(dictkey) == num_cameras:
                return
            else:
                dictkey.insert(pos, elem)
    
    if(opt.image_mode == 'LDR'):
        for f in tqdm.tqdm(img_paths, desc=f'Loading {ttype} data'):
            image = cv2.imread(f, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3: 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != H or image.shape[1] != W:
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
            
            image = image[..., ::-1]
            images.append(image / 255.0)
            
    elif(opt.image_mode == 'HDR'):
        fileformat = img_paths[0].split('.')[-1]
        
        for index, file in enumerate(tqdm.tqdm(img_paths, desc=f'Loading {ttype} exif data')):
            position = None
            
            if(ttype == 'val'):
                position = opt.val_ids[index]
           
            filename = os.path.splitext(os.path.basename(file))[0]
            append_meta(opt.metadict['filename'], filename, ttype, position)
            
            if(opt.rfield):
                led = int(file.rsplit('.', 1)[0].split('l')[-1])
                ldir = ldirs[led]
                append_meta(opt.metadict['ldirs'], ldir, ttype, position)
                
            if(fileformat != 'exr'):
                with open(file.rsplit('.', 1)[0] + '.json', 'rb') as e:
                    exif = json.load(e)[0]
                    exifs.append(exif)
                    append_meta(opt.metadict['ShutterSpeed'],(1. / float(exif['ShutterSpeed'].split('/')[1])), ttype, position)

            if(fileformat == 'exr'):
                # First try hardcoded metadata, when exif data exists, overwrite it
                if(opt.bracketing):
                    exposureValue = float(file.rsplit('.', 1)[0].split('e')[-1]) / 1000000
                    append_meta(opt.metadict['ShutterSpeed'], float(exposureValue), ttype, position)

                    if file_exists(file.rsplit('.', 1)[0] + '.json'):
                        with open(file.rsplit('.', 1)[0] + '.json', 'rb') as e:
                            exif = json.load(e)[0]
                            exifs.append(exif)
                            a = int(exif['Exposure_Time_0_0'].split(' ')[1])
                            b = int(exif['Exposure_Time_0_0'].split(' ')[0])
                            exposureValue = intPairToDouble(a, b) / 1000000 
                            append_meta(opt.metadict['ShutterSpeed'], exposureValue, ttype, position)
                else:
                    append_meta(opt.metadict['ShutterSpeed'], 1, ttype, position)
                    
        # sort the metadata when training and validation data are loaded
        if(ttype == 'val'):
            shutter_speeds = opt.metadict['ShutterSpeed']
            # Sort the shutter speeds from slowest (largest) to fastest (smallest).
            # This way index 0 will always correspond to the brightest image.
            unique_shutters = np.sort(np.unique(shutter_speeds))[::-1]
            exposure_idx = np.zeros_like(shutter_speeds, dtype=np.int32)
            for i, shutter in enumerate(unique_shutters):
                # Assign index `i` to all images with shutter speed `shutter`.
                exposure_idx[shutter_speeds == shutter] = i
            opt.metadict['exposure_idx'] = exposure_idx
            opt.metadict['unique_shutters'] = unique_shutters
            # Rescale to use relative shutter speeds, where 1. is the brightest.
            # This way the NeRF output with exposure=1 will always be reasonable.
            opt.metadict['exposure_values'] = (shutter_speeds / unique_shutters[0])
            opt.metadict['ldirs'] = np.array(opt.metadict['ldirs'])


        for index, file in enumerate(tqdm.tqdm(img_paths, desc=f'Loading {ttype} img data')):
            filename = os.path.splitext(os.path.basename(file))[0]
          
            if(fileformat != 'exr'):
                with open(file.rsplit('.', 1)[0] + '.dng', 'rb') as data:
                    #image = imageio.v2.imread(data) # 12 bit raw data   
                    image = rawpy.imread(data).raw_image # 12 bit raw data

            if(fileformat == 'exr'):               
                # hardcoded exif data for now
                with open(file, 'rb') as data:
                    image = imageio.v2.imread(data) # 12 bit raw data   

            image = image.astype(np.float32)
            
            if(opt.clip): # as been measured in lightstage data
                image = np.clip(image, 0, 1)
                blacklevel = 0.00024420026 # 0
                whitelevel =  1.0
            else:
                blacklevel = exif['BlackLevel']
                whitelevel = exif['WhiteLevel']
            
            image = (image - blacklevel) / (whitelevel - blacklevel)

            if(opt.mosaiced == False):
                image = raw_utils.bilinear_demosaic(image)
            
            # downsample image
            if image.shape[0] != H or image.shape[1] != W:
                image = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)

            if(opt.mosaiced):
                rgb_image = np.zeros((image.shape[0], image.shape[1], 3))
                rgb_image[0::2, 0::2, 0] = image[0::2, 0::2]  # R channel
                rgb_image[0::2, 1::2, 1] = image[0::2, 1::2]  # G channel (red row)
                rgb_image[1::2, 0::2, 1] = image[1::2, 0::2]  # G channel (blue row)
                rgb_image[1::2, 1::2, 2] = image[1::2, 1::2]  # B channel
                image = rgb_image
            
            if(index == 0):
                image0 = image
                cv2.imwrite(opt.debug_path + 'input_img_0.png', image0[...,::-1] * 255)
            # masking
            # check for correct image size
            # masked_filename = root_path + 'masked_img_data_' + opt.background + '/' + filename + '.npy'
            #if(file_exists(masked_filename)):
                #print('[DEBUG] Loading Masked Data')
                #image = np.load(masked_filename, allow_pickle=True)
            if(opt.masked):
                filename = filename.split('/')[-1]
                filename = filename.split('_e')[0]
                filename = filename.split('_l')[0]
                mask_path = root_path + '/mask/'
                mask = imageio.v2.imread(mask_path + filename + '.png')
                
                if mask.shape[0] != H or mask.shape[1] != W:
                    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_AREA)
                # Convert the image and mask to NumPy arrays
                image_np = np.array(image)
                mask_np = np.array(mask)

                # Apply the mask to the image
                if opt.background == 'black':                
                    masked_image_np = np.zeros_like(image_np)
                    masked_image_np[:,:,0] = np.where(mask_np > 0, image_np[:,:,0], 0) # 0 = black
                    masked_image_np[:,:,1] = np.where(mask_np > 0, image_np[:,:,1], 0)
                    masked_image_np[:,:,2] = np.where(mask_np > 0, image_np[:,:,2], 0)
                else:
                    masked_image_np = np.ones_like(image_np)
                    masked_image_np[:,:,0] = np.where(mask_np > 0, image_np[:,:,0], 1)
                    masked_image_np[:,:,1] = np.where(mask_np > 0, image_np[:,:,1], 1)
                    masked_image_np[:,:,2] = np.where(mask_np > 0, image_np[:,:,2], 1)
                
                # save this as np array
                #makedirs(root_path + 'masked_img_data_' + opt.background + '/')
                #np.save(masked_filename, masked_image_np, allow_pickle=True)
                image = masked_image_np
            
            if(fileformat != 'exr'):
                whitebalance = exif['AsShotNeutral']
                whitebalance = np.array(whitebalance.split()).astype(float)
                cam2camwb = np.array(np.diag(1. / whitebalance)) # 27
                xyz2camwb = exif['ColorMatrix2']
                xyz2camwb = np.array(xyz2camwb.split()).astype(float).reshape((3, 3))
                
                rgb2camwb = xyz2camwb @ _RGB2XYZ
                rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True) # 24

                cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb # 25
                append_meta(opt.metadict['cam2rgb'], cam2rgb, ttype, position)

            if(fileformat == 'exr'):
                # determine_wb(exif)
                cam2rgb = np.array([[ 0.00689549, -0.00128842, -0.00071225,],
                                    [-0.00200243,  0.00597485, -0.00057672],
                                    [ 0.00040781, -0.0030018,   0.00672216]])
                append_meta(opt.metadict['cam2rgb'], cam2rgb * 255., ttype, position)
                
            if(opt.expose):
                # Apply a color correction matrix (from camera RGB to canonical XYZ) and XYZ-to-RGB matrix
                rgb_linear = np.matmul(image, opt.cam2rgb.T) # 28

                # Adjust the exposure to set the white level to the p-th percentile, this can 
                # be done with a lambda function stored in exif
                exposure = np.percentile(rgb_linear, opt.exposure_percentile) # 29

                # "Expose" image by mapping the input exposure level to white and clipping.
                rgb_linear_scaled = np.clip(rgb_linear / exposure, 0,1) # 30
               
                # Apply sRGB gamma curve to serve as a simple tonemap.
                image = raw_utils.linear_to_srgb(rgb_linear_scaled) # 25 / 3

            images.append(image)
            
    images = np.stack(images, axis=0)
    return images

def depth_to_normal(depthmap):
    rows, cols = depthmap.shape

    # Calculate the partial derivatives of depth with respect to x and y
    dx = cv2.Sobel(depthmap, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(depthmap, cv2.CV_32F, 0, 1)

    # Compute the normal vector for each pixel
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

    # Map the normal vectors to the [0, 255] range and convert to uint8
    normal = (normal + 1) * 127.5
    normal = normal.clip(0, 255).astype(np.uint8)

    # Save the normal map to a file
    normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    return normal_bgr

def determine_wb(exif):
    # receive exr colorchecker capture
    path = '/home/lehmann/scratch2/captures/captures-f25-exp100000/exr/colorchecker/colorchecker.exr'
    image = imageio.v2.imread(path) # 12 bit raw data
    # cast to 32 bit floating point
    image = image.astype(np.float32)
    crop_box = (2280, 1065, 2890, 1982)  # (left, upper, right, lower)
    angle = -90
    image = Image.fromarray(image)
    cropped_image = image.crop(crop_box)
    image = cropped_image.rotate(angle, expand=True)
    
    blacklevel = exif['BlackLevel']
    whitelevel = exif['WhiteLevel']
    image = np.array(image)

    image = (image - blacklevel) / (whitelevel - blacklevel)
    
    #print('[DEBUG] image shape', image.shape)
    image = raw_utils.bilinear_demosaic(image)
    
    # Given b, coordinate array to be used further on to fill in cam matrix.reference colors!
    b = np.array( [[115, 82,  68]
                ,[194, 150, 130]
                ,[98,  122, 157]
                ,[87,  108, 67]
                ,[133, 128, 177]
                ,[103, 189, 170]
                ,[214, 126, 44]
                ,[80,  91,  166]
                ,[193, 90,  99]
                ,[94,  60,  108]
                ,[157, 188, 64]
                ,[224, 163, 46]
                ,[56,  61,  150]
                ,[70,  148, 73]
                ,[175, 54,  60]
                ,[231, 199, 31]
                ,[187, 86,  149]
                ,[8,   133, 161]
                ,[243, 243, 242]
                ,[200, 200, 200]
                ,[160, 160, 160]
                ,[122, 122, 121]
                ,[85,  85,  85]
                ,[52,  52,  52]] ,dtype=float)

    print("b shape", b.shape)
    b_t = np.transpose(b)
    print("b_t shape", b_t.shape)
    # print(b)
    b = (b / 255).astype(float)
    # print(b)
    print("image.shape:", image.shape)
    # Upper left and bottom right coordinates of the first patch.
    coords = np.array([[60,  50],
                    [140, 130]])
    print("coords shape:", coords.shape)
    # Delta, i.e. spacing for the patches in x and y directions.
    delta = 150.0

    # Color for each patch. Default dtype is float32.
    cam = np.zeros([24,3])
    print("cam shape:", cam.shape)
    cam_t = np.transpose(cam)
    print("cam_t shape:", cam_t.shape)

    # fill cam matrix
    counter_patch = 0
    y1 = 50
    y2 = 130
    while coords[0,0] < 630:
        while coords[0,1] < 900:
            sum = [0, 0, 0]
            counter_sum = 0
            for x in range(coords[0, 0], min(630, coords[1, 0]), 1):
                for y in range(coords[0, 1], min(900, coords[1, 1]), 1):
                    sum = [sum[z] + image[x, y, z] for z in range(len(sum))]
                    counter_sum += 1      
            sum[:] = [x / counter_sum for x in sum]
            cam[counter_patch, :] = sum
            counter_patch += 1
            coords[0, 1] += int(delta)
            coords[1, 1] += int(delta)
        coords[0, 0] += int(delta)
        coords[1, 0] += int(delta)
        coords[0, 1] = y1
        coords[1, 1] = y2    

    #print(cam)

    # b) TODO: O = C*WB^T. We can use the last formula from slide 70 here
    id = np.eye(3)
    mat = np.linalg.solve(np.dot(cam_t, cam), id)
    mat = np.matmul(mat, cam_t)
    mat = np.matmul (mat, b)
    mat = np.transpose(mat)    
    image = image[:,:,:3]
    image_white_balanced = np.dot(image.astype(float), mat.T) * 255.0
    imageio.imwrite('test.jpg', np.array(image_white_balanced).astype(np.uint8))
    return mat
