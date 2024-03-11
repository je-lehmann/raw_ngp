import sys, os, glob
import imageio
import numpy as np
from PIL import Image
import argparse
import cv2
from typing import Optional
import torch
sys.path.append('../raw')
import raw_utils

# Get the path of the current script (main_script.py)
current_path = os.path.dirname(os.path.abspath(__file__))

# Append the path of the directory containing the module (another_directory)
module_path = os.path.join(current_path, "/home/lehmann/code/Matte-Anything")
sys.path.append(module_path)
import matte_exr

def convert_exr_to_png(exr_file, png_file, wb_mat=None):
    if not os.path.isfile(exr_file):
        print(exr_file ,'does not exist')
        return False

    filename, extension = os.path.splitext(exr_file)
    if not extension.lower().endswith('.exr'):
        print(exr_file ,'is not exr')
        return False

    # imageio.plugins.freeimage.download() #DOWNLOAD IT
    image = imageio.v2.imread(exr_file)
    image = image.astype(np.float32)

    #normalize
    blacklevel = 0.0
    whitelevel = 1.0
    image = (image - blacklevel) / (whitelevel - blacklevel)

    # remove alpha channel for jpg conversion
    # image = image[:,:,:3]
    image = raw_utils.bilinear_demosaic(image)
    
    # Do whitebalancing
    if wb_mat is not None:
        image = np.dot(image.astype(float), wb_mat.T)

    exposure = np.percentile(image, 99.99) 
    image = np.clip(image / exposure, 0, 1) 
    image = raw_utils.linear_to_srgb(image)

    data = 255 * image # 65535 = 2^16 - 1, 4095 = 2^12 - 1
    data[data>255]=255
    rgb_image = data.astype('uint8')
    rgb_image = imageio.core.image_as_uint(rgb_image, bitdepth=8) # recorded with bitdepth 12

    imageio.imwrite(png_file, rgb_image, format='png')
    print(exr_file, '.exr converted to .png')
    return True

def apply_mask(image_path, mask_path, output_path):
    print(image_path)
    # Open the image and mask
    if not os.path.isfile(image_path):
        print('Image file not found')
        return False

    image = imageio.v2.imread(image_path)
    fileformat = os.path.splitext(image_path)[1]
    
    image = image.astype(np.float32)
    mask = imageio.v2.imread(mask_path)

    # Convert the image and mask to NumPy arrays
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # Apply the mask to the image
    masked_image_np = np.zeros_like(image_np)
    masked_image_np[:,:,0] = np.where(mask_np > 0, image_np[:,:,0], 0)
    masked_image_np[:,:,1] = np.where(mask_np > 0, image_np[:,:,1], 0)
    masked_image_np[:,:,2] = np.where(mask_np > 0, image_np[:,:,2], 0)
    
    # Save the masked image 
    imageio.imwrite(output_path, (masked_image_np).astype(np.uint8), format='.png')
    print("Masked image saved as", output_path, "with format .png")

def upscale_image(img, scale_factor):
    # Get the original width and height of the image
    image = Image.fromarray(img)

    original_width, original_height = image.size

    # Calculate the new width and height after upscaling
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor

    # Upscale the image using the BICUBIC resampling method
    upscaled_image = image.resize((new_width, new_height), Image.BICUBIC)

    return np.asarray(upscaled_image)

def matte(capture_path, png_path, output_path, prompt, downscale):
    fileformat = os.path.splitext(capture_path)[1]
    filename = os.path.splitext(os.path.basename(capture_path))[0]

    if fileformat == '.exr':
        if(os.path.isfile(png_path + filename + '.png') == False):
            print('Converting ' + filename + ' from exr to png for matting')
            convert_exr_to_png(capture_path, png_path + filename + '.png')
        
    image = imageio.v2.imread(png_path + filename + '.png')
    if downscale != 1:
        image = cv2.resize(image, (image.shape[1]//downscale, image.shape[0]//downscale), interpolation=cv2.INTER_AREA)
        
    mask = matte_exr.run_inference(image,[], 10, 10, 0.25, 0.25, prompt, 0.5, 0.25, "glass.lens.crystal.diamond.bubble.bulb.web.grid")
    # "glass.lens.crystal.diamond.bubble.bulb.web.grid"
     # [DEBUG] Input [] 10 10 0.25 0.25 object 0.5 0.25 glass.lens.crystal.diamond.bubble.bulb.web.grid
    image_array = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    # Scale the values to the range [0, 255] (unsigned 8-bit integer)
    image = (image_array * 255).astype(np.uint8)
    imageio.imwrite(output_path, upscale_image(image, downscale), format='.png')
    print("Binary Image Mask saved as", output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='object', help="text prompt the matting is based on")
    parser.add_argument('--downscale', type=int, default=4, help="downscale training images")
    parser.add_argument('--capture_dir', type=str, default='/home/lehmann/scratch2/captures/flowers/')
    parser.add_argument('--dataset_dir', type=str, default='/home/lehmann/scratch2/datasets/flowers/')

    opt = parser.parse_args()
    mask_path = opt.dataset_dir + "mask/"
    png_path = opt.dataset_dir + "png/"
    masked_path = opt.dataset_dir +  "masked/"
    
    if(os.path.isdir(opt.dataset_dir)== False):
        print('Creating base_path directory')
        os.mkdir(opt.dataset_dir)

    if(os.path.isdir(mask_path) == False):
        print('Creating mask_path directory')
        os.mkdir(mask_path)
        
    if(os.path.isdir(masked_path) == False):
        print('Creating masked_path directory')
        os.mkdir(masked_path)
    
    if(os.path.isdir(png_path) == False):
        print('Creating png directory')
        os.mkdir(png_path)
            
    image_files = glob.glob(os.path.join(opt.capture_dir, "*.exr"))
    matte_exr.init_models()
    # Read each image file
    for image_file in image_files:
        with open(image_file, "rb") as file:
            path = image_file
            filename = os.path.splitext(os.path.basename(image_file))[0]
            print("EXR to PNG Mask: Processing:" + filename)
            convert_exr_to_png(image_file, png_path + filename + '.png')
            matte(path, png_path, mask_path + filename + '.png', opt.prompt, opt.downscale)
            apply_mask(png_path + filename + '.png', mask_path + filename + '.png', masked_path + filename + '.png')
            
            # RField
            #matte(path, png_path, mask_path + filename + '.png', opt.prompt, opt.downscale)
            #apply_mask(png_path + filename + '.png', mask_path + 'lightmask' + '.png', masked_path + filename + '.png')
