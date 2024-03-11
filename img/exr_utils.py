import sys, os, glob
import imageio
import numpy as np
from PIL import Image
import cv2
from typing import Optional
sys.path.append('../raw')
import raw_utils


_Array = np.ndarray

def determineWBMat():
    #receive exr colorchecker capture
    path = '/home/lehmann/scratch2/captures/captures-f25-exp100000/exr/colorchecker/colorchecker.exr'
    convert_exr_to_png(path, 'colorchecker.png')
    crop_box = (2280, 1065, 2890, 1982)  # (left, upper, right, lower)
    angle = -90
    crop_and_rotate_image('colorchecker.png', crop_box, angle)
    image = imageio.imread('rotated_image.png')
    
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

def preprocess_exr(exr_file):
    image = imageio.v2.imread(exr_file)
    image = image.astype(np.float32)
    #normalize
    blacklevel = 0.0
    whitelevel = 1.0
    image = np.clip((image - blacklevel) / (whitelevel - blacklevel), 0, 1)

    image = raw_utils.bilinear_demosaic(image)

    return image

def convert_exr_to_png(exr_file, png_file, wb_mat=None):
    if not os.path.isfile(exr_file):
        return False

    filename, extension = os.path.splitext(exr_file)
    if not extension.lower().endswith('.exr'):
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
    return True

def rough_cut(image_path, x, y, width, height, output_path):
    if not os.path.isfile(image_path):
        return False
    
    image_data = imageio.v2.imread(image_path)
    image_data = image_data.astype(np.uint8)

    fileformat = os.path.splitext(image_path)[1]
    # Calculate the coordinates of the rough cut region
    left = x
    top = y
    right = x + width
    bottom = y + height
    
    # Crop the image
    image = Image.fromarray(image_data) # mode='F' 

    rough_cut_image = image.crop((left, top, right, bottom))
    
    # Save the cropped image
    imageio.imwrite(output_path, rough_cut_image, format=fileformat)
    print("Rough cut image saved as", output_path, "with format", fileformat)

def rough_cut_exr(image_path, x, y, width, height, output_path):
    if not os.path.isfile(image_path):
        return False
    
    image_data = imageio.v2.imread(image_path)
    image_data = image_data.astype(np.float32)

    fileformat = os.path.splitext(image_path)[1]
    # Calculate the coordinates of the rough cut region
    left = x
    top = y
    right = x + width
    bottom = y + height
    
    # Crop the image
    image = Image.fromarray(image_data, mode = 'F') # mode='F' 

    rough_cut_image = image.crop((left, top, right, bottom))
    
    # Save the cropped image
    imageio.imwrite(output_path, rough_cut_image, format=fileformat)
    print("Rough cut image saved as", output_path, "with format", fileformat)


def crop(image_path, output_path):
    if os.path.exists(image_path):
        
        image=Image.open(image_path)
        image.load()

        image_data = np.asarray(image)
        image_data_bw = 255 - image_data.max(axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
        cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

        image_data_new = image_data[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1 , :]

        new_image = Image.fromarray(image_data_new)
        new_image.save(output_path)
    
def add_padding(image_path, output_path, target_size):
    if os.path.exists(image_path):
        image = Image.open(image_path)        
        image_data = np.asarray(image)
        width, height = image.size

        # Calculate the difference in width and height between the image and target size
        width_diff = target_size[0] - width
        height_diff = target_size[1] - height

        # Calculate the amount of padding needed on each side
        left_padding = width_diff // 2
        right_padding = width_diff - left_padding
        top_padding = height_diff // 2
        bottom_padding = height_diff - top_padding

        # Create a new image with the desired target size and white background
        padded_image = Image.new(image.mode, target_size, (255, 255, 255))

        # Paste the original image onto the padded image
        padded_image.paste(image, (left_padding, top_padding))
        # Save the padded image to the output path
        padded_image.save(output_path)
        print("Padded image saved as", output_path)

def crop_and_rotate_image(image_path, crop_box, angle):
    # Open the image
    image = Image.open(image_path)
    
    # Crop the image
    cropped_image = image.crop(crop_box)
    
    # Rotate the cropped image
    rotated_image = cropped_image.rotate(angle, expand=True)
    
    # Save the rotated image
    rotated_image.save("rotated_image.png")
    
    # Close the image
    image.close()

def apply_mask(image_path, mask_path, output_path, downscale = 1):
    # Open the image and mask
    if not os.path.isfile(image_path):
        return False

    image = imageio.v2.imread(image_path)
    fileformat = os.path.splitext(image_path)[1]
    
    if fileformat == '.exr':
        image = preprocess_exr(image_path)
    
    image = image.astype(np.float32)
    mask = imageio.v2.imread(mask_path)

    # Create a white background image with the same size as the input image
    #background = Image.new("RGB", image.size, (255, 255, 255))
    # result = Image.new("RGB", (4096, 3000), (0, 0, 0))
    result = np.zeros((int(3000 / downscale), int(4096 / downscale), 3))
    #print(result.shape)

    # Convert the image and mask to NumPy arrays
    image_np = np.array(image)
    mask_np = np.array(mask)
    
    # Apply the mask to the image
    masked_image_np = np.zeros_like(image_np)
    masked_image_np[:,:,0] = np.where(mask_np > 0, image_np[:,:,0], 0)
    masked_image_np[:,:,1] = np.where(mask_np > 0, image_np[:,:,1], 0)
    masked_image_np[:,:,2] = np.where(mask_np > 0, image_np[:,:,2], 0)

    # Create a PIL image from the masked NumPy array
    #masked_image = Image.fromarray(masked_image_np)
    
    # Paste the masked image onto the background
    #image = cv2.copyMakeBorder(masked_image_np, 1000, 1000, 1000, 1000, cv2.BORDER_CONSTANT)
    #print(result.shape, masked_image_np.shape)
    #pos = (1300,600)
    #result[0+600:2400+600, 0+1300:1900+1300, :] = masked_image_np[:,:,:]
    result = masked_image_np
    #result.paste(masked_image, (1300, 600))        
    
    # Save the masked image 
    imageio.imwrite(output_path, (result).astype(np.uint8), format='.png') # *255.0 ?
    print("Masked image saved as", output_path, "with format .png")

if __name__ == '__main__':
    # cut at 1300, 600, 1900, 2400,
    # todo: add mkdir
    # use this for testing
    captures_dir = os.path.join(os.path.dirname(__file__), "/home/lehmann/scratch2/datasets/trooper_exp/raw/")
    base_path = "/home/lehmann/scratch2/datasets/trooper_exp/"
    base_path_cropped = "/home/lehmann/scratch2/datasets/trooper5_cropped/"

    raw_path = base_path + "raw/"

    mask_path = base_path + "mask/"
    png_path = base_path + "images/"
    masked_path = base_path +  "masked/"
    cropped_path = base_path_cropped + "cropped/"
    padded_path = base_path + "padded/"
    preds_path = "/home/lehmann/scratch2/ngp_raw_nerf_workspace/trooper_bl/validation/exp_color"
    cam2rgb = np.array([[ 0.00689549, -0.00128842, -0.00071225,],
                        [-0.00200243,  0.00597485, -0.00057672],
                        [ 0.00040781, -0.0030018,   0.00672216]])
    #target_size = (1500, 1500)
    #wb_mat = determineWBMat()
    #print(wb_mat)
    # Use glob to get a list of all image files in the captures directory
    image_files = glob.glob(os.path.join(captures_dir, "*.exr"))
    print(image_files)
    # Read each image file
    for image_file in image_files:
        with open(image_file, "rb") as file:
            filename = os.path.splitext(os.path.basename(image_file))[0]
            # def postprocess_raw(raw: np.ndarray,
            #camtorgb: np.ndarray,
            #exposure: Optional[float] = None) -> np.ndarray:
           
            
            #processed_image = raw_utils.postprocess_raw(image_file, cam2rgb, 99.9)
            #imageio.imwrite(preds_path + '/postprocessed/' + filename + '.jpg', processed_image, format='.jpg')
            if(filename.split('e')[1] == '100' or filename.split('e')[1] == '1000'):
                print("Processing:" + filename)
                convert_exr_to_png(image_file, png_path + filename + '.png', cam2rgb)
            # x,y,width,height
            #rough_cut(mask_path + filename + '.png', 1400, 600, 1700, 2200, base_path_cropped + 'mask/' + filename + '.png')
            #rough_cut_exr(raw_path + filename + '.exr', 1400, 600, 1700, 2200, base_path_cropped + 'raw/' + filename + '.exr')
            #matte(png_path + filename + '.png', mask_path + filename + '.png')
            #apply_mask(png_path + filename + '.png', mask_path + filename + '.png', masked_path + filename + '.png', 4)
            #if(os.listdir(mask_path + filename)[0] != 'metadata.csv'):
                #apply_mask(png_path + 'cutout/' + filename + '.png', mask_path + filename + '/' + os.listdir(mask_path + filename)[0], masked_path + filename + '.png')
            #crop(masked_path + filename + '.png', cropped_path + filename + '.png')
            #add_padding(cropped_path + filename + '.png', padded_path + filename + '.png', target_size)
            
            # apply masking to exr data
            #rough_cut(image_file, 1300, 600, 1900, 2400, base_path + 'cutout_exr/' + filename + '.exr')
            #if(os.listdir(mask_path + filename)[0] != 'metadata.csv'):
                #apply_mask( base_path + 'cutout_exr/' + filename + '.exr', mask_path + filename + '/' + os.listdir(mask_path + filename)[0], masked_path + filename + '.png')


