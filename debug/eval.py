import numpy as np
import os
import argparse
import cv2 
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import mean_squared_error as mse

                  
def bilinear_demosaic(bayer):
  """Converts Bayer data into a full RGB image using bilinear demosaicking.

  Input data should be ndarray of shape [height, width] with 2x2 mosaic pattern:
    -------------
    |red  |green|
    -------------
    |green|blue |
    -------------
  Red and blue channels are bilinearly upsampled 2x, missing green channel
  elements are the average of the neighboring 4 values in a cross pattern.

  Args:
    bayer: [H, W] array, Bayer mosaic pattern input image.
    xnp: either numpy or jax.numpy.

  Returns:
    rgb: [H, W, 3] array, full RGB image.
  """
  def reshape_quads(*planes):
    """Reshape pixels from four input images to make tiled 2x2 quads."""
    planes = np.stack(planes, -1)
    shape = planes.shape[:-1]
    # Create [2, 2] arrays out of 4 channels.
    zup = planes.reshape(shape + (2, 2,))
    # Transpose so that x-axis dimensions come before y-axis dimensions.
    zup = np.transpose(zup, (0, 2, 1, 3))
    # Reshape to 2D.
    zup = zup.reshape((shape[0] * 2, shape[1] * 2))
    return zup

  def bilinear_upsample(z):
    """2x bilinear image upsample."""
    # Using np.roll makes the right and bottom edges wrap around. The raw image
    # data has a few garbage columns/rows at the edges that must be discarded
    # anyway, so this does not matter in practice.
    # Horizontally interpolated values.
    zx = .5 * (z + np.roll(z, -1, axis=-1))
    # Vertically interpolated values.
    zy = .5 * (z + np.roll(z, -1, axis=-2))
    # Diagonally interpolated values.
    zxy = .5 * (zx + np.roll(zx, -1, axis=-2))
    return reshape_quads(z, zx, zy, zxy)

  def upsample_green(g1, g2):
    """Special 2x upsample from the two green channels."""
    z = np.zeros_like(g1)
    z = reshape_quads(z, g1, g2, z)
    alt = 0
    # Grab the 4 directly adjacent neighbors in a "cross" pattern.
    for i in range(4):
      axis = -1 - (i // 2)
      roll = -1 + 2 * (i % 2)
      alt = alt + .25 * np.roll(z, roll, axis=axis)
    # For observed pixels, alt = 0, and for unobserved pixels, alt = avg(cross),
    # so alt + z will have every pixel filled in.
    return alt + z

  r, g1, g2, b = [bayer[(i//2)::2, (i%2)::2] for i in range(4)]
  r = bilinear_upsample(r)
  # Flip in x and y before and after calling upsample, as bilinear_upsample
  # assumes that the samples are at the top-left corner of the 2x2 sample.
  b = bilinear_upsample(b[::-1, ::-1])[::-1, ::-1]
  g = upsample_green(g1, g2)
  rgb = np.stack([r, g, b], -1)
  return rgb

def linear_to_srgb(linear, eps = None) :
  """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = np.finfo(np.float32).eps
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * np.maximum(eps, linear)**(5 / 12) - 11) / 200
  return np.where(linear <= 0.0031308, srgb0, srgb1)

def postprocess_raw_hdr_output(raw, camtorgb, percentiles):
 
  if raw.shape[-1] != 3:
    raise ValueError(f'raw.shape[-1] is {raw.shape[-1]}, expected 3')
  if camtorgb.shape != (3, 3):
    raise ValueError(f'camtorgb.shape is {camtorgb.shape}, expected (3, 3)')
  # Convert from camera color space to standard linear RGB color space.
  rgb_linear = np.matmul(raw, camtorgb.T)

  # merge three different exposures 100 is 1/97
  exposed_images = []
  exposure_times = []
  for percentile in percentiles:
    exp = np.percentile(rgb_linear, percentile)
    if exp > 0:
      exposed_images.append(np.array(255. * np.clip(rgb_linear / exp, 0, 1)).astype(np.uint8))
      exposure_times.append(exp)
  
  exposure_times = np.array([1. / exposure_times[i] for i in range(len(exposure_times))], dtype=np.float32)

  # Create an HDR merge object, Merge the images
  merge_robertson = cv2.createMergeRobertson()
  cal_robertson = cv2.createCalibrateRobertson()
  crf_robertson = cal_robertson.process(exposed_images, times=exposure_times)
  hdr_robertson = merge_robertson.process(exposed_images, times=exposure_times, response=crf_robertson)
  merge = hdr_robertson[..., ::-1]
  tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=1, light_adapt=0, color_adapt=0.0)
  ldr_result = tonemap.process(merge)
  return ldr_result

def postprocess_raw(raw, camtorgb, exposure=None):
  
  if raw.shape[-1] != 3:
    raw = bilinear_demosaic(raw)
    #raise ValueError(f'raw.shape[-1] is {raw.shape[-1]}, expected 3')
  if camtorgb.shape != (3, 3):
    raise ValueError(f'camtorgb.shape is {camtorgb.shape}, expected (3, 3)')
  
  # Convert from camera color space to standard linear RGB color space.
  rgb_linear = np.matmul(raw, camtorgb.T)    
  # exposure = np.percentile(rgb_linear, percentile)

  rgb_linear_scaled = np.clip(rgb_linear / exposure, 0, 1)
  # Apply sRGB gamma curve to serve as a simple tonemap.
  srgb = linear_to_srgb(rgb_linear_scaled)
  srgb = srgb[..., ::-1]

  return srgb

def save_image(image, filename):
    # Normalize the image to the range [0, 255] for saving
    #norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    norm_image = np.clip(image,0,1) * 255.
    cv2.imwrite(filename, norm_image.astype(np.uint8))
    
def log(message, output_dir):
  print(message)
  filename = os.path.join(output_dir, 'metrics.txt')
  with open(filename, 'a') as file:
        file.write(message + '\n')
        
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder, filename)
            img = np.load(file_path)
            images.append(img)
    return images

def calculate_average_metrics(ground_truth_images, predicted_images, args):
    if len(ground_truth_images) != len(predicted_images):
        raise ValueError("The number of ground truth and predicted images does not match.")

    total_psnr = 0
    total_ssim = 0
    total_rmse = 0
    total_mse = 0
    index=0

    for gt_img, pred_img in zip(ground_truth_images, predicted_images):
        if args.demosaic:
          mosaic = np.zeros((gt_img.shape[0], gt_img.shape[1]))
          mosaic[0::2, 0::2] = gt_img[0::2, 0::2,0]  # R channel
          mosaic[0::2, 1::2] = gt_img[0::2, 1::2,1]  # G channel (red row)
          mosaic[1::2, 0::2] = gt_img[1::2, 0::2,1]  # G channel (blue row)
          mosaic[1::2, 1::2] = gt_img[1::2, 1::2,2]  # B channel
          gt_img = bilinear_demosaic(mosaic)
        
        if args.HDR:
          gt_img = postprocess_raw_hdr_output(gt_img, args.cam2rgb, args.exposure_levels)
          pred_img = postprocess_raw_hdr_output(pred_img, args.cam2rgb, args.exposure_levels)
        else:
          if args.cam2rgb is not None:
            gt_img = postprocess_raw(gt_img, args.cam2rgb, args.exposure_levels[args.level])
            pred_img = postprocess_raw(pred_img, args.cam2rgb, args.exposure_levels[args.level])

        data_range = gt_img.max() - gt_img.min()
        current_psnr = psnr(gt_img, pred_img, data_range=data_range)
        current_ssim = ssim(gt_img, pred_img, channel_axis=2, data_range=data_range)
        current_mse = mse(gt_img, pred_img)
        current_rmse = np.sqrt(current_mse)

        total_psnr += current_psnr
        total_ssim += current_ssim
        total_rmse += current_rmse
        total_mse += current_mse
        
        log(f'Evaluating image {index} at exposure level {args.level} with color correction for scene: {args.experiment}', args.dir)
        log(f'PSNR: {current_psnr} SSIM: {current_ssim}, RMSE: {current_rmse}, MSE: {current_mse}', args.dir)
        if index == 0 or args.save_all:
          save_image(gt_img[...], f"/home/lehmann/scratch2/debug/sample_ground_truth_{index}.png")
          save_image(pred_img[...], f"/home/lehmann/scratch2/debug/sample_predicted_{index}.png")
        index+=1
    
    log('Average PSNR: ' + str(total_psnr / len(ground_truth_images)), args.dir)
    log('Average SSIM: ' + str(total_ssim / len(ground_truth_images)), args.dir)
    log('Average RMSE: ' + str(total_rmse / len(ground_truth_images)), args.dir)
    log('Average MSE: ' + str(total_mse / len(ground_truth_images)), args.dir)

def main(args):
    ground_truth_dir = args.dir+'/eval/GT/'
    predicted_dir = args.dir+'/eval/pred/'
    ground_truth_images = load_images_from_folder(ground_truth_dir)
    predicted_images = load_images_from_folder(predicted_dir)

    calculate_average_metrics(ground_truth_images, predicted_images, args)

if __name__ == "__main__":
    # Initialize and parse arguments
    parser = argparse.ArgumentParser(description='Calculate average PSNR and SSIM for images stored as NumPy arrays.')
    parser.add_argument('dir', type=str, help='Path to the directory containing ground truth/predicted images in .npy format')
    parser.add_argument('--demosaic', action='store_true', help="demosaic GT")
    parser.add_argument('--HDR', action='store_true', help="demosaic GT")

    parser.add_argument('--level', type = float, default = 100, help="")
    parser.add_argument('--experiment', choices=['candlefiat', 'sharpshadow', 'trooper', 'gardenlights', 'stove'], help="")
    parser.add_argument('--save_all', action='store_true', help="save every image comp")


    args = parser.parse_args()
    
    if args.experiment == 'sharpshadow':
      args.cam2rgb = np.array([[ 3.28082413, -0.38636967, -0.1760492 ], 
                              [-0.24419113,  1.36912759, -0.4726144 ],
                              [ 0.03878405, -0.35183652,  2.43700175]])
      args.exposure_levels = {97: 0.07917889751493923, 99: 0.141523285806179, 99.9: 0.2773077885508566, 100: 0.46454960107803345}


    elif args.experiment == 'candlefiat':
      args.cam2rgb = np.array([[ 1.76584572, -0.38636967, -0.30940984],
                              [-0.13143157,  1.36912759, -0.83062885],
                              [ 0.02087483, -0.35183652,  4.28307722]])
                        
      #args.exposure_levels = {97: 0.006095239049755022, 99: 0.009976100064814086, 99.9: 0.3633142784238186, 100: 1.6786712408065796}
      args.exposure_levels = {97: 0.006095239049755022, 99: 0.020076100064814086, 99.9: 0.3633142784238186, 100: 1.6786712408065796}

    elif args.experiment == 'trooper':
      args.cam2rgb = np.array([[ 1.75834995, -0.3285471 , -0.18162375],
                            [-0.51061965,  1.52358675, -0.1470636 ],
                            [ 0.10399155, -0.765459  ,  1.7141508 ]])
        
      args.exposure_levels = {90: 0.05, 97: 0.11854784257709983, 99: 0.17611335217952728, 99.9: 0.2666314863562631, 100: 0.4046235978603363}
    
    elif args.experiment == 'gardenlights':
      args.cam2rgb= np.array([[ 3.23050589, -0.38636967, -0.176261],
                              [-0.24044595,  1.36912759, -0.47318299],
                              [ 0.03818921, -0.35183652,  2.43993364]])
      args.exposure_levels = {97: 0.014249206865206338, 99: 0.02871689369902014, 99.9: 0.7266747761369978, 100: 4.0183281898498535}
    
    elif args.experiment == 'stove':
      args.cam2rgb = np.array([[ 2.29436859, -0.38636967, -0.27180436],
                              [-0.17076943,  1.36912759, -0.72967473],
                              [ 0.02712273, -0.35183652,  3.76251465]])
      args.exposure_levels = {70: 0.001758907514158635, 80: 0.0031277706846594873, 90: 0.014507083874195813, 97: 0.07527966797351837, 99: 0.21920456230640362, 99.9: 1.3034300055505277, 100: 3.480355978012085}

    else:
      args.cam2rgb = None
      args.exposure_levels = {97: 0.07737476922571651, 99: 0.2247878850996492, 100: 3.698023796081543}
      
      
    main(args)
  
    