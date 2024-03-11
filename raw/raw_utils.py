# Adopted from multinerf to fit raw-ngp implementation

import numpy as np
import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import types
import glob
import json
import cv2
import torch

_Array = np.ndarray

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

def gaussian_weighting(values, peak_value=1, sigma=0.5, max_weight=1):
    weights = torch.exp(-(values - peak_value ** 2) / (2 * sigma ** 2))
    weighted_values = (max_weight * weights / torch.max(weights)).detach()
    return weighted_values

def hanning_weighting(values, max_weight=2):
    N = len(values)
    weights = 0.5 - 0.5 * torch.cos(2 * torch.pi * torch.arange(N) / (N - 1))
    # Scale the weights to have a maximum value of max_weight
    weighted_values = (max_weight * weights / torch.max(weights)).detach()
    
    # Replicate the weighted values across three dimensions
    weighted_values = weighted_values.unsqueeze(1).expand(-1, 3)
      
    return weighted_values

def planck_taper_weighting(values, peak_value=0.5, start_taper=0.95, end_taper=0.9, max_weight=2.0):
    # Create an array of weights based on a Planck-taper window
    weights = torch.where(values < (peak_value - start_taper), 
                          torch.tensor(0.0), 
                          torch.where((values >= (peak_value - start_taper)) & (values <= (peak_value + start_taper)), 
                          max_weight * (0.5 + 0.5 * torch.cos((values - peak_value) * (3.141592653589793 / (2 * start_taper)))), 
                          torch.tensor(0.0)))
    return weights

def linear_to_srgb(linear: _Array,
                   eps: Optional[float] = None) -> _Array:
  """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = np.finfo(np.float32).eps
  srgb0 = 323 / 25 * linear
  srgb1 = (211 * np.maximum(eps, linear)**(5 / 12) - 11) / 200
  return np.where(linear <= 0.0031308, srgb0, srgb1)


def srgb_to_linear(srgb: _Array,
                   eps: Optional[float] = None) -> _Array:
  """Assumes `srgb` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
  if eps is None:
    eps = np.finfo(np.float32).eps
  linear0 = 25 / 323 * srgb
  linear1 = np.maximum(eps, ((200 * srgb + 11) / (211)))**(12 / 5)
  return np.where(srgb <= 0.04045, linear0, linear1)

def bilinear_demosaic(bayer: _Array) -> _Array:
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

def pixels_to_bayer_mask(pix_x: np.ndarray, pix_y: np.ndarray) -> np.ndarray:
  """Computes binary RGB Bayer mask values from integer pixel coordinates."""
  # Red is top left (0, 0).
  r = (pix_x % 2 == 0) * (pix_y % 2 == 0)
  # Green is top right (0, 1) and bottom left (1, 0).
  g = (pix_x % 2 == 1) * (pix_y % 2 == 0) + (pix_x % 2 == 0) * (pix_y % 2 == 1)
  # Blue is bottom right (1, 1).
  b = (pix_x % 2 == 1) * (pix_y % 2 == 1)

  return np.stack([r, g, b], -1).astype(np.float32)

def generate_bayer_mask(size):
    n = int(np.log2(size))
    if size != 2 ** n:
        raise ValueError("Size must be a power of 2.")

    bayer_mask = np.zeros((size, size), dtype=int)
    bayer_mask[0, 0] = 0

    for i in range(1, n + 1):
        b = 2 ** (i - 1)
        for y in range(0, 2 ** i, 2):
            for x in range(0, 2 ** i, 2):
                bayer_mask[y, x] = (4 * bayer_mask[y // 2, x // 2] + b) % (4 * b)

                # Other sub-quadrants
                bayer_mask[y + b, x] = (4 * bayer_mask[y // 2, x // 2] + 2 * b) % (4 * b)
                bayer_mask[y, x + b] = (4 * bayer_mask[y // 2, x // 2] + 3 * b) % (4 * b)
                bayer_mask[y + b, x + b] = (4 * bayer_mask[y // 2, x // 2] + b) % (4 * b)

    return bayer_mask
  
def postprocess_raw(raw: np.ndarray,
                    camtorgb: np.ndarray,
                    exposure: Optional[float] = None) -> np.ndarray:
  
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
   
def postprocess_raw_hdr_output(raw, camtorgb, percentiles, merge_algo, tonemap_algo):
 
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
      #cv2.imwrite('./debug/exposed_images_' + str(percentile) + '.png', (exposed_images[-1][..., ::-1]))
      #np.save('./debug' + str(exposure) + '.png', img, allow_pickle=True)
  
  exposure_times = np.array([1. / exposure_times[i] for i in range(len(exposure_times))], dtype=np.float32)

  # Create an HDR merge object, Merge the images
  if merge_algo == 'debevec':
    merge_debevec = cv2.createMergeDebevec()
    cal_debevec = cv2.createCalibrateDebevec()
    crf_debevec = cal_debevec.process(exposed_images, times=exposure_times)
    hdr_debevec = merge_debevec.process(exposed_images, times=exposure_times, response=crf_debevec)    
    merge = hdr_debevec[..., ::-1]
  elif merge_algo == 'robertson':    
    merge_robertson = cv2.createMergeRobertson()
    cal_robertson = cv2.createCalibrateRobertson()
    crf_robertson = cal_robertson.process(exposed_images, times=exposure_times)
    hdr_robertson = merge_robertson.process(exposed_images, times=exposure_times, response=crf_robertson)
    merge = hdr_robertson[..., ::-1]
  if tonemap_algo == 'reinhard':
      tonemap = cv2.createTonemapReinhard(gamma=2.2, intensity=-1, light_adapt=0, color_adapt=0.0)
  elif tonemap_algo == 'mantiuk':
    tonemap = cv2.createTonemapMantiuk(gamma=2.2, scale=0.7, saturation=1.0)
  elif tonemap_algo == 'drago':
    tonemap = cv2.createTonemapDrago(gamma=2.2, saturation=1.0, bias=0.85)

  ldr_result = tonemap.process(merge)
  return ldr_result
