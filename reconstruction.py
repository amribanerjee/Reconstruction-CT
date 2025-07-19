import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, restoration, filters, util
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import convolve2d
import sys
import os

def load_image(file_path):
    try:
        raw_img = io.imread(file_path)
        if raw_img.ndim == 3:
            gray_img = color.rgb2gray(raw_img)
        else:
            gray_img = raw_img
        return util.img_as_float64(gray_img)
    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        sys.exit(1)

def denoise_fourier_transform(input_img):
    f_transform = np.fft.fft2(input_img)
    f_shift = np.fft.fftshift(f_transform)

    rows, cols = input_img.shape
    c_row, c_col = rows // 2, cols // 2
    radius = min(rows, cols) // 6

    y_coords, x_coords = np.ogrid[-c_row:rows-c_row, -c_col:cols-c_col]
    mask = (x_coords**2 + y_coords**2 <= radius**2)

    f_filtered = f_shift * mask
    f_filtered_shifted_back = np.fft.ifftshift(f_filtered)
    denoised_img = np.fft.ifft2(f_filtered_shifted_back)
    
    return np.real(denoised_img)

def remove_radon_streaks(input_img):
    angles = np.linspace(0., 180., max(input_img.shape), endpoint=False)
    sino = transform.radon(input_img, theta=angles)

    smoothed_sino = median_filter(sino, size=3)

    reconstructed_img = transform.iradon(smoothed_sino, theta=angles, filter_name='ramp')
    return np.clip(reconstructed_img, 0, 1)

def sharpen_image_unsharp(original_img):
    blurred_img = gaussian_filter(original_img, sigma=1)
    detail_mask = original_img - blurred_img

    sharpen_amt = 1.5
    sharpened_img = original_img + detail_mask * sharpen_amt
    
    return np.clip(sharpened_img, 0, 1)

def upscale_and_deconvolve_image(low_res_img, scale_factor=2):
    upscaled_img = transform.resize(
        low_res_img,
        (low_res_img.shape[0] * scale_factor, low_res_img.shape[1] * scale_factor),
        anti_aliasing=True
    )

    psf_size = 21
    psf_std = 3
    psf = np.zeros((psf_size, psf_size))
    psf[psf_size // 2, psf_size // 2] = 1
    psf = gaussian_filter(psf, sigma=psf_std)
    psf /= psf.sum()

    deblurred_img = restoration.wiener(upscaled_img, psf, balance=0.005)
    
    return np.clip(deblurred_img, 0, 1)

def run_image_enhancement_pipeline():
    in_file = ''
    out_file = 'enhanced_result.png'

    if not in_file:
        sys.exit(1)

    initial_img = load_image(in_file)
    
    current_img_state = initial_img

    current_img_state = denoise_fourier_transform(current_img_state)
    current_img_state = remove_radon_streaks(current_img_state)
    current_img_state = sharpen_image_unsharp(current_img_state)
    final_output = upscale_and_deconvolve_image(current_img_state)
    
    try:
        io.imsave(out_file, util.img_as_ubyte(final_output))
    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    run_image_enhancement_pipeline()
