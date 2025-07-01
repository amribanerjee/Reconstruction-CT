import numpy as np
from skimage import io, color, transform, util
from skimage.metrics import structural_similarity as ssim_metric
import sys

def load_image(image_path):
    print(f"Loading image from: {image_path}")
    try:
        original_image = io.imread(image_path)
        if original_image.ndim == 3:
            original_image = color.rgb2gray(original_image)
        original_image = util.img_as_float64(original_image)
        print(f"Image loaded successfully. Shape: {original_image.shape}, Dtype: {original_image.dtype}")
        return original_image
    except FileNotFoundError:
        print("Error: File not found. Please ensure the image path is correct.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        sys.exit()

def calculate_mse(image1, image2):
    if image1.shape != image2.shape:
        print("Warning: Images have different dimensions. Resizing the larger image to match the smaller one for MSE calculation.")
        if image1.size > image2.size:
            image1 = transform.resize(image1, image2.shape, anti_aliasing=True)
        else:
            image2 = transform.resize(image2, image1.shape, anti_aliasing=True)
            
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def calculate_rmse(image1, image2):
    mse = calculate_mse(image1, image2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')
    
    max_pixel_value = 1.0 
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def calculate_ssim(image1, image2):
    if image1.shape != image2.shape:
        print("Warning: Images have different dimensions. Resizing the larger image to match the smaller one for SSIM calculation.")
        if image1.size > image2.size:
            image1 = transform.resize(image1, image2.shape, anti_aliasing=True)
        else:
            image2 = transform.resize(image2, image1.shape, anti_aliasing=True)
            
    ssim_value = ssim_metric(image1, image2, data_range=image1.max() - image1.min())
    return ssim_value

def main():
    original_image_path = ''
    comparison_image_path = ''

    if not original_image_path or not comparison_image_path:
        print("Error: Image paths are not defined. Please set 'original_image_path' and 'comparison_image_path' variables in the main function.")
        sys.exit()

    original_image = load_image(original_image_path)
    comparison_image = load_image(comparison_image_path)

    mse_value = calculate_mse(original_image, comparison_image)
    print(f"\nMean Squared Error (MSE) between '{original_image_path}' and '{comparison_image_path}': {mse_value:.4f}")

    rmse_value = calculate_rmse(original_image, comparison_image)
    print(f"Root Mean Squared Error (RMSE) between '{original_image_path}' and '{comparison_image_path}': {rmse_value:.4f}")

    psnr_value = calculate_psnr(original_image, comparison_image)
    print(f"Peak Signal-to-Noise Ratio (PSNR) between '{original_image_path}' and '{comparison_image_path}': {psnr_value:.4f} dB")

    ssim_value = calculate_ssim(original_image, comparison_image)
    print(f"Structural Similarity Index (SSIM) between '{original_image_path}' and '{comparison_image_path}': {ssim_value:.4f}")

if __name__ == "__main__":
    main()
