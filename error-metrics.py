import numpy as np
from skimage import io, color, transform, util
from skimage.metrics import structural_similarity as ssim_metric
import sys
import os

def load_img(img_path):
    print(f"Loading image from: {img_path}")
    if not os.path.exists(img_path):
        print(f"Error: File not found at {img_path}.")
        return None
    try:
        raw_img_data = io.imread(img_path)
        if raw_img_data.ndim == 3:
            gray_img = color.rgb2gray(raw_img_data)
        else:
            gray_img = raw_img_data

        float_img = util.img_as_float64(gray_img)
        print(f"Image loaded. Shape: {float_img.shape}, Dtype: {float_img.dtype}")
        return float_img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def calc_mse(img1, img2):
    work_img1 = img1.copy()
    work_img2 = img2.copy()

    if work_img1.shape != work_img2.shape:
        print("Warning: Image dimensions differ. Resizing for MSE calculation.")
        if work_img1.size > work_img2.size:
            work_img1 = transform.resize(work_img1, work_img2.shape, anti_aliasing=True)
        else:
            work_img2 = transform.resize(work_img2, work_img1.shape, anti_aliasing=True)
            
    pix_diff_sq = (work_img1.astype("float") - work_img2.astype("float")) ** 2
    total_err_sum = np.sum(pix_diff_sq)
    num_pix = float(work_img1.shape[0] * work_img1.shape[1])
    mse_val = total_err_sum / num_pix
    return mse_val

def calc_rmse(img_a, img_b):
    mse = calc_mse(img_a, img_b)
    rmse_val = np.sqrt(mse)
    return rmse_val

def calc_psnr(img_one, img_two):
    curr_mse = calc_mse(img_one, img_two)
    if curr_mse == 0:
        return float('inf')
    
    max_pix_val = 1.0 
    psnr_out = 10 * np.log10((max_pix_val ** 2) / curr_mse)
    return psnr_out

def eval_ssim(img_src, img_tgt):
    work_src_img = img_src.copy()
    work_tgt_img = img_tgt.copy()

    if work_src_img.shape != work_tgt_img.shape:
        print("Warning: Image dimensions differ. Resizing for SSIM calculation.")
        if work_src_img.size > work_tgt_img.size:
            work_src_img = transform.resize(work_src_img, work_tgt_img.shape, anti_aliasing=True)
        else:
            work_tgt_img = transform.resize(work_tgt_img, work_src_img.shape, anti_aliasing=True)
            
    ssim_res = ssim_metric(work_src_img, work_tgt_img, data_range=work_src_img.max() - work_src_img.min())
    return ssim_res

def run_img_comp_metrics():
    orig_img_loc = ''
    comp_img_loc = ''

    if not orig_img_loc or not comp_img_loc:
        print("Error: Image paths not defined. Set 'orig_img_loc' and 'comp_img_loc'.")
        sys.exit()

    orig_img = load_img(orig_img_loc)
    comp_img = load_img(comp_img_loc)

    if orig_img is None or comp_img is None:
        print("Image loading failed. Exiting.")
        sys.exit()

    print("\n--- Image Comparison Results ---")

    mse = calc_mse(orig_img, comp_img)
    print(f"MSE ('{os.path.basename(orig_img_loc)}' vs '{os.path.basename(comp_img_loc)}'): {mse:.4f}")

    rmse = calc_rmse(orig_img, comp_img)
    print(f"RMSE: {rmse:.4f}")

    psnr = calc_psnr(orig_img, comp_img)
    if psnr == float('inf'):
        print(f"PSNR: Infinity (images identical)")
    else:
        print(f"PSNR: {psnr:.4f} dB")

    ssim = eval_ssim(orig_img, comp_img)
    print(f"SSIM: {ssim:.4f}")
    print("-----------------------------------\n")


if __name__ == "__main__":
    run_img_comp_metrics()
