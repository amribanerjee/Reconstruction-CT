import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform, restoration, filters, util
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import convolve2d
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
        print("File not found")
        sys.exit()
    except Exception as e:
        print(f"An error occurred during image loading: {e}")
        sys.exit()

def denoise_image_fourier(image):
    print("\n--- Applying Denoising (Fourier Transform) ---")
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    radius = min(rows, cols) // 6

    mask = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow)**2 + (j - ccol)**2) < radius:
                mask[i, j] = 1

    f_transform_filtered = f_transform_shifted * mask
    f_transform_filtered_shifted_back = np.fft.ifftshift(f_transform_filtered)
    denoised_image = np.fft.ifft2(f_transform_filtered_shifted_back)
    denoised_image = np.real(denoised_image)
    print("Denoising complete.")
    return denoised_image

def remove_artifacts_radon(image):
    print("\n--- Applying Artifact Removal (Radon Transform) ---")
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = transform.radon(image, theta=theta)

    filtered_sinogram = median_filter(sinogram, size=3)

    reconstructed_image_radon = transform.iradon(filtered_sinogram, theta=theta, filter_name='ramp')
    reconstructed_image_radon = np.clip(reconstructed_image_radon, 0, 1)
    print("Artifact removal (via Radon transform and sinogram filtering) complete.")
    return reconstructed_image_radon

def enhance_edges_unsharp_masking(image):
    print("\n--- Applying Edge Enhancement (Unsharp Masking) ---")
    blurred_for_mask = gaussian_filter(image, sigma=1)
    detail_mask = image - blurred_for_mask
    sharpening_amount = 1.5
    edge_enhanced_image = image + detail_mask * sharpening_amount
    edge_enhanced_image = np.clip(edge_enhanced_image, 0, 1)
    print("Edge enhancement (Unsharp Masking) complete.")
    return edge_enhanced_image

def improve_resolution(image, scale_factor=2):
    print("\n--- Applying Resolution Improvement ---")

    print("Interpolation for resolution improvement complete.")
    interpolated_image = transform.resize(
        image,
        (image.shape[0] * scale_factor, image.shape[1] * scale_factor),
        anti_aliasing=True
    )

    psf_size = 21
    psf_sigma = 3
    psf = np.zeros((psf_size, psf_size))
    psf[psf_size // 2, psf_size // 2] = 1
    psf = gaussian_filter(psf, sigma=psf_sigma)
    psf /= psf.sum()

    deconvolved_image = restoration.wiener(interpolated_image, psf, balance=0.005)
    deconvolved_image = np.clip(deconvolved_image, 0, 1)
    print("Deconvolution for sharpening complete.")
    return deconvolved_image

# Removed display_images function as it's no longer needed for direct output

def main():
    image_path = 'image-00200.jpg'
    output_image_path = 'final_processed_image.png' # Define output path

    original_image = load_image(image_path)
    current_image = original_image

    current_image = denoise_image_fourier(current_image)
    current_image = remove_artifacts_radon(current_image)
    current_image = enhance_edges_unsharp_masking(current_image)
    final_processed_image = improve_resolution(current_image)

    # Save the final processed image
    io.imsave(output_image_path, util.img_as_ubyte(final_processed_image))
    print(f"\nFinal processed image saved to: {output_image_path}")

    print("\nAll image processing techniques applied.")
    print("Only the Final Processed Image has been saved to a file.")

if __name__ == "__main__":
    main()
