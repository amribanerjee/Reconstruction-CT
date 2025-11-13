# **Advanced Image Reconstruction and Evaluation Toolkit** 🖼️
### *Multi-Step Image Enhancement and Quality Assessment Pipeline*

---

### **From the Developer**

This toolkit provides a robust, two-part Python solution for advanced image processing: **Enhancement** (denoising, streak removal, sharpening, and deconvolution) and **Evaluation** (calculating quality metrics like MSE, PSNR, and SSIM).

---

### **Project Files**
* **`reconstruction.py`**: The main image enhancement pipeline.
* **`error-metrics.py`**: Tools for comparing two images and calculating quality scores.

---

## 🚀 Tech Stack

| Area | Tools Used |
|------|------------|
| Core Language | **Python** |
| Image Processing | **Scikit-Image (skimage)** |
| Scientific Computing | **NumPy**, **SciPy** (ndimage, signal) |
| Visualization (Implied) | **Matplotlib** |

---

## ⚙️ Dependencies
Ensure the following Python packages are installed:
* `numpy`
* `scikit-image`
* `scipy`
* `matplotlib`

---

## 🧑‍💻 How to Run the Application

### 1. Installation

```bash
# Install the necessary Python libraries
pip install numpy scikit-image scipy matplotlib
```
###2. Image Enhancement (reconstruction.py)
```bash
#2. Image Enhancement (reconstruction.py)

This script runs the full refinement pipeline on a single image.

Edit reconstruction.py:
Set the path for your input image in the in_file variable and the desired output path in out_file.
# reconstruction.py snippet
in_file = 'path/to/your/input_image.png' 
out_file = 'enhanced_result.png' 


