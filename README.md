# Canny Edge Detector

This project is a Python implementation of the **Canny Edge Detection** algorithm, developed as part of *CAP 5415 – Computer Vision* (Programming Assignment 1).

## Overview

The program follows the standard Canny edge detection pipeline:

1. **Grayscale conversion** – load an image and represent it as a NumPy matrix.
2. **Gaussian smoothing** – build a 1D Gaussian kernel and compute its derivative.
3. **Convolution** – apply the kernels separately along the x and y axes.
4. **Gradient calculation** – compute gradient magnitude and direction.
5. **Non-maximum suppression** – thin edges to one-pixel width.
6. **Hysteresis thresholding** – use dual thresholds and connectivity analysis to keep only meaningful edges.

> Implemented from scratch (except for utility functions like **connected components** from OpenCV) to understand the steps behind Canny instead of relying on a single built-in function.

---

## Installation

**Requirements**

- Python 3.12+  
- Poetry (recommended)

**Install dependencies (Poetry):**
```bash
poetry install
```

**If Poetry fails, install manually (pip):**
```bash
pip install opencv-python matplotlib numpy
```

---

## Usage

**Run the main program:**
```bash
poetry run python -m src.pa1.main
```

**Steps:**
1. Place input images in `images/images/train/`.
2. Edit `src/pa1/main.py` to set the image name and parameters (`kernel_size`, `sigma`, `low_thresh`, `high_thresh`).
3. Outputs will be saved to `images/output/` with parameterized subfolders.
4. To view results from `.npy` arrays:
   ```bash
   poetry run python -m src.pa1.openNPYFiles
   ```
   You will be prompted to enter the output folder name (e.g., `513531_5_1_5_10`), then results are displayed one by one.

---

## Results and Findings

- Changing **sigma** affects edge clarity:
  - Higher sigma → more blur, softer gradients.
  - Lower sigma (e.g., `1`) → sharper, more defined edges.
- The combination of **low/high thresholds** controls edge continuity.

Example results are included in the report PDF (see `/docs`).

---

## Thoughts

Re-implementing Canny gave me a deeper appreciation for how much work underlies “simple” functions in computer vision libraries. It showed the role of parameters in balancing sensitivity and noise, and why Canny remains a robust edge detector decades after its introduction.
