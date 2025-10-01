from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.pa1.gaussian import perform_conv_1derivative, perform_conv
from src.pa1.nonmaxSuppression import perform_nonmax_suppression, get_gradient_magnitude
from src.pa1.hysteresisThresholding import perform_hysteresis_thresholding


def save_array(
    save_dir: Path,
    array_name: str,
    arr: np.ndarray,
    kernel_size: int,
    sigma: float,
    low_thresh: float,
    high_thresh: float,
    suffix: str = ""
) -> Path:
    """
    Save a NumPy array to a subdirectory named with parameters.
    Reuses the directory if it exists. Overwrites the file if it exists.
    """
    base_name = Path(array_name).stem

    # Directory encodes parameters
    dir_name = f"{base_name}_{kernel_size}_{sigma}_{low_thresh}_{high_thresh}"
    array_dir = save_dir / dir_name
    array_dir.mkdir(parents=True, exist_ok=True)

    # Create final file path
    file_name = f"{base_name}_{suffix}.npy"
    array_path = array_dir / file_name

    # Save as .npy (overwrite if exists)
    np.save(str(array_path), arr)

    return array_path


def main():
    # Create an image path object
    image_name = '41004'
    extension = '.jpg'
    test_image_dir = Path('images') / 'images' / 'train'
    image_path = test_image_dir / (image_name + extension)

    # Create save file path object
    save_dir = Path('images') / 'output'

    # Set configuration parameters
    kernel_size = 5
    sigma = 3
    low_thresh = 14
    high_thresh = 16

    # Read image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # Convert image to matrix
    image_arr = np.array(image)
    save_array(save_dir, image_name, image_arr, kernel_size, sigma, low_thresh, high_thresh, "original_greyscale")

    # Perform Gaussian blur
    image_conv_x, image_conv_y = perform_conv(image_arr, kernel_size, sigma)
    save_array(save_dir, image_name, image_conv_x, kernel_size, sigma, low_thresh, high_thresh, "conv_x")
    save_array(save_dir, image_name, image_conv_y, kernel_size, sigma, low_thresh, high_thresh, "conv_y")

    # Perform Gaussian blur 1st derivative
    image_conv_x, image_conv_y = perform_conv_1derivative(image_arr, kernel_size, sigma)
    save_array(save_dir, image_name, image_conv_x, kernel_size, sigma, low_thresh, high_thresh, "conv_x_deriv")
    save_array(save_dir, image_name, image_conv_y, kernel_size, sigma, low_thresh, high_thresh, "conv_y_deriv")

    # Perform gradient magnitude
    image_grad_mag = get_gradient_magnitude(image_conv_x, image_conv_y)
    save_array(save_dir, image_name, image_grad_mag, kernel_size, sigma, low_thresh, high_thresh, "grad_mag")

    # Perform non-maximum suppression
    image_nonmax_supp = perform_nonmax_suppression(image_conv_x, image_conv_y)
    save_array(save_dir, image_name, image_nonmax_supp, kernel_size, sigma, low_thresh, high_thresh, "nonmax_supp")

    # Perform hysteresis thresholding
    image_hysteresis = perform_hysteresis_thresholding(image_nonmax_supp, low_thresh, high_thresh)
    save_array(save_dir, image_name, image_hysteresis, kernel_size, sigma, low_thresh, high_thresh, "hysteresis")
    plt.imshow(image_hysteresis, cmap="gray")
    plt.show()


if __name__ == "__main__":
    main()