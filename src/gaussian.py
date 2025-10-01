import numpy as np

def get_gaussian_values(x, sigma):
    """
    Calculate the value of the Gaussian function for a given x and sigma.
    g(x) = (1 / (2 * pi * sigma^2) ^ (1/2)) * e(-1(x^2 / (2 * sigma^2)

    Parameters:
        x (float or int): The input value.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        float: The Gaussian value at x.
    """
    v = np.exp((-0.5) * ((x / sigma) ** 2))
    c = 1 / (np.sqrt(2 * np.pi * (sigma ** 2)))

    return c * v

def get_gaussian_1derivative_values(x, sigma):
    v = get_gaussian_values(x, sigma)
    c = (-1 * x) / (sigma ** 2)

    return c * v

def get_gaussian_1derivative_kernel(size, sigma):
    """
    Generate a 1D Gaussian 1st derivative kernel.

    Parameters:
        size (int): The size of the kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: The 1D Gaussian kernel as a NumPy array.
    """
    kernel = []

    for i in range(-1 * (size//2), (size//2) + 1):
        val = get_gaussian_1derivative_values(i, sigma)
        kernel.append(val)

    return np.array(kernel)

def get_gaussian_kernel(size, sigma):
    """
    Generate a 1D Gaussian kernel.

    Parameters:
        size (int): The size of the kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: The 1D Gaussian kernel as a NumPy array.
    """
    kernel = []

    for i in range(-1 * (size//2), (size//2) + 1):
        val = get_gaussian_values(i, sigma)
        kernel.append(val)

    return np.array(kernel)

def convolution_1d(x, kernel):
    """
    Perform a 1D convolution between an input array and a kernel.

    Parameters:
        x (np.ndarray): The input 1D array.
        kernel (np.ndarray): The 1D kernel array.

    Returns:
        np.ndarray: The result of the convolution.
    """
    # Perform padding
    padding = len(kernel) // 2
    x = np.pad(x, padding, mode='edge')

    y = []

    for i in range(len(x) - len(kernel) + 1):
        sum = 0

        for j in range(len(kernel)):
            sum += x[i + j] * kernel[-j - 1]

        y.append(sum)

    return np.array(y)

def conv_x(img, kernel):
    """
    Apply 1D convolution along the rows of a 2D image array.

    Parameters:
        img (np.ndarray): The 2D input image array.
        kernel (np.ndarray): The 1D kernel array.

    Returns:
        np.ndarray: The convolved image.
    """
    result = []
    for i in range(img.shape[0]):
        result.append(convolution_1d(img[i], kernel))
    return np.array(result)

def conv_y(img, kernel):
    """
    Apply 1D convolution along the columns of a 2D image array.

    Parameters:
        img (np.ndarray): The 2D input image array.
        kernel (np.ndarray): The 1D kernel array.

    Returns:
        np.ndarray: The convolved image.
    """
    return conv_x(img.T, kernel).T

def perform_conv_1derivative(img, kernel_size, sigma):
    """
    Apply a 1D Gaussian blur 1st derivative to a 2D image array using a specified kernel size and sigma.

    Parameters:
        img (np.ndarray): The 2D input image array.
        kernel_size (int): The size of the Gaussian kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: The blurred image after applying the Gaussian filter along the rows.
    """
    # Create a 1D Gaussian kernel
    kernel = get_gaussian_1derivative_kernel(kernel_size, sigma)

    # Apply 1D convolution along the rows of the image
    result_x = conv_x(img, kernel)

    # Apply 1D convolution along the columns of the image
    result_y = conv_y(img, kernel)

    return result_x, result_y

def perform_conv(img, kernel_size, sigma):
    """
    Apply a 1D Gaussian blur to a 2D image array using a specified kernel size and sigma.

    Parameters:
        img (np.ndarray): The 2D input image array.
        kernel_size (int): The size of the Gaussian kernel (should be odd).
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: The blurred image after applying the Gaussian filter along the rows.
    """
    # Create a 1D Gaussian kernel
    kernel = get_gaussian_kernel(kernel_size, sigma)

    # Apply 1D convolution along the rows of the image
    result_x = conv_x(img, kernel)

    # Apply 1D convolution along the columns of the image
    result_y = conv_y(img, kernel)

    return result_x, result_y