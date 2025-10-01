import numpy as np


def get_gradient_direction(x, y):
    """
    Calculate the gradient direction (angle) from x and y gradient components.

    Parameters:
        x (np.ndarray): Gradient in x direction
        y (np.ndarray): Gradient in y direction

    Returns:
        np.ndarray: Gradient direction in radians
    """
    return np.arctan2(y, x)


def get_gradient_magnitude(x, y):
    """
    Calculate the gradient magnitude from x and y gradient components.

    Parameters:
        x (np.ndarray): Gradient in x direction
        y (np.ndarray): Gradient in y direction

    Returns:
        np.ndarray: Gradient magnitude
    """
    return np.sqrt(x ** 2 + y ** 2)


def compute_neighbor_angles(theta):
    """
    Compute the x and y components of the gradient direction angle.

    Parameters:
        theta (float): Gradient direction angle in radians

    Returns:
        tuple: (dx, dy) - x and y components of the angle
    """
    dx = np.cos(theta)
    dy = np.sin(theta)

    return dx, dy


def linear_interpolation_weight(dx, dy):
    """
    Calculate the weight for linear interpolation based on direction components.

    Parameters:
        dx (float): x component of gradient direction
        dy (float): y component of gradient direction

    Returns:
        float: Interpolation weight between 0 and 1
    """
    w = np.abs(dy) / (np.abs(dx) + np.abs(dy))
    return w


def determine_direction(component):
    """
    Determine the direction (-1, 0, or 1) based on the sign of a component.

    Parameters:
        component (float): Direction component value

    Returns:
        int: -1 for negative, 0 for zero, 1 for positive
    """
    if component < 0:
        return -1
    elif component > 0:
        return 1
    else:
        return 0


def compute_neighbor_magnitude(magnitude_array, direction_array, row, column):
    """
    Compute interpolated magnitudes of neighboring pixels along gradient direction.

    Parameters:
        magnitude_array (np.ndarray): 2D array of gradient magnitudes
        direction_array (np.ndarray): 2D array of gradient directions
        row (int): Current pixel row index
        column (int): Current pixel column index

    Returns:
        tuple: (forward_val, backward_val) - interpolated magnitudes in forward and backward directions
    """
    # Retrieve the magnitude and direction of the current pixel
    current_magnitude = magnitude_array[row, column]
    theta = direction_array[row, column]

    # Retrieve components of angle theta
    dx, dy = compute_neighbor_angles(theta)

    # Determine the direction of adjacent pixels
    x_change = determine_direction(dx)
    y_change = determine_direction(dy)

    # No adjacent pixels to compare to, return
    if x_change == 0 and y_change == 0:
        return 0, 0

    # Compute the weight for linear interpolation
    weight = linear_interpolation_weight(dx, dy)

    # Compute the linear interpolation of the magnitudes of the adjacent pixels
    forward_val = ((1.0 - weight) * magnitude_array[row, column + x_change] +
                   weight * magnitude_array[row + y_change, column + x_change])
    backward_val = ((1.0 - weight) * magnitude_array[row, column - x_change] +
                    weight * magnitude_array[row - y_change, column - x_change])

    return forward_val, backward_val


def perform_nonmax_suppression(grad_x, grad_y):
    """
    Perform non-maximum suppression on gradient magnitudes to thin edges.

    Parameters:
        grad_x (np.ndarray): Gradient in x direction
        grad_y (np.ndarray): Gradient in y direction

    Returns:
        np.ndarray: Suppressed gradient magnitudes with thinned edges
    """
    # grad_x and grad_y must be the same shape
    assert grad_x.shape == grad_y.shape, "grad_x and grad_y must be the same shape"

    # Create magnitude and direction matrix
    magnitude_array = get_gradient_magnitude(grad_x, grad_y)
    direction_array = get_gradient_direction(grad_x, grad_y)

    # Retrieve the height and width of the image
    height, width = grad_x.shape

    # Output matrix
    out = magnitude_array.copy()

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            # For each pixel, compute the greatest magnitude in a neighborhood
            forward_val, backward_val = compute_neighbor_magnitude(magnitude_array, direction_array, row, column)
            if magnitude_array[row, column] >= max(forward_val, backward_val):
                out[row, column] = magnitude_array[row, column]
            else:
                out[row, column] = 0

    return out
