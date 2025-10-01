import numpy as np
import cv2


def perform_hysteresis_thresholding(img, low_threshold, high_threshold):
    # Determine pixels who meet minimum threshold
    candidates = (img >= low_threshold).astype(np.uint8)

    # Identify connected components and label each vein
    num_labels, labels = cv2.connectedComponents(candidates, connectivity=8)

    # Determine which labels have at least one high-threshold pixel
    strong_labels = np.unique(labels[img >= high_threshold])

    # Keep veins that have at least one high-threshold pixel
    keep = np.isin(labels, strong_labels)

    return keep.astype(np.uint8)
