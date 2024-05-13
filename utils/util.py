import cv2
import numpy as np
from skimage.transform import SimilarityTransform

reference = np.array([[[38.2946, 51.6963],
                       [73.5318, 51.5014],
                       [56.0252, 71.7366],
                       [41.5493, 92.3655],
                       [70.7299, 92.2041]]], dtype=np.float32)


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def estimate_matrix(kpt, image_size=112):
    assert kpt.shape == (5, 2)
    min_index = []
    min_error = float('inf')
    transform = SimilarityTransform()

    if image_size == 112:
        src = reference
    else:
        src = float(image_size) / 112 * reference

    min_matrix = []
    kpt_transform = np.insert(kpt, 2, values=np.ones(5), axis=1)
    for i in np.arange(src.shape[0]):
        transform.estimate(kpt, src[i])
        matrix = transform.params[0:2, :]
        results = np.dot(matrix, kpt_transform.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_index = i
            min_error = error
            min_matrix = matrix
    return min_matrix, min_index


def norm_crop_image(image, landmark, image_size=112):
    matrix, pose_index = estimate_matrix(landmark, image_size)
    warped = cv2.warpAffine(image, matrix, (image_size, image_size), borderValue=0.0)
    return warped
