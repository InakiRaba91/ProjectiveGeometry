import numpy as np

from projective_geometry.camera.camera import Camera
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry.point import Point2D


def calculate_focal_length_from_homography(
    camera: Camera,
    image_size: ImageSize,
) -> tuple[bool, np.ndarray]:
    """Estimate the calibration matrix from the homography matrix.

    This method initializes the focal length from the homography between the world plane
    of the pitch and the image sensor. The principal point is set to the center of the
    image sensor.

    It relies on retrieving the image of the absolute conic w. It solves a homogeneous
    system of equations A * w = 0, where A is a 5x6 matrix derived from certain conditions:
    1. First row: zero skew
    2. Second row: zero skew
    3. Third row: principal point at the center of the image sensor
    4. Fourth row: vanishing points for horizontal and vertical sets of lines
    5. Fifth row: vanishing points for 45º/135º lines

    Note: check README for more details on the derivation of the equations.

    Parameters
    ----------
    camera
        The camera for which the calibration matrix is estimated.
    image_size
        The size of the image sensor in pixels.

    Returns
    -------
    success
        Whether the calibration matrix was successfully estimated.
    focal_length_xy
        The estimated focal length.
    """
    principal_point = Point2D(x=image_size.width / 2, y=image_size.height / 2)

    H = np.reshape(camera.H, (9,))
    A = np.zeros((5, 6))
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    A[1, 2] = -1.0
    A[2, 3] = principal_point.y / principal_point.x
    A[2, 4] = -1.0
    A[3, 0] = H[0] * H[1]
    A[3, 1] = H[0] * H[4] + H[1] * H[3]
    A[3, 2] = H[3] * H[4]
    A[3, 3] = H[0] * H[7] + H[1] * H[6]
    A[3, 4] = H[3] * H[7] + H[4] * H[6]
    A[3, 5] = H[6] * H[7]
    A[4, 0] = H[0] * H[0] - H[1] * H[1]
    A[4, 1] = 2 * H[0] * H[3] - 2 * H[1] * H[4]
    A[4, 2] = H[3] * H[3] - H[4] * H[4]
    A[4, 3] = 2 * H[0] * H[6] - 2 * H[1] * H[7]
    A[4, 4] = 2 * H[3] * H[6] - 2 * H[4] * H[7]
    A[4, 5] = H[6] * H[6] - H[7] * H[7]

    _, _, vh = np.linalg.svd(A)
    w = vh[-1]
    W = np.zeros((3, 3))
    W[0, 0] = w[0] / w[5]
    W[0, 1] = w[1] / w[5]
    W[0, 2] = w[3] / w[5]
    W[1, 0] = w[1] / w[5]
    W[1, 1] = w[2] / w[5]
    W[1, 2] = w[4] / w[5]
    W[2, 0] = w[3] / w[5]
    W[2, 1] = w[4] / w[5]
    W[2, 2] = w[5] / w[5]

    try:
        Ktinv = np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        K = np.eye(3)
        return False, np.ones((2,))

    K = np.linalg.inv(np.transpose(Ktinv))
    K /= K[2, 2]
    focal_length_xy = np.array([K[0, 0], K[1, 1]])
    return True, focal_length_xy


def calculate_focal_length_from_homography2(
    camera: Camera,
    image_size: ImageSize,
) -> tuple[bool, np.ndarray]:
    H = camera.H
    w2, h2 = image_size.width / 2, image_size.height / 2
    vp1 = H[:2, 0] / H[2, 0]
    vp2 = H[:2, 1] / H[2, 1]
    f = (-(vp1[0] - w2) * (vp2[0] - w2) - (vp1[1] - h2) * (vp2[1] - h2)) ** 0.5
    return True, np.array([f, f])
