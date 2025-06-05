from __future__ import annotations

import logging
from typing import TypeAlias

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


KeyPoints3D: TypeAlias = np.ndarray
"""3D keypoints, a Nx3 array of points in 3D space."""
KeyPoints2D: TypeAlias = np.ndarray
"""2D keypoints, a Nx2 array of points in 2D space."""
HomographyMatrix: TypeAlias = np.ndarray
"""3x3 rotation matrix representing the homography transformation between two planes."""
CalibrationMatrix: TypeAlias = np.ndarray
"""3x3 calibration matrix containing intrinsic parameters."""
RotationMatrix3D: TypeAlias = np.ndarray
"""3x3 rotation matrix representing the orientation of the camera in 3D space."""
Vector2D: TypeAlias = np.ndarray
"""2D vector."""
Vector3D: TypeAlias = np.ndarray
"""3D vector."""
DistortionCoefficients: TypeAlias = np.ndarray
"""Distortion coefficients, 1D array of coefficients for lens distortion."""


def rotation_matrix_to_roll_tilt_pan(R_mat: RotationMatrix3D) -> tuple[float, float, float]:
    r = R.from_matrix(R_mat)
    roll, tilt, pan = r.as_euler("zyx", degrees=True)
    return roll, tilt, pan


def roll_tilt_pan_to_rotation_matrix(roll: float, tilt: float, pan: float) -> RotationMatrix3D:
    r = R.from_euler("zyx", [roll, tilt, pan], degrees=True)
    return r.as_matrix()


def calculate_homography(world_points: KeyPoints2D, image_points: KeyPoints2D, alpha: float = 0.5) -> HomographyMatrix:
    """Calculate the homography matrix between the source and destination points.

    Parameters
    ----------
    world_points
        The world points.
    image_points
        The image points.
    alpha
        Value between 0 and 1 that controls whether more importance is given to
        localization (world) or reprojection (image) errors. A value of 0 only uses the
        localization error, a value of 1 only uses the reprojection error, and any other
        value gives a weighted combination of the two.

    Returns
    -------
    homography_matrix
        The homography matrix mapping the source points to the destination points.
    """
    world_points = world_points.astype(np.float32)
    image_points = image_points.astype(np.float32)
    if alpha == 1:
        return cv2.findHomography(world_points, image_points)[0]  # type: ignore[return-value]
    if alpha == 0:
        return np.linalg.inv(cv2.findHomography(image_points, world_points)[0])
    return alpha * cv2.findHomography(world_points, image_points)[0] + (1 - alpha) * np.linalg.inv(
        cv2.findHomography(image_points, world_points)[0]
    )


def calculate_focal_length_from_homography(
    homography_matrix: HomographyMatrix,
    sensor_wh: tuple[int, int] | Vector2D,
) -> tuple[bool, Vector2D]:
    """Estimate the calibration matrix from the homography matrix.

    This method initializes the focal length from the homography between the world plane
    of the pitch and the image sensor. The principal point is set to the center of the
    image sensor.

    Parameters
    ----------
    homography_matrix
        The homography matrix between the world plane of the pitch and the image.
    sensor_wh
        The sensor width and height of the camera, i.e., the camera's resolution.

    Returns
    -------
    success
        Whether the calibration matrix was successfully estimated.
    focal_length_xy
        The estimated focal length.
    """
    # ruff: noqa: N806
    sensor_wh = np.asanyarray(sensor_wh)
    principal_point = sensor_wh / 2

    H = np.reshape(homography_matrix, (9,))
    A = np.zeros((5, 6))
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    A[1, 2] = -1.0
    A[2, 3] = principal_point[1] / principal_point[0]
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

    focal_length_xy = np.array([K[0, 0], K[1, 1]]) / sensor_wh

    return True, focal_length_xy


def convert_calibration_matrix_to_intrinsics(
    calibration_matrix: CalibrationMatrix,
) -> tuple[Vector2D, Vector2D]:
    """Calculate the intrinsics from the calibration matrix.

    Parameters
    ----------
    calibration_matrix
        The calibration matrix.

    Returns
    -------
    sensor_wh
        The sensor width and height of the camera, i.e., the camera's resolution.
    focal_length_xy
        The focal length of the camera.
    """
    sensor_wh = 2 * calibration_matrix[:2, 2]
    focal_length_xy = calibration_matrix[:2, :2] / sensor_wh
    return sensor_wh, focal_length_xy


def convert_intrinsics_to_calibration_matrix(sensor_wh: Vector2D, focal_length_xy: float | Vector2D) -> CalibrationMatrix:
    """Calculate the calibration matrix from intrinsics.

    Parameters
    ----------
    sensor_wh
        The sensor width and height of the camera, i.e., the camera's resolution.
    focal_length_xy
        The focal length of the camera.

    Returns
    -------
    calibration_matrix
        The calibration matrix.
    """
    if isinstance(focal_length_xy, (int, float)):
        focal_length_xy = np.array([focal_length_xy, focal_length_xy])
    return np.array(
        [
            [focal_length_xy[0] * sensor_wh[0], 0, sensor_wh[0] / 2],
            [0, focal_length_xy[1] * sensor_wh[1], sensor_wh[1] / 2],
            [0, 0, 1],
        ]
    )


def convert_rvec_tvec_to_camera_pose(rvec: Vector3D, tvec: Vector3D) -> tuple[RotationMatrix3D, Vector3D]:
    """Convert rvec (rotation vector) and tvec (translation vector) to a camera pose.

    Parameters
    ----------
    rvec
        The rotation vector of the camera.
    tvec
        The translation vector of the camera.

    Returns
    -------
    rotation_matrix
        The rotation matrix of the camera.
    position_xyz
        The position of the camera in the world.
    """
    rotation_matrix: RotationMatrix3D = cv2.Rodrigues(np.asarray(rvec))[0]  # type: ignore[return-value]
    position_xyz = -rotation_matrix.T @ np.asarray(tvec).flatten()
    return rotation_matrix, position_xyz


def calculate_camera_pose_and_distortion(
    world_points: KeyPoints3D,
    image_points: KeyPoints2D,
    calibration_matrix: CalibrationMatrix,
    distortion_coefficients: DistortionCoefficients | None = None,
) -> tuple[RotationMatrix3D, Vector3D, DistortionCoefficients]:
    """Calculate the camera pose and distortion coefficients.

    This function calculates the camera pose, rotation matrix and position vector, and
    distortion coefficients from the given world and image points correspondences and
    calibration matrix.

    Parameters
    ----------
    world_points
        The world points.
    image_points
        The image points.
    calibration_matrix
        The calibration matrix of the camera.
    distortion_coefficients
        The initial guess for the distortion coefficients. If not provided, defaults to
        None which sets the distortion coefficients to zero.

    Returns
    -------
    rotation_matrix
        The rotation matrix of the camera.
    position_xyz
        The position of the camera in the world.
    distortion_coefficients
        The distortion coefficients.
    """
    world_points = world_points.astype(np.float32)
    image_points = image_points.astype(np.float32)

    sensor_wh, _ = convert_calibration_matrix_to_intrinsics(calibration_matrix)

    rmsre, calibration_matrix, distortion_coefficients, rvec, tvec = cv2.calibrateCamera(  # type: ignore[assignment]
        objectPoints=[world_points],
        imagePoints=[image_points],
        imageSize=(int(sensor_wh[0]), int(sensor_wh[1])),
        cameraMatrix=calibration_matrix,
        distCoeffs=distortion_coefficients,  # type: ignore[arg-type]
        flags=(
            cv2.CALIB_USE_INTRINSIC_GUESS
            + cv2.CALIB_FIX_ASPECT_RATIO
            + cv2.CALIB_FIX_PRINCIPAL_POINT
            + cv2.CALIB_FIX_FOCAL_LENGTH
        ),
    )  # type: ignore[call-overload]

    rotation_matrix, position_xyz = convert_rvec_tvec_to_camera_pose(rvec, tvec)  # type: ignore[arg-type]

    logger.info(f"Root Mean Squared Re-Projection Error: {rmsre}")
    assert isinstance(distortion_coefficients, np.ndarray)
    return rotation_matrix, position_xyz, distortion_coefficients[0]
