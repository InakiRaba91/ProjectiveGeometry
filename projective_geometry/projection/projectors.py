import logging
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.camera.camera2 import Camera2
from projective_geometry.camera.geometry import convert_intrinsics_to_calibration_matrix
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Line, Point2D, Point3D
from projective_geometry.geometry.conic import Conic
from projective_geometry.geometry.line_segment import LineSegment
from projective_geometry.pitch_template.pitch_template import PitchTemplate

logger = logging.getLogger(__name__)


def project_points(camera: Camera, pts: Tuple[Point2D, ...]) -> Tuple[Point2D, ...]:
    """
    Project points using given camera

    Args:
        camera: Camera to project with
        pts: tuple of Points to project

    Returns:
        Tuple[Point, ...]: Projected points
    """
    pts_homogeneous = np.array([pt.to_homogeneous() for pt in pts])
    projected_pts_homogeneous = camera.H.dot(pts_homogeneous.T).T
    return tuple([Point2D.from_homogeneous(pt_homogeneous=pt) for pt in projected_pts_homogeneous])


def project_lines(camera: Camera, lns: Tuple[Line, ...]) -> Tuple[Line, ...]:
    """
    Project lines using given camera

    Args:
        camera: Camera to project with
        lines: Lines to project

    Returns:
        Tuple[Line, ...]: Projected lines
    """
    ln_matrix = np.stack([line.to_array() for line in lns], axis=0)
    projected_ln_matrix = np.linalg.inv(camera._H.T).dot(ln_matrix.T).T
    return tuple([Line(a=ln_arr[0], b=ln_arr[1], c=ln_arr[2]) for ln_arr in projected_ln_matrix])


def project_line_segments(camera: Camera, ln_segments: Tuple[LineSegment, ...]) -> Tuple[LineSegment, ...]:
    """
    Project lines using given camera

    Args:
        camera: Camera to project with
        ln_segments: LineSegments to project

    Returns:
        Tuple[Line, ...]: Projected lines
    """
    # extract all endpoints
    pts = []
    for ln_segment in ln_segments:
        pts.append(ln_segment.pt1)
        pts.append(ln_segment.pt2)
    proj_pts = project_points(camera=camera, pts=tuple(pts))
    return tuple([LineSegment(pt1=proj_pts[idx], pt2=proj_pts[idx + 1]) for idx in range(0, len(proj_pts), 2)])


def project_conics(camera: Camera, conics: Tuple[Conic, ...]) -> Tuple[Conic, ...]:
    """
    Method to project Conic objects

    Args:
        camera: Camera to project with
        conics: Conics to project

    Returns:
        Tuple[Conic, ...]: Projected conics
    """
    conic_mats = [conic.M for conic in conics]
    Hinv = np.linalg.inv(camera.H)
    projected_conic_mats = Hinv.T @ conic_mats @ Hinv
    return tuple([Conic(M=M) for M in projected_conic_mats])


def project_pitch_template(
    pitch_template: PitchTemplate,
    camera: Camera,
    image_size: ImageSize,
    frame: Optional[np.ndarray] = None,
    color: Tuple[Any, ...] = Color.RED,
    thickness: int = 3,
) -> np.ndarray:
    """
    Method to project a pitch template into an image viewed through the homography matrix. This projection
    generates a pitch template image from the point of view of the "camera" that the homography belongs to.

    Args:
        homography (Homography): Homography class instance
        image_size (ImageSize): Image size to generate the image in.
        frame (Optional[np.ndarray], optional):
            Optional background frame to layer the warped template image onto.
            If None, the method uses a black background. Defaults to None.
        color: BGR int tuple indicating the color of the geometric features
        thickness: int thickness of the drawn geometric features

    Returns:
        np.ndarray: Warped template image with or without background image.
    """
    # create the BEV image manually
    pitch_template_img = pitch_template.draw(image_size=image_size, color=color, thickness=thickness)

    # create BEV camera intrinsics
    pitch_width, pitch_height = pitch_template.pitch_dims.width, pitch_template.pitch_dims.height
    K_pitch_image_to_pitch_template = np.array(
        [
            [pitch_width / image_size.width, 0, -pitch_width / 2.0],
            [0, pitch_height / image_size.height, -pitch_height / 2.0],
            [0, 0, 1.0],
        ]
    )

    # create a chained homography projection that maps from BEV camera -> desired camera homography
    H_chained = camera.H.dot(K_pitch_image_to_pitch_template)

    projected_pitch_image = cv2.warpPerspective(
        src=pitch_template_img, M=H_chained, dsize=(image_size.width, image_size.height)
    )

    # optionally add the warped pitch template to a background image
    if isinstance(frame, np.ndarray):
        # checks
        frame_image_size = ImageSize(height=frame.shape[0], width=frame.shape[1])
        err_msg = f"Image size needs to match for frame ({frame_image_size}) and pitch ({image_size}) images"
        assert frame_image_size == image_size, err_msg

        # layer images
        return cv2.addWeighted(frame, 1, projected_pitch_image, 1, 0)

    return projected_pitch_image


_NEAR_ZERO_Z = 1e-6


def _distort(normalized_camera_points: np.ndarray, distortion_coefficients: np.ndarray) -> np.ndarray:
    """Apply distortion to normalized camera coordinates.

    Parameters
    ----------
    normalized_camera_points
        Normalized camera coordinates.
    distortion_coefficients
        Distortion coefficients [k1, k2, p1, p2, k3].

    Returns
    -------
    distorted_points
        Distorted normalized coordinates.
    """
    k1, k2, p1, p2, k3 = distortion_coefficients[:5]
    x, y = normalized_camera_points[:, 0], normalized_camera_points[:, 1]
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3

    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    return np.c_[x_distorted, y_distorted]


def project_to_sensor(camera: Camera2, world_points: Sequence[Point3D]) -> Tuple[Point2D, ...]:
    """Project points to the sensor plane of the camera.

    Args:
        camera: Camera instance to project with.
        pts: Points to project.

    Returns:
        Projected points on the sensor plane.
    """
    positions_xyz = camera.camera_params.camera_pose.postion_xyz
    rotation_matrix = camera.camera_params.camera_pose.rotation_matrix
    distortion_coefficients = camera.camera_params.camera_distorion.to_array()
    focal_length_xy = camera.camera_params.focal_length_xy
    sensor_wh = camera.sensor_wh
    principal_point = np.array([sensor_wh[0] / 2, sensor_wh[1] / 2])

    world_points_arr = np.array([pt.to_array() for pt in world_points])

    relative_points = world_points_arr - positions_xyz
    camera_points = rotation_matrix @ relative_points.T

    near_zero_z = abs(camera_points[2, :]) <= _NEAR_ZERO_Z
    if np.any(near_zero_z):
        logger.warning(
            f"{np.sum(near_zero_z)} world points are too close to the camera:\n"
            # f"{','.join(world_points[near_zero_z].tolist())}"
        )

    normalized_camera_points = camera_points / camera_points[2]
    normalized_camera_points = normalized_camera_points[:2].T

    distorted_points = _distort(normalized_camera_points, distortion_coefficients)

    x = distorted_points[..., 0] * (focal_length_xy[0] * sensor_wh[0]) + principal_point[0]
    y = distorted_points[..., 1] * (focal_length_xy[1] * sensor_wh[1]) + principal_point[1]
    image_points = np.c_[x, y]

    return tuple(Point2D.from_array(pt) for pt in image_points)


def _undistort(image_points: np.ndarray, distortion_coefficients: np.ndarray, calibration_matrix: np.ndarray) -> np.ndarray:
    """Remove distortion from image points.

    Parameters
    ----------
    image_points
        Distorted image points.
    distortion_coefficients
        Distortion coefficients [k1, k2, p1, p2, k3].
    calibration_matrix
        Calibration matrix of the camera.

    Returns
    -------
    normalized_camera_points
        Normalized camera coordinates.
    """
    normalized_camera_points = cv2.undistortPoints(
        image_points,
        calibration_matrix,
        distortion_coefficients,
        R=None,
        P=None,
    )
    return normalized_camera_points.squeeze(axis=1)


def project_to_world(camera: Camera2, image_points: Sequence[Point2D], z_plane: float = 0.0) -> Tuple[Point3D, ...]:
    """Project image points to world points.

    Parameters
    ----------
    camera
        The camera to project with.
    image_points
        The image points to project.
    z_plane
        The Z plane to project the points to.

    Returns
    -------
    world_points
        The projected world points.
    """
    position_xyz = camera.camera_params.camera_pose.postion_xyz
    rotation_matrix = camera.camera_params.camera_pose.rotation_matrix
    distortion_coefficients = camera.camera_params.camera_distorion.to_array()
    focal_length_xy = camera.camera_params.focal_length_xy
    sensor_wh = camera.sensor_wh
    calibration_matrix = convert_intrinsics_to_calibration_matrix(sensor_wh, focal_length_xy)

    image_points_arr = np.array([pt.to_array() for pt in image_points], dtype=np.float64)

    normalized_points = _undistort(image_points_arr, distortion_coefficients, calibration_matrix)
    xs, ys = normalized_points[..., 0], normalized_points[..., 1]

    normalized_points = np.c_[xs, ys, np.ones_like(xs)]

    world_vector = rotation_matrix.T @ normalized_points.T
    world_vector /= np.linalg.norm(world_vector)
    scale = (z_plane - position_xyz[2]) / world_vector[2]

    xs = position_xyz[0] + scale * world_vector[0]
    ys = position_xyz[1] + scale * world_vector[1]

    world_points = np.c_[xs, ys, np.full_like(xs, fill_value=z_plane)]
    return tuple(Point3D.from_array(pt) for pt in world_points)
