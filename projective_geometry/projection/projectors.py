from typing import Any, Optional, Tuple

import cv2
import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Line, Point
from projective_geometry.geometry.conic import Conic
from projective_geometry.geometry.line_segment import LineSegment
from projective_geometry.pitch_template.pitch_template import PitchTemplate


def project_points(camera: Camera, pts: Tuple[Point, ...]) -> Tuple[Point, ...]:
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
    return tuple([Point.from_homogeneous(pt_homogeneous=pt) for pt in projected_pts_homogeneous])


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


def project_sphere(pos: np.ndarray, radius: float, H: np.ndarray) -> Conic:
    """ Project a sphere in 3D space to an ellipse in the image plane using the camera model.

    Args:
        pos (np.ndarray): 3D position of the sphere center (x, y, z).
        radius (float): Radius of the sphere.
        camera (Camera): Camera object with homography matrix.

    Returns:
        Conic: Projected ellipse as a Conic object.
    """
    r = radius
    x, y, z = pos
    Q = np.array([
        [1, 0, 0, -x],
        [0, 1, 0, -y],
        [0, 0, 1, -z],
        [-x, -y, -z, x**2 + y**2 + z**2 - r**2]
    ])

    # obtain inverse
    Q_inv = np.linalg.inv(Q)
    C_inv = H.dot(Q_inv).dot(H.T)
    return Conic(M=np.linalg.inv(C_inv))