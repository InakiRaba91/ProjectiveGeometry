from typing import Tuple

import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.geometry import Line, Point


def project_points(camera: Camera, pts: Tuple[Point, ...]) -> Tuple[Point, ...]:
    """Project points in the 3D world ground (z=0) to image

    Args:
        pts_world_ground_array: tuple of Points in the ground of the 3D world (xy coordinates)

    Returns:
        tuple of projected points Points in the image (pixel coordinates)
    """

    # convert world points to homogeneous coordinates (z=1) and transform to array
    pts_homogeneous = np.array([pt.to_homogeneous() for pt in pts])

    # map them to image domain
    projected_pts_homogeneous = camera.H.dot(pts_homogeneous.T).T

    # convert back from homogeneous domain (units are pixels, we're in image domain after projection)
    projected_pts = tuple([Point.from_homogeneous(pt_homogeneous=pt) for pt in projected_pts_homogeneous])
    return projected_pts


def project_lines(camera: Camera, lns: Tuple[Line, ...]) -> Tuple[Line, ...]:
    """
    Method to project the "Line" type from image space to world space.

    Args:
        lines: Lines in pixel space to project to real world coordinates.

    Returns:
        Tuple[Line, ...]: Projected lines in real world pitch coordinates.
    """
    # create matrix with 3D line coordinates
    ln_matrix = np.stack([line.to_array() for line in lns], axis=0)

    # map to world domain
    # H.T (is the mapping to take image lines into world lines)
    # lines_image_mat.T is required to make the matrix 3 x N for multiplication with the 3 x 3 H^-1.T
    # the final transpose is for convenience when working with the coordinates
    projected_ln_matrix = camera._H.T.dot(ln_matrix.T).T

    # convert to world Line objects
    projected_lns = tuple([Line(a=ln_arr[0], b=ln_arr[1], c=ln_arr[2]) for ln_arr in projected_ln_matrix])
    return projected_lns
