from typing import Tuple

import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.geometry import Line, Point
from projective_geometry.geometry.conic import Conic


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
    projected_ln_matrix = camera._H.T.dot(ln_matrix.T).T
    return tuple([Line(a=ln_arr[0], b=ln_arr[1], c=ln_arr[2]) for ln_arr in projected_ln_matrix])


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
