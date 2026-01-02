# Reimplementation from: https://github.com/jingnanshi/recipnps
from typing import List

import numpy as np

from projective_geometry.camera import CameraPose
from projective_geometry.solvers import arun_camera_pose


def p3p_grunert(pts_world: np.ndarray, pts_img_cam: np.ndarray) -> List[CameraPose]:
    """
    Solve the P3P problem using Grunert's method.

    Args:
        pts_world: 3x3 matrix of world points (each point is a column)
        pts_img: 3x3 matrix of image points (each point is a column)

    Returns:
        List of valid CameraPose solutions
    """
    p1w, p2w, p3w = pts_world.T
    p1i, p2i, p3i = pts_img_cam.T
    a = np.linalg.norm(p2w - p3w)
    b = np.linalg.norm(p1w - p3w)
    c = np.linalg.norm(p1w - p2w)
    a_sq = a**2
    b_sq = b**2
    c_sq = c**2
    j1 = p1i / np.linalg.norm(p1i)
    j2 = p2i / np.linalg.norm(p2i)
    j3 = p3i / np.linalg.norm(p3i)
    cos_alpha = j2.dot(j3)
    cos_beta = j1.dot(j3)
    cos_gamma = j1.dot(j2)
    cos_alpha_sq = cos_alpha**2
    cos_beta_sq = cos_beta**2
    cos_gamma_sq = cos_gamma**2
    a_sq_minus_c_sq_div_b_sq = (a_sq - c_sq) / b_sq
    a_sq_plus_c_sq_div_b_sq = (a_sq + c_sq) / b_sq
    b_sq_minus_c_sq_div_b_sq = (b_sq - c_sq) / b_sq
    b_sq_minus_a_sq_div_b_sq = (b_sq - a_sq) / b_sq
    a4 = (a_sq_minus_c_sq_div_b_sq - 1) ** 2 - 4 * c_sq / b_sq * cos_alpha_sq
    a3 = 4 * (
        a_sq_minus_c_sq_div_b_sq * (1 - a_sq_minus_c_sq_div_b_sq) * cos_beta
        - (1 - a_sq_plus_c_sq_div_b_sq) * cos_alpha * cos_gamma
        + 2 * c_sq / b_sq * cos_alpha_sq * cos_beta
    )
    a2 = 2 * (
        (a_sq_minus_c_sq_div_b_sq) ** 2
        - 1
        + 2 * (a_sq_minus_c_sq_div_b_sq) ** 2 * cos_beta_sq
        + 2 * (b_sq_minus_c_sq_div_b_sq) * cos_alpha_sq
        - 4 * (a_sq_plus_c_sq_div_b_sq) * cos_alpha * cos_beta * cos_gamma
        + 2 * (b_sq_minus_a_sq_div_b_sq) * cos_gamma_sq
    )
    a1 = 4 * (
        -(a_sq_minus_c_sq_div_b_sq) * (1 + a_sq_minus_c_sq_div_b_sq) * cos_beta
        + 2 * a_sq / b_sq * cos_gamma_sq * cos_beta
        - (1 - (a_sq_plus_c_sq_div_b_sq)) * cos_alpha * cos_gamma
    )
    a0 = (1 + a_sq_minus_c_sq_div_b_sq) ** 2 - 4 * a_sq / b_sq * cos_gamma_sq

    def get_points_in_cam_frame_from_v(v):
        u = (
            (-1 + a_sq_minus_c_sq_div_b_sq) * v**2
            - 2 * (a_sq_minus_c_sq_div_b_sq) * cos_beta * v
            + 1
            + a_sq_minus_c_sq_div_b_sq
        ) / (2 * (cos_gamma - v * cos_alpha))
        s1 = (c_sq / (1 + u**2 - 2 * u * cos_gamma)) ** 0.5
        s2 = u * s1
        s3 = v * s1
        pts_cam = np.array([s1 * j1, s2 * j2, s3 * j3]).T
        return pts_cam

    all_roots = np.roots([a4, a3, a2, a1, a0])
    all_roots = np.real(all_roots[np.isreal(all_roots)])  # filter out complex roots
    camera_pose_candidates = []
    for i in range(all_roots.size):
        pts_cam = get_points_in_cam_frame_from_v(all_roots[i])
        camera_pose = arun_camera_pose(pts_world=pts_world, pts_cam=pts_cam)
        camera_pose_candidates.append(camera_pose)
    return camera_pose_candidates
