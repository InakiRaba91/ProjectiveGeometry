from errno import E2BIG
import numpy as np
import random
from typing import List, Callable, Optional

from projective_geometry.camera import CameraPose
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.projection.reproj_error import reproj_error_points_pose
from projective_geometry.solvers import p3p_grunert


def p4p(
    pts_world: np.ndarray,
    pts_img_cam: np.ndarray,
    solver: Callable[[np.ndarray, np.ndarray], List[CameraPose]] = p3p_grunert,
) -> Optional[CameraPose]:
    """
    P4P
    
    Args:
        world_points: 3xN matrix of world points
        bearing_vectors: 3xN matrix of bearing vectors
        solver: Function that computes models from point correspondences
        
    Returns:
        The best model found, or None if no model was found
    """    
    num_points = pts_world.shape[1]
    assert num_points == 4
    indices = list(range(num_points))
    
    # Sample 4 points (3 for P3P, 1 for model selection)
    sampled_indices = random.sample(indices, 4)

    # Compute camera pose candidates
    camera_pose_candidates = solver(pts_world=pts_world[:, sampled_indices[:-1]], pts_img_cam=pts_img_cam[:, sampled_indices[:-1]])
        
    # Skip if no camera poses were found
    if len(camera_pose_candidates) == 0:
        return
        
    # Select model with the smallest reprojection cost
    selected_model_idx = 0
    min_cost = float('inf')
    pt_idx = sampled_indices[3]
    
    for i, camera_pose in enumerate(camera_pose_candidates):
        cost = reproj_error_points_pose(camera_pose=camera_pose, pts_world=pts_world[:, [pt_idx]], pts_cam=pts_img_cam[:, [pt_idx]])
        if cost < min_cost:
            selected_model_idx = i
            min_cost = cost
    return camera_pose_candidates[selected_model_idx]
