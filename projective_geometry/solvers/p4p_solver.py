from errno import E2BIG
import numpy as np
import random
from typing import List, Callable, Optional

from projective_geometry.camera import CameraPose
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.projection.reproj_error import reproj_error_points_pose
from projective_geometry.solvers import p3p_grunert

def pnp_ransac(
    pts_world: np.ndarray,
    pts_img_cam: np.ndarray,
    solver: Callable[[np.ndarray, np.ndarray], List[CameraPose]] = p3p_grunert,
    max_iterations: int = 5,
    inlier_dist_threshold: float = 0.1,
    probability: float = 1,
) -> Optional[CameraPose]:
    """
    RANSAC implementation for PnP problem.
    
    Args:
        world_points: 3xN matrix of world points
        bearing_vectors: 3xN matrix of bearing vectors
        solver: Function that computes models from point correspondences
        max_iterations: Maximum number of iterations
        inlier_dist_threshold: Maximum distance for a point to be considered an inlier
        probability: Desired probability of finding the correct model
        
    Returns:
        The best model found, or None if no model was found
    """
    # Parameters
    iterations = 0
    best_inliers_count = 0
    k = 1.0
    skipped_count = 0
    max_skip = max_iterations * 10
    
    num_points = pts_world.shape[1]
    indices = list(range(num_points))
    
    best_model = None
    
    while (iterations < int(k)) and (iterations < max_iterations) and (skipped_count < max_skip):
        # Sample 4 points (3 for P3P, 1 for model selection)
        sampled_indices = random.sample(indices, 4)

        # Compute camera poses
        camera_poses = solver(pts_world=pts_world[:, sampled_indices[:-1]], pts_img_cam=pts_img_cam[:, sampled_indices[:-1]])
        
        # Skip if no camera poses were found
        if len(camera_poses) == 0:
            skipped_count += 1
            continue
        
        # Select model with the smallest reprojection cost
        selected_model_idx = 0
        min_cost = float('inf')
        pt_idx = sampled_indices[3]
        
        for i, camera_pose in enumerate(camera_poses):
            cost = reproj_error_points_pose(camera_pose=camera_pose, pts_world=pts_world[:, pt_idx], pts_cam=pts_world[:, pt_idx])
            if cost < min_cost:
                selected_model_idx = i
                min_cost = cost
        
        # Count inliers
        dists_to_all = reproj_error_points_pose(camera_pose=camera_poses[selected_model_idx], pts_world=pts_world, pts_cam=pts_img_cam, reduce=False)
        inlier_count = np.sum(dists_to_all < inlier_dist_threshold)
        
        # Update best model if we found a better one
        if inlier_count > best_inliers_count:
            best_inliers_count = inlier_count
            best_model = camera_poses[selected_model_idx]
            
            # Compute the k parameter (k=log(z)/log(1-w^n))
            w = float(best_inliers_count) / num_points
            p_no_outliers = 1.0 - w**len(sampled_indices)
            
            # Avoid numerical issues
            p_no_outliers = max(p_no_outliers, np.finfo(float).eps)
            p_no_outliers = min(p_no_outliers, 1.0 - np.finfo(float).eps)
            
            # Update required iterations
            k = np.log(1.0 - probability) / np.log(p_no_outliers)
        
        # Resample
        iterations += 1
    
    return best_model

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

if __name__ == "__main__":
    # Test
    from projective_geometry.pitch_template.basketball_template import BasketballCourtTemplate
    from projective_geometry.geometry import Point
    f = 350
    W2, H2 = BasketballCourtTemplate.PITCH_WIDTH / 2, BasketballCourtTemplate.PITCH_HEIGHT / 2
    pts_world = np.array([
        [-W2, -H2, 0],
        [-W2, H2, 0],
        [W2, -H2, 0],
        [W2, H2, 0]
    ]).T
    image_size = ImageSize(width=1280, height=720)
    w2, h2 = image_size.width / 2, image_size.height / 2
    # Top left:  [877.11741024  83.60096188]
    # Bottom left:  [1021.48619459  462.69981494]
    # Top right:  [296.31302725 252.83757669]
    # Bottom right:  [287.22444872 567.63319843]
    pts_img = [
        Point(x=877.11741024, y=83.60096188),
        Point(x=1021.48619459, y=462.69981494),
        Point(x=296.31302725, y=252.83757669),
        Point(x=287.22444872, y=567.63319843),
    ]
    # pts_img = [
    #     Point(x=285.6429465, y=631.96317038),
    #     Point(x=387.28434915, y=209.62881133),
    #     Point(x=1080.98176137, y=666.52982236),
    #     Point(x=1034.59125115, y=291.0014197),
    # ]
    pts_img_cam = np.array([[pt.x - w2, pt.y - h2, f] for pt in pts_img]).T
    camera_pose = p4p(pts_world, pts_img_cam)
    camera_pose2 = CameraPose(rx=20, ry=10, rz=170, tx=-5, ty=5, tz=15)
    K = np.array([[f, 0, w2], [0, f, h2], [0, 0, 1]])
    Rc, t = camera_pose.to_Rt()
    Rc2, t2 = camera_pose2.to_Rt()
    R = Rc.T
    T = -R.dot(t)
    R2 = Rc2.T
    T2 = -R2.dot(t2)
    E = np.concatenate((R, T), axis=1)
    E2 = np.concatenate((R2, T2), axis=1)
    pts_img_cam_proj = K.dot(E).dot(np.vstack((pts_world, np.ones((1, 4)))))
    pts_img_cam_proj = pts_img_cam_proj[:2] / pts_img_cam_proj[2]
    pts_img_cam_proj2 = K.dot(E2).dot(np.vstack((pts_world, np.ones((1, 4)))))
    pts_img_cam_proj2 = pts_img_cam_proj2[:2] / pts_img_cam_proj2[2]
    print(camera_pose)