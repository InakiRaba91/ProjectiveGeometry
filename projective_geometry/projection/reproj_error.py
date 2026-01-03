import numpy as np

from projective_geometry.camera import CameraPose


def reproj_error_points_pose(camera_pose: CameraPose, pts_world: np.ndarray, pts_cam: np.ndarray) -> float:
    """
    Calculate reprojection error for points using given camera pose

    Args:
        camera: Camera pose to calculate reprojection error with
        pts: tuple of Points to calculate reprojection error for

    Returns:
        Tuple[Point, ...]: Reprojected points
    """
    # Project
    Rc, t = camera_pose.to_Rt()
    R = Rc.T
    T = -R @ t
    pts_world_proj = R @ pts_world + T

    # Normalize projected points
    norm_proj = np.linalg.norm(pts_world_proj, axis=0)
    pts_world_proj_normalized = pts_world_proj / norm_proj

    # Normalize camera points
    norm_cam = np.linalg.norm(pts_cam, axis=0)
    pts_cam_normalized = pts_cam / norm_cam

    # Compute cosine of angles between corresponding vectors
    cos_angles = np.sum(pts_world_proj_normalized * pts_cam_normalized, axis=0)

    # Compute angular distances (1 - cos(angle))
    angular_errors = 1.0 - cos_angles

    # Return mean of angular errors
    # We use angular error because we assume no knowledge of the focal length, so points are aligned but not scaled
    return np.mean(angular_errors)
