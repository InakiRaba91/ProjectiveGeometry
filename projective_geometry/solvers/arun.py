# Code grabbed from: https://github.com/jingnanshi/aruns-method


import numpy as np

from projective_geometry.camera.camera_pose import CameraPose


def arun_camera_pose(pts_world: np.ndarray, pts_cam: np.ndarray) -> CameraPose:
    """
    Solve 3D registration using Arun's method: pts_cam = R @ pts_world + t

    Args:
        pts_world: 3xN matrix of world points
        pts_cam: 3xN matrix of camera points

    Returns:
        CameraPose object with estimated rotation and translation
    """
    N = pts_world.shape[1]
    assert pts_cam.shape[1] == N

    # calculate centroids
    world_centroid = np.reshape(1 / N * (np.sum(pts_world, axis=1)), (3, 1))
    cam_centroid = np.reshape(1 / N * (np.sum(pts_cam, axis=1)), (3, 1))

    # calculate the vectors from centroids
    q_world = pts_world - world_centroid
    q_cam = pts_cam - cam_centroid

    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = q_world[:, i]
        bi = q_cam[:, i]
        H = H + np.outer(ai, bi)
    U, _, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # translation estimation
    t = cam_centroid - R @ world_centroid

    return CameraPose.from_Rt(R=R.T, t=-R.T.dot(t))
