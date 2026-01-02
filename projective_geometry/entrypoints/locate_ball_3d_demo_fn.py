import numpy as np

from projective_geometry.camera import Camera, CameraParams, CameraPose
from projective_geometry.draw.image_size import BASE_IMAGE_SIZE
from projective_geometry.projection.projectors import project_sphere
from projective_geometry.entrypoints.utils import BALL_RADIUS


def locate_ball_3d_demo(bx: int, by: int, bz: int):
    tx, ty, tz = 0, -20, 25
    t = np.array([tx, ty, tz])
    rx, ry, rz = -142, 0, 0
    focal_length = 1100
    camera = Camera.from_camera_params(
        camera_params=CameraParams(
            camera_pose=CameraPose(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz),
            focal_length=focal_length,
        ),
        image_size=BASE_IMAGE_SIZE,
    )
    H_origin = Camera.full_homography_from_camera_params(
        camera_params=CameraParams(
            camera_pose=CameraPose(tx=0, ty=0, tz=0, rx=rx, ry=ry, rz=rz),
            focal_length=focal_length,
        ),
        image_size=BASE_IMAGE_SIZE,
    )

    # Project sphere to get 2D conic
    pos = np.array([bx, by, bz])
    C = project_sphere(pos=pos, radius=BALL_RADIUS, camera=camera)

    # Back project 2D ball to cone
    Q_co = H_origin.T.dot(C.M).dot(H_origin)
    Q = Q_co[:3, :3]

    # Get cone axis
    eigvals, eigvecs = np.linalg.eig(Q)
    for i in range(3):
        others = [eigvals[j] for j in range(3) if j != i]
        if np.sign(eigvals[i]) != np.sign(others[0]) and np.sign(others[0]) == np.sign(others[1]):
            eigvec_cone_axis = eigvecs[:, i]
            eigval_cone_axis = eigvals[i]
            eigval_perp = others[0]
            break

    # ensure cone axis points away from origin
    s = t.dot(eigvec_cone_axis)
    if s < 0:
        eigvec_cone_axis = -eigvec_cone_axis
    sin_theta = np.sqrt(eigval_cone_axis / (eigval_cone_axis - eigval_perp))
    d = BALL_RADIUS / sin_theta
    est_b = t - d * eigvec_cone_axis
    print(f"Estimated 3D ball location: ({est_b[0]}, {est_b[1]}, {est_b[2]})")
