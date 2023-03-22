from __future__ import annotations

from typing import List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from projective_geometry.camera.camera_params import CameraParams
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry.line import Line
from projective_geometry.geometry.point import Point


class Camera:
    """
    Class responsible for maintaining Homography and associated planar projection functions.

    Note:
        This class internally stores a 3 x 3 homography matrix. The team is concerned about mapping
        the ground plane in the 3D world to the 2D pixel image - all rays from the camera that
        intersect with the XY plane in 3D space.

        As a result, the full 3 x 4 projection matrix can be simplified. The height above ground
        (i.e. the z axis in 3D cartesian space) is zeroed out, when projecting points between
        pixel and pitch domains.
    """

    def __init__(self, H: np.ndarray) -> None:
        """
        Initialiser

        Args:
            H (np.ndarray): 3x3 Homography matrix.
        """
        # checks
        err_msg = f"Expected homography matrix of shapes: (3, 3); got {H.shape}"
        assert H.shape == (3, 3), err_msg

        # store state
        self._H = H.copy()

    @property
    def H(self) -> np.ndarray:
        return self._H.copy()

    def get_horizon_line(self) -> Line:
        """
        Computes horizon line in image. To do so, it leverages the fact that both points [1, 0, 0] and [0, 1, 0] are in the
        horizon. Therefore, the line that passes through both, given by [0, 0, 1] is in the horizon. As a consequence, it
        suffices to project that line into the image


        Returns:
            Line corresponding to horizon
        """
        # it can be obtained by projecting points [1, 0, 0] and [0, 1, 0]. Or alternatively projecting the line [0, 0, 1].
        # Projecting a point is p'=H·p, so projecting a line is l'=Hinv.T·l, which is equivalent to extracting the last row
        # of Hinv to determine the horizon line
        Hinv = np.linalg.inv(self.H)
        horizon_line = Hinv[-1]
        return Line(a=horizon_line[0], b=horizon_line[1], c=horizon_line[2])

    @classmethod
    def from_point_correspondences(
        cls,
        pts_source: List[Point],
        pts_target: List[Point],
        ransac: bool = False,
    ) -> "Camera":
        """
        Method to generate the homography from point correspondences. The method uses the cv2.findHomography
        method to retrieve the homography matrix.

        Args:
            pts_source (List[Point]): List of points in first image
            pts_target (List[Point]): List of points in second image
            ransac (bool): Boolean of whether to use RANSAC optimisation to get homography. Defaults to False.

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # checks
        num_points = len(pts_target)
        assert num_points >= 4, f"At least 4 points are required for camera calibration, only {num_points} were given"

        # generate homography using opencv
        # cv2.findHomography requires float32 arrays as arguments
        template_pts_f32 = np.array([pt.to_array() for pt in pts_source]).astype(np.float32)
        image_pts_f32 = np.array([pt.to_array() for pt in pts_target]).astype(np.float32)

        if ransac:
            homography_matrix, _ = cv2.findHomography(template_pts_f32, image_pts_f32, cv2.RANSAC)
        else:
            homography_matrix, _ = cv2.findHomography(template_pts_f32, image_pts_f32)

        if homography_matrix is None:
            err_msg = "Homography calculation from point correspondences failed!"
            raise ValueError(f"{err_msg}")

        return cls(H=homography_matrix)

    @staticmethod
    def intrinsic_matrix_from_focal_length(focal_length: float, image_size: ImageSize) -> np.ndarray:
        """Compute intrinsic matrix from focal length
        Args:
            focal_length: Distance focal length
            image_size: ImageSize with the size of the image the camera is projecting the 3D world into
        Returns:
            3x3 ndarray intrinsic matrix from given focal length assuming negligible skewness
        """
        return np.array(
            [
                [focal_length, 0, image_size.width / 2.0],
                [0, focal_length, image_size.height / 2.0],
                [0, 0, 1],
            ]
        )

    @classmethod
    def from_camera_params(cls, camera_params: CameraParams, image_size: ImageSize) -> "Camera":
        """Build camera components from parameters
        The camera can be shifted to a 3D location [tx, ty, tz], and the image can undergo a 3D rotation.
        The extrinsic matrix E = [R | T] undoes the rotation and shifts everything to the origin of
        coordinates before projecting the 3D world into the image.
        If we have a point p=[x,y,z] in the 3d world, we can transform it to homogeneous coordinates
        ph = [x, y, z, 1]. Before being projected with the intrinsic matrix, it undergoes the transform
        p'= E * ph = (R * p) + T = R * (p - t) = Rc^-1 * (p - t)
        Therefore
           Rc = R^-1 = R^T (for orthogonal matrices M^-1=M^T)
           t = -R*T => T = -Rc*t
        Source: https://ksimek.github.io/2012/08/22/extrinsic/
        Note:
            The three rotations are given in a global frame of reference (extrinsic), and the order is x->y->z
        Args:
            camera_params: CameraParams with 3D location, rotation angles and focal length of the camera
            image_size: ImageSize of the image we're projecting into with the camera
        Returns:
            Camera from given params
        """
        camera_pose = camera_params.camera_pose
        rot_angles = [camera_pose.roll, camera_pose.tilt, camera_pose.pan]
        Rc = Rotation.from_euler("xyz", rot_angles, degrees=True).as_matrix()
        R = Rc.T  # transpose
        t = np.array([[camera_pose.tx], [camera_pose.ty], [camera_pose.tz]])
        T = -Rc.T.dot(t)
        K = cls.intrinsic_matrix_from_focal_length(focal_length=camera_params.focal_length, image_size=image_size)
        E = np.concatenate((R, T), axis=1)

        # get rid of third column
        E = E[:, [0, 1, 3]]
        H = K.dot(E)
        return cls(H=H)

    def __repr__(self):
        return f"Camera(H={self.H})"
