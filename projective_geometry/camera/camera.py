from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from projective_geometry.camera.camera_params import CameraParams
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry.ellipse import Ellipse
from projective_geometry.geometry.line import Line
from projective_geometry.geometry.point import Point2D
from projective_geometry.image_registration.register import (
    ImageRegistrator,
    MatchedKeypoints,
)


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

    @staticmethod
    def _get_matrix_from_point_correspondences(pts_source: Sequence[Point2D], pts_target: Sequence[Point2D]) -> np.ndarray:
        """
        Method to generate the auxiliary matrix for point correspondences defined in Eq.(10) in
        https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            pts_source (List[Point]): List of points in first image
            pts_target (List[Point]): List of points in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # generate auxiliary matrix
        matrices_pt = []
        assert len(pts_source) == len(
            pts_target
        ), f"Number of points in source ({len(pts_source)}) and target ({len(pts_target)}) must be equal"
        if len(pts_source) == 0:
            return np.array([], dtype=np.int64).reshape(0, 9)

        for pt_source, pt_target in zip(pts_source, pts_target):
            x_s, y_s = pt_source.x, pt_source.y
            x_t, y_t = pt_target.x, pt_target.y
            matrix_pt = np.array(
                [
                    [x_s, y_s, 1, 0, 0, 0, -x_s * x_t, -y_s * x_t, -x_t],
                    [0, 0, 0, x_s, y_s, 1, -x_s * y_t, -y_s * y_t, -y_t],
                ]
            )
            matrices_pt.append(matrix_pt)
        return np.concatenate(matrices_pt, axis=0)

    @staticmethod
    def _get_matrix_from_line_correspondences(lines_source: Sequence[Line], lines_target: Sequence[Line]) -> np.ndarray:
        """
        Method to generate the auxiliary matrix for line correspondences defined in
        https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            lines_source (List[Line]): List of lines in first image
            lines_target (List[Line]): List of lines in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # checks
        assert len(lines_source) == len(
            lines_target
        ), f"Number of lines in source ({len(lines_source)}) and target ({len(lines_target)}) must be equal"
        if len(lines_source) == 0:
            return np.array([], dtype=np.int64).reshape(0, 9)

        # generate auxiliary matrix
        matrices_line = []
        for line_source, line_target in zip(lines_source, lines_target):
            a_s, b_s, c_s = line_source.a, line_source.b, line_source.c
            a_t, b_t, c_t = line_target.a, line_target.b, line_target.c
            matrix_line = np.array(
                [
                    [a_t, 0, a_t * a_s / c_s, b_t, 0, b_t * a_s / c_s, c_t, 0, c_t * a_s / c_s],
                    [0, a_t, a_t * b_s / c_s, 0, b_t, b_t * b_s / c_s, 0, c_t, c_t * b_s / c_s],
                ]
            )
            matrices_line.append(matrix_line)
        return np.concatenate(matrices_line, axis=0)

    @staticmethod
    def _get_matrix_from_multiple_ellipse_correspondences(
        ellipses_source: Sequence[Ellipse], ellipses_target: Sequence[Ellipse]
    ) -> np.ndarray:
        """
        Method to generate the auxiliary matrix for ellipse correspondences defined in
        https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            ellipses_source (List[Ellipse]): List of ellipses in first image
            ellipses_target (List[Ellipse]): List of ellipses in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # checks
        assert len(ellipses_source) == len(
            ellipses_target
        ), f"Number of lines in source ({len(ellipses_source)}) and target ({len(ellipses_target)}) must be equal"
        if len(ellipses_source) < 2:
            return np.array([], dtype=np.int64).reshape(0, 9)

        # normalize to unit frobenius norm and compute scaling factors
        M_source, M_target, Minv_source, Minv_target, k = [], [], [], [], []
        for ell_source, ell_target in zip(ellipses_source, ellipses_target):
            M_s, M_t = ell_source.to_matrix(), ell_target.to_matrix()
            M_s = M_s / np.linalg.norm(M_s)
            M_t = M_t / np.linalg.norm(M_t)
            Minv_s, Minv_t = np.linalg.inv(M_s), np.linalg.inv(M_t)
            M_source.append(M_s)
            M_target.append(M_t)
            Minv_source.append(Minv_s)
            Minv_target.append(Minv_t)
            k.append(np.cbrt(np.linalg.det(M_s) / np.linalg.det(M_t)))

        # generate auxiliary matrix
        matrices_ellipses = []
        Id = np.eye(3)
        # order in a pair matters (i, j) != (j, i)
        for idx1, (Minv_s1, Minv_t1) in enumerate(zip(Minv_source, Minv_target)):
            for idx2, (M_s2, M_t2) in enumerate(zip(M_source, M_target)):
                A = k[idx1] * Minv_s1.dot(M_s2)
                B = k[idx2] * Minv_t1.dot(M_t2)
                b = B.flatten()
                matrix_ellipse = np.concatenate(
                    (
                        np.concatenate((A.T - b[0] * Id, -b[1] * Id, -b[2] * Id), axis=1),
                        np.concatenate((-b[3] * Id, A.T - b[4] * Id, -b[5] * Id), axis=1),
                        np.concatenate((-b[6] * Id, -b[7] * Id, A.T - b[8] * Id), axis=1),
                    ),
                    axis=0,
                )
                matrices_ellipses.append(matrix_ellipse)
        return np.concatenate(matrices_ellipses, axis=0)

    @classmethod
    def _compute_homography_from_aux_matrix(cls, A: np.ndarray) -> "Camera":
        """
        Method to compute the homography from the auxiliary matrix based on the SVD as explained in
        https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            A (np.ndarray): Auxiliary matrix

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # compute SVD
        _, _, Vt = np.linalg.svd(A)

        # get homography as eigenvector corresponding to min eigenvalue
        H = Vt[-1, :].reshape(3, 3)
        return cls(H=H)

    @classmethod
    def from_point_correspondences(cls, pts_source: Sequence[Point2D], pts_target: Sequence[Point2D]) -> "Camera":
        """
        Method to generate the homography from point correspondences. The method uses the
        SVD as explained in https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            pts_source (List[Point]): List of points in first image
            pts_target (List[Point]): List of points in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # checks
        num_points_target = len(pts_target)
        num_points_source = len(pts_source)
        assert (
            num_points_target >= 4
        ), f"At least 4 points are required for camera calibration, only {num_points_target} were given"
        assert (
            num_points_target == num_points_source
        ), f"Number of points in source ({num_points_source}) and target ({num_points_target}) must be equal"

        A = cls._get_matrix_from_point_correspondences(pts_source=pts_source, pts_target=pts_target)
        return cls._compute_homography_from_aux_matrix(A=A)

    @classmethod
    def from_line_correspondences(cls, lines_source: Sequence[Line], lines_target: Sequence[Line]) -> "Camera":
        """
        Method to generate the homography from line correspondences. The method uses the
        SVD as explained in https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            lines_source (List[Line]): List of lines in first image
            lines_target (List[Line]): List of lines in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # checks
        num_lines_target = len(lines_target)
        num_lines_source = len(lines_source)
        assert (
            num_lines_target >= 4
        ), f"At least 4 lines are required for camera calibration, only {num_lines_target} were given"
        assert (
            num_lines_target == num_lines_source
        ), f"Number of lines in source ({num_lines_source}) and target ({num_lines_target}) must be equal"

        A = cls._get_matrix_from_line_correspondences(lines_source=lines_source, lines_target=lines_target)
        return cls._compute_homography_from_aux_matrix(A=A)

    @classmethod
    def from_multiple_ellipse_correspondences(
        cls, ellipses_source: Sequence[Ellipse], ellipses_target: Sequence[Ellipse]
    ) -> "Camera":
        """
        Method to generate the homography from point correspondences. The method uses the
        SVD as explained in https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            pts_source (List[Point]): List of points in first image
            pts_target (List[Point]): List of points in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        # checks
        num_ell_target = len(ellipses_source)
        num_ell_source = len(ellipses_target)
        assert num_ell_target >= 3, f"At least 3 ellipses are required for camera calibration, only {num_ell_target} were given"
        assert (
            num_ell_target == num_ell_source
        ), f"Number of ellipses in source ({num_ell_source}) and target ({num_ell_target}) must be equal"

        A = cls._get_matrix_from_multiple_ellipse_correspondences(
            ellipses_source=ellipses_source, ellipses_target=ellipses_target
        )
        return cls._compute_homography_from_aux_matrix(A=A)

    @classmethod
    def from_correspondences(
        cls,
        pts_source: Sequence[Point2D],
        pts_target: Sequence[Point2D],
        lines_source: Sequence[Line],
        lines_target: Sequence[Line],
        ellipses_source: Sequence[Ellipse],
        ellipses_target: Sequence[Ellipse],
    ) -> "Camera":
        """
        Method to generate the homography from geometric correspondences. The method uses the
        SVD as explained in https://inakiraba91.github.io/projective-geometry-estimating-the-homography-matrix.html

        Args:
            pts_source (List[Point]): List of points in first image
            pts_target (List[Point]): List of points in second image
            lines_source (List[Line]): List of lines in first image
            lines_target (List[Line]): List of lines in second image
            ellipses_source (List[Ellipse]): List of ellipses in first image
            ellipses_target (List[Ellipse]): List of ellipses in second image

        Returns:
            Homography: 3 x 3 Homography object.
        """
        A_pts = cls._get_matrix_from_point_correspondences(pts_source=pts_source, pts_target=pts_target)
        A_lines = cls._get_matrix_from_line_correspondences(lines_source=lines_source, lines_target=lines_target)
        A_ellipses = cls._get_matrix_from_multiple_ellipse_correspondences(
            ellipses_source=ellipses_source, ellipses_target=ellipses_target
        )
        A = np.concatenate((A_pts, A_lines, A_ellipses), axis=0)
        return cls._compute_homography_from_aux_matrix(A=A)

    @classmethod
    def from_point_correspondences_cv2(
        cls,
        pts_source: Sequence[Point2D],
        pts_target: Sequence[Point2D],
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

    @classmethod
    def from_image_registration(cls, target_image: np.ndarray, source_image: np.ndarray) -> Tuple["Camera", MatchedKeypoints]:
        """Compute homography matrix from image registration

        Args:
            target_image: ndarray reference image that will remain static
            source_image: ndarray image that will undergo a homography transform in order to align with target image

        Returns:
            Camera with homography matrix from image registration
        """
        matched_keypoints, H_registration = ImageRegistrator().register(target_image=target_image, source_image=source_image)
        return cls(H=H_registration), matched_keypoints

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
