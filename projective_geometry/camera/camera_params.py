from __future__ import annotations

from typing import Any

import numpy as np

from projective_geometry.camera.camera_distortion import CameraDistortion
from projective_geometry.camera.camera_pose import CameraPose


class CameraParams:
    """Class with camera parameters including camera pose, distortion coefficients and focal length

    Attributes:
        camera_pose: contains extrinsic parameters including 3D location/orientation of the camera
        focal_length: distance between camera pinhole and film (analog camera)
        camera_distorion: contains the distortion coefficients
    """

    def __init__(
        self,
        camera_pose: CameraPose,
        focal_length: float,
        camera_distortion: CameraDistortion | None = None,
    ) -> None:
        self.camera_pose = camera_pose
        self.focal_length = focal_length
        self.camera_distorion = camera_distortion or CameraDistortion(0.0, 0.0, 0.0, 0.0, 0.0)

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [tx, ty, tz, rx, ry, rz, focal_length]
        """
        return np.concatenate(
            (self.camera_pose.to_array(), self.camera_distorion.to_array(), np.array([self.focal_length])), axis=0
        )

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.
        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal
        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(self, other.__class__):
            return (
                (self.camera_pose == other.camera_pose)
                and (self.camera_distorion == other.camera_distortion)
                and ((self.focal_length - other.focal_length) < tol)
            )
        return False

    def __repr__(self):
        return f"CameraParams(camera_pose={str(self.camera_pose)}, camera_distorion={str(self.camera_distorion)},focal_length={self.focal_length})"


class CameraParams2:
    """Class with camera parameters including camera pose, distortion coefficients and focal length

    Attributes:
        camera_pose: contains extrinsic parameters including 3D location/orientation of the camera
        focal_length_xy: distance between camera pinhole and film (analog camera)
        camera_distorion: contains the distortion coefficients
    """

    def __init__(
        self,
        camera_pose: CameraPose,
        focal_length_xy: float | tuple[float, float] | np.ndarray,
        camera_distortion: CameraDistortion | None = None,
    ) -> None:
        if isinstance(focal_length_xy, (int, float)):
            focal_length_xy = focal_length_xy, focal_length_xy
        if isinstance(focal_length_xy, tuple):
            focal_length_xy = np.array(focal_length_xy)

        self.camera_pose = camera_pose
        self.focal_length_xy = focal_length_xy
        self.camera_distorion = camera_distortion or CameraDistortion(0.0, 0.0, 0.0, 0.0, 0.0)

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [tx, ty, tz, roll, tilt, pan, k1, k2, p1, p2, k3, focal_length]
        """
        return np.concatenate((self.camera_pose.to_array(), self.camera_distorion.to_array(), self.focal_length_xy), axis=0)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.
        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal
        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(self, other.__class__):
            return (
                (self.camera_pose == other.camera_pose)
                and (self.camera_distorion == other.camera_distortion)
                and (np.sum(self.focal_length_xy - other.focal_length) < tol)
            )
        return False

    def __repr__(self):
        return f"CameraParams(camera_pose={str(self.camera_pose)}, camera_distorion={str(self.camera_distorion)},focal_length={self.focal_length_xy})"
