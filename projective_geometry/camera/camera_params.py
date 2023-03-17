from __future__ import annotations

from typing import Any

import numpy as np

from projective_geometry.camera import CameraPose


class CameraParams:
    """Class with camera parameters including camera pose and focal length

    Attributes:
        camera_pose: contains extrinsic parameters including 3D location/orientation of the camera
        focal_length: distance between camera pinhole and film (analog camera)
    """

    def __init__(self, camera_pose: CameraPose, focal_length: float):
        self.camera_pose = camera_pose
        self.focal_length = focal_length

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [tx, ty, tz, roll, tilt, pan, focal_length]
        """
        return np.concatenate((self.camera_pose.to_array(), np.array([self.focal_length])), axis=0)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.
        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal
        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(self, other.__class__):
            return (self.camera_pose == other.camera_pose) and ((self.focal_length - other.focal_length) < tol)
        return False
