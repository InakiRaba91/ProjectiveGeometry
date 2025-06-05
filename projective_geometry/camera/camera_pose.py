from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import roll_tilt_pan_to_rotation_matrix


class CameraPose:
    """Class with camera pose parameters including 3D location/orientation of the camera.

    Attributes:
        tx: x-location of camera in 3D world
        ty: y-location of camera in 3D world
        tz: z-location of camera in 3D world
        rx: rotation around x-axis
        ry: rotation around y-axis
        rz: rotation around z-axis
    """

    def __init__(self, tx: float, ty: float, tz: float, roll: float, tilt: float, pan: float):
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.roll = roll
        self.tilt = tilt
        self.pan = pan

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [tx, ty, tz, rx, ry, rz]
        """
        return np.array([self.tx, self.ty, self.tz, self.roll, self.tilt, self.pan])

    @property
    def postion_xyz(self) -> np.ndarray:
        """Returns the position of the camera in 3D space as a numpy array."""
        return np.array([self.tx, self.ty, self.tz])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Calculates the rotation matrix based on roll, tilt, and pan angles.
        Returns:
            3x3 numpy array representing the rotation matrix.
        """
        return roll_tilt_pan_to_rotation_matrix(self.roll, self.tilt, self.pan)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.
        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal
        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(self, other.__class__):
            return max(np.abs(self.to_array() - other.to_array())) < tol
        return False

    def __repr__(self):
        return f"CameraPose(tx={self.tx}, ty={self.ty}, tz={self.tz}, roll={self.roll}, tilt={self.tilt}, pan={self.pan})"
