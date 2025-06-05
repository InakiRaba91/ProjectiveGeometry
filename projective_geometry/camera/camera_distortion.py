from __future__ import annotations

from typing import Any

import numpy as np


class CameraDistortion:
    """Class representing camera distortion parameters."""

    def __init__(self, k1: float, k2: float, p1: float, p2: float, k3: float) -> None:
        self.k1 = k1
        """firth radial distortion coefficient"""
        self.k2 = k2
        """second radial distortion coefficient"""
        self.p1 = p1
        """x tangential distortion coefficient"""
        self.p2 = p2
        """y tangential distortion coefficient"""
        self.k3 = k3
        """third radial distortion coefficient"""

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            [k1, k2, p1, p2, k3]
        """
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])

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
        return f"CameraDistortion(k1={self.k1}, k2={self.k2}, p1={self.p1}, p2={self.p2}, k3={self.k3})"
