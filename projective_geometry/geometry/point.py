from __future__ import annotations

from typing import Any, Tuple

import cv2
import numpy as np

from projective_geometry.draw import Color


class Point:
    """2D point"""

    def __init__(self, x: float, y: float):
        """
        Args:
            x: x-coordinate
            y: y-coordinate
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        """x-coordinate"""
        return self._x

    @property
    def y(self) -> float:
        """y-coordinate"""
        return self._y

    def to_homogeneous(self) -> np.ndarray:
        """Converts to numpy array in homogenous coordinates

        Returns:
            ndarray in homogeneous coordinates [x, y, 1]
        """
        return np.array([self.x, self.y, 1.0])

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    @classmethod
    def from_homogeneous(cls, pt_homogeneous: np.ndarray) -> "Point":
        """Converts from numpy array in homogenous coordinates to 2D Point

        Args:
            homogeneous_pt: ndarray in homogeneous coordinates [x, y, 1]

        Returns:
            2D Point
        """
        assert len(pt_homogeneous) == 3, pt_homogeneous
        return cls(x=pt_homogeneous[0] / pt_homogeneous[2], y=pt_homogeneous[1] / pt_homogeneous[2])

    def draw(self, img: np.ndarray, color: Tuple[Any, ...] = Color.RED, radius: int = 3, thickness: int = 3):
        """Draws the point as a circle within the given image in-place

        Note:
            This method modifies the provided image in place

        Args:
            img: ndarray image to draw the point in
            color: BGR int tuple indicating the color of the point
            radius: int radius of the drawn circle
            thickness: int thickness of the drawn circle

        Returns: None
        """
        cv2.circle(
            img=img,
            center=(round(self.x), round(self.y)),
            color=color,
            radius=radius,
            thickness=thickness,
        )
