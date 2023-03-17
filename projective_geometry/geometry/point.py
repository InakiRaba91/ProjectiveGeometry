from __future__ import annotations

from typing import Any, Tuple, Union

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

    def __add__(self, other: Union[Point, float]) -> Point:  # type: ignore
        """Adds two points

        Args:
            other: Point/float to add

        Returns:
            Point resulting from the sum
        """
        assert isinstance(other, (float, Point))
        if isinstance(other, Point):
            return Point(x=self.x + other.x, y=self.y + other.y)
        else:
            return Point(x=self.x + other, y=self.y + other)

    def __sub__(self, other: Union[Point, float]) -> Point:
        """Substracts a point from another

        Args:
            other: Point/float to substract

        Returns:
            Point resulting from the substraction
        """
        assert isinstance(other, (float, Point))
        if isinstance(other, Point):
            return Point(x=self.x - other.x, y=self.y - other.y)
        else:
            return Point(x=self.x - other, y=self.y - other)

    def __mul__(self, other: Union[float, Point]) -> Union[float, Point]:  # type: ignore
        """Multiplies two points or a point by a scalar

        Args:
            other: Point/float to multiply by

        Returns:
            result of multiplication:
              float resulting from the dot product of two points or
              Point resulting from the multiplication by a scalar
        """
        if isinstance(other, Point):
            return self.x * other.x + self.y * other.y
        # Mypy does not recognize the if clause, so it doesn't realize
        # other can only be an int/float at this point and raises an error
        # because the operation can't be performed if other was a Point
        assert isinstance(other, (float, int))
        return Point(x=self.x * other, y=self.y * other)

    def __rmul__(self, other: Union[float, Point]) -> Union[float, Point]:  # type: ignore
        """Applies left multiplication for scalars (in case we do 2*pt instead of pt*2)

        Args:
            other: Point/float to multiply by

        Returns:
            result of multiplication:
              float resulting from the dot product of two points or
              Point resulting from the multiplication by a scalar
        """
        return self * other

    def __truediv__(self, value: float) -> Union[float, Point]:
        """Divides a point by a scalar (division between points is not defined)

        Args:
            other: float scalar to multiply by

        Returns:
             Point resulting from the division by a scalar
        """
        return self * (1 / value)

    def __neg__(self) -> Point:
        """Flips a point 180º w.r.t. the origin of coordinates
        Args: None
        Returns:
            flipped Point
        """
        return Point(x=-self.x, y=-self.y)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.

        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal

        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(self, other.__class__):
            return (self - other).length() < tol
        return False

    def length(self) -> float:
        """Computes the length of the vector from the origin of coordinates to the point

        Returns:
            float length of the vector from the origin of coordinates to the point
        """
        return np.sqrt(self.x**2 + self.y**2)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"

    def scale(self, pt: Point) -> Point:
        """Provides the point after applying a scaling of the 2D space with
        the scaling given in each coordinate of point
        The 2D x-y space is scaled by
        x' = x * pt.x
        y' = y * pt.y

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            LineSegment resulting from scaling the 2D space
        """
        return Point(x=self.x * pt.x, y=self.y * pt.y)

    def rotate(self, angle: float) -> Point:
        """Rotates a point by the degrees given in angle counter clock-wise
        A point is rotated by applying
        x' = x * cos(angle) - y * sin(angle)
        y' = x * sin(angle) + y * cos(angle)
        Args:
            angle: float indicating rotation w.r.t. x-axis counter clock-wise in degrees
        Returns:
            rotated Point
        """
        rads = np.deg2rad(angle)
        x = self.x * float(np.cos(rads)) - self.y * float(np.sin(rads))
        y = self.x * float(np.sin(rads)) + self.y * float(np.cos(rads))
        return Point(x=x, y=y)

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
