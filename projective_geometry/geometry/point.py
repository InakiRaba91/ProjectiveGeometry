from __future__ import annotations

from typing import Any, Literal, Tuple, TypeAlias, Union

import cv2
import numpy as np

from projective_geometry.draw import Color


class Point2D:
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

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [x, y]
        """
        return np.array([self.x, self.y])

    def to_homogeneous(self) -> np.ndarray:
        """Converts to numpy array in homogenous coordinates

        Returns:
            ndarray in homogeneous coordinates [x, y, 1]
        """
        return np.array([self.x, self.y, 1.0])

    def __add__(self, other: Union[Point2D, float]) -> Point2D:  # type: ignore
        """Adds two points

        Args:
            other: Point/float to add

        Returns:
            Point resulting from the sum
        """
        assert isinstance(other, (float, Point2D))
        if isinstance(other, Point2D):
            return Point2D(x=self.x + other.x, y=self.y + other.y)
        else:
            return Point2D(x=self.x + other, y=self.y + other)

    def __sub__(self, other: Union[Point2D, float]) -> Point2D:
        """Substracts a point from another

        Args:
            other: Point/float to substract

        Returns:
            Point resulting from the substraction
        """
        assert isinstance(other, (float, Point2D))
        if isinstance(other, Point2D):
            return Point2D(x=self.x - other.x, y=self.y - other.y)
        else:
            return Point2D(x=self.x - other, y=self.y - other)

    def __mul__(self, other: Union[float, Point2D]) -> Union[float, Point2D]:  # type: ignore
        """Multiplies two points or a point by a scalar

        Args:
            other: Point/float to multiply by

        Returns:
            result of multiplication:
              float resulting from the dot product of two points or
              Point resulting from the multiplication by a scalar
        """
        if isinstance(other, Point2D):
            return self.x * other.x + self.y * other.y
        # Mypy does not recognize the if clause, so it doesn't realize
        # other can only be an int/float at this point and raises an error
        # because the operation can't be performed if other was a Point
        assert isinstance(other, (float, int))
        return Point2D(x=self.x * other, y=self.y * other)

    def __rmul__(self, other: Union[float, Point2D]) -> Union[float, Point2D]:  # type: ignore
        """Applies left multiplication for scalars (in case we do 2*pt instead of pt*2)

        Args:
            other: Point/float to multiply by

        Returns:
            result of multiplication:
              float resulting from the dot product of two points or
              Point resulting from the multiplication by a scalar
        """
        return self * other

    def __truediv__(self, value: float) -> Union[float, Point2D]:
        """Divides a point by a scalar (division between points is not defined)

        Args:
            other: float scalar to multiply by

        Returns:
             Point resulting from the division by a scalar
        """
        return self * (1 / value)

    def __neg__(self) -> Point2D:
        """Flips a point 180º w.r.t. the origin of coordinates
        Args: None
        Returns:
            flipped Point
        """
        return Point2D(x=-self.x, y=-self.y)

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

    def scale(self, pt: Point2D) -> Point2D:
        """Provides the point after applying a scaling of the 2D space with
        the scaling given in each coordinate of point
        The 2D x-y space is scaled by
        x' = x * pt.x
        y' = y * pt.y

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            Point resulting from scaling the 2D space
        """
        return Point2D(x=self.x * pt.x, y=self.y * pt.y)

    def rotate(self, angle: float) -> Point2D:
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
        return Point2D(x=x, y=y)

    @classmethod
    def from_array(cls, pt: np.ndarray) -> Point2D:
        """Converts from numpy array to 2D Point

        Args:
            pt: ndarray [x, y]

        Returns:
            2D Point
        """
        assert len(pt) == 2, pt
        return cls(x=pt[0], y=pt[1])

    @classmethod
    def from_homogeneous(cls, pt_homogeneous: np.ndarray) -> Point2D:
        """Converts from numpy array in homogenous coordinates to 2D Point

        Args:
            homogeneous_pt: ndarray in homogeneous coordinates [x, y, 1]

        Returns:
            2D Point
        """
        assert len(pt_homogeneous) == 3, pt_homogeneous
        return cls(x=pt_homogeneous[0] / pt_homogeneous[2], y=pt_homogeneous[1] / pt_homogeneous[2])

    def draw(self, img: np.ndarray, color: Tuple[Any, ...] | None = Color.RED, radius: int = 3, thickness: int = 3):
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
            color=color or Color.RED,
            radius=radius,
            thickness=thickness,
        )


Axis: TypeAlias = Literal["x", "y", "z"]
"""Axis of rotation for 3D points, can be 'x', 'y', or 'z'."""


class Point3D:
    """3D point"""

    def __init__(self, x: float, y: float, z: float = 0.0):
        """
        Args:
            x: x-coordinate
            y: y-coordinate
            z: z-coordinate (default is 0)
        """
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self) -> float:
        """x-coordinate"""
        return self._x

    @property
    def y(self) -> float:
        """y-coordinate"""
        return self._y

    @property
    def z(self) -> float:
        """z-coordinate"""
        return self._z

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [x, y]
        """
        return np.array([self.x, self.y, self.z])

    def to_homogeneous(self) -> np.ndarray:
        """Converts to numpy array in homogenous coordinates

        Returns:
            ndarray in homogeneous coordinates [x, y, z, 1]
        """
        return np.array([self.x, self.y, self.z, 1.0])

    def __add__(self, other: Union[Point3D, float]) -> Point3D:
        """Adds two points

        Args:
            other: Point/float to add

        Returns:
            Point resulting from the sum
        """
        assert isinstance(other, (float, Point3D))
        if isinstance(other, Point3D):
            return Point3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)
        else:
            return Point3D(x=self.x + other, y=self.y + other, z=self.z + other)

    def __sub__(self, other: Union[Point3D, float]) -> Point3D:
        """Substracts a point from another

        Args:
            other: Point/float to substract

        Returns:
            Point resulting from the substraction
        """
        assert isinstance(other, (float, Point3D))
        if isinstance(other, Point3D):
            return Point3D(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)
        else:
            return Point3D(x=self.x - other, y=self.y - other, z=self.z - other)

    def __mul__(self, other: Union[float, Point3D]) -> Union[float, Point3D]:  # type: ignore
        """Multiplies two points or a point by a scalar

        Args:
            other: Point/float to multiply by

        Returns:
            result of multiplication:
              float resulting from the dot product of two points or
              Point resulting from the multiplication by a scalar
        """
        if isinstance(other, Point3D):
            return self.x * other.x + self.y * other.y + self.z * other.z
        # Mypy does not recognize the if clause, so it doesn't realize
        # other can only be an int/float at this point and raises an error
        # because the operation can't be performed if other was a Point
        assert isinstance(other, (float, int))
        return Point3D(x=self.x * other, y=self.y * other)

    def __rmul__(self, other: Union[float, Point3D]) -> Union[float, Point3D]:
        """Applies left multiplication for scalars (in case we do 2*pt instead of pt*2)

        Args:
            other: Point/float to multiply by

        Returns:
            result of multiplication:
              float resulting from the dot product of two points or
              Point resulting from the multiplication by a scalar
        """
        return self * other

    def __truediv__(self, value: float) -> Union[float, Point3D]:
        """Divides a point by a scalar (division between points is not defined)

        Args:
            other: float scalar to multiply by

        Returns:
             Point resulting from the division by a scalar
        """
        return self * (1 / value)

    def __neg__(self) -> Point3D:
        """Flips a point 180º w.r.t. the origin of coordinates
        Args: None
        Returns:
            flipped Point
        """
        return Point3D(x=-self.x, y=-self.y, z=-self.z)

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
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"

    def scale(self, pt: Point3D) -> Point3D:
        """Provides the point after applying a scaling of the 2D space with
        the scaling given in each coordinate of point
        The 2D x-y space is scaled by
        x' = x * pt.x
        y' = y * pt.y
        z' = z * pt.z

        Args:
            pt: Point defining the scaling of each axis in the 3D space

        Returns:
            Point resulting from scaling the 3D space
        """
        return Point3D(x=self.x * pt.x, y=self.y * pt.y, z=self.z * pt.z)

    def rotate(self, angle: float, axis: Axis) -> "Point3D":
        """
        Rotate the point around the specified axis (x, y, or z) by angle in degrees.
        Rotation is counter-clockwise when looking along the positive axis toward the origin.

        Args:
            angle: Rotation angle in degrees.
            axis: Axis of rotation ('x', 'y', or 'z').

        Returns:
            Rotated Point3D instance.
        """
        rads = np.deg2rad(angle)
        c, s = np.cos(rads), np.sin(rads)

        if axis == "x":
            y = self.y * c - self.z * s
            z = self.y * s + self.z * c
            return Point3D(self.x, y, z)
        elif axis == "y":
            x = self.x * c + self.z * s
            z = -self.x * s + self.z * c
            return Point3D(x, self.y, z)
        elif axis == "z":
            x = self.x * c - self.y * s
            y = self.x * s + self.y * c
            return Point3D(x, y, self.z)
        else:
            raise ValueError(f"Invalid axis '{axis}', expected one of: 'x', 'y', 'z'")

    @classmethod
    def from_array(cls, pt: np.ndarray) -> Point3D:
        """Converts from numpy array to 3D Point

        Args:
            pt: ndarray [x, y, z]

        Returns:
            3D Point
        """
        assert len(pt) == 3, pt
        return cls(x=pt[0], y=pt[1], z=pt[2])

    @classmethod
    def from_homogeneous(cls, pt_homogeneous: np.ndarray) -> Point3D:
        """Converts from numpy array in homogenous coordinates to 3D Point

        Args:
            homogeneous_pt: ndarray in homogeneous coordinates [x, y, z, C]

        Returns:
            2D Point
        """
        assert len(pt_homogeneous) == 4, pt_homogeneous
        x, y, z = pt_homogeneous[:3] / pt_homogeneous[3]
        return cls(x=x, y=y, z=z)

    # FIXME: This is a quick hack!
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
