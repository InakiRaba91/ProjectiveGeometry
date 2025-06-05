# Copyrights (C) StatsBomb Services Ltd. 2021. - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly
# prohibited. Proprietary and confidential
from typing import Any, Tuple, Union

import cv2
import numpy as np

from ..draw import Color
from .exceptions import LineFromEqualPointsException
from .point import Point2D


class LineSegment:
    """2D line segment defined by its two endpoints

    Args:
        pt1: first endpoint
        pt2: second endpoint
    """

    tol = 1e-6

    def __init__(self, pt1: Point2D, pt2: Point2D):
        if (pt1 - pt2).length() <= self.tol:
            raise LineFromEqualPointsException("Both points are equal.")
        self._pt1 = pt1
        self._pt2 = pt2

    @property
    def pt1(self) -> Point2D:
        """Returns 1st point"""
        return self._pt1

    @property
    def pt2(self) -> Point2D:
        """Returns 2nd point"""
        return self._pt2

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.

        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal

        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(other, LineSegment):
            pt1_equal = (self.pt1 - other.pt1).length() < tol
            pt2_equal = (self.pt2 - other.pt2).length() < tol
            return pt1_equal and pt2_equal
        return False

    def __neg__(self) -> "LineSegment":
        """Flips a line segment180º w.r.t. the origin of coordinates

        Args: None

        Returns:
            flipped LineSegment
        """
        return LineSegment(pt1=-self.pt1, pt2=-self.pt2)

    def __add__(self, other: Union["LineSegment", Point2D]) -> "LineSegment":  # type: ignore
        """Adds a point or line segment to line segment

        Args:
            other: Point to add

        Returns:
            LineSegment resulting from the sum
        """
        if isinstance(other, Point2D):
            return LineSegment(pt1=self.pt1 + other, pt2=self.pt2 + other)
        elif isinstance(other, LineSegment):
            return LineSegment(pt1=self.pt1 + other.pt1, pt2=self.pt2 + other.pt2)

    def length(self) -> float:
        """Computes the length of the line segment

        Args: None

        Returns:
            float length of the line segment
        """
        return (self.pt1 - self.pt2).length()

    def scale(self, pt: Point2D) -> "LineSegment":
        """Provides the 2D segment after applying a scaling of the 2D space with
        the scaling given in each coordinate of point

        The 2D x-y space is scaled by
        x' = x * pt.x
        y' = y * pt.y

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            LineSegment resulting from scaling the 2D space
        """
        scale_x = pt.x
        scale_y = pt.y
        return LineSegment(
            pt1=Point2D(x=self.pt1.x * scale_x, y=self.pt1.y * scale_y),
            pt2=Point2D(x=self.pt2.x * scale_x, y=self.pt2.y * scale_y),
        )

    def keypoints(self, distance: float = 0.1) -> list[Point2D]:
        """Generates a list of keypoints along the line segment"""
        num_points = int(self.length() / distance) + 1
        dx = (self.pt2.x - self.pt1.x) / num_points
        dy = (self.pt2.y - self.pt1.y) / num_points
        return [Point2D(x=self.pt1.x + i * dx, y=self.pt1.y + i * dy) for i in range(num_points + 1)]

    def draw(self, img: np.ndarray, color: Tuple[Any, ...] = Color.RED, thickness: int = 3):
        """Draws the segment within the given image in-place

        Note:
            This method modifies the provided image in place

        Args:
            img: ndarray image to draw the segment in
            color: BGR int tuple indicating the color of the segment
            thickness: int thickness of the drawn segment

        Returns: None
        """
        cv2.line(
            img=img,
            pt1=(round(self.pt1.x), round(self.pt1.y)),
            pt2=(round(self.pt2.x), round(self.pt2.y)),
            color=color,
            thickness=thickness,
        )
