from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from ..draw import Color
from .exceptions import InvalidLineException, LineFromEqualPointsException
from .point import Point


class Line:
    """2D line parametrized in the general form ax+by+c=0"""

    def __init__(self, a: float, b: float, c: float):
        """Initializes a line with the given coefficients

        Args:
            a: x-weight in equation Ax + by + c.
            b: y-weight in equation ax + By + c.
            c: (float): constant weight in equation ax + by + c.
        """

        self._a = a
        self._b = b
        self._c = c

        self._check_valid_parametrization()

    @property
    def a(self) -> float:
        """Returns the x-weight in equation Ax + by + c"""
        return self._a

    @property
    def b(self) -> float:
        """Returns the y-weight in equation ax + By + c"""
        return self._b

    @property
    def c(self) -> float:
        """Returns the constant weight in equation ax + by + C"""
        return self._c

    def _check_valid_parametrization(self) -> None:
        """Checks the line parametrization is not invalid (a=b=0)

        Raises:
            ValueError: if invalid parametrization is provided (a=b=0)
        """
        tol = 1e-6
        if (np.abs(self.a - self.b) <= tol) and (np.abs(self.a) <= tol):
            raise InvalidLineException("Invalid line parametrization (if a=b=0 -> c=0).")

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [a, b, c]
        """
        return np.array([self.a, self.b, self.c])

    @classmethod
    def from_points(cls, pt1: Point, pt2: Point, tol: float = 1e-6) -> "Line":
        """Computes the line that passes through two given points

        Args:
            pt1: first Point the line passes through
            pt2: second Point the line passes through
            tol: float error tolerance for considering two points are equal

        Returns:
            line parametrization (ax+by+c) passing through the provided points
        """
        if (pt1 - pt2).length() <= tol:
            raise LineFromEqualPointsException("Both points are equal.")
        a, b, c = np.cross(pt1.to_homogeneous(), pt2.to_homogeneous())
        return cls(a=a, b=b, c=c)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.

        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal

        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(other, Line):
            # we normalize both line arrays
            normalized_self_line = self.to_array() / np.max(np.abs(self.to_array()))
            normalized_other_line = other.to_array() / np.max(np.abs(other.to_array()))
            # they need to be equal except maybe for the sign
            proportional_lines = (np.abs(np.abs(normalized_self_line) - np.abs(normalized_other_line)) < tol).all()
            return proportional_lines
        return False

    def __neg__(self) -> Line:
        """Flips a line 180º w.r.t. the origin of coordinates

        Returns:
            flipped Line
        """
        return Line(a=self._a, b=self._b, c=-self._c)

    def __add__(self, pt: Point) -> Line:  # type: ignore
        """Adds a point to line

        When adding a point, the resulting line is parallel to the original one.
        We can compute a point in the original line, i.e. (-c/a, 0) or (0, -c/b),
        add the given point and force the new line to pass through this point (x', y')
        The original line equation evaluated at the point is
        a*x'+b*y'+c=0
        If we do c' = -(a*x'+b*y')
        the new line
        ax+by+c'=0
        fulfills the definition

        Args:
            pt: Point to add

        Returns:
            Line resulting from the sum
        """
        if np.abs(self._a) > 1e-8:
            pt_line = Point(x=-self._c / self._a, y=0)
        else:
            pt_line = Point(x=0, y=-self._c / self._b)
        pt_new_line = pt_line + pt
        return Line(a=self._a, b=self._b, c=-(self._a * pt_new_line.x + self._b * pt_new_line.y))

    def __repr__(self):
        return f"Line(a={self.a}, b={self.b}, c={self.c})"

    def contains_pt(self, pt: Point, tol: float = 1e-6) -> bool:
        """Determines whether a point belongs to a line or not

        The points that belong to the line satisfy the general equation
        ax + by +c =0

        Args:
            pt: Point to check whether or not it belongs to the line
            tol: float error tolerance for considering a point belongs to the line

        Returns:
            boolean indicating if the point belongs to the line
        """
        return np.abs(np.dot(self.to_array(), pt.to_homogeneous())) <= tol

    def intersection_line(self, other: Line, tol: float = 1e-6) -> Optional[Point]:
        """Find the point of intersection between two lines
        Explanation: https://www.cuemath.com/geometry/intersection-of-two-lines/

        Args:
            other: second Line to find the intersection with
            tol: float error tolerance for considering a point belongs to the line

        Returns:
            None if they don't intersect or are the same line, Point of intersection otherwise
        """
        if np.abs((self._a * other.b) - (other.a * self._b)) > tol:
            x = (other.c * self._b - self._c * other.b) / (self._a * other.b - other.a * self._b)
            y = (self._c * other.a - other.c * self._a) / (self._a * other.b - other.a * self._b)
            return Point(x=x, y=y)
        else:
            return None

    def draw(self, img: np.ndarray, color: Tuple[Any, ...] = Color.RED, thickness: int = 3, tol: float = 1e-6):
        """Draws the line within the given image in-place

        Note:
            This method modifies the provided image in place

        Args:
            img: ndarray image to draw the line in
            color: BGR int tuple indicating the color of the point
            thickness: int thickness of the drawn line
            tol: float error tolerance for considering a line is vertical

        Returns: None
        """
        image_height, image_width = img.shape[0], img.shape[1]
        if np.abs(self._b) > tol:
            # If the line is not vertical, we find the intersection between
            # the line and the two vertical edges
            line_left_edge_image = Line(a=1, b=0, c=0)
            line_right_edge_image = Line(a=1, b=0, c=-image_width)
            pt1 = self.intersection_line(line_left_edge_image)
            pt2 = self.intersection_line(line_right_edge_image)
        else:
            # If the line is vertical, the general equation is ax+c=0
            # which means x is set at x=-c/s
            pt1 = Point(x=-self._c / self._a, y=0)
            pt2 = Point(x=-self._c / self._a, y=image_height)
        assert pt1 is not None and pt2 is not None  # Otherwise, mypy raises an error in next line
        cv2.line(
            img=img,
            pt1=(round(pt1.x), round(pt1.y)),
            pt2=(round(pt2.x), round(pt2.y)),
            color=color,
            thickness=thickness,
        )
