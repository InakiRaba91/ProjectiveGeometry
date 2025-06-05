from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from ..draw import Color
from .ellipse import Ellipse
from .exceptions import InvalidEllipseArcException
from .line import Line
from .point import Point2D


class EllipseArc:
    """2D ellipse arc

    It is defined by:
    -> Ellipse parametrized as in OpenCV. It is built rotating first, shifting then
    https://docs.opencv.org/3.4.13/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
    -> Half plane fulfilling the greater inequality from given line ax+by+c>0
    Line needs to intersect with the ellipse (non-intersecting or tangent lines are invalid)

    Note:
        The following two equations are equivalent ax+by+c=0, -ax-by-c=0. However, due to
        our convention (line specifies half-plane fulfilling greater inequality), they
        define opposite half planes
        1) ax+by+c>0
        2) -ax-by-c>0 => ax+by+c<0

        Another way to interpret this is using normal vectors. A line ax+by+c=0 has two perpendicular
        normal vectors (c/b, c/a) and (-c/b, -c/a). We will use the convention that the line parametrizes
        the half plane in the direction of the first one. So negating the line would have the opposite
        normal vector
    """

    def __init__(self, ellipse: Ellipse, line: Line, start_angle: Optional[float] = None, end_angle: Optional[float] = None):
        """Initializes an ellipse arc

        Args:
            ellipse (Ellipse): ellipse the arc belongs to.
            line (Line): line intersecting the ellipse at two points to define the arc
            start_angle (float): angle in degrees at which the ellipse arc starts. If not provided,
                it will be computed from points of intersection
            end_angle (float): angle in degrees at which the ellipse arc ends. If not provided,
                it will be computed from points of intersection
        """

        self._ellipse = ellipse
        self._line = line
        self._check_line_and_ellipse_intersect()

        self._start_angle, self._end_angle = self._check_and_set_valid_angles(start_angle, end_angle)

    def _check_and_set_valid_angles(
        self,
        start_angle: Optional[float] = None,
        end_angle: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Checks and sets valid angles for the ellipse arc.

        Args:
            start_angle (float): angle in degrees at which the ellipse arc starts.
            end_angle (float): angle in degrees at which the ellipse arc ends.
        """

        tol: float = 1e-6
        computed_start_angle, computed_end_angle = EllipseArc._angle_covered(ellipse=self.ellipse, line=self.line)

        if start_angle is not None:
            assert (
                np.abs((computed_start_angle - start_angle) % 360) <= tol
            ), f"Provided start_angle {start_angle} differs from start angle of intersection {computed_start_angle}"
        else:
            start_angle = computed_start_angle

        if end_angle is not None:
            assert (
                np.abs((computed_end_angle - end_angle) % 360) <= tol
            ), f"Provided end_angle {end_angle} differs from end angle of intersection {computed_end_angle}"
        else:
            end_angle = computed_end_angle

        return start_angle, end_angle

    def _check_line_and_ellipse_intersect(self) -> None:
        """Checks ellipse and line do intersect at two different points

        Raises:
            ValueError: if ellipse and line do not intersect or are tangent
        """

        pts_intersection = self.ellipse.intersection_line(line=self.line)

        if len(pts_intersection) == 0:
            raise InvalidEllipseArcException("Ellipse and line do not intersect.")
        elif len(pts_intersection) == 1:
            raise InvalidEllipseArcException("Ellipse and line are tangent.")

    @property
    def ellipse(self) -> Ellipse:
        """Ellipse the arc belongs to"""
        return self._ellipse

    @property
    def line(self) -> Line:
        """Line intersecting the ellipse arc at two points"""
        return self._line

    @property
    def start_angle(self) -> float:
        """Angle in degrees at which the ellipse arc starts"""
        return self._start_angle

    @property
    def end_angle(self) -> float:
        """Angle in degrees at which the ellipse arc ends"""
        return self._end_angle

    def __add__(self, pt: Point2D) -> "EllipseArc":  # type: ignore
        """Adds a point to ellipse arc, which simply shifts it

        Args:
            pt: Point to add

        Returns:
            EllipseArc resulting from sum
        """
        return EllipseArc(ellipse=self.ellipse + pt, line=self.line + pt)

    def scale(self, pt: Point2D) -> "EllipseArc":
        """Provides the ellipse arc after applying a scaling of the 2D space with
        the scaling given in each coordinate of point

        It is simply the result of scaling the ellipse and line that define it

        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            EllipseArc resulting from scaling the 2D space
        """
        return EllipseArc(ellipse=self.ellipse.scale(pt=pt), line=self.line.scale(pt=pt))

    @classmethod
    def _angle_covered(cls, ellipse: Ellipse, line: Line) -> Tuple[float, float]:
        """Sets the start and end angles of the arc

        A rigid transform is applied to both the ellipse and the line in order
        to center the ellipse at the origin of coordinates and align its axes
        with the xy-axes. This implies applying the two operations sequentially
        in reverse order w.r.t. the way the ellipse is built: shift first, rotate
        then.

        Afterwards, the points of intersection are obtained, which allows to
        compute the start and end angle

        Args: None

        Returns: None
        """
        # We center both the ellipse and line applying the rigid transform
        line_rigid = line.rigid_transform(pt_shift=-ellipse.center, angle=-ellipse.angle)
        ellipse_rigid = Ellipse(center=Point2D(x=0, y=0), axes=ellipse.axes, angle=0)
        pts_intersection = ellipse_rigid.intersection_line(line=line_rigid)
        angles: List[float] = []
        for pt in pts_intersection:
            angles.append(ellipse_rigid.circle_angle_from_ellipse_point(pt=pt))

        # We assume the arc goes from the smallest to the biggest angle
        start_angle = min(angles)
        end_angle = max(angles)
        # We generate a point in the ellipse at the intermediate angle
        # which can be done operating in polar coordinates and then reverting
        # rigid transform
        mid_pt_rigid = ellipse_rigid.ellipse_point_from_circle_angle(gamma=((start_angle + end_angle) / 2))
        mid_pt = mid_pt_rigid.rotate(angle=ellipse.angle) + ellipse.center
        # if the half plane matches the desired one, we assigned angles right. Otherwise, we flip them
        if line.is_in_greater_half_plane(pt=mid_pt):
            # The angles are assigned correctly
            return start_angle, end_angle
        else:
            # Otherwise, we revert them. OpenCV reorders them, so we need to sum 360 to
            # the start angle
            return end_angle, start_angle + 360

    def __repr__(self):
        return f"EllipseArc(ellipse={str(self.ellipse)}, line={str(self.line)})"

    def keypoints(self, num_points: int = 100) -> List[Point2D]:
        """Generates a list of points in the ellipse

        Args:
            num_points: int number of points to generate in the ellipse

        Returns:
            List of points in the ellipse
        """
        return [
            self.ellipse.ellipse_point_from_circle_angle(gamma=gamma)
            for gamma in np.linspace(self.start_angle, self.end_angle, num=num_points, endpoint=True)
        ]

    def draw(self, img: np.ndarray, color: Tuple[Any, ...] = Color.RED, thickness: int = 3):
        """Draws the ellipse arc within the given image in-place

        Note:
            This method modifies the provided image in place

        Args:
            img: ndarray image to draw the ellipse in
            color: BGR int tuple indicating the color of the point
            thickness: int thickness of the drawn ellipse

        Returns: None
        """
        cv2.ellipse(
            img=img,
            center=(round(self.ellipse.center.x), round(self.ellipse.center.y)),
            axes=(round(self.ellipse.axes.x), round(self.ellipse.axes.y)),
            angle=self.ellipse.angle,
            startAngle=self.start_angle,
            endAngle=self.end_angle,
            color=color,
            thickness=thickness,
        )
