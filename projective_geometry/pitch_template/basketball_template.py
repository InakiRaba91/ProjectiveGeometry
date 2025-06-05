from typing import Any, Tuple, Union

import numpy as np

from projective_geometry.geometry import Ellipse, EllipseArc, Line, LineSegment, Point2D

from ..utils.distances import FOOT, INCH
from .pitch_dims import PitchDims
from .pitch_template import PitchTemplate


class BasketballCourtTemplate(PitchTemplate):
    """2D basketball court template

    Args:
        pitch_dims: PitchDims of the template

    Attributes:
        pitch_dims: PitchDims of the template
        line_segments: Tuple of LineSegments with all pitch markings corresponding to lines
        curved_lines: Tuple of Ellipse and EllipseArcs with all pitch markings corresponding to curved lines
        keypoints: Tuple of Points with keypoints like penalty spots and the center of the pitch
    """

    PITCH_WIDTH = 94 * FOOT
    PITCH_HEIGHT = 50 * FOOT

    def __init__(self):
        pitch_dims = PitchDims(width=self.PITCH_WIDTH, height=self.PITCH_HEIGHT)
        super(BasketballCourtTemplate, self).__init__(pitch_dims=pitch_dims)

    def _geometric_features(self) -> Tuple[Any, ...]:
        """Compute all geometric features (line segments, ellipses, ellipse arcs and keypoints)
        in the soccer template

        Args: None

        Returns:
            Tuple of geometric features (Point, LineSegment, Ellipse, EllipseArc)
        """
        self.line_segments = self._line_segments()
        self.curved_lines = self._curved_lines()
        return self.line_segments + self.curved_lines

    def _keypoints(self):
        keypoints = []
        for geometric_feature in self.geometric_features:
            keypoints += geometric_feature.keypoints()
        return keypoints

    def _line_segments(self) -> Tuple[LineSegment]:
        """Compute all line segments corresponding to pitch markings and sets them as attribute

        Args: None

        Returns: None
        """
        H2 = self.pitch_dims.height / 2
        sidelines = self._sidelines()
        left_side_lines = self._left_side()
        right_side_lines = tuple([-line for line in left_side_lines])
        halfway_line = (LineSegment(pt1=Point2D(x=0, y=-H2), pt2=Point2D(x=0, y=H2)),)
        return sidelines + left_side_lines + right_side_lines + halfway_line

    def _curved_lines(self) -> Tuple[Union[Ellipse, EllipseArc], ...]:
        """Compute all curved lines (ellipses and ellipse arcs) corresponding
        to pitch markings

        Returns: tuple with Ellipses and EllipseArcs
        """
        W2 = self.pitch_dims.width / 2
        central_circles = (
            Ellipse(center=Point2D(x=0, y=0), axes=Point2D(x=6 * FOOT, y=6 * FOOT), angle=0),
            Ellipse(center=Point2D(x=0, y=0), axes=Point2D(x=2 * FOOT, y=2 * FOOT), angle=0),
        )
        free_throw_circles = (
            Ellipse(center=Point2D(x=-W2 + 19 * FOOT, y=0), axes=Point2D(x=6 * FOOT, y=6 * FOOT), angle=0),
            Ellipse(center=Point2D(x=W2 - 19 * FOOT, y=0), axes=Point2D(x=6 * FOOT, y=6 * FOOT), angle=0),
        )
        three_point_arcs = (
            EllipseArc(
                ellipse=Ellipse(
                    center=Point2D(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
                    axes=Point2D(x=23 * FOOT + 9 * INCH, y=23 * FOOT + 9 * INCH),
                    angle=0,
                ),
                line=Line(a=1, b=0, c=W2 - 14 * FOOT),
            ),
            EllipseArc(
                ellipse=Ellipse(
                    center=Point2D(x=W2 - 5 * FOOT - 3 * INCH, y=0),
                    axes=Point2D(x=23 * FOOT + 9 * INCH, y=23 * FOOT + 9 * INCH),
                    angle=0,
                ),
                line=Line(a=-1, b=0, c=W2 - 14 * FOOT),
            ),
        )
        restricted_arcs = (
            EllipseArc(
                ellipse=Ellipse(
                    center=Point2D(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
                    axes=Point2D(x=4 * FOOT, y=4 * FOOT),
                    angle=0,
                ),
                line=Line(a=1, b=0, c=W2 - 5 * FOOT - 3 * INCH),
            ),
            EllipseArc(
                ellipse=Ellipse(
                    center=Point2D(x=W2 - 5 * FOOT - 3 * INCH, y=0),
                    axes=Point2D(x=4 * FOOT, y=4 * FOOT),
                    angle=0,
                ),
                line=Line(a=-1, b=0, c=W2 - 5 * FOOT - 3 * INCH),
            ),
        )
        return central_circles + free_throw_circles + three_point_arcs + restricted_arcs

    def _sidelines(self):
        """Compute all sidelines

        Args: None

        Returns:
            Tuple of LineSegments corresponding to sidelines
        """
        W2, H2 = self.pitch_dims.width / 2, self.pitch_dims.height / 2
        return tuple(
            [
                LineSegment(pt1=Point2D(x=-W2, y=-H2), pt2=Point2D(x=W2, y=-H2)),
                LineSegment(pt1=Point2D(x=W2, y=-H2), pt2=Point2D(x=W2, y=H2)),
                LineSegment(pt1=Point2D(x=W2, y=H2), pt2=Point2D(x=-W2, y=H2)),
                LineSegment(pt1=Point2D(x=-W2, y=H2), pt2=Point2D(x=-W2, y=-H2)),
            ]
        )

    def _left_side(self):
        """Compute all line segments from left box

        Args: None

        Returns:
            Tuple of LineSegments corresponding to left box
        """
        W2, H2 = self.pitch_dims.width / 2, self.pitch_dims.height / 2
        paint = (
            # outter
            LineSegment(pt1=Point2D(x=-W2, y=-8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=-8 * FOOT)),
            LineSegment(pt1=Point2D(x=-W2 + 19 * FOOT, y=-8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT)),
            LineSegment(pt1=Point2D(x=-W2, y=8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT)),
            # inner
            LineSegment(pt1=Point2D(x=-W2, y=-6 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=-6 * FOOT)),
            LineSegment(pt1=Point2D(x=-W2, y=6 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=6 * FOOT)),
        )
        three_points = (
            LineSegment(pt1=Point2D(x=-W2, y=-22 * FOOT), pt2=Point2D(x=-W2 + 14 * FOOT, y=-22 * FOOT)),
            LineSegment(pt1=Point2D(x=-W2, y=22 * FOOT), pt2=Point2D(x=-W2 + 14 * FOOT, y=22 * FOOT)),
        )
        free_throw_marks = tuple(
            [
                LineSegment(pt1=Point2D(x=-W2 + x * FOOT, y=y * FOOT), pt2=Point2D(x=-W2 + x * FOOT, y=(y + np.sign(y)) * FOOT))
                for x in (7, 8, 11, 14)
                for y in (-8, 8)
            ]
        )
        coaching_marks = (
            LineSegment(pt1=Point2D(x=-W2 + 28 * FOOT, y=-H2), pt2=Point2D(x=-W2 + 28 * FOOT, y=-H2 + 3 * FOOT)),
            LineSegment(pt1=Point2D(x=-W2 + 28 * FOOT, y=H2), pt2=Point2D(x=-W2 + 28 * FOOT, y=H2 - 3 * FOOT)),
        )
        return paint + three_points + free_throw_marks + coaching_marks
