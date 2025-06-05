from __future__ import annotations

from projective_geometry.geometry import Point2D


class PitchDims:
    """
    Model to capture the dimensions of a pitch.
    """

    def __init__(self, width: float, height: float):
        """
        Initializes the pitch dimensions

        Args:
            width: width of the pitch
            height: height of the pitch
        """

        self._width = width
        self._height = height

    def __hash__(self) -> int:
        return hash((self.width, self.height))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PitchDims) and other.width == self.width and other.height == self.height

    @property
    def width(self) -> float:
        """Returns the width of the pitch"""
        return self._width

    @property
    def height(self) -> float:
        """Returns the height of the pitch"""
        return self._height

    def on_pitch(self, pt: Point2D) -> bool:
        """Checks if a point is within the pitch

        The pitch is defined within
            x in [-width/2, width/2]
            y in [-height/2, height/2]

        Args:
            pt: Point to check whether it lies within the pitch

        Returns:
            boolean indicating whether it lies within the pitch
        """
        return (-self.width / 2.0 <= pt.x <= self.width / 2.0) and (-self.height / 2.0 <= pt.y <= self.height / 2.0)

    def __repr__(self):
        return f"PitchDims(width={self.width}, height={self.height})"
