class InvalidLineException(Exception):
    """Exception raised when a line is created with a=b=0."""


class LineFromEqualPointsException(Exception):
    """Exception raised you try to create a line from two equal points."""


class InvalidEllipseArcException(Exception):
    """Exception raised when an ellipse arc is created from an ellipse and line that don't intersect or ar tangent"""


class PointNotInEllipseException(Exception):
    """Exception raised when operating with a point that is not in the ellipse"""


class InvalidConicMatrixEllipseException(Exception):
    """Exception raised when operating with an invalid matrix representation of an ellipse"""
