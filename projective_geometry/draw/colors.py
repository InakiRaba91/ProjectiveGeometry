import inspect
from typing import Any, Tuple


class Color(object):
    """Class with BGR values for different colors"""

    RED: Tuple = (0, 0, 255)
    GREEN: Tuple = (0, 255, 0)
    BLUE: Tuple = (255, 0, 0)
    PURPLE: Tuple = (255, 0, 255)
    YELLOW: Tuple = (0, 255, 255)
    ORANGE: Tuple = (0, 128, 255)
    WHITE: Tuple = (255, 255, 255)
    BLACK: Tuple = (0, 0, 0)
    CYAN: Tuple = (249, 235, 180)
    LIGHT_GREEN: Tuple = (187, 249, 180)

    @classmethod
    def get_colors(cls) -> Tuple[Tuple[Any, ...]]:
        """Returns list with all available colors"""
        attributes = inspect.getmembers(Color, lambda a: not (inspect.isroutine(a)))
        color_attributes = [a for a in attributes if not (a[0].startswith("__") and a[0].endswith("__"))]
        return tuple([rgb for (name, rgb) in color_attributes])  # type: ignore
