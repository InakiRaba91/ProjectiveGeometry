from __future__ import annotations

from typing import Any


class ImageSize:
    def __init__(self, width: int, height: int):
        """
        Image size class initializer

        Args:
            width: Distance image width
            height: Distance image height
        """
        self._width = width
        self._height = height

    def __hash__(self):
        return hash((self.width, self.height))

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def __eq__(self, other: Any) -> bool:
        return self.width == other.width and self.height == other.height


BASE_IMAGE_SIZE = ImageSize(width=1280, height=720)
