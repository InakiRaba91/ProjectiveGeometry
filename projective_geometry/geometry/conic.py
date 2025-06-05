from typing import Any, Optional, Tuple

import numpy as np

from ..draw import Color
from .point import Point2D


def check_symmetric_and_non_degenerate(mat: np.ndarray, tol: float, ndim: Optional[int] = None) -> bool:
    """
    Helper method to check whether a matrix is:
    - symmetric i.e. elements above and below the diagonal are equal
    - non degenerate i.e. the matrix determinant is non zero

    Args:
        mat (np.ndarray): Matrix to check.
        tol (float): Tolerance within which to check zero equivalence.
        ndim (Optional[int]): Dimensionality to check for in square matrix. Defaults to None.

    Returns:
        bool: Result if matrix is non degenerate and symmetric.
    """
    is_square = mat.shape[0] == mat.shape[1]

    if ndim is not None:
        is_square = is_square and (mat.shape[0] == ndim)

    if not is_square:
        return False

    symmetric_matrix = np.isclose(mat, mat.T).all()
    non_degenerate_matrix = np.abs(np.linalg.det(mat)) > tol

    return is_square and symmetric_matrix and non_degenerate_matrix  # type: ignore


class Conic:
    def __init__(self, M: np.ndarray):
        """
        Initializes the conic with its matrix representation

        Args:
            M: matrix representation of the conic
        """
        self._M = M

    @property
    def M(self) -> np.ndarray:
        """Returns the matrix representation"""
        return self._M

    def __add__(self, pt: Point2D) -> "Conic":  # type: ignore
        """Adds a point to conic

        Adding a point is decribed in homogeneous coordinates by x'=Tx

        Therefore M'=(T')^-T·M·(T')^-1

        Args:
            pt: Point to add

        Returns:
            Line resulting from the sum
        """
        Tinv = np.eye(3)
        Tinv[0, 2], Tinv[1, 2] = -pt.x, -pt.y
        M_shifted = Tinv.T.dot(self.M).dot(Tinv)
        return Conic(M=M_shifted)

    def scale(self, pt: Point2D) -> "Conic":
        """Provides the conic after applying a scaling of the 2D space with
        the scaling given in each coordinate of point

        If a transform can be characterized as an homography, a conic defined by
        its matrix representation M will undergo the following transform
        M' = H^-T * M * H^-1

        A scaling transform can be characterized as
            | pt.x   0    0 |
        H = |  0   pt.y   0 |
            |  0     0    1 |


        Args:
            pt: Point defining the scaling of each axis in the 2D space

        Returns:
            Ellipse resulting from scaling the 2D space
        """
        Hinv = np.array([[1 / pt.x, 0, 0], [0, 1 / pt.y, 0], [0, 0, 1]])  # symmetric
        M_scaled = Hinv.T.dot(self.M).dot(Hinv)
        return Conic(M=M_scaled)

    def __repr__(self):
        return f"Conic(M={self.M})"

    def draw(self, img: np.ndarray, color: Tuple[Any, ...] = Color.RED):
        """Draws the filled conic within the given image in-place

        Note:
            This method modifies the provided image in place

        Args:
            img: ndarray image to draw the conic in
            color: BGR int tuple indicating the color of the conic
        """
        ny, nx = img.shape[:2]
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xx, yy = np.meshgrid(x, y)
        xx = xx.reshape(nx * ny, 1)
        yy = yy.reshape(nx * ny, 1)
        locs = np.concatenate((xx, yy, np.ones_like(xx)), axis=1)
        q = np.einsum("ij,ij->i", locs, (self.M @ locs.T).T).reshape(ny, nx)
        img_conic = q <= 0
        img[img_conic] = color
