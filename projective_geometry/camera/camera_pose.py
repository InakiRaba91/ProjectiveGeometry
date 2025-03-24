from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


class CameraPose:
    """Class with camera pose parameters including 3D location/orientation of the camera.

    Attributes:
        tx: x-location of camera in 3D world
        ty: y-location of camera in 3D world
        tz: z-location of camera in 3D world
        rx: rotation angle around y-axis
        ry: rotation angle around x-axis
        rz: rotation angle around z-axis
    """

    def __init__(self, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float):
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.rx = rx
        self.ry = ry
        self.rz = rz

    def to_array(self) -> np.ndarray:
        """Converts to numpy array
        Returns:
            ndarray  [tx, ty, tz, rx, ry, rz]
        """
        return np.array([self.tx, self.ty, self.tz, self.rx, self.ry, self.rz])

    def to_Rt(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts to rotation matrix and translation vector
        Returns:
            tuple of rotation matrix and translation vector
        """
        rot_angles = [self.rx, self.ry, self.rz]
        Rc = Rotation.from_euler("xyz", rot_angles, degrees=True).as_matrix()
        t = np.array([[self.tx], [self.ty], [self.tz]])
        return Rc, t
    
    @classmethod
    def from_Rt(cls, R: np.ndarray, t: np.ndarray, tol: float = 1e-6) -> CameraPose:
        """Creates a CameraPose object from rotation matrix and translation vector

        The camera can be shifted to a 3D location [tx, ty, tz], and the image can undergo a 3D rotation.
        The extrinsic matrix E = [R | T] undoes the rotation and shifts everything to the origin of
        coordinates before projecting the 3D world into the image.

        If we have a point p=[x,y,z] in the 3d world, we can transform it to homogeneous coordinates
        ph = [x, y, z, 1]. Before being projected with the intrinsic matrix, it undergoes the transform
        p'= E * ph = (R * p) + T = R * (p - t) = Rc^-1 * (p - t)

        Therefore
        t = -R*T
        Rc = R^-1 = R^T (for orthogonal matrices M^-1=M^T)

        Scipy has a method to retrieve euler angles from rotation matrix. However, there's an ambiguity
        if one of the angles is 90º, known as the Gimbal lock. Scipy has its own mechanism to deal with it,
        but we'll impose our constraint in that case.

        The rotation matrix is given by the equation in
        https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
        R=Rz(z)Ry(y)Rx(x) =
        | cos(z)cos(y)    cos(z)sin(y)sin(x)-sin(z)cos(x)    cos(z)sin(y)cos(x)+sin(z)sin(x) |
        | cos(z)cos(y)    sin(z)sin(y)sin(x)+cos(z)cos(x)    sin(z)sin(y)cos(x)-cos(z)sin(x) |
        |   -sin(y)              cos(y)sin(x)                         cos(y)cos(x)           |

        We can see the ambiguity from the equation if R[2, 0] = +-1 --> y=+-90, because in that case
        R[0, 0] = R[1, 0] = R[2, 1] = R[2, 2] = 0
        and there are infinite solutions to satisfy the system given by R[[0,1], [1,2]] (R[2, 0]=-sin(y))
        R[0, 1] = -R[2, 0]cos(z)sin(x)-sin(z)cos(x)
        R[0, 2] = -R[2, 0]cos(z)cos(x)+sin(z)sin(x)
        R[1, 1] = -R[2, 0]sin(z)sin(x)+cos(z)cos(x)
        R[1, 2] = -R[2, 0]sin(z)cos(x)-cos(z)sin(x)
        where we can see the first and third eqs are equivalent, and so are the second and fourth.
        The system
        R[0, 1] = -R[2, 0]cos(z)sin(x)-sin(z)cos(x)
        R[0, 2] = -R[2, 0]cos(z)cos(x)+sin(z)sin(x)
        has infinite solutions cus we can simply set x and solve for y.

        Given how broadcast works, it seems reasonable to assume roll is negligible, so we will force
        x=0 and then solve for
        R[0, 1] = -sin(z)
        R[0, 2] = -R[2, 0]cos(z)
        which leads to
        R[0, 1] / R[0, 2] = tg(z) / R[2, 0]  --> z = arctan2( R[0, 1]*R[2, 0], R[0, 2])

        Note:
            The three rotations are given in a global frame of reference (extrinsic), and the order is x->y->z
            Gimbal lock explained: https://www.youtube.com/watch?v=zc8b2Jo7mno

        Args:
            R: rotation matrix
            t: translation vector
            tol: float error tolerance for detecting Gimbal lock

        Returns:
            Camera from given params
        """
        # For Gimbal lock, we force roll=0
        # https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
        if np.abs(np.abs(R[2, 0]) - 1) < tol:
            ry = -np.sign(R[2, 0]) * 90
            rx = 0
            rz = np.rad2deg(np.arctan2(R[0, 1] * R[2, 0], R[0, 2]))
        else:
            rx, ry, rz = Rotation.from_matrix(R).as_euler("xyz", degrees=True)
        tx, ty, tz = t.squeeze()
        return cls(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz)

    def __eq__(self, other: Any, tol: float = 1e-6):
        """Performs the equality comparison between current object and passed one.
        Args:
            other: object to compare against
            tol: float error tolerance for considering two cameras equal
        Returns:
            boolean indicating if two objects are equal
        """
        if isinstance(self, other.__class__):
            return max(np.abs(self.to_array() - other.to_array())) < tol
        return False

    def __repr__(self):
        return f"CameraPose(tx={self.tx}, ty={self.ty}, tz={self.tz}, rx={self.rx}, ry={self.ry}, rz={self.rz})"
