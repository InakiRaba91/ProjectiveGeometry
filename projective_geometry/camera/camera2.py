from __future__ import annotations

from typing import Self, Sequence, Tuple

import numpy as np

from projective_geometry.camera.camera_distortion import CameraDistortion
from projective_geometry.camera.camera_params import CameraParams2
from projective_geometry.camera.camera_pose import CameraPose
from projective_geometry.camera.geometry import (
    calculate_camera_pose_and_distortion,
    calculate_focal_length_from_homography,
    calculate_homography,
    convert_intrinsics_to_calibration_matrix,
    rotation_matrix_to_roll_tilt_pan,
)
from projective_geometry.geometry.point import Point2D, Point3D


class Camera2:
    """Simple class to represent a camera"""

    def __init__(self, sensor_wh: tuple[int, int], camera_params: CameraParams2) -> None:
        self.sensor_wh = np.asarray(sensor_wh)
        """Height and width of the camera sensor in pixels"""
        self.camera_params = camera_params
        """Camera parameters including camera pose, distortion coefficients and focal length"""

    @classmethod
    def from_keypoint_correspondences(
        cls,
        world_points: Sequence[Point3D],
        image_points: Sequence[Point2D],
        sensor_wh: Tuple[int, int],
    ) -> Self:
        """Create a Camera object from keypoint correspondences and sensor size.

        This method automatically estimates the focal length from the homography between
        the world plane of the pitch and the image sensor. The principal point is set to
        the center of the image sensor.

        Parameters
        ----------
        world_points
            The world points.
        image_points
            The image points.
        sensor_wh
            The sensor height and width of the camera, i.e., the camera's resolution.

        Returns
        -------
        camera
            The Camera object.
        """
        world_points_arr = np.array([pt.to_array() for pt in world_points])
        image_points_arr = np.array([pt.to_array() for pt in image_points])

        world_points_arr = world_points_arr.astype(np.float32)
        image_points_arr = image_points_arr.astype(np.float32)

        homography_matrix = calculate_homography(world_points_arr, image_points_arr, alpha=0)
        success, focal_length_xy = calculate_focal_length_from_homography(homography_matrix, sensor_wh)
        if not success:
            msg = "Failed to estimate focal length automatically."
            raise ValueError(msg)

        calibration_matrix = convert_intrinsics_to_calibration_matrix(np.asarray(sensor_wh), focal_length_xy)

        rotation_matrix, position_xyz, distortion_coefficients = calculate_camera_pose_and_distortion(
            world_points_arr,
            image_points_arr,
            calibration_matrix,
        )

        roll, tilt, pan = rotation_matrix_to_roll_tilt_pan(rotation_matrix)

        return cls(
            sensor_wh=sensor_wh,
            camera_params=CameraParams2(
                camera_pose=CameraPose(
                    tx=position_xyz[0],
                    ty=position_xyz[1],
                    tz=position_xyz[2],
                    roll=roll,
                    tilt=tilt,
                    pan=pan,
                ),
                camera_distortion=CameraDistortion(
                    k1=distortion_coefficients[0],
                    k2=distortion_coefficients[1],
                    p1=distortion_coefficients[2],
                    p2=distortion_coefficients[3],
                    k3=distortion_coefficients[4],
                ),
                focal_length_xy=focal_length_xy,
            ),
        )

    def __repr__(self):
        return f"Camera(sensor_wh={self.sensor_wh}, camera_params={str(self.camera_params)})"
