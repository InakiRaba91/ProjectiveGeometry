from pathlib import Path

import cv2
import numpy as np

from projective_geometry.draw import Color
from projective_geometry.entrypoints.utils import (
    PROJECT_LOCATION,
    get_2d_homography_between_planes,
    get_intrinsic_from_2d_homographies,
    project_3d_points,
)


def intrinsic_from_three_planes_demo(image_path: Path = PROJECT_LOCATION / "results/SoccerPitchCalibration.png"):
    # ground truth
    f, tx, ty, tz, rx, ry, rz = 4763, -21, -110, 40, 250, 2, 13
    image = cv2.imread(image_path.as_posix())
    image_width, image_height = image.shape[1], image.shape[0]
    pitch_width = 120
    K = np.array([[f, 0, image_width // 2], [0, f, image_height // 2], [0, 0, 1]])
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(rx * np.pi / 180), -np.sin(rx * np.pi / 180)],
            [0, np.sin(rx * np.pi / 180), np.cos(rx * np.pi / 180)],
        ]
    )
    Ry = np.array(
        [
            [np.cos(ry * np.pi / 180), 0, np.sin(ry * np.pi / 180)],
            [0, 1, 0],
            [-np.sin(ry * np.pi / 180), 0, np.cos(ry * np.pi / 180)],
        ]
    )
    Rz = np.array(
        [
            [np.cos(rz * np.pi / 180), -np.sin(rz * np.pi / 180), 0],
            [np.sin(rz * np.pi / 180), np.cos(rz * np.pi / 180), 0],
            [0, 0, 1],
        ]
    )
    R = Rz.dot(Ry).dot(Rx)
    T = np.array([[tx], [ty], [tz]])
    E = np.concatenate((R.T, -R.T.dot(T)), axis=1)
    H = K.dot(E)

    # relevant keypoints
    goal_width = 8
    goal_height = 8 / 3

    # goal points
    bottom_left_world = np.array([[-pitch_width / 2, -goal_width / 2, 0, 1]]).T
    top_left_world = np.array([[-pitch_width / 2, -goal_width / 2, goal_height, 1]]).T
    bottom_right_world = np.array([[-pitch_width / 2, goal_width / 2, 0, 1]]).T
    top_right_world = np.array([[-pitch_width / 2, goal_width / 2, goal_height, 1]]).T
    pts_goal = np.concatenate((bottom_left_world, top_left_world, bottom_right_world, top_right_world), axis=1)
    bottom_left_image, top_left_image, bottom_right_image, top_right_image = project_3d_points(H=H, pts=pts_goal).T

    # box points
    box_height, box_width = 10, 6
    bottom_left_box_world = np.array([[-pitch_width / 2, -box_height, 0, 1]]).T
    top_left_box_world = np.array([[-pitch_width / 2, box_height, 0, 1]]).T
    bottom_right_box_world = np.array([[-pitch_width / 2 + box_width, -box_height, 0, 1]]).T
    top_right_box_world = np.array([[-pitch_width / 2 + box_width, box_height, 0, 1]]).T
    pts_box = np.concatenate((bottom_left_box_world, top_left_box_world, bottom_right_box_world, top_right_box_world), axis=1)
    bottom_left_box_image, top_left_box_image, bottom_right_box_image, top_right_box_image = project_3d_points(
        H=H, pts=pts_box
    ).T

    # group keypoints for three identified planes
    plane_points_world = [
        # goal plane
        [bottom_left_world, top_left_world, bottom_right_world, top_right_world],
        # ground plane
        [bottom_left_box_world, top_left_box_world, bottom_right_box_world, top_right_box_world],
        # inclined plane
        [top_left_world, top_right_world, bottom_right_box_world, top_right_box_world],
    ]
    plane_points_image = [
        # goal plane
        [bottom_left_image, top_left_image, bottom_right_image, top_right_image],
        # ground plane
        [bottom_left_box_image, top_left_box_image, bottom_right_box_image, top_right_box_image],
        # inclined plane
        [top_left_image, top_right_image, bottom_right_box_image, top_right_box_image],
    ]

    # find homography for each plane
    H_planes = []
    colors = [Color.BLUE, Color.ORANGE, Color.RED]
    images = []
    for pts_world, pts_image, color in zip(plane_points_world, plane_points_image, colors):
        image_plane = image.copy()
        for pt in pts_image:
            cv2.circle(image_plane, (int(pt[0]), int(pt[1])), 15, color, -1)
        images.append(image_plane)
        H_planes.append(get_2d_homography_between_planes(H=H, pts_world=pts_world))
    cv2.imwrite((PROJECT_LOCATION / "results/SoccerPitchCalibrationPlanes.png").as_posix(), np.concatenate(images, axis=1))

    # build system of equations with constraints for image of the absolute conic
    K_estimate = get_intrinsic_from_2d_homographies(H_planes=H_planes)
    print(f"Ground truth intrinsic matrix: \n{K}")
    print(f"Estimated intrinsic matrix: \n{K_estimate}")
