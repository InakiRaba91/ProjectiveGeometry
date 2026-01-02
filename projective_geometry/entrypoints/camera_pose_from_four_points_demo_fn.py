from pathlib import Path

import cv2
import numpy as np

from projective_geometry.entrypoints.utils import PROJECT_LOCATION
from projective_geometry.geometry import Point
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.solvers import p4p


def camera_pose_from_four_points_demo(image_path: Path = PROJECT_LOCATION / "results/BasketballCourtCalibration.png"):
    focal_length = 350
    image = cv2.imread(image_path.as_posix())
    image_width, image_height = image.shape[1], image.shape[0]
    w2, h2 = image_width / 2, image_height / 2
    W2, H2 = BasketballCourtTemplate.PITCH_WIDTH / 2, BasketballCourtTemplate.PITCH_HEIGHT / 2
    pts_world = np.array([[-W2, -H2, 0], [-W2, H2, 0], [W2, -H2, 0], [W2, H2, 0]]).T
    pts_img = [
        Point(x=877.11741024, y=83.60096188),
        Point(x=1021.48619459, y=462.69981494),
        Point(x=296.31302725, y=252.83757669),
        Point(x=287.22444872, y=567.63319843),
    ]
    pts_img_cam = np.array([[pt.x - w2, pt.y - h2, focal_length] for pt in pts_img]).T
    camera_pose = p4p(pts_world, pts_img_cam)
    print(camera_pose)
