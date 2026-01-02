from pathlib import Path

import cv2

from projective_geometry.geometry import Point
from projective_geometry.entrypoints.utils import PROJECT_LOCATION


def focal_length_from_orthogonal_vanishing_points_demo(image: Path = PROJECT_LOCATION / "results/BasketballCourtCalibration.png"):
    image = cv2.imread(image.as_posix())
    width, height = image.shape[1], image.shape[0]
    vp1 = Point(x=-1815.15874324, y=868.08488743)
    vp2 = Point(x=341.78456657, y=-1322.13274968)
    focal_length = (-(vp1.x - width / 2) * (vp2.x - width / 2) - (vp1.y - height / 2) * (vp2.y - height / 2)) ** 0.5
    print(f"Focal length: {focal_length}")
    pass
