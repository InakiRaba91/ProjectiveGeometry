from pathlib import Path

import cv2
import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Point
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.projection.projectors import project_pitch_template
from projective_geometry.utils.distances import FOOT, INCH
from projective_geometry.entrypoints.utils import (
    PROJECT_LOCATION,
    IMG_CELTICS_FPATH,
    PT_RADIUS,
    PT_THICKNESS,
)


def homography_from_point_correspondences_demo(
    output: Path = PROJECT_LOCATION / "results/celtics_with_projected_court_from_points.png",
):
    # obtain camera from annotated points
    W2, H2 = BasketballCourtTemplate.PITCH_WIDTH / 2, BasketballCourtTemplate.PITCH_HEIGHT / 2
    points_frame = [
        Point(x=845, y=290),
        Point(x=126, y=872),
        Point(x=1692, y=367),
        Point(x=1115, y=707),
        Point(x=1560, y=644),
    ]
    points_template = [
        Point(x=-W2, y=-H2),
        Point(x=-W2, y=H2),
        Point(x=-W2 + 28 * FOOT, y=-H2),
        Point(x=-W2 + 19 * FOOT, y=8 * FOOT),
        Point(x=-W2 + 28 * FOOT + 12 * INCH, y=0),
    ]
    camera = Camera.from_point_correspondences(pts_source=points_template, pts_target=points_frame)

    # project basketball court template
    basketball_court = BasketballCourtTemplate()
    frame = cv2.imread(IMG_CELTICS_FPATH.as_posix())
    image_size = ImageSize(width=frame.shape[1], height=frame.shape[0])
    frame_with_projected_court = project_pitch_template(
        pitch_template=basketball_court,
        camera=camera,
        image_size=image_size,
        frame=frame,
        thickness=12,
        color=Color.BLUE,
    )
    for pt in points_frame:
        pt.draw(img=frame_with_projected_court, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)

    # draw basketball court for reference
    basketball_court_img = basketball_court.draw(image_size=image_size, color=Color.WHITE)
    for pt in points_template:
        # map the the point from pitch coordinates to pitch image (no projection involved, just shift and scaling)
        pt_img = basketball_court.pitch_template_to_pitch_image(geometric_feature=pt, image_size=image_size)
        pt_img.draw(img=basketball_court_img, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)  # type: ignore

    cv2.imwrite(output.as_posix(), np.concatenate((frame_with_projected_court, basketball_court_img), axis=1))
