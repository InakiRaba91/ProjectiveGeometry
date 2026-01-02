from pathlib import Path

import cv2
import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Line, Point
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.projection.projectors import project_pitch_template
from projective_geometry.utils.distances import FOOT
from projective_geometry.entrypoints.utils import (
    PROJECT_LOCATION,
    IMG_CELTICS_FPATH,
    PT_THICKNESS,
)


def homography_from_line_correspondences_demo(
    output: Path = PROJECT_LOCATION / "results/celtics_with_projected_court_from_lines.png",
):
    # obtain camera from annotated points
    W2, H2 = BasketballCourtTemplate.PITCH_WIDTH / 2, BasketballCourtTemplate.PITCH_HEIGHT / 2
    lines_template = [
        Line.from_points(pt1=Point(x=-W2, y=-H2), pt2=Point(x=W2, y=-H2)),
        Line.from_points(pt1=Point(x=W2, y=H2), pt2=Point(x=-W2, y=H2)),
        Line.from_points(pt1=Point(x=-W2, y=H2), pt2=Point(x=-W2, y=-H2)),
        Line.from_points(pt1=Point(x=-W2, y=-8 * FOOT), pt2=Point(x=-W2 + 19 * FOOT, y=-8 * FOOT)),
        Line.from_points(pt1=Point(x=-W2 + 19 * FOOT, y=-8 * FOOT), pt2=Point(x=-W2 + 19 * FOOT, y=8 * FOOT)),
        Line.from_points(pt1=Point(x=-W2, y=8 * FOOT), pt2=Point(x=-W2 + 19 * FOOT, y=8 * FOOT)),
    ]
    lines_frame = [
        Line(a=193.86019201178604, b=-2080.7791860169154, c=438295.7707991526),
        Line(a=-186.00789437021066, b=1430.181983413831, c=-1224782.6368186155),
        Line(a=-319.57933618979035, b=-393.6058971593335, c=383963.87263253756),
        Line(a=38.64487027084424, b=-375.8717723989944, c=142641.7531879748),
        Line(a=102.77328087499652, b=83.87270632687233, c=-173739.07908707517),
        Line(a=38.13697697658064, b=-333.79059163488, c=193512.39303263795),
    ]
    camera = Camera.from_line_correspondences(lines_source=lines_template, lines_target=lines_frame)
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
    for line in lines_frame:
        line.draw(img=frame_with_projected_court, color=Color.RED, thickness=PT_THICKNESS)

    # draw basketball court for reference
    basketball_court_img = basketball_court.draw(image_size=image_size, color=Color.WHITE)
    for line in lines_template:
        # map the the point from pitch coordinates to pitch image (no projection involved, just shift and scaling)
        line_img = basketball_court.pitch_template_to_pitch_image(geometric_feature=line, image_size=image_size)
        line_img.draw(img=basketball_court_img, color=Color.RED, thickness=PT_THICKNESS)

    cv2.imwrite(output.as_posix(), np.concatenate((frame_with_projected_court, basketball_court_img), axis=1))
