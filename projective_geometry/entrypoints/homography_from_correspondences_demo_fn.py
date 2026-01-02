from pathlib import Path

import cv2
import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Ellipse, Line, Point
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


def homography_from_correspondences_demo(
    output: Path = PROJECT_LOCATION / "results/celtics_with_projected_court_from_multiple_features.png",
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
    ellipses_frame = [
        # free-throw circle
        Ellipse(
            center=Point(x=1198.1113211177799, y=606.6167316662579),
            axes=Point(x=77.11168956853989, y=222.0629912180788),
            angle=90.70347611420907,
        ),
        # three-point arc
        Ellipse(
            center=Point(x=707.2445757522346, y=600.6708571711737),
            axes=Point(x=296.0026379644702, y=871.9410745372605),
            angle=88.64701264444109,
        ),
        # restricted arc
        Ellipse(
            center=Point(x=720.7694262254354, y=553.058980020591),
            axes=Point(x=48.53764408421055, y=144.87141097409037),
            angle=88.70747899257509,
        ),
        # corner arcs
        Ellipse(
            center=Point(x=810.80940320813, y=325.6083962774958),
            axes=Point(x=127.8838414420805, y=438.8219609157189),
            angle=89.32970121679617,
        ),
        Ellipse(
            center=Point(x=172.0013839803821, y=851.1080576625786),
            axes=Point(x=227.28737736672736, y=601.9016883331738),
            angle=85.72932402628315,
        ),
    ]
    ellipses_template = [
        # free-throw circle
        Ellipse(
            center=Point(x=-W2 + 19 * FOOT, y=0),
            axes=Point(x=6 * FOOT, y=6 * FOOT),
            angle=0,
        ),
        # three-point arc
        Ellipse(
            center=Point(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
            axes=Point(x=23 * FOOT + 9 * INCH, y=23 * FOOT + 9 * INCH),
            angle=0,
        ),
        # restricted arc
        Ellipse(
            center=Point(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
            axes=Point(x=4 * FOOT, y=4 * FOOT),
            angle=0,
        ),
        # corner arcs
        Ellipse(
            center=Point(x=-W2, y=-H2 + 3 * FOOT),
            axes=Point(x=14 * FOOT, y=14 * FOOT),
            angle=0,
        ),
        Ellipse(
            center=Point(x=-W2, y=H2 - 3 * FOOT),
            axes=Point(x=14 * FOOT, y=14 * FOOT),
            angle=0,
        ),
    ]

    camera = Camera.from_correspondences(
        pts_source=points_template,
        pts_target=points_frame,
        lines_source=lines_template,
        lines_target=lines_frame,
        ellipses_source=ellipses_template,
        ellipses_target=ellipses_frame,
    )

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
        line.draw(img=frame_with_projected_court, color=Color.GREEN, thickness=PT_THICKNESS)
    for ellipse in ellipses_frame:
        ellipse.draw(img=frame_with_projected_court, color=Color.YELLOW, thickness=PT_THICKNESS)
    for pt in points_frame:
        pt.draw(img=frame_with_projected_court, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)

    # draw basketball court for reference
    basketball_court_img = basketball_court.draw(image_size=image_size, color=Color.WHITE)
    for line in lines_template:
        line_img = basketball_court.pitch_template_to_pitch_image(geometric_feature=line, image_size=image_size)
        line_img.draw(img=basketball_court_img, color=Color.GREEN, thickness=PT_THICKNESS)
    for ellipse in ellipses_template:
        ellipse_img = basketball_court.pitch_template_to_pitch_image(geometric_feature=ellipse, image_size=image_size)
        ellipse_img.draw(img=basketball_court_img, color=Color.YELLOW, thickness=PT_THICKNESS)
    for pt in points_template:
        pt_img = basketball_court.pitch_template_to_pitch_image(geometric_feature=pt, image_size=image_size)
        pt_img.draw(img=basketball_court_img, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)  # type: ignore
    cv2.imwrite(output.as_posix(), np.concatenate((frame_with_projected_court, basketball_court_img), axis=1))
