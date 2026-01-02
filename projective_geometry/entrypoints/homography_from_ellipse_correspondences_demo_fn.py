from pathlib import Path

import cv2
import numpy as np

from projective_geometry.camera import Camera
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.entrypoints.utils import (
    IMG_CELTICS_FPATH,
    PROJECT_LOCATION,
    PT_THICKNESS,
)
from projective_geometry.geometry import Ellipse, Point
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.projection.projectors import project_pitch_template
from projective_geometry.utils.distances import FOOT, INCH


def homography_from_ellipse_correspondences_demo(
    output: Path = PROJECT_LOCATION / "results/celtics_with_projected_court_from_ellipses.png",
):
    W2, H2 = BasketballCourtTemplate.PITCH_WIDTH / 2, BasketballCourtTemplate.PITCH_HEIGHT / 2
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

    camera = Camera.from_multiple_ellipse_correspondences(ellipses_source=ellipses_template, ellipses_target=ellipses_frame)

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
    for ellipse in ellipses_frame:
        ellipse.draw(img=frame_with_projected_court, color=Color.RED, thickness=PT_THICKNESS)

    # draw basketball court for reference
    basketball_court_img = basketball_court.draw(image_size=image_size, color=Color.WHITE)
    for ellipse in ellipses_template:
        ellipse_img = basketball_court.pitch_template_to_pitch_image(geometric_feature=ellipse, image_size=image_size)
        ellipse_img.draw(img=basketball_court_img, color=Color.RED, thickness=PT_THICKNESS)
    cv2.imwrite(output.as_posix(), np.concatenate((frame_with_projected_court, basketball_court_img), axis=1))
