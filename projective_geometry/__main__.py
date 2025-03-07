from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from typer import Typer

from projective_geometry.camera import Camera, CameraParams, CameraPose
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Ellipse, Line, Point
from projective_geometry.geometry.conic import Conic
from projective_geometry.geometry.line_segment import LineSegment
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.projection.projectors import (
    project_conics,
    project_pitch_template,
)
from projective_geometry.utils.distances import FOOT, INCH

PINHOLE_SVG = Point(x=497.18973, y=33.56244)
IMG_SVG_SIZE = ImageSize(width=993.77657, height=287.99746)
Y_GROUND_SVG = 90.30712
X_FILM_SVG = 468.59334
UNIT = 28.585252
FRISBEE_COLOR = (0, 199, 137)
FRISBEE_DISTANCE_TO_PINHOLE = 12
FRISBEE_BORDER_COLOR = (0, 128, 0)
RADIUS_FRISBEE = 5
SIZE_FRISBEE_PERSEPECTIVE = Point(x=RADIUS_FRISBEE * UNIT, y=18.33109)
IMG_DISPLAY_UNIT = 500
BORDER = 15
CAMERA_HEIGHT = 2 * IMG_DISPLAY_UNIT
CAMERA = Camera.from_camera_params(
    camera_params=CameraParams(
        camera_pose=CameraPose(tx=0, ty=0, tz=CAMERA_HEIGHT, roll=0, tilt=90, pan=0),
        focal_length=IMG_DISPLAY_UNIT,
    ),
    image_size=ImageSize(width=IMG_DISPLAY_UNIT, height=IMG_DISPLAY_UNIT),
)
CORNERS_FILM = [
    Point(x=1755, y=96),
    Point(x=1784, y=69),
    Point(x=1755, y=194),
    Point(x=1784, y=165),
]
CORNERS_DISPLAY = [
    Point(x=0, y=0),
    Point(x=IMG_DISPLAY_UNIT, y=0),
    Point(x=0, y=IMG_DISPLAY_UNIT),
    Point(x=IMG_DISPLAY_UNIT, y=IMG_DISPLAY_UNIT),
]
CAMERA_FILM = Camera.from_point_correspondences(CORNERS_DISPLAY, CORNERS_FILM)
OVERRIDEN_CAMERA_LINE = LineSegment(
    pt1=Point(x=1754, y=95),
    pt2=Point(x=1860, y=95),
)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 3
THICKNESS_TEXT = 2
BACKGROUND_COLOR = {
    "Ellipse": Color.YELLOW,
    "Parabola": Color.LIGHT_GREEN,
    "Hyperbola": Color.CYAN,
    "Emphasis": Color.GREEN,
}
MARGIN_TEXT = 5
PROJECT_LOCATION = Path(__file__).parent.parent
IMG_TEMPLATE_FPATH = PROJECT_LOCATION / "results/animation_template.png"
IMG_CELTICS_FPATH = PROJECT_LOCATION / "results/celtics.png"
PT_RADIUS = 10
PT_THICKNESS = 7
BORDER_SIZE = 10

cli_app = Typer()


def label_conic_type(img: np.ndarray, conic_type: str, background_color: Tuple[Any, ...]):
    (width, height), baseline = cv2.getTextSize(conic_type, FONT, FONT_SCALE, THICKNESS_TEXT)
    text_patch = (np.ones((height + baseline, width, 3)) * background_color).astype(np.uint8)
    cv2.putText(text_patch, conic_type, (0, height), FONT, FONT_SCALE, Color.BLACK, THICKNESS_TEXT)
    x = (img.shape[1] - width) // 2
    y = img.shape[0] - (IMG_DISPLAY_UNIT + BORDER * 2 + MARGIN_TEXT + height + baseline)
    img[y : (y + height + baseline), x : (x + width), :] = text_patch
    return img


def get_projection_film(pt: Point, pinhole: Point, x_film: float) -> LineSegment:
    if abs(pt.x - pinhole.x) < 1e-5:
        pt2 = Point(x=pinhole.x, y=0)
    elif pt.x < pinhole.x:
        pt2 = pinhole
    else:
        line = Line.from_points(pt1=pt, pt2=pinhole)
        a, b, c = line.to_array()
        pt2 = Point(x=x_film, y=-(a * x_film + c) / b)
    return LineSegment(pt1=pt, pt2=pt2)


def generate_projected_conic_img(x_frisbee: float):
    img_captured = np.ones((IMG_DISPLAY_UNIT, IMG_DISPLAY_UNIT, 3)) * 255
    x_display = IMG_DISPLAY_UNIT * x_frisbee / UNIT
    ellipse = Ellipse(
        center=Point(x=x_display, y=0),
        axes=Point(x=RADIUS_FRISBEE * IMG_DISPLAY_UNIT, y=RADIUS_FRISBEE * IMG_DISPLAY_UNIT),
        angle=0,
    )
    conic = Conic(M=ellipse.to_matrix())
    projected_conic = project_conics(camera=CAMERA, conics=(conic,))[0]
    projected_conic.draw(img=img_captured, color=FRISBEE_COLOR)
    return img_captured


def project_image_to_camera_film(img: np.ndarray, img_captured: np.ndarray):
    img_flipped = np.flip(img_captured.transpose((1, 0, 2)), (0, 1))
    dsize = (img.shape[1], img.shape[0])
    img_film = cv2.warpPerspective(src=img_flipped, M=CAMERA_FILM.H, dsize=dsize)
    mask = img_film[:, :, 1] > 0
    img[mask] = img_film[mask]

    # add overriden camera line
    OVERRIDEN_CAMERA_LINE.draw(img=img, color=Color.BLACK, thickness=2)
    return img


def generate_frame(img: np.ndarray, x_frisbee: float):
    # define scale w.r.t. svg size
    img_size = ImageSize(width=img.shape[1], height=img.shape[0])
    sx, sy = img_size.width / IMG_SVG_SIZE.width, img_size.height / IMG_SVG_SIZE.height
    scale = Point(x=sx, y=sy)

    # create ellipse
    pinhole = PINHOLE_SVG.scale(pt=scale)
    ellipse = Ellipse(
        center=Point(x=x_frisbee, y=Y_GROUND_SVG).scale(pt=scale),
        axes=SIZE_FRISBEE_PERSEPECTIVE.scale(pt=scale),
        angle=0,
    )

    # get amplified captured image
    img_captured = generate_projected_conic_img(x_frisbee=x_frisbee - PINHOLE_SVG.x)

    # display on the camera film
    img = project_image_to_camera_film(img=img, img_captured=img_captured)

    # add projection rays
    x_film = X_FILM_SVG * sx
    pts_ellipse_axes = [
        Point(x=x_frisbee + SIZE_FRISBEE_PERSEPECTIVE.x, y=Y_GROUND_SVG).scale(pt=scale),
        Point(x=x_frisbee - SIZE_FRISBEE_PERSEPECTIVE.x, y=Y_GROUND_SVG).scale(pt=scale),
    ]
    ellipse.draw(img=img, color=FRISBEE_COLOR, thickness=-1)
    # ellipse.draw(img=img, color=FRISBEE_BORDER_COLOR)
    for pt in pts_ellipse_axes:
        projection_ray = get_projection_film(pt=pt, pinhole=pinhole, x_film=x_film)
        projection_ray.draw(img=img, color=Color.ORANGE, thickness=2)

    # display with border on the right
    img_display = np.zeros((IMG_DISPLAY_UNIT + BORDER * 2, IMG_DISPLAY_UNIT + BORDER * 2, 3))
    img_display[BORDER:-BORDER, BORDER:-BORDER, :] = img_captured
    img_display = img_display.transpose((1, 0, 2))
    h, w = img_display.shape[:2]
    x = int((img_size.width - w) / 2)
    img[-h:, x : w + x, :] = img_display
    return img


@cli_app.command()
def frisbee_demo(output: Path = PROJECT_LOCATION / "results/frisbee.mp4"):
    img = cv2.imread(IMG_TEMPLATE_FPATH.as_posix())
    output_size = (img.shape[1], img.shape[0])
    fps = 3
    n_frames = 49
    out = cv2.VideoWriter(output.as_posix(), cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, output_size)  # type: ignore
    x_start = PINHOLE_SVG.x + FRISBEE_DISTANCE_TO_PINHOLE * UNIT
    x_end = PINHOLE_SVG.x - FRISBEE_DISTANCE_TO_PINHOLE * UNIT
    # we ensure the number of points is odd so we can see the parabola
    for x in np.linspace(x_start, x_end, n_frames):
        frame = generate_frame(img=img.copy(), x_frisbee=x)

        # label it
        x_left = x - RADIUS_FRISBEE * UNIT
        x_right = x + RADIUS_FRISBEE * UNIT
        if abs(x_left - PINHOLE_SVG.x) < 1e-5 or abs(x_right - PINHOLE_SVG.x) < 1e-5:
            conic_type = "Parabola"
            frame = label_conic_type(img=frame, conic_type=conic_type, background_color=BACKGROUND_COLOR[conic_type])
            # emphasize
            frame2 = label_conic_type(
                img=frame.copy(),
                conic_type=conic_type,
                background_color=BACKGROUND_COLOR["Emphasis"],
            )
            frames = ([frame] * 2 + [frame2] * 2) * 2 + [frame]
            for f in frames:
                out.write(f)
        elif (x_left > PINHOLE_SVG.x) or (x_right < PINHOLE_SVG.x):
            conic_type = "Ellipse"
            label_conic_type(img=frame, conic_type=conic_type, background_color=BACKGROUND_COLOR[conic_type])
        else:
            conic_type = "Hyperbola"
            label_conic_type(img=frame, conic_type=conic_type, background_color=BACKGROUND_COLOR[conic_type])
        out.write(frame)
    out.release()


@cli_app.command()
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


@cli_app.command()
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


@cli_app.command()
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


@cli_app.command()
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


@cli_app.command()
def homography_from_image_registration(
    target: Path = PROJECT_LOCATION / "results/target.png",
    source: Path = PROJECT_LOCATION / "results/source.png",
):
    target_image = cv2.imread(target.as_posix())
    source_image = cv2.imread(source.as_posix())
    camera, matched_keypoints = Camera.from_image_registration(target_image=target_image, source_image=source_image)

    # visualize matched keypoints
    img_matches = cv2.drawMatches(
        target_image,
        matched_keypoints.target_keypoints,
        source_image,
        matched_keypoints.source_keypoints,
        matched_keypoints.matches,
        None,
    )
    cv2.imwrite((PROJECT_LOCATION / "results/matches.png").as_posix(), img_matches)

    # visualize warped image
    height, width, _ = source_image.shape
    source_image_border = cv2.copyMakeBorder(
        source_image[BORDER_SIZE:-BORDER_SIZE, BORDER_SIZE:-BORDER_SIZE],
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv2.BORDER_CONSTANT,
        value=Color.RED,
    )
    warped_image = cv2.warpPerspective(source_image_border, camera.H, (width, height))
    # add red border to warped image without changing its size
    target_and_warped_images = cv2.addWeighted(target_image, 1, warped_image, 0.5, 0)
    cv2.imwrite((PROJECT_LOCATION / "results/warped.png").as_posix(), target_and_warped_images)

@cli_app.command()
def focal_length_from_orthogonal_vanishing_points_demo(image: Path = PROJECT_LOCATION / "results/BasketballCourtCalibration.png"):
    image = cv2.imread(image.as_posix())
    width, height = image.shape[1], image.shape[0]
    vp1 = Point(x=7239.60, y=875.45)
    vp2 = Point(x=754.46, y=-1758.11)
    focal_length = (-(vp1.x - width / 2) * (vp2.x - width / 2) - (vp1.y - height / 2) * (vp2.y - height / 2)) ** 0.5
    print(f"Focal length: {focal_length}")
    pass

@cli_app.command()
def intrinsic_from_three_planes():
    pass

# Program entry point redirection
if __name__ == "__main__":
    cli_app()
