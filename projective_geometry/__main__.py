from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from typer import Typer

from projective_geometry.camera import Camera, CameraParams, CameraPose
from projective_geometry.camera.camera2 import Camera2
from projective_geometry.draw import Color
from projective_geometry.draw.image_size import ImageSize
from projective_geometry.geometry import Ellipse, Line, Point2D, Point3D
from projective_geometry.geometry.conic import Conic
from projective_geometry.geometry.line_segment import LineSegment
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.projection.projectors import (
    project_conics,
    project_pitch_template,
    project_to_sensor,
    project_to_world,
)
from projective_geometry.utils.distances import FOOT, INCH
from projective_geometry.visualization.virtual_trajectory import (
    generate_video_virtual_trajectory_camera,
)
from projective_geometry.visualization.visualizer import show_camera_visualisation

PINHOLE_SVG = Point2D(x=497.18973, y=33.56244)
IMG_SVG_SIZE = ImageSize(width=993.77657, height=287.99746)
Y_GROUND_SVG = 90.30712
X_FILM_SVG = 468.59334
UNIT = 28.585252
FRISBEE_COLOR = (0, 199, 137)
FRISBEE_DISTANCE_TO_PINHOLE = 12
FRISBEE_BORDER_COLOR = (0, 128, 0)
RADIUS_FRISBEE = 5
SIZE_FRISBEE_PERSEPECTIVE = Point2D(x=RADIUS_FRISBEE * UNIT, y=18.33109)
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
    Point2D(x=1755, y=96),
    Point2D(x=1784, y=69),
    Point2D(x=1755, y=194),
    Point2D(x=1784, y=165),
]
CORNERS_DISPLAY = [
    Point2D(x=0, y=0),
    Point2D(x=IMG_DISPLAY_UNIT, y=0),
    Point2D(x=0, y=IMG_DISPLAY_UNIT),
    Point2D(x=IMG_DISPLAY_UNIT, y=IMG_DISPLAY_UNIT),
]
CAMERA_FILM = Camera.from_point_correspondences(CORNERS_DISPLAY, CORNERS_FILM)
OVERRIDEN_CAMERA_LINE = LineSegment(
    pt1=Point2D(x=1754, y=95),
    pt2=Point2D(x=1860, y=95),
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


def project_3d_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Project 3D points using given homography matrix"

    Args:
        H: Homography matrix
        pts: 3D points to project

    Returns:
        np.ndarray: Projected 2D points
    """
    projected_pts_homogeneous = H.dot(pts)
    return projected_pts_homogeneous / projected_pts_homogeneous[-1]


def get_2d_homography_between_planes(H: np.ndarray, pts_world: list[np.ndarray]) -> np.ndarray:
    """
    Get 2D homography between two planes

    Args:
        H: original Homography matrix mapping world to image
        pts_world: List of 3D points in world plane
        pts_image: List of 2D points in image plane

    Returns:
        np.ndarray: Homography matrix
    """
    # get a 3D basis in the plane
    p1, p2, p3, p4 = [pt[:-1, :] for pt in pts_world]
    v1 = p2 - p1
    v2 = p3 - p1

    # turn it orthonormal via Gram-Schmidt
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(v2.T, v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    M = np.concatenate([v1, v2], axis=1)

    # find the homography between the two planes
    H_aux = np.concatenate((np.concatenate([M, p1], axis=1), np.array([[0, 0, 1]])), axis=0)
    return H.dot(H_aux)


def get_intrinsic_from_2d_homographies(H_planes: list[np.ndarray]) -> np.ndarray:
    """
    Get intrinsic matrix from 2D homographies

    Args:
        H_planes: List of 2D homographies

    Returns:
        np.ndarray: Intrinsic matrix
    """
    A = np.zeros((6, 4))
    for i, Hi in enumerate(H_planes):
        u = Hi[:, 0]
        v = Hi[:, 1]
        A[2 * i, :] = np.array([u[0] * v[0] + u[1] * v[1], u[2] * v[0] + u[0] * v[2], u[2] * v[1] + u[1] * v[2], u[2] * v[2]])
        a1 = np.array([u[0] ** 2 + u[1] ** 2, 2 * u[0] * u[2], 2 * u[1] * u[2], u[2] ** 2])
        a2 = np.array([v[0] ** 2 + v[1] ** 2, 2 * v[0] * v[2], 2 * v[1] * v[2], v[2] ** 2])
        A[2 * i + 1, :] = a1 - a2
    # # Solve system A*w = 0
    U, S, Vt = np.linalg.svd(A)
    w = Vt[-1, :] * np.sign(Vt[-1, 0])  # to ensure the first element is positive, since it's f**2
    omega = np.array([[w[0], 0, w[1]], [0, w[0], w[2]], [w[1], w[2], w[3]]])
    f_estimate = 1 / (omega[0, 0] ** 0.5)
    px_estimate = -omega[0, 2] / omega[0, 0]
    py_estimate = -omega[1, 2] / omega[1, 1]
    return np.array([[f_estimate, 0, px_estimate], [0, f_estimate, py_estimate], [0, 0, 1]])


def label_conic_type(img: np.ndarray, conic_type: str, background_color: Tuple[Any, ...]):
    (width, height), baseline = cv2.getTextSize(conic_type, FONT, FONT_SCALE, THICKNESS_TEXT)
    text_patch = (np.ones((height + baseline, width, 3)) * background_color).astype(np.uint8)
    cv2.putText(text_patch, conic_type, (0, height), FONT, FONT_SCALE, Color.BLACK, THICKNESS_TEXT)
    x = (img.shape[1] - width) // 2
    y = img.shape[0] - (IMG_DISPLAY_UNIT + BORDER * 2 + MARGIN_TEXT + height + baseline)
    img[y : (y + height + baseline), x : (x + width), :] = text_patch
    return img


def get_projection_film(pt: Point2D, pinhole: Point2D, x_film: float) -> LineSegment:
    if abs(pt.x - pinhole.x) < 1e-5:
        pt2 = Point2D(x=pinhole.x, y=0)
    elif pt.x < pinhole.x:
        pt2 = pinhole
    else:
        line = Line.from_points(pt1=pt, pt2=pinhole)
        a, b, c = line.to_array()
        pt2 = Point2D(x=x_film, y=-(a * x_film + c) / b)
    return LineSegment(pt1=pt, pt2=pt2)


def generate_projected_conic_img(x_frisbee: float):
    img_captured = np.ones((IMG_DISPLAY_UNIT, IMG_DISPLAY_UNIT, 3)) * 255
    x_display = IMG_DISPLAY_UNIT * x_frisbee / UNIT
    ellipse = Ellipse(
        center=Point2D(x=x_display, y=0),
        axes=Point2D(x=RADIUS_FRISBEE * IMG_DISPLAY_UNIT, y=RADIUS_FRISBEE * IMG_DISPLAY_UNIT),
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
    scale = Point2D(x=sx, y=sy)

    # create ellipse
    pinhole = PINHOLE_SVG.scale(pt=scale)
    ellipse = Ellipse(
        center=Point2D(x=x_frisbee, y=Y_GROUND_SVG).scale(pt=scale),
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
        Point2D(x=x_frisbee + SIZE_FRISBEE_PERSEPECTIVE.x, y=Y_GROUND_SVG).scale(pt=scale),
        Point2D(x=x_frisbee - SIZE_FRISBEE_PERSEPECTIVE.x, y=Y_GROUND_SVG).scale(pt=scale),
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
        Point2D(x=845, y=290),
        Point2D(x=126, y=872),
        Point2D(x=1692, y=367),
        Point2D(x=1115, y=707),
        Point2D(x=1560, y=644),
    ]
    points_template = [
        Point2D(x=-W2, y=-H2),
        Point2D(x=-W2, y=H2),
        Point2D(x=-W2 + 28 * FOOT, y=-H2),
        Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT),
        Point2D(x=-W2 + 28 * FOOT + 12 * INCH, y=0),
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
        Line.from_points(pt1=Point2D(x=-W2, y=-H2), pt2=Point2D(x=W2, y=-H2)),
        Line.from_points(pt1=Point2D(x=W2, y=H2), pt2=Point2D(x=-W2, y=H2)),
        Line.from_points(pt1=Point2D(x=-W2, y=H2), pt2=Point2D(x=-W2, y=-H2)),
        Line.from_points(pt1=Point2D(x=-W2, y=-8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=-8 * FOOT)),
        Line.from_points(pt1=Point2D(x=-W2 + 19 * FOOT, y=-8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT)),
        Line.from_points(pt1=Point2D(x=-W2, y=8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT)),
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
            center=Point2D(x=1198.1113211177799, y=606.6167316662579),
            axes=Point2D(x=77.11168956853989, y=222.0629912180788),
            angle=90.70347611420907,
        ),
        # three-point arc
        Ellipse(
            center=Point2D(x=707.2445757522346, y=600.6708571711737),
            axes=Point2D(x=296.0026379644702, y=871.9410745372605),
            angle=88.64701264444109,
        ),
        # restricted arc
        Ellipse(
            center=Point2D(x=720.7694262254354, y=553.058980020591),
            axes=Point2D(x=48.53764408421055, y=144.87141097409037),
            angle=88.70747899257509,
        ),
        # corner arcs
        Ellipse(
            center=Point2D(x=810.80940320813, y=325.6083962774958),
            axes=Point2D(x=127.8838414420805, y=438.8219609157189),
            angle=89.32970121679617,
        ),
        Ellipse(
            center=Point2D(x=172.0013839803821, y=851.1080576625786),
            axes=Point2D(x=227.28737736672736, y=601.9016883331738),
            angle=85.72932402628315,
        ),
    ]
    ellipses_template = [
        # free-throw circle
        Ellipse(
            center=Point2D(x=-W2 + 19 * FOOT, y=0),
            axes=Point2D(x=6 * FOOT, y=6 * FOOT),
            angle=0,
        ),
        # three-point arc
        Ellipse(
            center=Point2D(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
            axes=Point2D(x=23 * FOOT + 9 * INCH, y=23 * FOOT + 9 * INCH),
            angle=0,
        ),
        # restricted arc
        Ellipse(
            center=Point2D(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
            axes=Point2D(x=4 * FOOT, y=4 * FOOT),
            angle=0,
        ),
        # corner arcs
        Ellipse(
            center=Point2D(x=-W2, y=-H2 + 3 * FOOT),
            axes=Point2D(x=14 * FOOT, y=14 * FOOT),
            angle=0,
        ),
        Ellipse(
            center=Point2D(x=-W2, y=H2 - 3 * FOOT),
            axes=Point2D(x=14 * FOOT, y=14 * FOOT),
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
        Point2D(x=845, y=290),
        Point2D(x=126, y=872),
        Point2D(x=1692, y=367),
        Point2D(x=1115, y=707),
        Point2D(x=1560, y=644),
    ]
    points_template = [
        Point2D(x=-W2, y=-H2),
        Point2D(x=-W2, y=H2),
        Point2D(x=-W2 + 28 * FOOT, y=-H2),
        Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT),
        Point2D(x=-W2 + 28 * FOOT + 12 * INCH, y=0),
    ]
    lines_template = [
        Line.from_points(pt1=Point2D(x=-W2, y=-H2), pt2=Point2D(x=W2, y=-H2)),
        Line.from_points(pt1=Point2D(x=W2, y=H2), pt2=Point2D(x=-W2, y=H2)),
        Line.from_points(pt1=Point2D(x=-W2, y=H2), pt2=Point2D(x=-W2, y=-H2)),
        Line.from_points(pt1=Point2D(x=-W2, y=-8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=-8 * FOOT)),
        Line.from_points(pt1=Point2D(x=-W2 + 19 * FOOT, y=-8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT)),
        Line.from_points(pt1=Point2D(x=-W2, y=8 * FOOT), pt2=Point2D(x=-W2 + 19 * FOOT, y=8 * FOOT)),
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
            center=Point2D(x=1198.1113211177799, y=606.6167316662579),
            axes=Point2D(x=77.11168956853989, y=222.0629912180788),
            angle=90.70347611420907,
        ),
        # three-point arc
        Ellipse(
            center=Point2D(x=707.2445757522346, y=600.6708571711737),
            axes=Point2D(x=296.0026379644702, y=871.9410745372605),
            angle=88.64701264444109,
        ),
        # restricted arc
        Ellipse(
            center=Point2D(x=720.7694262254354, y=553.058980020591),
            axes=Point2D(x=48.53764408421055, y=144.87141097409037),
            angle=88.70747899257509,
        ),
        # corner arcs
        Ellipse(
            center=Point2D(x=810.80940320813, y=325.6083962774958),
            axes=Point2D(x=127.8838414420805, y=438.8219609157189),
            angle=89.32970121679617,
        ),
        Ellipse(
            center=Point2D(x=172.0013839803821, y=851.1080576625786),
            axes=Point2D(x=227.28737736672736, y=601.9016883331738),
            angle=85.72932402628315,
        ),
    ]
    ellipses_template = [
        # free-throw circle
        Ellipse(
            center=Point2D(x=-W2 + 19 * FOOT, y=0),
            axes=Point2D(x=6 * FOOT, y=6 * FOOT),
            angle=0,
        ),
        # three-point arc
        Ellipse(
            center=Point2D(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
            axes=Point2D(x=23 * FOOT + 9 * INCH, y=23 * FOOT + 9 * INCH),
            angle=0,
        ),
        # restricted arc
        Ellipse(
            center=Point2D(x=-W2 + 5 * FOOT + 3 * INCH, y=0),
            axes=Point2D(x=4 * FOOT, y=4 * FOOT),
            angle=0,
        ),
        # corner arcs
        Ellipse(
            center=Point2D(x=-W2, y=-H2 + 3 * FOOT),
            axes=Point2D(x=14 * FOOT, y=14 * FOOT),
            angle=0,
        ),
        Ellipse(
            center=Point2D(x=-W2, y=H2 - 3 * FOOT),
            axes=Point2D(x=14 * FOOT, y=14 * FOOT),
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
        np.zeros_like(target_image, dtype=np.uint8),
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
def focal_length_from_orthogonal_vanishing_points_demo(
    image_path: Path = PROJECT_LOCATION / "results/BasketballCourtCalibration.png",
):
    image = cv2.imread(image_path.as_posix())
    width, height = image.shape[1], image.shape[0]
    vp1 = Point2D(x=7239.60, y=875.45)
    vp2 = Point2D(x=754.46, y=-1758.11)
    focal_length = (-(vp1.x - width / 2) * (vp2.x - width / 2) - (vp1.y - height / 2) * (vp2.y - height / 2)) ** 0.5
    print(f"Focal length: {focal_length}")


@cli_app.command()
def intrinsic_from_three_planes_demo(image_path: Path = PROJECT_LOCATION / "results/SoccerPitchCalibration.png"):
    # ground truth
    f, tx, ty, tz, rx, ry, rz = 4763, -21, -110, 40, 250, 2, 13
    image = cv2.imread(image_path.as_posix())
    image_width, image_height = image.shape[1], image.shape[0]
    pitch_width, _ = 120, 80
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


@cli_app.command()
def camera_calibration_test(
    image_path: Path = PROJECT_LOCATION / "results/Broadcast.png",
):

    image_points = tuple(
        [
            Point2D(x=1410.0393, y=293.5348),
            Point2D(x=737.2433, y=251.7833),
            Point2D(x=450.1709, y=329.2394),
            Point2D(x=1186.5133, y=381.2686),
            Point2D(x=926.6083, y=485.5929),
            Point2D(x=410.3828, y=427.8692),
            Point2D(x=441.6741, y=533.5641),
            Point2D(x=533.7554, y=641.9421),
            Point2D(x=31.9192, y=842.0333),
        ]
    )

    world_points = tuple(
        [
            Point3D(x=-36.1188, y=33.8328),
            Point3D(x=-52.578, y=33.8328),
            Point3D(x=-52.578, y=20.1168),
            Point3D(x=-36.1188, y=20.1168),
            Point3D(x=-36.1188, y=7.3152),
            Point3D(x=-47.0916, y=9.144),
            Point3D(x=-41.6052, y=0.0),
            Point3D(x=-36.1188, y=-7.3152),
            Point3D(x=-36.1188, y=-20.1168),
        ]
    )

    image = cv2.imread(image_path.as_posix())

    for pt in image_points:
        pt.draw(img=image, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)
    cv2.imwrite((PROJECT_LOCATION / "results/BroadcastManualImagePointsTest.png").as_posix(), image)

    sensor_wh = image.shape[1], image.shape[0]
    camera = Camera2.from_keypoint_correspondences(world_points, image_points, sensor_wh)
    print(f"Intrinsic matrix: \n{camera}")

    image_point2 = project_to_sensor(camera, world_points)

    rmse_image = np.mean(
        [np.linalg.norm(pt1.to_array() - pt2.to_array()) for pt1, pt2 in zip(image_points, image_point2, strict=True)]
    )
    print(f"RMSE Image: {rmse_image} px")

    for pt_i1, pt_i2 in zip(image_points, image_point2, strict=True):
        pt_i1.draw(img=image, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)
        pt_i2.draw(img=image, color=Color.BLUE, radius=PT_RADIUS, thickness=PT_THICKNESS)
    cv2.imwrite((PROJECT_LOCATION / "results/BroadcastProjectedImagePointsTest.png").as_posix(), image)

    world_point2 = project_to_world(camera, image_points)

    rmse_world = np.mean(
        [np.linalg.norm(pt1.to_array() - pt2.to_array()) for pt1, pt2 in zip(world_points, world_point2, strict=True)]
    )
    print(f"RMSE World: {rmse_world} m")

    FIELD_HEIGHT = 68
    FIELD_WIDTH = 105
    FIELD_IMAGE_FACTOR = 10

    field_image = (
        np.ones((FIELD_HEIGHT * FIELD_IMAGE_FACTOR, FIELD_WIDTH * FIELD_IMAGE_FACTOR, 3)) * 255
    )  # create a white field image
    for pt_w1, pt_w2 in zip(world_points, world_point2, strict=True):
        pt_w1 += Point3D(x=FIELD_WIDTH / 2, y=FIELD_HEIGHT / 2)
        pt_w1 *= FIELD_IMAGE_FACTOR  # type: ignore[assignment]
        pt_w1 = Point3D(pt_w1.x, -pt_w1.y, pt_w1.z) + Point3D(0, FIELD_HEIGHT * FIELD_IMAGE_FACTOR, 0)
        pt_w2 += Point3D(x=FIELD_WIDTH / 2, y=FIELD_HEIGHT / 2)
        pt_w2 *= FIELD_IMAGE_FACTOR  # type: ignore[assignment]
        pt_w2 = Point3D(pt_w2.x, -pt_w2.y, pt_w2.z) + Point3D(0, FIELD_HEIGHT * FIELD_IMAGE_FACTOR, 0)
        pt_w1.draw(img=field_image, color=Color.RED, radius=PT_RADIUS, thickness=PT_THICKNESS)
        pt_w2.draw(img=field_image, color=Color.BLUE, radius=PT_RADIUS, thickness=PT_THICKNESS)

    cv2.imwrite((PROJECT_LOCATION / "results/BroadcastProjectedFieldPointsTest.png").as_posix(), field_image)


@cli_app.command()
def camera_retrieval_test(output: Path = PROJECT_LOCATION / "results/celtics_retrieval.png"):
    W2, H2 = BasketballCourtTemplate.PITCH_WIDTH / 2, BasketballCourtTemplate.PITCH_HEIGHT / 2
    points_frame = [
        Point2D(x=845, y=290),
        Point2D(x=126, y=872),
        Point2D(x=1692, y=367),
        Point2D(x=1115, y=707),
        Point2D(x=1560, y=644),
        Point2D(x=1270, y=510),
    ]
    points_template = [
        Point3D(x=-W2, y=-H2),
        Point3D(x=-W2, y=H2),
        Point3D(x=-W2 + 28 * FOOT, y=-H2),
        Point3D(x=-W2 + 19 * FOOT, y=8 * FOOT),
        Point3D(x=-W2 + 28 * FOOT + 12 * INCH, y=0),
        Point3D(x=-W2 + 19 * FOOT, y=-8 * FOOT),
    ]
    points_template_2d = tuple(Point2D(x=pt.x, y=pt.y) for pt in points_template)

    camera = Camera.from_point_correspondences(pts_source=points_template_2d, pts_target=points_frame)

    # project basketball court template
    basketball_court = BasketballCourtTemplate()
    frame = cv2.imread(IMG_CELTICS_FPATH.as_posix())
    sensor_wh = frame.shape[1], frame.shape[0]
    image_size = ImageSize(width=frame.shape[1], height=frame.shape[0])
    basketball_court_img = basketball_court.draw(image_size=image_size, color=Color.WHITE)

    camera2 = Camera2.from_keypoint_correspondences(points_template, points_frame, sensor_wh)

    keypoints = tuple(Point3D(x=pt.x, y=pt.y) for pt in basketball_court.keypoints)
    keypoints_frame = project_to_sensor(camera2, keypoints)
    for pt in keypoints_frame:
        pt.draw(img=frame, color=Color.GREEN, radius=PT_RADIUS, thickness=PT_THICKNESS)

    points_frame2 = project_to_sensor(camera2, points_template)
    rmses_image = []
    for pt_i1, pt_i2 in zip(points_frame, points_frame2, strict=True):
        pt_i1.draw(img=frame, color=Color.RED, radius=2 * PT_RADIUS, thickness=-1)
        pt_i2.draw(img=frame, color=Color.BLUE, radius=PT_RADIUS, thickness=PT_THICKNESS)
        rmses_image.append(np.linalg.norm(pt_i1.to_array() - pt_i2.to_array()))
    rmse_image = np.mean(rmses_image)
    print(f"RMSE Image: {rmse_image} px")

    points_template2 = project_to_world(camera2, points_frame)
    rmses_world = []
    for pt_w1, pt_w2 in zip(points_template, points_template2, strict=True):
        new_pt_w1 = Point2D(x=pt_w1.x, y=pt_w1.y)
        new_pt_w2 = Point2D(x=pt_w2.x, y=pt_w2.y)
        pt_img1 = basketball_court.pitch_template_to_pitch_image(geometric_feature=new_pt_w1, image_size=image_size)
        pt_img1.draw(img=basketball_court_img, color=Color.RED, radius=2 * PT_RADIUS, thickness=-1)
        pt_img2 = basketball_court.pitch_template_to_pitch_image(geometric_feature=new_pt_w2, image_size=image_size)
        pt_img2.draw(img=basketball_court_img, color=Color.BLUE, radius=PT_RADIUS, thickness=PT_THICKNESS)  # type: ignore
        rmses_world.append(np.linalg.norm(new_pt_w1.to_array() - new_pt_w2.to_array()))
    rmse_world = np.mean(rmses_world)
    print(f"RMSE World: {rmse_world} m")

    cv2.imwrite(output.as_posix(), np.concatenate((frame, basketball_court_img), axis=1))


def visualize():
    show_camera_visualisation()


@cli_app.command()
def virtual_camera_trajectory():
    generate_video_virtual_trajectory_camera(video_path=PROJECT_LOCATION / "results/virtual_camera_trajectory.mp4")


# Program entry point redirection
if __name__ == "__main__":
    cli_app()
