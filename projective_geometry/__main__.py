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
from projective_geometry.projection.projectors import project_conics

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
    out = cv2.VideoWriter(output.as_posix(), cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, output_size)
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


# Program entry point redirection
if __name__ == "__main__":
    cli_app()
