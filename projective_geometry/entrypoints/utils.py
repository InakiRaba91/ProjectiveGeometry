from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np

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
        camera_pose=CameraPose(tx=0, ty=0, tz=CAMERA_HEIGHT, rx=0, ry=90, rz=0),
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
PROJECT_LOCATION = Path(__file__).parent.parent.parent
IMG_TEMPLATE_FPATH = PROJECT_LOCATION / "results/animation_template.png"
IMG_CELTICS_FPATH = PROJECT_LOCATION / "results/celtics.png"
PT_RADIUS = 10
PT_THICKNESS = 7
BORDER_SIZE = 10
BALL_CIRCUMFERENCE = 29.5 / 36.0  
BALL_RADIUS = BALL_CIRCUMFERENCE / (2 * np.pi)


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
        A[2*i, :] = np.array([u[0]*v[0] + u[1]*v[1], u[2]*v[0] + u[0]*v[2], u[2]*v[1] + u[1]*v[2], u[2]*v[2]])
        a1 = np.array([u[0]**2 + u[1]**2, 2*u[0]*u[2], 2*u[1]*u[2], u[2]**2])
        a2 = np.array([v[0]**2 + v[1]**2, 2*v[0]*v[2], 2*v[1]*v[2], v[2]**2])
        A[2*i+1, :] = a1 - a2
    # # Solve system A*w = 0
    U, S, Vt = np.linalg.svd(A)
    w = Vt[-1, :] * np.sign(Vt[-1, 0])  # to ensure the first element is positive, since it's f**2
    omega = np.array([[w[0], 0, w[1]], [0, w[0], w[2]], [w[1], w[2], w[3]]])
    f_estimate = 1 / (omega[0,0] ** 0.5)
    px_estimate = - omega[0, 2] / omega[0, 0]
    py_estimate = - omega[1, 2] / omega[1, 1]
    return np.array([[f_estimate, 0, px_estimate], [0, f_estimate, py_estimate], [0, 0, 1]])


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
