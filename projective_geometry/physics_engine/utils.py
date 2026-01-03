from pathlib import Path

import cv2
import numpy as np

from projective_geometry.geometry.conic import Conic
from projective_geometry.geometry.ellipse import Ellipse
from projective_geometry.geometry.point import Point


def corrupt_ellipse(conic: Conic, stds: np.ndarray, random_state: np.random.RandomState) -> Ellipse:
    """Adds Gaussian noise to ellipse parameters

    Args:
        conic: Original conic
        stds: Standard deviations for (a, b, x0, y0, angle)

    Returns:
        Noisy conic
    """
    ellipse = Ellipse.from_matrix(conic.M, check_validity=False)
    center = ellipse.center
    axes = ellipse.axes
    angle = ellipse.angle
    x0 = center.x + random_state.normal(0, stds[0])
    y0 = center.y + random_state.normal(0, stds[1])
    a = max(1, axes.x + random_state.normal(0, stds[2]))
    b = max(1, axes.y + random_state.normal(0, stds[3]))
    angle = angle + random_state.normal(0, stds[4])
    return Ellipse(center=Point(x=x0, y=y0), axes=Point(x=a, y=b), angle=angle)


def sample_ellipse(ellipse: Ellipse, num_points: int = 10) -> np.ndarray:
    """Samples points on the ellipse

    Args:
        ellipse: Ellipse to sample from
        num_points: Number of points to sample
    Returns:
        points: Array of shape (num_points, 2)
    """
    center = ellipse.center
    axes = ellipse.axes
    angle = ellipse.angle
    x0, y0 = center.x, center.y
    a, b = axes.x, axes.y

    theta = np.linspace(0, 2 * np.pi, num_points)
    x = x0 + a * np.cos(theta) * np.cos(np.deg2rad(angle)) - b * np.sin(theta) * np.sin(np.deg2rad(angle))
    y = y0 + a * np.cos(theta) * np.sin(np.deg2rad(angle)) + b * np.sin(theta) * np.cos(np.deg2rad(angle))
    return np.vstack((x, y)).T


def get_bbox_ellipse(conic: Ellipse | Conic) -> np.ndarray:
    """Computes axis-aligned bounding box of an ellipse

    Args:
        conic: Ellipse or Conic
        tol: Tolerance for numerical stability
    Returns:
        (bottom-left), (top-right): Tuple of coordinates defining the bounding box
    """
    if isinstance(conic, Conic):
        ellipse = Ellipse.from_matrix(conic.M, check_validity=False)
        x0, y0 = ellipse.center.x, ellipse.center.y
        a, b = ellipse.axes.x, ellipse.axes.y
        angle = ellipse.angle
    else:
        a, b, x0, y0, angle = conic.center.x, conic.center.y, conic.axes.x, conic.axes.y, conic.angle
    theta = np.deg2rad(angle)
    x_extent = np.sqrt((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2)
    y_extent = np.sqrt((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2)
    x_left = x0 - x_extent
    x_right = x0 + x_extent
    y_bottom = y0 - y_extent
    y_top = y0 + y_extent
    return np.array([x_left, y_bottom, x_right, y_top])


def draw_trajectory_comparison(
    gt_trajectory_3d_positions: dict,
    est_trajectory_3d_positions: dict,
    noisy_trajectory_ellipse_observations: dict,
    H: np.ndarray,
    H2: np.ndarray,
    template_image_path: Path,
    output_prefix: str = "locate_3d_ball_trajectory",
):
    """Draw ground truth, estimated, and noisy trajectory observations.

    Args:
        gt_trajectory_3d_positions: Ground truth 3D positions keyed by time
        est_trajectory_3d_positions: Estimated 3D positions keyed by time
        noisy_trajectory_ellipse_observations: Noisy ellipse observations keyed by time
        H: Homography for broadcast camera
        H2: Homography for birdseye camera
        template_image_path: Path to the pitch template image
        output_prefix: Prefix for output image files
    """
    image = cv2.imread(template_image_path.as_posix())
    image_size = (image.shape[1], image.shape[0])
    pitch_width, pitch_height = 120, 80
    K_pitch_image_to_pitch_template = np.array(
        [
            [pitch_width / image_size[0], 0, 0, -pitch_width / 2.0],
            [0, pitch_height / image_size[1], 0, -pitch_height / 2.0],
            [0, 0, 1.0, 0],
            [0, 0, 0, 1.0],
        ]
    )

    # create a chained homography projection that maps from BEV camera -> desired camera homography
    H_chained = H.dot(K_pitch_image_to_pitch_template)
    H_chained = H_chained[:, [0, 1, 3]]
    H_chained2 = H2.dot(K_pitch_image_to_pitch_template)
    H_chained2 = H_chained2[:, [0, 1, 3]]
    image_broadcast = cv2.warpPerspective(src=image, M=H_chained, dsize=image_size)
    image_birdseye = cv2.warpPerspective(src=image, M=H_chained2, dsize=image_size)
    image_birdseye_obs = image_birdseye.copy()
    image_broadcast_obs = image_broadcast.copy()

    color_gt = (229, 228, 226)
    color_obs = (0, 0, 255)
    color_est = (255, 0, 0)
    cross_size = 7

    for t, pt in gt_trajectory_3d_positions.items():
        # Draw ground truth on broadcast camera
        pth_broadcast = np.array([pt[0], pt[1], pt[2], 1])
        pt_img_broadcast = H.dot(pth_broadcast)
        pt_img_broadcast /= pt_img_broadcast[2]

        # Draw ground truth on birdseye camera
        pth_birdseye = np.array([pt[0], pt[1], 0, 1])
        pt_img_birdseye = H2.dot(pth_birdseye)
        pt_img_birdseye /= pt_img_birdseye[2]

        cv2.circle(image_broadcast, (int(pt_img_broadcast[0]), int(pt_img_broadcast[1])), 5, color_gt, -1)
        cv2.circle(image_broadcast_obs, (int(pt_img_broadcast[0]), int(pt_img_broadcast[1])), 5, color_gt, -1)
        cv2.circle(image_birdseye_obs, (int(pt_img_birdseye[0]), int(pt_img_birdseye[1])), 5, color_gt, -1)

        # Draw noisy observations on separate image
        noisy_obs = noisy_trajectory_ellipse_observations[t]
        if noisy_obs is not None:
            noisy_obs.draw(image_broadcast_obs, color=color_obs, thickness=-1)

        # Draw estimated trajectory on both cameras
        pt_est = est_trajectory_3d_positions[t]
        pth_est_broadcast = np.array([pt_est[0], pt_est[1], pt_est[2], 1])
        pt_img_est_broadcast = H.dot(pth_est_broadcast)
        pt_img_est_broadcast /= pt_img_est_broadcast[2]

        pth_est_birdseye = np.array([pt_est[0], pt_est[1], 0, 1])
        pt_img_est_birdseye = H2.dot(pth_est_birdseye)
        pt_img_est_birdseye /= pt_img_est_birdseye[2]

        # Draw diagonal cross for estimated point on broadcast
        x_bc, y_bc = int(pt_img_est_broadcast[0]), int(pt_img_est_broadcast[1])
        cv2.line(image_broadcast, (x_bc - cross_size, y_bc - cross_size), (x_bc + cross_size, y_bc + cross_size), color_est, 2)
        cv2.line(image_broadcast, (x_bc - cross_size, y_bc + cross_size), (x_bc + cross_size, y_bc - cross_size), color_est, 2)

        # Draw diagonal cross for estimated point on birdseye
        x_be, y_be = int(pt_img_est_birdseye[0]), int(pt_img_est_birdseye[1])
        cv2.line(image_birdseye, (x_be - cross_size, y_be - cross_size), (x_be + cross_size, y_be + cross_size), color_est, 2)
        cv2.line(image_birdseye, (x_be - cross_size, y_be + cross_size), (x_be + cross_size, y_be - cross_size), color_est, 2)

    output_dir = template_image_path.parent
    cv2.imwrite(
        (output_dir / f"{output_prefix}_observations.png").as_posix(),
        np.concatenate([image_broadcast_obs, image_birdseye_obs], axis=1),
    )
    cv2.imwrite(
        (output_dir / f"{output_prefix}_estimated.png").as_posix(), np.concatenate([image_broadcast, image_birdseye], axis=1)
    )
