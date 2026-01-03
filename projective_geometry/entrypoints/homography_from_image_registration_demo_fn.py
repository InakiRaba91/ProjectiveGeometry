from pathlib import Path

import cv2

from projective_geometry.camera import Camera
from projective_geometry.draw import Color
from projective_geometry.entrypoints.utils import BORDER_SIZE, PROJECT_LOCATION


def homography_from_image_registration_demo(
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
