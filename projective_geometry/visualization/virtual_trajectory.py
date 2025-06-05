import cv2
import numpy as np

from projective_geometry.camera.camera import Camera
from projective_geometry.camera.camera_params import CameraParams
from projective_geometry.camera.camera_pose import CameraPose
from projective_geometry.draw.colors import Color
from projective_geometry.draw.image_size import BASE_IMAGE_SIZE
from projective_geometry.pitch_template.basketball_template import (
    BasketballCourtTemplate,
)
from projective_geometry.projection.projectors import project_pitch_template


def generate_video_virtual_trajectory_camera(video_path: str):
    video_length = 5  # seconds
    fps = 25  # frames per second
    num_frames = video_length * fps
    basketball_court = BasketballCourtTemplate()
    f_start, f_end = 900, 700
    tx_start, tx_end = 5, -5
    ty = 13
    tz = 9
    rx = -128
    ry = 0
    rz = 180
    txs = np.linspace(tx_start, tx_end, num_frames)
    fs = np.linspace(f_start, f_end, num_frames)
    image_size = BASE_IMAGE_SIZE
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter.fourcc("M", "J", "P", "G"),
        fps,
        (image_size.width, image_size.height),
    )
    for tx, f in zip(txs, fs):
        frame = np.zeros((image_size.height, image_size.width, 3), dtype=np.uint8)
        camera = Camera.from_camera_params(
            camera_params=CameraParams(
                camera_pose=CameraPose(tx=tx, ty=ty, tz=tz, roll=rx, tilt=ry, pan=rz),
                focal_length=f,
            ),
            image_size=image_size,
        )
        frame = project_pitch_template(
            pitch_template=basketball_court,
            camera=camera,
            image_size=image_size,
            frame=frame,
            thickness=12,
            color=Color.BLUE,
        )
        out.write(frame)
    out.release()
