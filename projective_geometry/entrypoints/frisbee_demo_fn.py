from pathlib import Path

import cv2
import numpy as np

from projective_geometry.entrypoints.utils import (
    BACKGROUND_COLOR,
    FRISBEE_DISTANCE_TO_PINHOLE,
    IMG_TEMPLATE_FPATH,
    PINHOLE_SVG,
    PROJECT_LOCATION,
    RADIUS_FRISBEE,
    UNIT,
    generate_frame,
    label_conic_type,
)


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
