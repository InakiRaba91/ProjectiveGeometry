import logging
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSlider,
    QVBoxLayout,
    QWidget,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraVisualiser(QMainWindow):
    """Basic video player with frame seeking."""

    def __init__(self, screen_height: int):
        super().__init__()
        self.setWindowTitle("Video Player with Frame Controls")

        # Add keyboard shortcut handling
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.resize(800, 300)
        # Layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Video Label (for displaying frames)
        self.label_camera_view = QLabel()
        self.label_camera_view.setScaledContents(True)
        self.label_frame_view = QLabel()
        self.label_frame_view.setScaledContents(True)

        # Scrubbing Slider
        self.keys = ["f", "tx", "ty", "tz", "rx", "ry", "rz"]
        self.sliders = {key: QSlider(Qt.Orientation.Horizontal) for key in self.keys}
        self.slider_labels = {key: QLabel() for key in self.keys}
        self.f_view, self.tx_view, self.ty_view, self.tz_view, self.rx_view, self.ry_view, self.rz_view = (
            480,
            0,
            -30,
            45,
            -130,
            0,
            0,
        )

        self.sliders["f"].setRange(200, 1000)
        self.sliders["f"].setValue(350)
        self.sliders["tx"].setRange(-100, 100)
        self.sliders["tx"].setValue(-5)
        self.sliders["ty"].setRange(-100, 100)
        self.sliders["ty"].setValue(5)
        self.sliders["tz"].setRange(0, 50)
        self.sliders["tz"].setValue(15)
        self.sliders["rx"].setRange(-180, 180)
        self.sliders["rx"].setValue(-170)
        self.sliders["ry"].setRange(-180, 180)
        self.sliders["ry"].setValue(10)
        self.sliders["rz"].setRange(-180, 180)
        self.sliders["rz"].setValue(170)

        self.frame = cv2.imread("data/BaseketballCourtTemplate.png")
        self.pitch_width, self.pitch_height = 94 / 3, 50 / 3
        self.frame_height, self.frame_width, _ = self.frame.shape
        self.frames_layout = QHBoxLayout()
        self.frames_layout.addWidget(self.label_camera_view)
        self.frames_layout.addWidget(self.label_frame_view)
        self.main_layout.addLayout(self.frames_layout)
        for key, slider in self.sliders.items():
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(slider)
            slider_layout.addWidget(self.slider_labels[key])
            self.main_layout.addLayout(slider_layout)
            self.update_slider_label(slider.value(), key)
            slider.valueChanged.connect(self.update_slider_label)
            slider.sliderReleased.connect(self.display)

        self.set_video_dimensions(screen_height)
        self.display()

    def set_video_dimensions(self, screen_height: int, aspect_ratio=16 / 9):
        """Calculate the desired dimensions for the video_label."""
        video_height = int(screen_height * 0.4)
        video_width = int(video_height * aspect_ratio)
        self.video_width = video_width
        self.video_height = video_height
        self.label_camera_view.setFixedSize(self.video_width, self.video_height)
        self.label_frame_view.setFixedSize(self.video_width, self.video_height)

    def update_slider_label(self, value, key=None):
        """Update the label showing the current value of the slider."""
        if key is None:
            slider = self.sender()
            key = next(k for k, v in self.sliders.items() if v == slider)
        self.slider_labels[key].setText(f"{key}: {value}")
        self.display()

    def display(self):
        self.display_camera_view()
        self.display_frame_view()

    def display_frame_view(self):
        """Render a frame on the video label, overlaying tracking data."""
        f, tx, ty, tz, rx, ry, rz = [self.sliders[key].value() for key in self.keys]
        R = self.rotation_matrix_from_angles(rx, ry, rz)
        K = np.array([[f, 0, self.frame_width // 2], [0, f, self.frame_height // 2], [0, 0, 1]])
        T = np.array([[tx], [ty], [tz]])
        E = np.concatenate((R.T, -R.T.dot(T)), axis=1)
        H = K.dot(E)
        # H = Camera.from_camera_params(
        #     camera_params=CameraParams(
        #         camera_pose=CameraPose(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz),
        #         focal_length=f,
        #     ),
        #     image_size=ImageSize(width=self.frame_width, height=self.frame_height),
        # )
        K_pitch_image_to_pitch_template = np.array(
            [
                [self.pitch_width / self.frame_width, 0, 0, -self.pitch_width / 2.0],
                [0, self.pitch_height / self.frame_height, 0, -self.pitch_height / 2.0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )

        # create a chained homography projection that maps from BEV camera -> desired camera homography
        H_chained = H.dot(K_pitch_image_to_pitch_template)
        H_chained = H_chained[:, [0, 1, 3]]
        frame = cv2.warpPerspective(src=self.frame, M=H_chained, dsize=(self.frame_width, self.frame_height))

        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label_frame_view.setPixmap(pixmap)

    def rotation_matrix_from_angles(self, rx, ry, rz):
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
        return Rz.dot(Ry).dot(Rx)

    def get_visible_edges(self, points_3d_cube, faces_cube, indices_edges_cube, camera_location, ref_location):
        visible_edges = []
        for edges_face in faces_cube:
            indices_points_face = []
            for edge_idx in edges_face:
                i, j = indices_edges_cube[edge_idx]
                if i not in indices_points_face:
                    indices_points_face.append(i)
                if j not in indices_points_face:
                    indices_points_face.append(j)
            face_center = np.mean(points_3d_cube[:, indices_points_face], axis=1)[:, None]
            face_center = face_center[:3] / face_center[3]
            face_normal = face_center - camera_location
            face_ref = face_center - ref_location
            if np.dot(face_normal.T, face_ref) < 0:
                for edge_idx in edges_face:
                    if edge_idx not in visible_edges:
                        visible_edges.append(edge_idx)
        return visible_edges

    def draw_camera(self, H_view, T_view, frame):
        f, tx, ty, tz, rx, ry, rz = [self.sliders[key].value() for key in self.keys]
        R = self.rotation_matrix_from_angles(rx, ry, rz)
        T = np.array([[tx], [ty], [tz]])
        E = np.concatenate((np.concatenate((R, T), axis=1), np.array([[0, 0, 0, 1]])), axis=0)
        size = 10
        z_film = 0.5 * (f - self.sliders["f"].maximum()) / (self.sliders["f"].minimum() - self.sliders["f"].maximum())
        points_3d_cube = size * np.array(
            [
                # cube
                [-1, -1, -1],
                [-1, 1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, 1],
                # film
                [-1, -1, z_film],
                [-1, 1, z_film],
                [1, -1, z_film],
                [1, 1, z_film],
            ]
        )
        idx_vertices_edges_cube = [
            (0, 1),
            (1, 3),
            (3, 2),
            (2, 0),
            (4, 5),
            (5, 7),
            (7, 6),
            (6, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        idx_edges_faces_cube = [
            (0, 1, 3, 2),  # z=-1
            (4, 5, 7, 6),  # z=1
            (3, 7, 8, 10),  # y=-1
            (1, 5, 9, 11),  # y=1
            (0, 4, 8, 9),  # x=-1
            (2, 6, 10, 11),  # x=1
        ]
        idx_vertices_edges_film = [
            (8, 9),
            (9, 11),
            (11, 10),
            (10, 8),
        ]
        point_3d_pinhole = size * np.array([[0, 0, 1]])

        # project
        points_3d_cube = E.dot(np.concatenate([points_3d_cube, np.ones((len(points_3d_cube), 1))], axis=1).T)
        points_2d_cube = H_view.dot(points_3d_cube)
        point_3d_pinhole = E.dot(np.concatenate([point_3d_pinhole, np.ones((1, 1))], axis=1).T)
        point_2d_pinhole = H_view.dot(point_3d_pinhole)

        # normalize
        points_2d_cube = points_2d_cube[:2] / points_2d_cube[2]
        point_2d_pinhole = point_2d_pinhole[:2] / point_2d_pinhole[2]

        # visible faces
        visible_edges = self.get_visible_edges(points_3d_cube, idx_edges_faces_cube, idx_vertices_edges_cube, T, T_view)

        # draw cube
        indices_visible_edges_cube, indices_covered_edges_cube = [], []
        for idx in range(len(idx_vertices_edges_cube)):
            if idx in visible_edges:
                indices_visible_edges_cube.append(idx_vertices_edges_cube[idx])
            else:
                indices_covered_edges_cube.append(idx_vertices_edges_cube[idx])
        colors = (
            [(0, 0, 255)] * len(indices_visible_edges_cube)
            + [(255, 0, 255)] * len(indices_covered_edges_cube)
            + [(206, 209, 0)] * len(idx_vertices_edges_film)
        )
        thicknesses = (
            [3] * len(indices_visible_edges_cube) + [1] * len(indices_covered_edges_cube) + [2] * len(idx_vertices_edges_film)
        )
        for (i, j), color, thickness in zip(
            indices_visible_edges_cube + indices_covered_edges_cube + idx_vertices_edges_film, colors, thicknesses
        ):
            cv2.line(
                frame,
                (int(points_2d_cube[0, i]), int(points_2d_cube[1, i])),
                (int(points_2d_cube[0, j]), int(points_2d_cube[1, j])),
                color,
                thickness,
            )
        cv2.circle(frame, (int(point_2d_pinhole[0]), int(point_2d_pinhole[1])), 5, (0, 0, 255), 2)
        return frame

    def display_camera_view(self):
        f_view = self.f_view
        tx_view, ty_view, tz_view = self.tx_view, self.ty_view, self.tz_view
        rx_view, ry_view, rz_view = self.rx_view, self.ry_view, self.rz_view
        K_view = np.array([[f_view, 0, self.frame_width // 2], [0, f_view, self.frame_height // 2], [0, 0, 1]])
        R_view = self.rotation_matrix_from_angles(rx_view, ry_view, rz_view)
        T_view = np.array([[tx_view], [ty_view], [tz_view]])
        E_view = np.concatenate((R_view.T, -R_view.T.dot(T_view)), axis=1)
        H_view = K_view.dot(E_view)

        # draw pitch
        K_pitch_image_to_pitch_template = np.array(
            [
                [self.pitch_width / self.frame_width, 0, 0, -self.pitch_width / 2.0],
                [0, self.pitch_height / self.frame_height, 0, -self.pitch_height / 2.0],
                [0, 0, 1.0, 0],
                [0, 0, 0, 1.0],
            ]
        )

        # create a chained homography projection that maps from BEV camera -> desired camera homography
        H_chained = H_view.dot(K_pitch_image_to_pitch_template)
        H_chained = H_chained[:, [0, 1, 3]]
        frame = cv2.warpPerspective(src=self.frame, M=H_chained, dsize=(self.frame_width, self.frame_height))
        frame = self.draw_camera(H_view=H_view, T_view=T_view, frame=frame)

        # Convert frame to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.label_camera_view.setPixmap(pixmap)


def show_camera_visualisation():
    """Launch the video player."""
    app = QApplication(sys.argv)
    screen_size = app.primaryScreen().size()
    player = CameraVisualiser(screen_height=screen_size.height())
    player.show()
    logger.debug("Started video player.")
    sys.exit(app.exec())
