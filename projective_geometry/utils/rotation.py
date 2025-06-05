import numpy as np


def rotation_matrix_from_angles(rx, ry, rz):
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
