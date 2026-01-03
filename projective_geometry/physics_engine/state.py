import numpy as np


def pack_state(pos, vel, ang_vel):
    return np.concatenate((pos, vel, ang_vel))


def unpack_state(s):
    return s[:3], s[3:6], s[6:9]
