from abc import ABC, abstractmethod

import numpy as np

from projective_geometry.physics_engine.config import PhysicsConfig
from projective_geometry.physics_engine.state import pack_state, unpack_state


class SimEvent(ABC):
    terminal = True
    direction = 0

    @abstractmethod
    def __call__(self, t, s, sim_config: PhysicsConfig):
        pass

    @abstractmethod
    def handle(self, t, s, sim_config: PhysicsConfig):
        pass


class EndEvent(SimEvent):
    terminal = True
    direction = -1

    def __call__(self, t, s, sim_config: PhysicsConfig):
        return sim_config.t_end - t

    def handle(self, t, s, sim_config: PhysicsConfig):
        raise StopIteration


class ObjAtRestEvent(SimEvent):
    terminal = True
    direction = -1

    def __call__(self, t, s, sim_config: PhysicsConfig):
        _, vel, _ = unpack_state(s)
        return np.linalg.norm(vel) - sim_config.min_vel

    def handle(self, t, s, sim_config: PhysicsConfig):
        raise StopIteration


class GroundEvent(SimEvent):
    terminal = True
    direction = -1

    def __call__(self, t, s, sim_config: PhysicsConfig):
        pos, vel, _ = unpack_state(s)
        if vel[-1] < 0 and np.abs(vel[-1]) > sim_config.min_vel:
            return pos[2] - sim_config.ball_radius
        return 1.0

    def handle(self, t, s, sim_config: PhysicsConfig):
        pos, vel, ang_vel = unpack_state(s)
        if vel[-1] < 0:
            vel[-1] = -sim_config.coef_restitution * vel[-1]
            if np.abs(vel[-1]) < sim_config.min_vel:
                vel[-1] = 0.0
            s = pack_state(pos, vel, ang_vel)
        return t, s
