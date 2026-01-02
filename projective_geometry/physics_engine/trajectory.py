import numpy as np
from scipy.integrate import solve_ivp

from projective_geometry.geometry.conic import Conic
from projective_geometry.physics_engine.config import PhysicsConfig
from projective_geometry.physics_engine.events import (
    EndEvent,
    GroundEvent,
    ObjAtRestEvent,
)
from projective_geometry.physics_engine.motion_equations import motion_eq
from projective_geometry.physics_engine.state import unpack_state
from projective_geometry.projection.projectors import project_sphere


def compute_trajectory(s: np.ndarray, ts: np.ndarray, sim_config: PhysicsConfig) -> dict[float, np.ndarray]:
    pos_t = {}
    t_start, t_end = ts[0], ts[-1]
    current_t = t_start
    current_s = s.copy()

    # Create event instances
    events = [GroundEvent(), EndEvent(), ObjAtRestEvent()]

    while current_t < t_end:
        sol = solve_ivp(motion_eq, [current_t, t_end], current_s, args=(sim_config,), events=events, dense_output=True)

        # Sample trajectory for this segment
        for t in ts:
            if current_t <= t <= sol.t[-1]:
                if float(t) not in pos_t:
                    pos, _, _ = unpack_state(sol.sol(t))
                    pos_t[float(t)] = pos

        # Determine which event triggered (if any)
        event_triggered = None
        event_time = t_end

        for i, event in enumerate(events):
            if sol.t_events[i].size > 0:
                if sol.t_events[i][0] < event_time:
                    event_time = sol.t_events[i][0]
                    event_triggered = event

        # Handle the event
        if event_triggered is not None:
            current_s = sol.sol(event_time)
            try:
                current_t, current_s = event_triggered.handle(event_time, current_s, sim_config)
            except StopIteration:
                break
        else:
            # No event triggered, simulation completed
            current_t = t_end
            break

    # Fill in any remaining timesteps with last known position
    last_known_pos = None
    for t in sorted(pos_t.keys()):
        last_known_pos = pos_t[t]

    if last_known_pos is not None:
        for t in ts:
            if float(t) not in pos_t:
                pos_t[float(t)] = last_known_pos

    return pos_t


def project_ball_trajectory(
    pos_trajectory: dict[float, np.ndarray],
    H: np.ndarray,
    radius: float,
) -> dict[float, Conic]:
    ellipse_trajectory = {}
    for t, pos in pos_trajectory.items():
        ellipse_trajectory[t] = project_sphere(pos, radius, H)
    return ellipse_trajectory
