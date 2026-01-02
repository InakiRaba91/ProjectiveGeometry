
import numpy as np
from projective_geometry.physics_engine.config import PhysicsConfig
from projective_geometry.physics_engine.state import pack_state, unpack_state


def get_gravity_acc(sim_config: PhysicsConfig) -> float:
    return np.array([0, 0, -sim_config.gravity_yds_per_sec2])

def get_friction_acc(vel, sim_config: PhysicsConfig) -> float:
    v_xy = vel[:2]
    v_norm = np.linalg.norm(v_xy)
    if v_norm == 0:
        return np.array([0, 0, 0])
    a_fric_mag = sim_config.mu_friction * sim_config.gravity_yds_per_sec2
    a_fric_xy = -a_fric_mag * (v_xy / v_norm)
    return np.array([a_fric_xy[0], a_fric_xy[1], 0])

def get_drag_acc(vel, sim_config: PhysicsConfig) -> float:
    # F = 0.5 * rho * v^2 * Cd * A
    # a = F/m
    v_norm = np.linalg.norm(vel)
    if v_norm < sim_config.eps:
        return np.array([0, 0, 0])
    drag_force_mag = sim_config.air_ball_const * sim_config.quadratic_drag_coeff * (v_norm ** 2)
    drag_acc_mag = drag_force_mag / sim_config.ball_mass_kgs
    drag_acc = -drag_acc_mag * (vel / v_norm)
    return drag_acc  

def get_magnus_acc(vel, omega, sim_config: PhysicsConfig) -> float:
    return sim_config.magnus_lift_coeff * sim_config.air_ball_const * np.cross(omega, vel) / sim_config.ball_mass_kgs


def motion_eq(t, s, sim_config: PhysicsConfig) -> np.ndarray:
    pos, vel, omega = unpack_state(s)
    
    if pos[2] <= sim_config.ball_radius and vel[2] < sim_config.eps:
        gravity_acc = np.array([0, 0, 0])
        friction_acc = np.array([0, 0, 0])
        if np.abs(vel[2]) < sim_config.eps:
            friction_acc = get_friction_acc(vel, sim_config)
    else:
        gravity_acc = get_gravity_acc(sim_config)
        friction_acc = np.array([0, 0, 0])

    drag_acc = get_drag_acc(vel, sim_config)
    magnus_acc = get_magnus_acc(vel, omega, sim_config)
    acc = gravity_acc + friction_acc + drag_acc + magnus_acc
    ang_acc = np.array([0, 0, 0])
    
    return pack_state(vel, acc, ang_acc)
