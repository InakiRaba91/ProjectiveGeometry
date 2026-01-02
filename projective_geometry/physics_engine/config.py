from dataclasses import dataclass
from functools import cached_property
from typing import Tuple

import numpy as np


@dataclass
class PhysicsConfig:
    # ball
    ball_circumference: float = 29.5 / 36.0
    ball_mass_kgs: float = 0.62

    # physics parameters
    air_temperature_c: float = 20.0
    air_pressure_mb: float = 1013.25
    r_dry: float = 287.05
    gravity_yds_per_sec2: float = 10.72
    mu_friction: float = 0.1
    quadratic_drag_coeff: float = 0.47
    magnus_lift_coeff: float = 0.1
    coef_restitution: float = 0.7
    min_vel: float = 0.1

    # simulation parameters
    t_start: float = 0.0
    t_end: float = 4.0
    time_step: float = 0.1
    eps: float = 1e-3

    @cached_property
    def ball_radius(self):
        return self.ball_circumference / (2 * np.pi)

    @cached_property
    def ball_area(self):
        return np.pi * self.ball_radius**2

    @cached_property
    def air_pressure_pascals(self):
        return self.air_pressure_mb * 100

    @cached_property
    def air_temperature_kelvin(self):
        return self.air_temperature_c + 273.15

    @cached_property
    def rho_air_kg_m3(self):
        return self.air_pressure_pascals / (self.r_dry * self.air_temperature_kelvin)

    @cached_property
    def rho_air_kg_yard3(self):
        return self.rho_air_kg_m3 / (0.9144**3)

    @cached_property
    def air_ball_const(self):
        return 0.5 * self.rho_air_kg_yard3 * self.ball_area


@dataclass
class IVPConfig:
    """Configuration for inverse problem optimization"""

    # Optimization bounds (x, y, z, vx, vy, vz, wx, wy, wz)
    bounds: Tuple[Tuple[float, float], ...] = (
        (-60, 60),  # x-position bounds
        (-40, 40),  # y-position bounds
        (0, 2),  # z-position bounds (height)
        (-120, 120),  # vx velocity bounds
        (-120, 120),  # vy velocity bounds
        (0, 100),  # vz velocity bounds
        (-5, 5),  # wx angular velocity bounds
        (-5, 5),  # wy angular velocity bounds
        (-40, 40),  # wz angular velocity bounds
    )

    # Optimization method
    method: str = "L-BFGS-B"

    # Optimization tolerances
    ftol: float = 1e-12  # Function tolerance
    gtol: float = 1e-10  # Gradient tolerance

    # Optimization limits
    maxiter: int = 10000  # Maximum iterations
    maxfun: int = 15000  # Maximum function evaluations


@dataclass
class DiffEvolConfig:
    """Configuration for differential evolution optimization"""

    # Optimization bounds (x, y, z, vx, vy, vz, wx, wy, wz)
    bounds: Tuple[Tuple[float, float], ...] = (
        (-60, 60),  # x-position bounds
        (-40, 40),  # y-position bounds
        (0, 2),  # z-position bounds (height)
        (-120, 120),  # vx velocity bounds
        (-120, 120),  # vy velocity bounds
        (0, 100),  # vz velocity bounds
        (-5, 5),  # wx angular velocity bounds
        (-5, 5),  # wy angular velocity bounds
        (-40, 40),  # wz angular velocity bounds
    )

    # Random seed for reproducibility
    seed: int = 42

    # Maximum iterations
    maxiter: int = 50

    # Absolute tolerance for convergence
    atol: float = 1e-4

    # Relative tolerance for convergence
    tol: float = 0.01

    # Whether to use L-BFGS-B polishing
    polish: bool = True

    # Number of workers for parallelization
    workers: int = 1
