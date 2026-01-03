from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize

from projective_geometry.physics_engine.config import DiffEvolConfig, IVPConfig


def estimate_trajectory(
    s0: np.ndarray,
    cost_function: Callable[[np.ndarray], float],
    minimization_type: str = "convex",
) -> np.ndarray:
    """
    Estimate the initial state by optimizing to minimize the cost function.

    Args:
        s0: Initial guess for the state (used only for convex optimization)
        cost_function: Cost function to minimize, takes state array as input
        minimization_type: Type of optimization - "convex" or "non_convex"

    Returns:
        np.ndarray: Optimized state vector

    Raises:
        ValueError: If minimization_type is not recognized
    """
    if minimization_type == "convex":
        ivp_config = IVPConfig()
        result = minimize(
            cost_function,
            s0,
            method=ivp_config.method,
            bounds=ivp_config.bounds,
            options={
                "ftol": ivp_config.ftol,
                "gtol": ivp_config.gtol,
                "maxiter": ivp_config.maxiter,
                "maxfun": ivp_config.maxfun,
            },
        )
        return result.x

    elif minimization_type == "non_convex":
        diff_eval_config = DiffEvolConfig()
        result = differential_evolution(
            cost_function,
            diff_eval_config.bounds,
            seed=diff_eval_config.seed,
            maxiter=diff_eval_config.maxiter,
            atol=diff_eval_config.atol,
            tol=diff_eval_config.tol,
            polish=diff_eval_config.polish,
            workers=diff_eval_config.workers,
        )
        return result.x

    else:
        raise ValueError(f"Unknown minimization_type '{minimization_type}'. Valid options: 'convex', 'non_convex'")
