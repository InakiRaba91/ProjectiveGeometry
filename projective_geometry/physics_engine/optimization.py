from typing import Callable

import numpy as np
from scipy.optimize import minimize, differential_evolution

from projective_geometry.physics_engine.config import IVPConfig, DiffEvolConfig


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
        config = IVPConfig()
        result = minimize(
            cost_function,
            s0,
            method=config.method,
            bounds=config.bounds,
            options={
                'ftol': config.ftol,
                'gtol': config.gtol,
                'maxiter': config.maxiter,
                'maxfun': config.maxfun,
            },
        )
        return result.x
    
    elif minimization_type == "non_convex":
        config = DiffEvolConfig()
        result = differential_evolution(
            cost_function,
            config.bounds,
            seed=config.seed,
            maxiter=config.maxiter,
            atol=config.atol,
            tol=config.tol,
            polish=config.polish,
            workers=config.workers,
        )
        return result.x
    
    else:
        raise ValueError(f"Unknown minimization_type '{minimization_type}'. Valid options: 'convex', 'non_convex'")
