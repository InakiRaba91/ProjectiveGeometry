from typing import Callable

import numpy as np

from projective_geometry.geometry.conic import Conic
from projective_geometry.physics_engine.config import PhysicsConfig
from projective_geometry.physics_engine.trajectory import (
    compute_trajectory,
    project_ball_trajectory,
)
from projective_geometry.physics_engine.utils import get_bbox_ellipse, sample_ellipse


def cost_algebraic(ellipse: Conic, points: np.ndarray):
    """Computes algebraic cost of points wrt conic

    Args:
        ellipse: Conic
        points: Iterable of 2D points [(x, y), ...]
    Returns:
        total_cost: float
    """
    total_cost = 0.0
    for p in points:
        x, y = p
        vec = np.array([x, y, 1])
        cost = vec.T.dot(ellipse.M).dot(vec)
        total_cost += cost**2
    return total_cost


def cost_sampson(ellipse: Conic, points: np.ndarray):
    """Computes Sampson cost of points wrt conic

    Args:
        ellipse: Conic
        points: Iterable of 2D points [(x, y), ...]
    Returns:
        total_cost: float
    """
    total_cost = 0.0
    for p in points:
        x, y = p
        vec = np.array([x, y, 1])
        num = (vec.T.dot(ellipse.M).dot(vec)) ** 2
        grad = 2 * ellipse.M.dot(vec)
        denom = grad[0] ** 2 + grad[1] ** 2
        cost = 0.0
        if denom != 0:
            cost = num / denom
        total_cost += cost
    return total_cost


def cost_bbox_l2(bbox_pred: np.ndarray, bbox_obs: np.ndarray) -> float:
    """Computes L2 distance between two bounding boxes.

    The cost is the sum of L2 distances between top-left and bottom-right corners:
    e = ||tl_pred - tl_obs||^2 + ||br_pred - br_obs||^2

    Args:
        bbox_pred: Predicted bbox [tlx, tly, brx, bry]
        bbox_obs: Observed bbox [tlx, tly, brx, bry]

    Returns:
        cost: L2 distance cost
    """
    return float(np.linalg.norm(bbox_pred - bbox_obs) ** 2)


def cost_bbox_yolo(bbox_pred: np.ndarray, bbox_obs: np.ndarray, lambda_c: float = 1.0, lambda_s: float = 1.0) -> float:
    """Computes YOLO-style loss for bounding boxes.

    The cost combines center loss and size loss:
    e = lambda_c * ||c_pred - c_obs||^2 + lambda_s * (||sqrt(w_pred) - sqrt(w_obs)||^2 + ||sqrt(h_pred) - sqrt(h_obs)||^2)

    Args:
        bbox_pred: Predicted bbox [tlx, tly, brx, bry]
        bbox_obs: Observed bbox [tlx, tly, brx, bry]
        lambda_c: Weight for center loss term
        lambda_s: Weight for size loss term

    Returns:
        cost: YOLO loss
    """
    # Extract corners
    tlx_pred, tly_pred, brx_pred, bry_pred = bbox_pred
    tlx_obs, tly_obs, brx_obs, bry_obs = bbox_obs

    # Compute centers
    center_pred = np.array([(tlx_pred + brx_pred) / 2, (tly_pred + bry_pred) / 2])
    center_obs = np.array([(tlx_obs + brx_obs) / 2, (tly_obs + bry_obs) / 2])

    # Center loss
    center_loss = np.linalg.norm(center_pred - center_obs) ** 2

    # Compute dimensions
    dims_pred = np.array([brx_pred - tlx_pred, bry_pred - tly_pred])
    dims_obs = np.array([brx_obs - tlx_obs, bry_obs - tly_obs])

    # Size loss (with square root)
    size_loss = np.linalg.norm(np.sqrt(np.abs(dims_pred)) - np.sqrt(np.abs(dims_obs))) ** 2

    return float(lambda_c * center_loss + lambda_s * size_loss)


def cost_ellipse_trajectories(
    s: np.ndarray,
    ts: np.ndarray,
    observed_trajectory_points: dict[float, np.ndarray],
    H: np.ndarray,
    objective_fn: Callable[[Conic, np.ndarray], float],
    sim_config: PhysicsConfig,
) -> float:
    """Computes cost between predicted and observed ellipse trajectories.

    Args:
        s: Initial state vector
        ts: Time points
        observed_trajectory_points: Dict mapping time to observed 2D points
        H: Homography matrix
        objective_fn: Ellipse cost function taking (conic, points)
        sim_config: Physics configuration
    Returns:
        Total cost across all timesteps
    """
    trajectory_positions = compute_trajectory(s, ts, sim_config)
    trajectory_ellipses = project_ball_trajectory(trajectory_positions, H, radius=sim_config.ball_radius)

    cost = 0.0
    for t, observed_points in observed_trajectory_points.items():
        if observed_points is None:
            continue
        conic = trajectory_ellipses[t]
        cost += objective_fn(conic, observed_points)
    return cost


def cost_bbox_trajectories(
    s: np.ndarray,
    ts: np.ndarray,
    observed_trajectory_bboxes: dict[float, np.ndarray],
    H: np.ndarray,
    objective_fn: Callable[[np.ndarray, np.ndarray], float],
    sim_config: PhysicsConfig,
) -> float:
    """Computes cost between predicted and observed bounding box trajectories.

    Args:
        s: Initial state vector
        ts: Time points
        observed_trajectory_bboxes: Dict mapping time to observed bbox [tlx, tly, brx, bry]
        H: Homography matrix
        objective_fn: Bbox cost function taking (bbox_pred, bbox_obs)
        sim_config: Physics configuration

    Returns:
        Total cost across all timesteps
    """
    trajectory_positions = compute_trajectory(s, ts, sim_config)
    trajectory_ellipses = project_ball_trajectory(trajectory_positions, H, radius=sim_config.ball_radius)

    cost = 0.0
    for t, observed_bbox in observed_trajectory_bboxes.items():
        if observed_bbox is None:
            continue
        conic = trajectory_ellipses[t]
        predicted_bbox = get_bbox_ellipse(conic)
        cost += objective_fn(predicted_bbox, observed_bbox)
    return cost


# Cost function catalogue
COST_CATALOGUE = {
    "algebraic": {
        "cost_fn": cost_algebraic,
        "trajectory_cost_fn": cost_ellipse_trajectories,
        "observation_extractor": sample_ellipse,
    },
    "sampson": {
        "cost_fn": cost_sampson,
        "trajectory_cost_fn": cost_ellipse_trajectories,
        "observation_extractor": sample_ellipse,
    },
    "l2": {
        "cost_fn": cost_bbox_l2,
        "trajectory_cost_fn": cost_bbox_trajectories,
        "observation_extractor": get_bbox_ellipse,
    },
    "yolo": {
        "cost_fn": cost_bbox_yolo,
        "trajectory_cost_fn": cost_bbox_trajectories,
        "observation_extractor": get_bbox_ellipse,
    },
}


def get_cost_config(cost_type: str) -> dict:
    """Get cost function configuration for given cost type.

    Args:
        cost_type: One of 'algebraic', 'sampson', 'l2', 'yolo'

    Returns:
        Dict with 'cost_fn', 'trajectory_cost_fn', and 'observation_extractor'

    Raises:
        ValueError: If cost_type is not recognized
    """
    if cost_type not in COST_CATALOGUE:
        valid_types = ", ".join(COST_CATALOGUE.keys())
        raise ValueError(f"Unknown cost type '{cost_type}'. Valid options: {valid_types}")
    return COST_CATALOGUE[cost_type]
