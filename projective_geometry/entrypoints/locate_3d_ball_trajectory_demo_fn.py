import cv2
import numpy as np

from projective_geometry.camera import Camera, CameraParams, CameraPose
from projective_geometry.draw.image_size import BASE_IMAGE_SIZE
from projective_geometry.entrypoints.utils import PROJECT_LOCATION
from projective_geometry.physics_engine.state import pack_state
from projective_geometry.physics_engine.trajectory import compute_trajectory, project_ball_trajectory
from projective_geometry.physics_engine.utils import corrupt_ellipse, draw_trajectory_comparison
from projective_geometry.physics_engine.cost_functions import get_cost_config
from projective_geometry.physics_engine.optimization import estimate_trajectory
from projective_geometry.physics_engine.config import PhysicsConfig


def locate_3d_ball_trajectory_demo(
    pos0_x: float = -54,
    pos0_y: float = -8,
    pos0_z: float = 0,
    vel0_x: float = 110,
    vel0_y: float = 10,
    vel0_z: float = 65,
    ang0_x: float = 0,
    ang0_y: float = 0,
    ang0_z: float = 20,
    cost_type: str = "algebraic",
    minimization_type: str = "convex",
    std: float = 2.0,
    prob_miss: float = 0.1,
    t_start: float = 0.0,
    t_end: float = 4.0,
    delta_t: float = 0.1,
):
    """
    Run trajectory optimization with noisy observations.
    
    Args:
        pos0_x, pos0_y, pos0_z: Initial position components
        vel0_x, vel0_y, vel0_z: Initial velocity components
        ang0_x, ang0_y, ang0_z: Initial angular velocity components
        cost_type: Cost function type ('algebraic', 'sampson', 'l2', 'yolo')
        minimization_type: Optimization type ('convex' or 'non_convex')
        std: Standard deviation for noise
        prob_miss: Probability of missing an observation
        t_start: Start time
        t_end: End time
        delta_t: Time step
    """
    # Set up initial state
    initial_pos = np.array([pos0_x, pos0_y, pos0_z])
    initial_vel = np.array([vel0_x, vel0_y, vel0_z])
    initial_ang_vel = np.array([ang0_x, ang0_y, ang0_z])
    
    # Create cameras
    # 1.Broadcast camera
    f, tx, ty, tz, rx, ry, rz = 265, 0, -28, 33, -157, 0, 0
    H = Camera.full_homography_from_camera_params(
        camera_params=CameraParams(
            camera_pose=CameraPose(tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz),
            focal_length=f,
        ),
        image_size=BASE_IMAGE_SIZE,
    )
    # 2. Birdseye camera
    f2, tx2, ty2, tz2, rx2, ry2, rz2 = 250, 0, 0, 30, -180, 0, 0
    H2 = Camera.full_homography_from_camera_params(
        camera_params=CameraParams(
            camera_pose=CameraPose(tx=tx2, ty=ty2, tz=tz2, rx=rx2, ry=ry2, rz=rz2),
            focal_length=f2,
        ),
        image_size=BASE_IMAGE_SIZE,
    )
    
    # Generate ground truth trajectory
    s = pack_state(initial_pos, initial_vel, initial_ang_vel)
    num_t = int((t_end - t_start) / delta_t + 1)
    ts = np.arange(0, num_t) * (t_end - t_start) / (num_t - 1)
    
    sim_config = PhysicsConfig()
    gt_trajectory_3d_positions = compute_trajectory(s, ts, sim_config)
    gt_trajectory_2d_ellipses = project_ball_trajectory(
        gt_trajectory_3d_positions, H, sim_config.ball_radius
    )
    
    # Get cost configuration
    cost_config = get_cost_config(cost_type)
    
    # Add noise to observations
    stds = np.ones(5) * std
    random_state = np.random.RandomState(seed=42)
    noisy_trajectory_ellipse_observations = {}
    noisy_trajectory_observations = {}
    for key, conic in gt_trajectory_2d_ellipses.items():
        noisy_conic = corrupt_ellipse(conic=conic, stds=stds, random_state=random_state)
        noisy_trajectory_ellipse_observations[key] = noisy_conic
        if random_state.rand() > prob_miss:
            observation = cost_config['observation_extractor'](noisy_conic)
            noisy_trajectory_observations[key] = observation
        else:
            noisy_trajectory_observations[key] = None
    # Define cost function for optimization
    def cost_function(s_opt):
        return cost_config['trajectory_cost_fn'](
            s_opt, ts, noisy_trajectory_observations, H, cost_config['cost_fn'], sim_config
        )
    
    # Run optimization
    s0 = np.concatenate([
        s[:2] + random_state.uniform(-5, 5, size=s[:2].shape),
        s[2:3] + random_state.uniform(0, 5, size=s[2:3].shape),
        s[3:6] + random_state.uniform(-10, 10, size=s[3:6].shape),
        s[6:9] + random_state.uniform(-1, 1, size=s[6:9].shape),
    ])
    result = estimate_trajectory(s0, cost_function, minimization_type)
    est_trajectory_3d_positions = compute_trajectory(result, ts, sim_config)
    
    print("Optimized state:")
    print(result)

    # Draw trajectory comparison
    template_image_path = PROJECT_LOCATION / "results/SoccerPitchTemplate.png"
    draw_trajectory_comparison(
        gt_trajectory_3d_positions=gt_trajectory_3d_positions,
        est_trajectory_3d_positions=est_trajectory_3d_positions,
        noisy_trajectory_ellipse_observations=noisy_trajectory_ellipse_observations,
        H=H,
        H2=H2,
        template_image_path=template_image_path,
        output_prefix="locate_3d_ball_trajectory"
    )

locate_3d_ball_trajectory_demo()