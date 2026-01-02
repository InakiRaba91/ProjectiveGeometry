import inspect
import types
from projective_geometry.entrypoints.frisbee_demo_fn import frisbee_demo
from projective_geometry.entrypoints.homography_from_point_correspondences_demo_fn import (
    homography_from_point_correspondences_demo,
)
from projective_geometry.entrypoints.homography_from_line_correspondences_demo_fn import (
    homography_from_line_correspondences_demo,
)
from projective_geometry.entrypoints.homography_from_ellipse_correspondences_demo_fn import (
    homography_from_ellipse_correspondences_demo,
)
from projective_geometry.entrypoints.homography_from_correspondences_demo_fn import (
    homography_from_correspondences_demo,
)
from projective_geometry.entrypoints.homography_from_image_registration_demo_fn import (
    homography_from_image_registration_demo,
)
from projective_geometry.entrypoints.focal_length_from_orthogonal_vanishing_points_demo_fn import (
    focal_length_from_orthogonal_vanishing_points_demo,
)
from projective_geometry.entrypoints.intrinsic_from_three_planes_demo_fn import (
    intrinsic_from_three_planes_demo,
)
from projective_geometry.entrypoints.camera_pose_from_four_points_demo_fn import (
    camera_pose_from_four_points_demo,
)
from projective_geometry.entrypoints.locate_ball_3d_demo_fn import locate_ball_3d_demo
from projective_geometry.entrypoints.locate_3d_ball_trajectory_demo_fn import locate_3d_ball_trajectory_demo


def register_entrypoints(module, decorator):
    current_func = inspect.currentframe().f_code
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, types.FunctionType) and obj.__code__ is not current_func:
            setattr(module, name, decorator(obj))

