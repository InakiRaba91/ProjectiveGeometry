import random
from typing import Any, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np

from projective_geometry.draw import Color
from projective_geometry.draw.image_size import BASE_IMAGE_SIZE, ImageSize
from projective_geometry.geometry import Ellipse, EllipseArc, Line, Point2D
from projective_geometry.geometry.line_segment import LineSegment
from projective_geometry.pitch_template.pitch_dims import PitchDims

T = TypeVar("T", bound=Union[Point2D, Line, LineSegment, Ellipse, EllipseArc])


class PitchTemplate(object):
    """Base Pitch Template

    Args:
        pitch_dims: PitchDims of the template

    Attributes:
        pitch_dims: PitchDims of the template
    """

    def __init__(self, pitch_dims: PitchDims):
        self.pitch_dims = pitch_dims
        self.geometric_features = self._geometric_features()
        self.keypoints = self._keypoints()

    def pitch_template_to_pitch_image(
        self,
        geometric_feature: T,
        image_size: ImageSize = BASE_IMAGE_SIZE,
    ) -> T:
        """Map geometric feature (Point, Line, LineSegment, Ellipse, EllipseArc)
         in pitch template to point in pitch image

        Pitch template:
            center: (0, 0)
            xrange: (-pitch_dims.width/2, pitch_dims.width/2)
            yrange: (-pitch_dims.height/2, pitch_dims.height/2)
        Pitch image:
            center: (image_size.width/2, image_shape.height/2)
            xrange: (0, image_size.width/2)
            yrange: (0, image_size.height/2)

        Args:
            geometric_feature: geometric feature (Point, Line, LineSegment, Ellipse, EllipseArc) pitch image
            image_size: ImageSize of the image to which the geometric feature belongs to

        Returns:
            geometric feature (Point, Line, LineSegment, Ellipse, EllipseArc) pitch image
        """
        # create translation vector that points to the centre of the pitch (real world)
        pt_shift = Point2D(x=self.pitch_dims.width / 2, y=self.pitch_dims.height / 2)  # world

        # translate the real_world_feature to make the pitch centre the origin
        real_world_feature = geometric_feature + pt_shift  # world

        # create a ratio between the proposed image size and real world pitch dimensions
        # Note: the ratio is maintained as yards to conform to the "scale" operation in geometric feature
        scaling_2d = Point2D(
            x=image_size.width / self.pitch_dims.width,  # image
            y=image_size.height / self.pitch_dims.height,  # image
        )  # world

        # use the ratio to move the geometric feature from the image domain to the real world domain
        image_feature = real_world_feature.scale(pt=scaling_2d)  # image

        return cast(T, image_feature)

    def pitch_image_to_pitch_template(
        self,
        geometric_feature: Union[Point2D, Line, LineSegment, Ellipse, EllipseArc],
        image_size: ImageSize = BASE_IMAGE_SIZE,
    ) -> Union[Point2D, Line, LineSegment, Ellipse, EllipseArc]:
        """Map geometric feature (Point, Line, LineSegment, Ellipse, EllipseArc)
         in pitch image to point in pitch image template

        Pitch template:
            center: (0, 0)
            xrange: (-pitch_dims.width/2, pitch_dims.width/2)
            yrange: (-pitch_dims.height/2, pitch_dims.height/2)
        Pitch image:
            center: (image_size.width/2, image_shape.height/2)
            xrange: (0, image_size.width/2)
            yrange: (0, image_size.height/2)

        Args:
            geometric_feature: geometric feature (Point, Line, LineSegment, Ellipse, EllipseArc) pitch image
            image_size: ImageSize of the image to which the geometric feature belongs to

        Returns:
            geometric feature (Point, Line, LineSegment, Ellipse, EllipseArc) pitch template
        """
        # create a ratio between real world pitch dimensions and the proposed image size
        # Note: the ratio is maintained as pixels to conform to the "scale" operation in geometric feature
        scaling_2d = Point2D(
            x=self.pitch_dims.width / image_size.width,  # world
            y=self.pitch_dims.height / image_size.height,  # world
        )  # switch to image

        # use the ratio to move the geometric feature from the image domain to the real world domain
        real_world_feature = geometric_feature.scale(pt=scaling_2d)  # world

        # create translation vector that points to the centre of the pitch (real world)
        pt_shift = -Point2D(x=self.pitch_dims.width / 2, y=self.pitch_dims.height / 2)  # world

        # translate the real_world_feature to make the pitch centre the origin
        real_world_feature = real_world_feature + pt_shift  # world

        return real_world_feature

    def draw(
        self,
        image_size: ImageSize = BASE_IMAGE_SIZE,
        color: Optional[Tuple[Any, ...]] = None,
        thickness: int = 3,
    ) -> np.ndarray:
        """Draw pitch template in an image with given size

        Args:
            image_size: ImageSize indicating shape of the image where template will be displayed on
            color: BGR int tuple indicating the color of the geometric features
            thickness: int thickness of the drawn geometric features

        Returns:
            np.uint8 array image
        """
        # We need to shift and scale because the pitch is centered at (0, 0), whereas
        # the image is centered at (image_size.width/2, image_shape.height/2)
        # conversion to int is not strictly necessary, but mypy complains otherwise
        img = np.zeros((int(image_size.height), int(image_size.width), 3), dtype=np.uint8)
        for idx, geometric_feature in enumerate(self.geometric_features):
            geometric_feature_image = self.pitch_template_to_pitch_image(
                geometric_feature=geometric_feature, image_size=image_size
            )
            color_feature: Tuple[Any, ...] = random.choice(Color.get_colors()) if color is None else color  # ignore
            geometric_feature_image.draw(img, color=color_feature, thickness=thickness)

        return img

    def draw_keypoints(
        self,
        image_size: ImageSize = BASE_IMAGE_SIZE,
        color: Optional[Tuple[Any, ...]] = Color.WHITE,
        radius: int = 3,
        thickness: int = -1,
    ) -> np.ndarray:
        img = np.zeros((int(image_size.height), int(image_size.width), 3), dtype=np.uint8)
        for keypoint in self.keypoints:
            keypoint_image = self.pitch_template_to_pitch_image(geometric_feature=keypoint, image_size=image_size)
            keypoint_image.draw(img, color=color, radius=radius, thickness=thickness)
        return img

    def _keypoints(self) -> List[Point2D]:
        """Computes all geometric relevant keypoints identifiable in the template, which are usually defined as
        the intersections between the different geometric features

        Args: None

        Returns:
            Tuple of Point
        """
        raise NotImplementedError("Must override keypoints")

    def _geometric_features(self):
        """Compute all geometric features (line segments, ellipses, ellipse arcs and keypoints)
        that define the template

        Args: None

        Returns:
            Tuple of geometric features (Point, LineSegment, Ellipse, EllipseArc)
        """
        raise NotImplementedError("Must override __set_geometric_features")
