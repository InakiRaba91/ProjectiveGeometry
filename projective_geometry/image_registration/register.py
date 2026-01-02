# Copyrights (C) StatsBomb Services Ltd. 2021. - All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly
# prohibited. Proprietary and confidential

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

MAX_FEATURES_ORB = 500
MATCHER_TYPE: int = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING


@dataclass
class MatchedKeypoints:
    """Class to store matched keypoints and descriptors for target and source images"""

    target_keypoints: Tuple[cv2.KeyPoint, ...]
    source_keypoints: Tuple[cv2.KeyPoint, ...]
    matches: Tuple[cv2.DMatch, ...]


class ImageRegistrator:
    def __init__(self, max_features_detector: int = MAX_FEATURES_ORB, matcher_type: int = MATCHER_TYPE):
        self._orb = cv2.ORB_create(max_features_detector)  # type: ignore
        self._matcher = cv2.DescriptorMatcher_create(matcher_type)  # type: ignore

    def register(
        self,
        target_image: np.ndarray,
        source_image: np.ndarray,
        target_mask: Optional[np.ndarray] = None,
        source_mask: Optional[np.ndarray] = None,
    ) -> Tuple[MatchedKeypoints, np.ndarray]:
        """Register two images of the same scene using a projective transform

        Note: If not enough keypoints are detected, it will return an identity matrix as the homography

        Args:
            target_image: ndarray reference image that will remain static
            source_image: ndarray image that will undergo a homography transform in order to align with target image
            target_mask: bool ndarray mask indicating valid region for keypoint detection in target image
            source_mask: bool ndarray mask indicating valid region for keypoint detection in source image

        Returns:
            matched_keypoints: MatchedKeypoints containing list of matches as well as all detected keypoints and descriptors for both images
            homography_image_registration: ndarray 3x3 homography matrix that maps homogeneous points from source image to target image
        """
        # find matches
        matched_keypoints = self._match_keypoints(
            target_image=target_image, source_image=source_image, target_mask=target_mask, source_mask=source_mask
        )
        # filter top matches to remove noisy outliers. It returns a minimum of 4 points to compute homography
        matched_keypoints = self._filter_top_matches(matched_keypoints=matched_keypoints)

        # Raise an error if it doesn't get at least 4 point
        assert (
            len(matched_keypoints.matches) >= 4
        ), f"Only {len(matched_keypoints.matches)} keypoints have been detected. A minimum of 4 is required for the algorithm to succeed"

        # compute homography
        homography = self._homography_from_matched_keypoints(matched_keypoints=matched_keypoints)
        return matched_keypoints, homography

    def _detect_keypoints(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[cv2.KeyPoint, ...]:
        """
        Detects relevant keypoints constraining the search within the mask if provided

        Args:
            image: ndarray BGR image to find keypoints in
            mask: bool ndarray mask indicating valid region for keypoint detection

        Returns:
            keypoints: Tuple[cv2.KeyPoint, ...] with keypoints to extract descriptors for
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        assert len(gray_image.shape) == 2, "Images must be in gray scale"
        # check mask has correct size and type if provided and convert to np.uint8 as required by opencv
        if isinstance(mask, np.ndarray):
            assert mask.dtype == bool, f"Mask must a boolean array (type is {mask.dtype})"
            assert gray_image.shape == mask.shape, f"Image {gray_image.shape} and mask {mask.shape} shapes must match"
            mask = mask.astype(np.uint8)

        # use ORB to detect keypoints and extract (binary) local invariant features
        return self._orb.detect(gray_image, mask)

    def _describe_keypoints(self, image: np.ndarray, keypoints: Tuple[cv2.KeyPoint, ...]) -> np.ndarray:
        """
        Extracts feature descriptors of given keypoints from image

        Args:
            image: ndarray BGR image to extract descriptors from
            keypoints: Tuple[cv2.KeyPoint, ...] with keypoints to extract descriptors for

        Returns:
            descriptive_keypoints_target_image: DescriptiveKeypoints containing keypoints and corresponding descriptors for target image
            descriptive_keypoints_source_image: DescriptiveKeypoints containing keypoints and corresponding descriptors for source image
        """
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        assert len(gray_image.shape) == 2, "Images must be in gray scale"

        # use ORB to detect keypoints and extract (binary) local invariant features
        _, descriptors = self._orb.compute(gray_image, keypoints)
        return descriptors

    def _match_keypoints(
        self,
        target_image: np.ndarray,
        source_image: np.ndarray,
        target_mask: Optional[np.ndarray] = None,
        source_mask: Optional[np.ndarray] = None,
    ) -> MatchedKeypoints:
        """
        Process target and source images to get descriptive keypoints and match them

        Args:
            target_image: ndarray reference image that will remain static
            source_image: ndarray image that will undergo a homography transform in order to align with target image
            target_mask: bool ndarray mask indicating valid region for keypoint detection in target image
            source_mask: bool ndarray mask indicating valid region for keypoint detection in source image

        Returns:
            MatchedKeypoints containing found matches
        """
        # detect keypoints
        target_keypoints = self._detect_keypoints(image=target_image, mask=target_mask)
        source_keypoints = self._detect_keypoints(image=source_image, mask=source_mask)

        # compute descriptors
        target_descriptive_keypoints = self._describe_keypoints(image=target_image, keypoints=target_keypoints)
        source_descriptive_keypoints = self._describe_keypoints(image=source_image, keypoints=source_keypoints)

        # find matches
        matches = self._matcher.match(target_descriptive_keypoints, source_descriptive_keypoints, None)
        return MatchedKeypoints(target_keypoints=target_keypoints, source_keypoints=source_keypoints, matches=matches)

    def _filter_top_matches(
        self,
        matched_keypoints: MatchedKeypoints,
        min_matches: int = 4,
        keep_percent: float = 0.1,
    ) -> MatchedKeypoints:
        """
        Sort matches based on similarity of their descriptors and filter to preserve the closest ones

        Args:
            matches: MatchedKeypoints containing list of matches after removing static ones, as well as all detected keypoints and descriptors for both images
            min_matches: int minimum number of matches to preserve
            keep_percent: float percentage of keypoint matches to keep, effectively allowing us to eliminate noisy keypoint matching results

        Returns:
            MatchedKeypoints containing list of matches sorted sorted by similarity after filtering to preserve only the
              most similar ones. It contains as well as all detected keypoints and descriptors for both images
        """
        assert 0 <= keep_percent <= 1, f"keep_percent must be in range [0, 1] (keep_percent={keep_percent})"
        assert (
            len(matched_keypoints.matches) >= min_matches
        ), f"Only {len(matched_keypoints.matches)} keypoints have been considered for image registration. A minimum of 4 is required for the algorithm to succeed"
        # sort the matches by their distance (the smaller the distance, the "more similar" the features are)
        matches_sorted = sorted(matched_keypoints.matches, key=lambda x: x.distance)

        # keep only the top matches
        keep_matches_indices = max(min_matches, int(len(matches_sorted) * keep_percent))
        top_matches: Tuple[cv2.DMatch, ...] = tuple(matches_sorted[:keep_matches_indices])
        return MatchedKeypoints(
            matches=top_matches,
            target_keypoints=matched_keypoints.target_keypoints,
            source_keypoints=matched_keypoints.source_keypoints,
        )

    def _homography_from_matched_keypoints(self, matched_keypoints: MatchedKeypoints) -> np.ndarray:
        """
        Compute homography matrix to map points from source image to target image

        Args:
            matched_keypoints: MatchedKeypoints matches as well as all detected keypoints for both images

        Returns:
            homography_image_registration: ndarray 3x3 homography matrix that maps homogeneous points from source image to target image
        """
        # Extract location of good matches
        keypoints_target_image_array = np.zeros((len(matched_keypoints.matches), 2), dtype=np.float32)
        keypoints_source_image_array = np.zeros((len(matched_keypoints.matches), 2), dtype=np.float32)
        for i, match in enumerate(matched_keypoints.matches):
            keypoints_target_image_array[i, :] = matched_keypoints.target_keypoints[match.queryIdx].pt
            keypoints_source_image_array[i, :] = matched_keypoints.source_keypoints[match.trainIdx].pt

        # Find homography
        homography_matrix_registration, _ = cv2.findHomography(
            keypoints_source_image_array, keypoints_target_image_array, cv2.RANSAC
        )
        return homography_matrix_registration
