"""Trajectory data augmentation strategies."""

import numpy as np
import torch


class TrajectoryAugmentor:
    """
    Applies stochastic augmentations to trajectory data.
    
    Augmentations preserve the physical plausibility of trajectories
    while increasing data diversity.
    """

    def __init__(self, angle_range: float = 15.0, flip_prob: float = 0.5):
        self.angle_range = np.radians(angle_range)
        self.flip_prob = flip_prob

    def __call__(self, obs: np.ndarray, pred: np.ndarray) -> tuple:
        """
        Apply random augmentations.
        
        Args:
            obs: (T_obs, 2)
            pred: (T_pred, 2)
        Returns:
            Augmented (obs, pred)
        """
        # Random rotation
        angle = np.random.uniform(-self.angle_range, self.angle_range)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ], dtype=np.float32)
        obs = obs @ rot.T
        pred = pred @ rot.T

        # Random horizontal flip
        if np.random.random() < self.flip_prob:
            obs[:, 0] *= -1
            pred[:, 0] *= -1

        # Random scaling (subtle)
        scale = np.random.uniform(0.9, 1.1)
        obs *= scale
        pred *= scale

        return obs, pred
