"""
ETH/UCY Pedestrian Trajectory Dataset.

Supports the five standard scenes from the ETH (Pellegrini et al., 2009)
and UCY (Lerner et al., 2007) datasets:
  - ETH, Hotel (ETH dataset)
  - Univ, Zara1, Zara2 (UCY dataset)

Data format: Each line is [frame_id, pedestrian_id, x, y]
Trajectories are extracted using a sliding window approach with
obs_len observed steps and pred_len prediction steps.

For demo purposes, this module also includes synthetic trajectory
generation that mimics real pedestrian motion patterns.
"""

import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class ETHUCYDataset(Dataset):
    """
    ETH/UCY trajectory dataset.

    Extracts trajectory sequences using a sliding window. Each sample
    contains an ego agent's observed + future trajectory and the
    trajectories of all co-occurring neighbors.

    Parameters
    ----------
    data_dir : str
        Path to dataset directory.
    scene : str
        Scene name (eth, hotel, univ, zara1, zara2).
    obs_len : int
        Number of observed timesteps.
    pred_len : int
        Number of predicted timesteps.
    skip : int
        Frame skip rate.
    min_ped : int
        Minimum pedestrians per scene to include.
    max_neighbors : int
        Maximum neighbors to track.
    split : str
        "train" or "test".
    augment : bool
        Whether to apply data augmentation.
    """

    def __init__(
        self,
        data_dir: str = "data/eth_ucy",
        scene: str = "eth",
        obs_len: int = 8,
        pred_len: int = 12,
        skip: int = 1,
        min_ped: int = 2,
        max_neighbors: int = 10,
        split: str = "train",
        augment: bool = False,
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.max_neighbors = max_neighbors
        self.augment = augment

        data_path = Path(data_dir) / f"{scene}_{split}.txt"
        if data_path.exists():
            self._load_real_data(data_path, skip, min_ped)
        else:
            # Generate synthetic data for demo
            self._generate_synthetic_data(scene, split)

    def _load_real_data(self, path: Path, skip: int, min_ped: int):
        """Load real ETH/UCY data from text file."""
        data = np.loadtxt(path)
        frames = np.unique(data[:, 0]).tolist()
        frame_data = [data[data[:, 0] == f, :] for f in frames]

        self.samples = []

        for start in range(0, len(frames) - self.seq_len + 1, skip):
            # Get pedestrians present in the full window
            window = frame_data[start : start + self.seq_len]
            peds_in_all = set(window[0][:, 1].tolist())
            for frame in window[1:]:
                peds_in_all &= set(frame[:, 1].tolist())

            if len(peds_in_all) < min_ped:
                continue

            peds = sorted(peds_in_all)
            for ego_id in peds:
                ego_traj = []
                neighbor_trajs = {pid: [] for pid in peds if pid != ego_id}

                for frame in window:
                    for pid in peds:
                        row = frame[frame[:, 1] == pid][0]
                        pos = row[2:4]
                        if pid == ego_id:
                            ego_traj.append(pos)
                        elif pid in neighbor_trajs:
                            neighbor_trajs[pid].append(pos)

                ego_traj = np.array(ego_traj)
                n_trajs = [np.array(v) for v in neighbor_trajs.values()]

                self.samples.append({
                    "ego_traj": ego_traj,
                    "neighbor_trajs": n_trajs,
                })

    def _generate_synthetic_data(self, scene: str, split: str):
        """
        Generate realistic synthetic pedestrian trajectories.
        
        Creates diverse motion patterns including:
        - Linear walking
        - Curved paths
        - Stopping / starting
        - Group behaviors
        - Collision avoidance maneuvers
        """
        random.seed(42 if split == "train" else 123)
        np.random.seed(42 if split == "train" else 123)

        num_samples = 2000 if split == "train" else 500

        # Scene-specific parameters
        scene_params = {
            "eth": {"scale": 1.0, "density": 0.7, "speed": (0.5, 1.5)},
            "hotel": {"scale": 0.5, "density": 0.5, "speed": (0.3, 0.8)},
            "univ": {"scale": 1.2, "density": 0.9, "speed": (0.4, 1.2)},
            "zara1": {"scale": 0.8, "density": 0.6, "speed": (0.4, 1.0)},
            "zara2": {"scale": 0.7, "density": 0.5, "speed": (0.3, 0.9)},
        }
        params = scene_params.get(scene, scene_params["eth"])

        self.samples = []

        for _ in range(num_samples):
            num_agents = max(2, int(np.random.poisson(3 * params["density"])) + 1)
            num_agents = min(num_agents, self.max_neighbors + 1)

            trajectories = []
            for _ in range(num_agents):
                traj = self._generate_single_trajectory(params)
                trajectories.append(traj)

            # Apply simple collision avoidance
            trajectories = self._apply_social_forces(trajectories)

            # Create sample for each agent as ego
            ego_idx = random.randint(0, num_agents - 1)
            ego_traj = trajectories[ego_idx]
            n_trajs = [trajectories[i] for i in range(num_agents) if i != ego_idx]

            self.samples.append({
                "ego_traj": ego_traj,
                "neighbor_trajs": n_trajs,
            })

    def _generate_single_trajectory(self, params: dict) -> np.ndarray:
        """Generate a single realistic trajectory."""
        scale = params["scale"]
        min_speed, max_speed = params["speed"]

        # Random start position
        start = np.random.uniform(-5 * scale, 5 * scale, size=2)

        # Random motion type
        motion_type = random.choices(
            ["linear", "curved", "accelerating", "stopping"],
            weights=[0.4, 0.3, 0.15, 0.15],
        )[0]

        speed = np.random.uniform(min_speed, max_speed)
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])

        trajectory = [start.copy()]

        for t in range(1, self.seq_len):
            dt = 0.4  # 2.5 Hz sampling

            if motion_type == "linear":
                noise = np.random.normal(0, 0.02, size=2)
                pos = trajectory[-1] + speed * dt * direction + noise

            elif motion_type == "curved":
                # Gradual turn
                angle_rate = np.random.uniform(-0.1, 0.1)
                angle += angle_rate
                direction = np.array([np.cos(angle), np.sin(angle)])
                noise = np.random.normal(0, 0.02, size=2)
                pos = trajectory[-1] + speed * dt * direction + noise

            elif motion_type == "accelerating":
                accel = 0.05 if t < self.seq_len // 2 else -0.03
                speed = max(0.1, min(max_speed * 1.5, speed + accel))
                noise = np.random.normal(0, 0.02, size=2)
                pos = trajectory[-1] + speed * dt * direction + noise

            elif motion_type == "stopping":
                if t > self.seq_len * 0.6:
                    speed = max(0.0, speed - 0.08)
                noise = np.random.normal(0, 0.01, size=2)
                pos = trajectory[-1] + speed * dt * direction + noise

            trajectory.append(pos)

        return np.array(trajectory)

    def _apply_social_forces(
        self, trajectories: list, threshold: float = 0.5
    ) -> list:
        """Apply simple repulsive social forces for collision avoidance."""
        trajs = [t.copy() for t in trajectories]
        n = len(trajs)

        for t in range(1, self.seq_len):
            for i in range(n):
                force = np.zeros(2)
                for j in range(n):
                    if i == j:
                        continue
                    diff = trajs[i][t] - trajs[j][t]
                    dist = np.linalg.norm(diff)
                    if dist < threshold and dist > 0.01:
                        force += 0.1 * diff / (dist**2)
                trajs[i][t] += force * 0.1

        return trajs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        ego_traj = sample["ego_traj"].astype(np.float32)
        n_trajs = sample["neighbor_trajs"]

        # Split into observed and predicted
        obs_traj = ego_traj[: self.obs_len]
        pred_traj = ego_traj[self.obs_len :]

        # Pad neighbors to max_neighbors
        N = len(n_trajs)
        neighbor_obs = np.zeros(
            (self.max_neighbors, self.obs_len, 2), dtype=np.float32
        )
        neighbor_mask = np.zeros(self.max_neighbors, dtype=bool)

        for i in range(min(N, self.max_neighbors)):
            neighbor_obs[i] = n_trajs[i][: self.obs_len].astype(np.float32)
            neighbor_mask[i] = True

        # Data augmentation
        if self.augment:
            obs_traj, pred_traj, neighbor_obs = self._augment(
                obs_traj, pred_traj, neighbor_obs, neighbor_mask
            )

        return {
            "obs_traj": torch.from_numpy(obs_traj),
            "pred_traj": torch.from_numpy(pred_traj),
            "neighbor_obs": torch.from_numpy(neighbor_obs),
            "neighbor_mask": torch.from_numpy(neighbor_mask),
        }

    def _augment(self, obs, pred, neighbors, mask):
        """Apply random rotation and flipping augmentation."""
        # Random rotation
        angle = np.random.uniform(-np.pi, np.pi)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ], dtype=np.float32)

        obs = obs @ rot.T
        pred = pred @ rot.T
        for i in range(len(neighbors)):
            if mask[i]:
                neighbors[i] = neighbors[i] @ rot.T

        # Random flip
        if random.random() > 0.5:
            obs[:, 0] *= -1
            pred[:, 0] *= -1
            neighbors[:, :, 0] *= -1

        return obs, pred, neighbors


def eth_ucy_collate_fn(batch: list) -> dict:
    """Custom collate function for DataLoader."""
    return {
        "obs_traj": torch.stack([b["obs_traj"] for b in batch]),
        "pred_traj": torch.stack([b["pred_traj"] for b in batch]),
        "neighbor_obs": torch.stack([b["neighbor_obs"] for b in batch]),
        "neighbor_mask": torch.stack([b["neighbor_mask"] for b in batch]),
    }
