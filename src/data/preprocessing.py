"""Trajectory preprocessing utilities."""

import numpy as np
import torch


def normalize_trajectories(
    trajectories: np.ndarray, method: str = "last_obs"
) -> tuple[np.ndarray, dict]:
    """
    Normalize trajectories to a common reference frame.

    Args:
        trajectories: (N, T, 2) trajectory array
        method: "last_obs" (translate so last obs = origin),
                "min_max" (scale to [0, 1]),
                "standard" (zero mean, unit variance)

    Returns:
        normalized: (N, T, 2)
        params: dict with normalization parameters for inverse transform
    """
    if method == "last_obs":
        # Center at last observed position (common in trajectory prediction)
        offset = trajectories[:, -1:, :]
        normalized = trajectories - offset
        return normalized, {"offset": offset, "method": method}

    elif method == "min_max":
        mins = trajectories.min(axis=(0, 1))
        maxs = trajectories.max(axis=(0, 1))
        scale = maxs - mins
        scale[scale == 0] = 1.0
        normalized = (trajectories - mins) / scale
        return normalized, {"min": mins, "scale": scale, "method": method}

    elif method == "standard":
        mean = trajectories.mean(axis=(0, 1))
        std = trajectories.std(axis=(0, 1))
        std[std == 0] = 1.0
        normalized = (trajectories - mean) / std
        return normalized, {"mean": mean, "std": std, "method": method}

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def denormalize_trajectories(
    normalized: np.ndarray, params: dict
) -> np.ndarray:
    """Inverse of normalize_trajectories."""
    method = params["method"]

    if method == "last_obs":
        return normalized + params["offset"]
    elif method == "min_max":
        return normalized * params["scale"] + params["min"]
    elif method == "standard":
        return normalized * params["std"] + params["mean"]
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess_trajectories(
    raw_data: np.ndarray,
    obs_len: int = 8,
    pred_len: int = 12,
    skip: int = 1,
) -> list[dict]:
    """
    Preprocess raw trajectory data into training samples.

    Args:
        raw_data: (N_frames, 4) array with [frame, ped_id, x, y]
        obs_len: observed timesteps
        pred_len: predicted timesteps
        skip: frame skip

    Returns:
        List of sample dicts with obs/pred trajectories
    """
    seq_len = obs_len + pred_len
    frames = np.unique(raw_data[:, 0])
    
    samples = []
    for start_idx in range(0, len(frames) - seq_len + 1, skip):
        frame_window = frames[start_idx : start_idx + seq_len]
        
        # Find pedestrians present in all frames
        peds = None
        for f in frame_window:
            frame_peds = set(raw_data[raw_data[:, 0] == f, 1].tolist())
            peds = frame_peds if peds is None else peds & frame_peds
        
        if len(peds) < 2:
            continue
        
        for pid in sorted(peds):
            traj = []
            for f in frame_window:
                row = raw_data[(raw_data[:, 0] == f) & (raw_data[:, 1] == pid)]
                traj.append(row[0, 2:4])
            
            traj = np.array(traj)
            samples.append({
                "obs": traj[:obs_len],
                "pred": traj[obs_len:],
                "ped_id": pid,
                "start_frame": frame_window[0],
            })
    
    return samples
