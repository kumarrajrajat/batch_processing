import torch
import numpy as np

def postprocess_batch(
    outputs: Tuple[np.ndarray, np.ndarray],   # simcc_x, simcc_y; both [N,K,W]
    center: np.ndarray,                      # [N, 2]
    scale: np.ndarray,                       # [N, 2]
    model_input_size: Tuple[int, int],
    simcc_split_ratio: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fully batch postprocess for RTMPose, NumPy in/out, Torch internal.
    Returns rescaled keypoints [N,K,2] and scores [N,K] as NumPy.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    simcc_x = torch.from_numpy(outputs[0]).float().to(device)
    simcc_y = torch.from_numpy(outputs[1]).float().to(device)
    center_t = torch.from_numpy(center).float().to(device)
    scale_t = torch.from_numpy(scale).float().to(device)
    input_size_t = torch.tensor(model_input_size, dtype=torch.float32, device=device)
    # SimCC maximum locations
    locs, scores = get_simcc_maximum_torch(simcc_x, simcc_y)  # [N,K,2], [N,K]
    keypoints = locs / simcc_split_ratio
    keypoints = keypoints / input_size_t * scale_t.unsqueeze(1)
    keypoints = keypoints + center_t.unsqueeze(1) - scale_t.unsqueeze(1) / 2
    return keypoints.cpu().numpy(), scores.cpu().numpy()



import torch
import numpy as np
from typing import Tuple

def get_simcc_maximum_batch(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch SimCC Maximum with NumPy input/output and Torch tensor internals on GPU.
    Args:
        simcc_x (np.ndarray): shape [N, K, Wx]
        simcc_y (np.ndarray): shape [N, K, Wy]
    Returns:
        locs (np.ndarray): [N, K, 2] float32
        vals (np.ndarray): [N, K] float32
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Convert to torch tensor on device
    simcc_x_t = torch.from_numpy(simcc_x).float().to(device)
    simcc_y_t = torch.from_numpy(simcc_y).float().to(device)

    N, K, Wx = simcc_x_t.shape
    _, _, Wy = simcc_y_t.shape
    simcc_x_flat = simcc_x_t.view(N * K, Wx)
    simcc_y_flat = simcc_y_t.view(N * K, Wy)

    x_locs = torch.argmax(simcc_x_flat, dim=1)
    y_locs = torch.argmax(simcc_y_flat, dim=1)
    locs = torch.stack((x_locs, y_locs), dim=-1).float().view(N, K, 2)

    max_val_x = torch.amax(simcc_x_flat, dim=1)
    max_val_y = torch.amax(simcc_y_flat, dim=1)
    vals = 0.5 * (max_val_x + max_val_y)
    vals = vals.view(N, K)

    locs[vals <= 0.] = -1

    # Return numpy arrays (copied to CPU)
    return locs.cpu().numpy(), vals.cpu().numpy()




def convert_coco_to_openpose_batch(keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts batch of COCO-format keypoints/scores to OpenPose. I/O NumPy; torch intermediate.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keypoints_t = torch.from_numpy(keypoints).float().to(device)   # [N,17,2]
    scores_t = torch.from_numpy(scores).float().to(device)         # [N,17]
    keypoints_info = torch.cat([keypoints_t, scores_t.unsqueeze(-1)], dim=-1)  # [N,17,3]
    neck = keypoints_info[:, [5, 6]].mean(dim=1, keepdim=True)                # [N,1,3]
    neck[:, :, 2] = torch.minimum(keypoints_info[:, 5:6, 2], keypoints_info[:, 6:7, 2])
    new_keypoints_info = torch.cat([keypoints_info, neck], dim=1)              # [N,18,3]
    mmpose_idx = torch.tensor([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3], device=device)
    openpose_idx = torch.tensor([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17], device=device)
    new_keypoints_info[:, openpose_idx] = new_keypoints_info[:, mmpose_idx]
    keypoints_out = new_keypoints_info[..., :2].cpu().numpy()
    scores_out = new_keypoints_info[..., 2].cpu().numpy()
    return keypoints_out, scores_out




def get_simcc_maximum_torch(simcc_x: torch.Tensor, simcc_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch max response from SimCC. Torch tensors; output torch.
    """
    N, K, Wx = simcc_x.shape
    _, _, Wy = simcc_y.shape
    simcc_x_flat = simcc_x.view(N*K, Wx)
    simcc_y_flat = simcc_y.view(N*K, Wy)
    x_locs = torch.argmax(simcc_x_flat, dim=1)
    y_locs = torch.argmax(simcc_y_flat, dim=1)
    locs = torch.stack((x_locs, y_locs), dim=-1).float().view(N, K, 2)
    max_val_x = torch.amax(simcc_x_flat, dim=1)
    max_val_y = torch.amax(simcc_y_flat, dim=1)
    vals = 0.5 * (max_val_x + max_val_y)
    vals = vals.view(N, K)
    locs[vals <= 0.] = -1
    return locs, valss
