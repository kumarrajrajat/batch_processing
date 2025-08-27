import torch
import numpy as np
import torch.nn.functional as F

def preprocess_batch(self, imgs: np.ndarray):
    """
    Batch preprocessing for RTMPose model inference.
    Args:
        imgs (np.ndarray): Input images of shape [N,H,W,C]. Can have variable H and W per image, but will output fixed shape for all.
    Returns:
        tuple:
        - resized_imgs (np.ndarray): Preprocessed images, shape [N, model_H, model_W, 3].
        - ratios (np.ndarray): Resizing ratios per image, shape [N].
    """
    # Desired model input size
    model_H, model_W = self.model_input_size[:2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = imgs.shape[0]
    
    # Convert input to torch tensor and permute to [N, C, H, W]
    imgs_t = torch.from_numpy(imgs).float().permute(0, 3, 1, 2).to(device)  # [N,3,H,W]

    # Compute original H and W for each image
    orig_H = torch.tensor([img.shape[0] for img in imgs], dtype=torch.float32, device=device)  # [N]
    orig_W = torch.tensor([img.shape[1] for img in imgs], dtype=torch.float32, device=device)  # [N]

    # Compute resize ratios per image
    ratio_H = model_H / orig_H
    ratio_W = model_W / orig_W
    ratios = torch.min(torch.stack([ratio_H, ratio_W], dim=1), dim=1)[0]  # [N]

    # Compute new sizes for all images
    new_H = (orig_H * ratios).round().long()  # [N]
    new_W = (orig_W * ratios).round().long()  # [N]

    # Resize all images to model input size
    resized_imgs = F.interpolate(
        imgs_t, 
        size=(model_H, model_W), 
        mode='bilinear', 
        align_corners=False
    )  # [N,3,model_H,model_W]

    # Convert back to [N, model_H, model_W, 3], uint8, numpy
    resized_imgs_np = resized_imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    ratios_np = ratios.cpu().numpy()
    return resized_imgs_np, ratios_np
