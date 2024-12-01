import torch

def rotation_2d_torch(x, theta, origin=None):
    """
    here we do rotation for augmentation. since we use norm_rot_x as the input to the decoder, we 
    have to sum it up with scene_mean which is origin, before finding the A again (repair_A)
    """
    if origin is None:
        origin = torch.zeros(2).to(x.device).to(x.dtype)
    norm_x = x - origin
    norm_rot_x = torch.zeros_like(x)
    norm_rot_x[..., 0] = norm_x[..., 0] * torch.cos(theta) - norm_x[..., 1] * torch.sin(theta)
    norm_rot_x[..., 1] = norm_x[..., 0] * torch.sin(theta) + norm_x[..., 1] * torch.cos(theta)
    rot_x = norm_rot_x + origin
    return rot_x, norm_rot_x