import matplotlib.pyplot as plt
import sys
import numpy as np
import torch

sys.path.append("../")
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

# ARKit Blendshape Names (51 total) - CORRECTED ORDER to match Blender
ARKIT_BLENDSHAPE_NAMES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
    'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper',
    'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight',
    'noseSneerLeft', 'noseSneerRight'
]

# ARKit category indices - CORRECTED to match new order
ARKIT_CATEGORIES = {
    'brow': [0, 1, 2, 3, 4],  # browDownLeft, browDownRight, browInnerUp, browOuterUpLeft, browOuterUpRight
    'cheek': [5, 6, 7],  # cheekPuff, cheekSquintLeft, cheekSquintRight
    'eye_blink': [8, 9],  # eyeBlinkLeft, eyeBlinkRight
    'eye_look': [10, 11, 12, 13, 14, 15, 16, 17],  # eyeLookDown/In/Out/Up Left/Right
    'eye_shape': [18, 19, 20, 21],  # eyeSquintLeft/Right, eyeWideLeft/Right
    'jaw': [22, 23, 24, 25],  # jawForward, jawLeft, jawOpen, jawRight
    'mouth': list(range(26,49)),  # indices 26-48: mouthClose through mouthUpperUpRight
    'nose': [49, 50]  # noseSneerLeft, noseSneerRight
}

def get_arkit_mask(categories=['jaw', 'mouth'], as_tensor=False):
    """
    Get mask for specific ARKit blendshape categories
    
    Args:
        categories: List of category names to keep (e.g., ['jaw', 'mouth'])
        as_tensor: If True, return torch.Tensor, else numpy array
    
    Returns:
        mask: Boolean array/tensor of shape (51,)
    """
    mask = np.zeros(51, dtype=bool)
    
    for category in categories:
        if category not in ARKIT_CATEGORIES:
            raise ValueError(f"Unknown category: {category}. Valid: {list(ARKIT_CATEGORIES.keys())}")
        indices = ARKIT_CATEGORIES[category]
        mask[indices] = True
    
    if as_tensor:
        return torch.from_numpy(mask)
    return mask

def plot_losses(train_losses, val_losses, save_name="losses"):
    print(train_losses)
    print(val_losses)
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig(f"{save_name}.png")
    plt.close()


def create_gaussian_diffusion(args):
    # default params
    sigma_small = True
    predict_xstart = False  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diff_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule("cosine", steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )