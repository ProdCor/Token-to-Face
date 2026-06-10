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

'''#!/usr/bin/env python3
"""
Utility functions for ARKit blendshape masking and FLAME transformations
CORRECTED VERSION - matches Blender's ARKit order
"""
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


def get_arkit_indices(categories=['jaw', 'mouth']):
    """
    Get indices of specific ARKit blendshape categories
    
    Args:
        categories: List of category names
    
    Returns:
        indices: List of indices
    """
    indices = []
    for category in categories:
        if category not in ARKIT_CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        indices.extend(ARKIT_CATEGORIES[category])
    return sorted(indices)


def apply_arkit_mask(blendshapes, mask):
    """
    Apply mask to ARKit blendshapes (zero out non-masked blendshapes)
    
    Args:
        blendshapes: (*, 51) tensor/array of ARKit blendshapes
        mask: (51,) boolean tensor/array
    
    Returns:
        masked_blendshapes: Same shape as input with mask applied
    """
    if isinstance(blendshapes, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).to(blendshapes.device)
        return blendshapes * mask
    else:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        return blendshapes * mask


def load_flame_to_arkit_transform(path='arkit_to_flame.npy'):
    """
    Load and compute FLAME to ARKit transformation matrix
    
    Args:
        path: Path to arkit_to_flame.npy (shape: 51, 103)
    
    Returns:
        transform: (103, 51) transformation matrix for FLAME -> ARKit
    """
    arkit_to_flame = np.load(path)  # (51, 103)
    flame_to_arkit = np.linalg.pinv(arkit_to_flame)  # (103, 51)
    return flame_to_arkit


def flame_to_arkit_blendshapes(flame_params, transform_matrix):
    """
    Transform FLAME parameters to ARKit blendshapes
    
    Args:
        flame_params: (*, 103) tensor/array of FLAME parameters
        transform_matrix: (103, 51) transformation matrix
    
    Returns:
        arkit_blendshapes: (*, 51) tensor/array
    """
    if isinstance(flame_params, torch.Tensor):
        if not isinstance(transform_matrix, torch.Tensor):
            transform_matrix = torch.from_numpy(transform_matrix).float().to(flame_params.device)
        return torch.matmul(flame_params, transform_matrix)
    else:
        if isinstance(transform_matrix, torch.Tensor):
            transform_matrix = transform_matrix.cpu().numpy()
        return np.dot(flame_params, transform_matrix)


def print_arkit_mask_info(mask):
    """Print information about ARKit mask"""
    print(f"ARKit Mask Info:")
    print(f"  Total active: {mask.sum()}/51")
    
    for category, indices in ARKIT_CATEGORIES.items():
        category_mask = mask[indices] if isinstance(mask, np.ndarray) else mask[indices].cpu().numpy()
        active = category_mask.sum()
        total = len(indices)
        print(f"  {category:15s}: {active}/{total} active")
    
    print(f"\nActive blendshapes:")
    for i, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
        is_active = mask[i] if isinstance(mask, np.ndarray) else mask[i].item()
        if is_active:
            print(f"  [{i:2d}] {name}")


# Verification function to check if order is correct
def verify_blendshape_order():
    """
    Verify that the blendshape order matches expectations
    """
    print("Verifying ARKit blendshape order...")
    print(f"Total blendshapes: {len(ARKIT_BLENDSHAPE_NAMES)}")
    
    # Check some key positions
    checks = [
        (0, 'eyeBlinkLeft'),
        (14, 'jawForward'),
        (16, 'jawOpen'),
        (18, 'mouthClose'),
        (41, 'browDownLeft'),
        (49, 'noseSneerLeft'),
    ]
    
    all_correct = True
    for idx, expected_name in checks:
        actual_name = ARKIT_BLENDSHAPE_NAMES[idx]
        status = "✓" if actual_name == expected_name else "✗"
        print(f"  [{idx:2d}] {status} Expected: {expected_name:20s} | Actual: {actual_name}")
        if actual_name != expected_name:
            all_correct = False
    
    if all_correct:
        print("\n✅ Blendshape order is CORRECT!")
    else:
        print("\n❌ Blendshape order has ERRORS!")
    
    return all_correct


if __name__ == "__main__":
    print("="*60)
    print("ARKit Blendshape Utilities - CORRECTED VERSION")
    print("="*60)
    
    # Verify order
    verify_blendshape_order()
    
    print("\n" + "="*60)
    
    # Test mask for mouth + jaw only
    mask = get_arkit_mask(['jaw', 'mouth'])
    print_arkit_mask_info(mask)
    
    print(f"\n{'='*60}")
    
    # Get indices
    indices = get_arkit_indices(['jaw', 'mouth'])
    print(f"\nMouth + Jaw indices: {indices}")
    print(f"Total: {len(indices)} blendshapes")'''