import torch
import random
from collections import Counter


def shuffle_colors(image):
    """Apply a random color mapping to shuffle integers from 0 to 9 in place, avoiding double-mapping issues."""
    original_values = list(range(10))
    shuffled_values = torch.randperm(10).tolist()
    mapping = {original: shuffled for original, shuffled in zip(original_values, shuffled_values)}
    mapped_image = image.clone()
    for original_value, new_value in mapping.items():
        mapped_image[image == original_value] = new_value
    return mapped_image, 1


def rotate(image):
    """Takes in a torch tensor of shape 'H x W' and returns the rotated version (randomly rotated 90, 180, or 270 degrees -- measured counterclockwise)
    """
    rotations = random.choice([1, 2, 3])
    rotated_image = torch.rot90(image, k=rotations, dims=(0, 1))
    if torch.equal(image, rotated_image):
        return image, 0
    return rotated_image, rotations
    

def vertical_mirror(image):
    """Takes in a torch tensor of shape 'H x W' and returns the mirrored version (mirrored over vertical axis). Returns '1' as the second argument."""
    mirrored_image = torch.flip(image, dims=[1])
    if torch.equal(image, mirrored_image):
        return image, 0
    return mirrored_image, 1


def horizontal_mirror(image):
    """Takes in a torch tensor of shape 'H x W' and returns the mirrored version (mirrored over horizontaL axis). Returns '1' as the second argument."""
    mirrored_image = torch.flip(image, dims=[0])
    if torch.equal(image, mirrored_image):
        return image, 0
    return mirrored_image, 1


def resize(image):
    """
    Resizes a torch tensor of integers based on the eligibility for scaling up or down.
    
    Args:
        image (torch.Tensor): A 2D tensor of integers with shape (H, W).
    
    Returns:
        torch.Tensor: Resized image (as a grid of integers).
        int: Scaling factor applied (0 for no scaling, 1 for 2x, 2 for 4x, 3 for 8x).
    """
    H, W = image.shape
    
    # Scaling up
    if H < 16 and W < 16:
        max_scale = 8 if H < 4 and W < 4 else 4 if H < 8 and W < 8 else 2
        scale_factors = [2 ** i for i in range(1, max_scale.bit_length())]  # [2, 4, 8]
        scale = random.choices(scale_factors, k=1)[0]
        scaled_image = image.repeat_interleave(scale, dim=0).repeat_interleave(scale, dim=1)
        return scaled_image, scale_factors.index(scale)+1
    
    # Scaling down
    elif H > 16 and W > 16:
        if random.random() < 0.5:
            pixel_counts = Counter(image.flatten().tolist())
            most_common_value, count = pixel_counts.most_common(1)[0]
            total_pixels = H * W
            
            background_value = most_common_value if count > total_pixels / 2 else -1
            
            downscaled_image = []
            for i in range(0, H, 2):
                row = []
                for j in range(0, W, 2):
                    patch = image[i:i+2, j:j+2].flatten().tolist()                    
                    patch_counts = Counter(x for x in patch if x != background_value)

                    if patch_counts:
                        most_common_non_bg, _ = patch_counts.most_common(1)[0]
                        row.append(most_common_non_bg)
                    else:
                        # Default to the background value if no other values are present
                        row.append(background_value)
                downscaled_image.append(row)
            
            downscaled_image = torch.tensor(downscaled_image, dtype=image.dtype)
            return downscaled_image, -1
    
    # No scaling needed
    return image, 0