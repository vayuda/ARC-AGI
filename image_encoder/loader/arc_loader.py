import os
import sys
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

import training_dsl as training_dsl


# ========================================================================
# Easy calls for external use
# ========================================================================
def _get_data_fp(filename):
    """
        Function to return the filepath to our training data.
        Pass in the filename of the dataset you want to use - e.g., 'ibot_traindata_aggregate.parquet'
    """
    try:
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_file_dir = os.getcwd()
    
    root = os.path.abspath(os.path.join(current_file_dir, '..'))
    file = os.path.join(root, 'train_data', filename)
    return file


def get_dataloader(filename, loader_params):
    """
        Function for easy external calling that returns our dataloader given loader params and a filename.

        Inputs:
            filename (str): Name of file we'll load our data from. Input to 'get_data_fp'
            loader_params (dict): Dictionary of our loader parameters
    """
    fp = _get_data_fp(filename)
    dataset = ARCDatasetWrapper(TxtDictDataset(fp), pad_images=loader_params['pad_images'], percent_mask=loader_params['percent_mask'], place_central=loader_params['place_central'])
    dataloader = ARCDataLoader(dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle'])
    return dataloader


# ========================================================================
# Custom Dataset, Dataset Wrapper, and DataLoader
# ========================================================================
class TxtDictDataset(Dataset):
    def __init__(self, fp):
        """
        Custom dataset to load in a .txt file of dict objects.
        Returns a tuple of (key, tensor).
        """
        self.fp = fp
        self.samples = self.load_samples(fp)

    def load_samples(self, fp):
        samples = []
        with open(fp, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                for key, value in entry.items():
                    tensor = torch.tensor(value, dtype=torch.int32)
                    samples.append((key, tensor))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class ARCDatasetWrapper:
    def __init__(self, dataset, evaluate=False, pad_images=False, percent_mask=0.1, pad_to_32x32=False, place_central=False):
        """
        Wrapper for TxtDictDataset to apply additional processing.
        
        Parameters:
        - dataset (Dataset): The base dataset to wrap, which should yield tuples of (key, image).
            * key (str): Identifier for the data sample.
            * image (torch.Tensor): 2D tensor representing the image, expected to be in shape (H, W).
        - pad_images (bool): If True, pads images to 32x32 using pad_to_32x32 function.
        - percent_mask (float): Probability of each pixel in the mask being set to 1.
        """
        self.dataset = dataset
        self.pad_images = pad_images
        self.percent_mask = percent_mask
        self.place_central = place_central
        self.evaluate = evaluate
        self.crop_args = {
            'min_crop_shape': 3,
            'max_crop': 5
        }
        self.init_transformations = {
            'mask_only': False,
            'u_offset': None,
            'v_offset': None,
            'translation': 0,
            'shuffle_colors': 0,
            'rotate': 0,
            'vertical_mirror': 0,
            'horizontal_mirror': 0,
            'resize': 0
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        problem_id, u = self.dataset[idx]        
        u_mask = self.get_image_mask(u)

        # Will be storing transformations applied to each image in a dict
        transformations = self.init_transformations.copy()
        transformations['mask_only'] = self.evaluate
        
        # Create transformed image
        v = u.clone()
        if transformations['mask_only']:
            v_mask = self.get_image_mask(v)
        else:
            v, transformations = self.apply_encoder_dsl(v, transformations)
            v_mask = self.get_image_mask(v)

        # Pad images if set to self_padding
        if self.pad_images:
            u, u_offset = self.pad_to_32x32(u)
            u_mask = self.pad_mask_to_32x32(u_mask, u_offset)
            
            transformations['u_offset'] = u_offset
            v_offset = u_offset if transformations['mask_only'] else None

            v, v_offset = self.pad_to_32x32(v, v_offset)
            v_mask = self.pad_mask_to_32x32(v_mask, v_offset)
            transformations['v_offset'] = v_offset
            t_magnitude = (((u_offset[0] - v_offset[0])**2 + (u_offset[1] - v_offset[1])**2)**0.5)
            transformations['translation'] = t_magnitude if t_magnitude < 1 else t_magnitude**0.5
        return problem_id, u, u_mask, v, v_mask, transformations

    def apply_encoder_dsl(self, image, transformations):
        """Accepts a 2-D pytorch tensor (H x W) and returns the transformed version of that image and the transformations applied"""
        if random.random() < 0.4:
            image, i = training_dsl.shuffle_colors(image)
            transformations['shuffle_colors'] = i
        if random.random() < 0.4:
            image, i = training_dsl.rotate(image)
            transformations['rotate'] = i
        if random.random() < 0.4:
            image, i = training_dsl.vertical_mirror(image)
            transformations['vertical_mirror'] = i
        if random.random() < 0.4:
            image, i = training_dsl.horizontal_mirror(image)
            transformations['horizontal_mirror'] = i
        if random.random() < 0.7:
            image, i = training_dsl.resize(image)
            transformations['resize'] = i
        return image, transformations
    
    def pad_to_32x32(self, image, offset=None, rand_pos=True):
        """Pads the input image tensor to 32x32 with custom padding rules."""
        height, width = image.shape        
        if height == 32 and width == 32:
            return image, (0, 0)
        
        # Create a 1-pixel border of '11' around the image
        bordered_image = torch.nn.functional.pad(image, (1, 1, 1, 1), value=11)
        cropped_image = bordered_image[:32, :32]    # Ensure 32 x 32
        height, width = cropped_image.shape
        pad_bottom = max(0, 32 - height)
        pad_right = max(0, 32 - width)
        
        # Determine offset
        if self.place_central:
            x = pad_right // 2
            y = pad_bottom // 2
        elif offset is not None:
            x, y = offset
        elif rand_pos:
            x = random.randint(0, pad_right)
            y = random.randint(0, pad_bottom)
        else:
            x, y = 0, 0
        
        # Pad to place the image within the 32x32 frame
        padded_image = torch.nn.functional.pad(
            cropped_image, 
            (x, pad_right - x, y, pad_bottom - y), 
            value=12
        )
        return padded_image, (x, y)        

    def pad_mask_to_32x32(self, mask, offset):
        x, y = offset
        x += 1
        y += 1  # Need to account for the 1 pixel border
        H, W = mask.shape
        padded_mask = torch.zeros((32, 32), dtype=mask.dtype, device=mask.device)
        padded_mask[y:y+H, x:x+W] = mask
        return padded_mask

    def shuffle_colors(self, image):
        """Apply a random color mapping to shuffle integers from 0 to 9 in place, avoiding double-mapping issues."""
        original_values = list(range(10))
        shuffled_values = torch.randperm(10).tolist()
        mapping = {original: shuffled for original, shuffled in zip(original_values, shuffled_values)}
        mapped_image = image.clone()    # Create clone to prevent double mapping
        for original_value, new_value in mapping.items():
            mapped_image[image == original_value] = new_value
        return mapped_image

    def global_crop(self, image):
        """Apply a global cropping transformation based on specified conditions."""
        height, width = image.shape
        if width > self.crop_args['min_crop_shape']:
            max_width_crop = min(self.crop_args['max_crop'], int((width * 1.5) / self.crop_args['max_crop']))
            crop_width = random.randint(0, max_width_crop)
            i = random.randint(0, crop_width)
            j = width - crop_width
            image = image[:, i:j]
    
        if height > self.crop_args['min_crop_shape']:
            max_height_crop = min(self.crop_args['max_crop'], int((height * 1.5) / self.crop_args['max_crop']))
            crop_height = random.randint(0, max_height_crop)
            i = random.randint(0, crop_height)
            j = height - crop_height
            image = image[i:j, :]
    
        return image

    def get_image_mask(self, image):
        """Generate an image mask with values 0 and 1 based on the percent_mask probability."""
        height, width = image.shape
        mask = torch.bernoulli(torch.full((height, width), self.percent_mask)).int()
        return mask



class ARCDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, collate_fn=None):
        if collate_fn is None:
            collate_fn = self.custom_collate_fn
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    @staticmethod
    def custom_collate_fn(batch):
        keys = [item[0] for item in batch]
        images = [item[1] for item in batch]
        image_masks = [item[2] for item in batch]
        transformed_images = [item[3] for item in batch]
        transformed_image_masks = [item[4] for item in batch]
        transformations = [item[5] for item in batch]
        return keys, torch.stack(images), torch.stack(image_masks), torch.stack(transformed_images), torch.stack(transformed_image_masks), transformations