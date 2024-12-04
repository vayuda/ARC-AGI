import os
import sys
import json
import random
import numpy as np
from copy import deepcopy
from typing import List, Set, Callable

import torch
from torch.utils.data import Dataset, DataLoader


# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))
if root not in sys.path:
    sys.path.append(root)

import source as source
from phog import synthetic_generator as sg
from utility_v2 import upsize_image




# ===========================================
# Externalized Functions
# ===========================================
def get_dataset(folder_path, only_inputs, transformation_samples, transformation_depth, p_use_base, percent_mask, pad_images):
    json_dataset = JSONDataset(folder_path, only_inputs=only_inputs)
    arc_nce_dataset = ARC_NCE_Dataset_Wrapper(json_dataset, transformation_samples, transformation_depth, p_use_base, percent_mask, pad_images)
    return arc_nce_dataset

def get_dataloader(dataset, shuffle=True):
    return ARC_Yielder_DataLoader(dataset, shuffle=shuffle)

def get_pcfg_datahandlers(folder_path, only_inputs, percent_mask, pad_images, transformation_depth, transformation_samples, p_use_base, shuffle):
    dataset = get_dataset(folder_path, only_inputs, transformation_samples, transformation_depth, p_use_base, percent_mask, pad_images)
    dataloader = get_dataloader(dataset, shuffle)
    return dataset, dataloader


# ===========================================
# Custom Torch Dataset and Dataloader
# ===========================================
class JSONDataset(Dataset):
    def __init__(self, folder_path, only_inputs=True):
        """
        Custom dataset to load in all json files within a target folder.
        Returns a tuple of (key, ARC_Object).
        """
        self.folder_path = os.path.join(root, folder_path)
        self.only_inputs = only_inputs
        self.samples = self.load_samples(self.folder_path)

    def load_samples(self, folder_path):
        samples = []
        viable_keys = ['input'] if self.only_inputs else ['input', 'output']
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                file_path = os.path.join(folder_path, file_name)        
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Iterate through 'train' and 'test' keys in the JSON
                file_id = file_name.split('.')[0]
                for split in ['train', 'test']:
                    if split in data:
                        for i, sample in enumerate(data[split]):
                            for key in viable_keys:
                                if key in sample:
                                    image = np.array(sample[key], dtype=int)
                                    arc_object = source.ARC_Object(image)
                                    samples.append((f"{file_id}-{key}-{i}", arc_object))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ARC_NCE_Dataset_Wrapper:
    def __init__(self, dataset, transformation_samples, transformation_depth, p_use_base, percent_mask, pad_images):
        self.dataset = dataset
        self.transformation_samples = transformation_samples
        self.transformation_depth = transformation_depth
        self.p_use_base = p_use_base
        self.percent_mask = percent_mask
        self.pad_images = pad_images

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        key, sample = self.dataset[idx]
        samples = generate_samples(sample, p_use_base=self.p_use_base, depth=self.transformation_depth, new_samples=self.transformation_samples)
        us, u_masks, vs, v_masks = [], [], [], []
        
        # Get maximum rows / cols of all samples then use to get offset
        max_r, max_c = 0, 0
        for arc_obj in samples:
            max_r = max(max_r, arc_obj.height)
            max_c = max(max_c, arc_obj.width)
        scale_factor = int(32 / max(max_r, max_c))
        offset = (random.randint(0, 32 - (max_c*scale_factor)), random.randint(0, 32 - (max_r*scale_factor)))
        
        # Turn to tensors, pad, and get masks
        for arc_obj in samples:
            u, u_mask, v, v_mask = self.arc_to_tensor(arc_obj, scale_factor, offset)
            us.append(u)
            u_masks.append(u_mask)
            vs.append(v)
            v_masks.append(v_mask)
            
        us = torch.stack(us)
        u_masks = torch.stack(u_masks)
        vs = torch.stack(vs)
        v_masks = torch.stack(v_masks)
        
        return key, us, u_masks, vs, v_masks
    
    def arc_to_tensor(self, arc_obj: source.ARC_Object, scale_factor, offset) -> torch.Tensor:
        """ Given an ARC Object, it applies various data augmentations and returns a tensor of shape [H, W] """
        base_image = torch.tensor(arc_obj.grid)   # [h, w]
        base_image = upsize_image(base_image, scale_factor)
        
        # Pad to max_r x max_c
        u_mask = self.get_image_mask(base_image)
        u, u_mask = self.pad_to_32x32(base_image.clone(), u_mask, offset)
        v_mask = self.get_image_mask(base_image)
        v, v_mask = self.pad_to_32x32(base_image.clone(), v_mask, offset)
        
        return u, u_mask, v, v_mask

    def pad_to_32x32(self, image, image_mask, offset):
        """ Pads the input image tensor to 32x32 with custom padding rules. """
        # Create a 1-pixel border of '11' around the image
        bordered_image = torch.nn.functional.pad(image, (1, 1, 1, 1), value=11)
        bordered_mask = torch.nn.functional.pad(image_mask, (1, 1, 1, 1), value=0)

        cropped_image = bordered_image[:32, :32]
        cropped_mask = bordered_mask[:32, :32]

        x, y = offset
        height, width = cropped_image.shape
        pad_bottom = max(0, 32 - height)
        pad_right = max(0, 32 - width)
        
        padded_image = torch.nn.functional.pad(
            cropped_image, 
            (x, pad_right - x, y, pad_bottom - y), 
            value=12
        )
        padded_mask = torch.nn.functional.pad(
            cropped_mask, 
            (x, pad_right - x, y, pad_bottom - y), 
            value=0
        )

        return padded_image, padded_mask
    
    def get_image_mask(self, image):
        """Generate an image mask with values 0 and 1 based on the percent_mask probability."""
        height, width = image.shape
        mask = torch.bernoulli(torch.full((height, width), self.percent_mask)).int()
        return mask


# Or can use DataYielder to get single step of each transformation
class ARC_Yielder_DataLoader(DataLoader):
    def __init__(self, dataset, shuffle=True):
        super().__init__(dataset, batch_size=1, shuffle=shuffle, collate_fn=self.default_collate_fn)
        
    @staticmethod
    def default_collate_fn(batch):
        keys, us, u_masks, vs, v_masks = batch[0]
        return (keys, us, u_masks, vs, v_masks)



# ===========================================
# Random Transformation Function
# ===========================================
def generate_samples(base_obj: source.ARC_Object, p_use_base: float =0.15, depth: int =2, new_samples: int =4) -> List[source.ARC_Object]:
    """
    Function to generate a number of samples from a base object.

    Args:
        base_obj (ARC_Object): The base object to generate samples from.
        depth (int): Number of transformations to apply.
        new_samples (int): The number of samples to generate.
    """
    samples = [base_obj]
    for _ in range(new_samples):
        sample = deepcopy(base_obj)
        use_base = random.random() < p_use_base
        _, transformed_objs, _, _ = sg.rand_transform(sample, depth=depth, use_base=use_base)
        samples.append(transformed_objs[0])
    return samples

