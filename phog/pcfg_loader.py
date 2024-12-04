import os
import sys
import json
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

from phog.synthetic_generator import rand_transform, get_dsl_operations
import source as source


# ========================================================================
# Easy calls for external use
# ========================================================================
def _get_data_fp(foldername):
    """
        Function to return the filepath to our training data.
        Pass in the name of the folder of the dataset you want to use - e.g., 'training'
    """
    folder_path = os.path.join(root, 'data', foldername)
    return folder_path


def get_pcfg_datayielder(datafolder, batch_size, moves_per_step, max_steps, p_use_base, shuffle=True, only_inputs=True, print_steps=False, **kwargs):
    """ Function for easy external calling that returns our dataloader given loader params and a filename. """
    fp = _get_data_fp(datafolder)
    yielder_dataset = JSONDataset(fp, only_inputs=only_inputs)
    yielder_dataloader = ARC_Yielder_DataLoader(yielder_dataset, batch_size=batch_size, shuffle=shuffle)
    data_yielder = ARC_DataYielder(yielder_dataloader, rand_transform, get_dsl_operations(), step_depth=moves_per_step, max_steps=max_steps, p_use_base=p_use_base, print_steps=print_steps)
    return data_yielder


# Can use something like this to load in the broader loader instead of the iterator
# train_dataset = JSONDataset("data/training", only_inputs=True)
# train_dataset = PHOG_ARC_DatasetWrapper(train_dataset, get_dsl_operations(), rand_transform, elems_per_image=16)
# train_loader = PHOG_ARC_Dataloader(train_dataset, batch_size=1, shuffle=True)

class SampleExhausted(Exception):
    """Custom exception to signal that the current sample is exhausted."""
    pass


# ========================================================================
# Custom Dataset, Dataset Wrapper, and DataLoader
# ========================================================================
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

# Can use dataloader to get batch of full transformation sequence
# class PCFG_ARC_DatasetWrapper(Dataset):
#     def __init__(self, dataset, label_operations, rand_transform, elems_per_image=16, max_labels=8):
#         self.dataset = dataset
#         self.label_operations = {label_operation: i for i, label_operation in enumerate(label_operations)}
#         self.elems_per_image = elems_per_image
#         self.max_labels = max_labels
#         self.rand_transform = rand_transform
#         self.sep_token = "<SEP>"
#         self.pad_token = "<PAD>"

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         key, image = self.dataset[idx]
#         use_base = random.random() < 0.2
#         input_objs, transformed_objs, transforms = self.rand_transform(image, depth=5, use_base=use_base)
#         x = input_objs
#         x.extend([self.pad_token] * (self.elems_per_image - len(x)) + [self.sep_token])
#         x.extend(transformed_objs)
#         x.extend([self.pad_token] * ((self.elems_per_image*2 + 1) - len(x)))
#         y = list(map(lambda x: self.label_operations[x], transforms))
#         y.extend([-1] * (self.max_labels - len(y)))
#         return key, x, y


# class PCFG_ARC_Dataloader(DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=True):
#         super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

#     @staticmethod
#     def collate_fn(batch):
#         keys, x, y = zip(*batch)
#         keys = list(keys)
#         x_batch = list(x)
#         y_batch = list(y)
#         return keys, x_batch, y_batch



# Or can use DataYielder to get single step of each transformation
class ARC_Yielder_DataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True):
        collate_fn = self.default_collate_fn
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    @staticmethod
    def default_collate_fn(batch):
        keys, samples = zip(*batch)
        keys = list(keys)
        samples = list(samples)
        return keys, samples


class ARC_DataYielder:
    sep_token = "<SEP>"
    pad_token = "<PAD>"
    max_labels = 8
    elems_per_image = 16

    def __init__(self, dataloader, rand_transform, dsl, step_depth=1, max_steps=5, p_use_base=0.2, print_steps=True):
        self.dataloader_iter = iter(dataloader)
        self.dataloader = dataloader
        self.rand_transform = rand_transform
        self.dsl = dsl
        self.label_ops = {label_operation: i for i, label_operation in enumerate(dsl)}
        
        self.cur_sample = None
        self.cur_key = None
        
        self.step_depth = step_depth
        self.p_use_base = p_use_base
        self.max_steps = max_steps
        self.epoch = 0
        self.cur_sample_step = 0

        self.print_steps = print_steps

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur_sample is None or self.cur_sample_step >= self.max_steps:
            self.cur_sample_step = 0
            try:
                self.cur_key, self.cur_sample = next(self.dataloader_iter)
                self.cur_key, self.cur_sample = self.cur_key[0], self.cur_sample[0]
            except StopIteration:
                # Start a new epoch
                self.epoch += 1
                self.dataloader_iter = iter(self.dataloader)
                raise StopIteration

        use_base = random.random() < self.p_use_base
        input_objs, transformed_objs, transforms, obj_indices = self.rand_transform(self.cur_sample, depth=self.step_depth, use_base=use_base)
        self.cur_sample_step += len(transforms)
        
        if len(transforms) == 0:
            if self.print_steps:
                print("Moving to next sample.")
            self.cur_sample_step = 0
            self.cur_sample, self.cur_key = None, None
            raise SampleExhausted  # Properly signal the end of iteration

        self.cur_sample = transformed_objs[0]   # Set to the next sample
        x, y, obj_idx = self._process_inputs(input_objs, transformed_objs, transforms, obj_indices)

        if self.print_steps:
            print(f"Problem ID: {self.cur_key}, [{self.cur_sample_step}/{self.max_steps}]")
            print(f"Transformation applied: {', '.join([t.__name__ for t in transforms])} on {'base' if use_base else 'objects'}.")
            print("Input:")
            x[0].plot_grid()
            print("Output:")
            x[self.elems_per_image + 1].plot_grid()

        return self.cur_key, x, y, obj_idx

    def _process_inputs(self, input_objs, transformed_objs, transforms, obj_indices):
        """ Given the inputs from rand_transform, returns the data as 'x' and 'y'.
        
        x: List of ARC_Objects or <PAD> and <SEP> tokens. Length is elems_per_image*2 + 1.
        y: List of integers representing the label operations. Length is 1.
        """
        x = input_objs
        x.extend([self.pad_token] * (self.elems_per_image - len(x)) + [self.sep_token])
        x.extend(transformed_objs)
        x.extend([self.pad_token] * ((self.elems_per_image*2 + 1) - len(x)))
        
        y = [self.label_ops[t] for t in transforms]
        
        return x, y, obj_indices