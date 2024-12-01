import os
import sys
import random
from typing import List, Set, Callable

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

from source import *


VALID_COLORS = range(1, 10)
BORDER_PADDING_COLORS = {10, 11, 12}

dsl_base_operations = [
    recolor,
    rotate, 
    flip
]

dsl_obj_operations = [
    color,
    recolor,
    rotate, 
    flip,
    delete,
    translate,
    single_copy,
    copy_translate,
    draw_line,
]

def get_dsl_operations():
    """ Returns the list of unique DSL operations for base objects and individual objects. """
    dsl_operations = []
    for func in dsl_base_operations + dsl_obj_operations:
        if func not in dsl_operations:
            dsl_operations.append(func)
    
    return dsl_operations


def rand_transform(base_obj: ARC_Object, seg_method:int =0, depth:int =6, use_base:bool =True):
    '''
    Input:
    - base_obj: An ARC Object (base object)
    - seg_method: 0 = random, 1 = color, 2 = contour_scale, 3 = contour_color
    - depth: Number of transforms to chain
    - use_base: Whether to only use base for your transforms or individual objects
    '''
    obj_list, seg_method = _extract_from_base(base_obj, seg_method=0)
    original_objs = [base_obj] + obj_list
    transforms = []

    if use_base:
        num_transforms = random.randint(1, int(depth/2))
        for _ in range(num_transforms):
            func = random.choice(dsl_base_operations)
            transforms.append(func)
            base_obj = _apply_base_transform(base_obj, func)
        obj_list, _ = _extract_from_base(base_obj, seg_method)
    else:
        num_transforms = random.randint(1, depth)
        for _ in range(num_transforms):
            if (len(obj_list) == 0):
                break
            func = random.choice(dsl_obj_operations)
            transforms.append(func)
            temp_obj_list = _apply_obj_transform(obj_list, func)
            if temp_obj_list is None or len(temp_obj_list) == 0:
                transforms.pop()
            else:
                obj_list = temp_obj_list
        base_obj = _flatten_objects(obj_list)
    
    transformed_objs = [base_obj] + obj_list
    return original_objs, transformed_objs, transforms


# =====================================
# Helper Functions
# =====================================

def _apply_base_transform(base_obj: ARC_Object, transform: Callable) -> ARC_Object:
    if transform.__name__ == 'recolor':
        old_color, new_color = _get_recolor_colors(base_obj)
        base_obj = recolor(base_obj, old_color, new_color)
    elif transform.__name__ == 'rotate':
        base_obj = rotate(base_obj)
    elif transform.__name__ == 'flip':
        base_obj = flip(base_obj)
    else:
        print(f'{transform.__name__} not viable base transformation.')
        return base_obj

    return base_obj

def _apply_obj_transform(obj_list: List[ARC_Object], transform: Callable) -> List[ARC_Object]:
    obj_idx = random.randint(0, len(obj_list) - 1)
    t_obj = obj_list[obj_idx]
    del obj_list[obj_idx]

    if transform.__name__ == 'delete':
        if len(obj_list) == 0:  # Cannot delete if only object
            return None
        t_obj = None

    if transform.__name__ == 'color':
        new_color = random.choice(VALID_COLORS)
        t_obj = color(t_obj, new_color)
    elif transform.__name__ == 'recolor':
        old_color, new_color = _get_recolor_colors(t_obj)
        t_obj = recolor(t_obj, old_color, new_color)
    elif transform.__name__ == 'rotate':
        t_obj = rotate(t_obj)
    elif transform.__name__ == 'flip':
        t_obj = flip(t_obj)
    elif transform.__name__ == 'delete':
        pass   # Need to do this since getting issues with returning early if 'delete' checked in this
    elif transform.__name__ == 'translate':
        transform_x, transform_y = _get_transform_bounds(t_obj, obj_list)
        translate_x, translate_y = random.randint(*transform_x), random.randint(*transform_y)
        if translate_x == 0 and translate_y == 0:
            return None
        t_obj = translate(t_obj, (translate_x, translate_y))
    elif transform.__name__ == 'single_copy':
        new_base = _get_background(obj_list + [t_obj])
        transform_x, transform_y = _get_transform_bounds(t_obj, obj_list)
        translate_x, translate_y = random.randint(*transform_x), random.randint(*transform_y)
        if translate_x == 0 and translate_y == 0:
            return None
        t_obj = single_copy(new_base, t_obj, direction=(translate_x, translate_y))
    elif transform.__name__ == 'copy_translate':
        new_base = _get_background(obj_list + [t_obj])
        transform_x, transform_y = _get_transform_bounds(t_obj, obj_list)
        translate_x, translate_y = _get_translation(t_obj.width, t_obj.height, transform_x, transform_y)
        if translate_x == 0 and translate_y == 0:
            return None
        t_obj = copy_translate(new_base, t_obj, direction=(translate_x, translate_y), end=0)
    elif transform.__name__ == 'draw_line':
        obj_list.append(t_obj)
        new_base = _get_background(obj_list)
        t_obj = draw_line(new_base, [random.randint(0, t_obj.height), random.randint(0, t_obj.width)], [random.randint(0, t_obj.height), random.randint(0, t_obj.width)], random.randint(1, 9))
    else:
        print(f'{transform.__name__} not viable object transformation.')
        return obj_list
    
    if t_obj is not None:
        obj_list.append(t_obj)
    return obj_list


def _extract_from_base(base_obj: ARC_Object, seg_method:int =0):
    if seg_method == 0:
        seg_method = random.randint(1, 2)
    obj_list = extract_objects(base_obj, method=SEG_METHODS[seg_method])
    if len(obj_list) > 12 or len(obj_list) == 0:
        seg_method = 1   # Color segmentation -- has bound of 12 colors
        obj_list = extract_objects(base_obj, method=SEG_METHODS[1])
    return obj_list, seg_method

def _get_background(obj_list: List[ARC_Object]) -> ARC_Object:
    """ Returns a background object from a list of objects. """
    background_color = _get_background_color(obj_list)
    max_x, max_y = 0, 0
    for obj in obj_list:
        max_x = max(max_x, obj.width + obj.top_left[0])
        max_y = max(max_y, obj.height + obj.top_left[1])
    new_base = ARC_Object(np.full((max_x, max_y), background_color))
    return new_base

def _get_background_color(obj_list: List[ARC_Object]) -> int:
    """ Gets the int associated with the most common color across the ARC_Objects (can be 0) """
    pixel_counts = [0]*13
    for obj in obj_list:
        counts = np.bincount(obj.grid.flatten(), minlength=13)
        pixel_counts = [x + y for x, y in zip(pixel_counts, counts)]
    pixel_counts = pixel_counts[0:10]  # Exclude border and padding
    return pixel_counts.index(max(pixel_counts))

def _flatten_objects(obj_list: List[ARC_Object]) -> ARC_Object:
    """ 
    Flattens a list of ARC_Objects into a single ARC_Object.

    Merges the objects in order of list, so the last object in the list will be on top.
    """
    new_base = _get_background(obj_list)
    for obj in obj_list:
        new_base = tile(new_base, obj, direction=(0,0), end=1)
    return new_base

def _get_recolor_colors(recolor_obj: ARC_Object) -> tuple[int, int]:
    """ Returns a tuple of old and new colors for recoloring an object. """
    active_colors: Set[int] = set()
    active_colors.update(recolor_obj.grid.flatten())
    active_colors -= BORDER_PADDING_COLORS
    old_color = random.choice(list(active_colors))
    new_color = None
    while new_color is None or new_color == old_color:
        new_color = random.choice(VALID_COLORS)
    return old_color, new_color

def _get_transform_bounds(base_obj: ARC_Object, obj_list: List[ARC_Object]) -> tuple[(int, int), (int, int)]:
    """ Given a base object (object to be translated) and list of objects, returns bounds of valid translations. """
    min_x, min_y = 32, 32
    max_x, max_y = 0, 0
    obj_list.append(base_obj)
    for obj in obj_list:
        min_x = min(min_x, obj.top_left[0])
        min_y = min(min_y, obj.top_left[1])
        max_x = max(max_x, obj.top_left[0] + obj.width)
        max_y = max(max_y, obj.top_left[1] + obj.height)
    
    x_offset_min = min_x - base_obj.top_left[0]
    y_offset_min = min_y - base_obj.top_left[1]
    x_offset_max = max_x - (base_obj.top_left[0] + base_obj.width)
    y_offset_max = max_y - (base_obj.top_left[1] + base_obj.height)
    return ((x_offset_min, x_offset_max), (y_offset_min, y_offset_max))

def _get_translation(obj_width, obj_height, range_x, range_y, sharpness=5.0):
    """
    Function that takes object dimensions and valid ranges for new coordinates.

    It uses a biased sampling that favors values close to the object's dimensions, so effective for copy_translate.
    """
    if range_x == (0, 0) and range_y == (0, 0):
        return 0, 0   # Will end up voiding this operation

    sampled_width = random.randint(*range_x)
    sampled_height = random.randint(*range_y)
    sign_width = np.sign(sampled_width)
    sign_height = np.sign(sampled_height)

    x_range = (range_x[0], 0) if sign_width < 0 else (0, range_x[1])
    y_range = (range_y[0], 0) if sign_height < 0 else (0, range_y[1])

    x_value = _biased_value(x_range, sharpness, obj_width)
    y_value = _biased_value(y_range, sharpness, obj_height)

    return x_value, y_value

def _biased_value(value_range, object_length, sharpness):
    if value_range == (0, 0):
        return 0
    values = np.arange(value_range[0], value_range[1] + 1)
    
    # Define the center of the normal distribution
    center = object_length + 2 if value_range[1] > 0 else -(object_length + 2)

    # Compute probabilities of normal distribution, normalize and return a random sample
    probs = np.exp(-((values - center) ** 2) / (2 * sharpness ** 2))    
    probs /= probs.sum()
    return np.random.choice(values, p=probs)
