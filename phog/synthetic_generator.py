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

class InvalidTransformationError(Exception):
    pass

def get_dsl_operations():
    """ Deterministically returns the list of unique DSL operations for base objects and individual objects. """
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
    intermediate_objs = [deepcopy(base_obj)]
    attempts, max_attempts = 0, 15
    obj_idx = 0

    if use_base:
        num_transforms = random.randint(1, 3) if depth == None else depth

        while len(transforms) < num_transforms:
            attempts += 1
            if attempts > max_attempts:
                break
            
            func = random.choice(dsl_base_operations)
            try:
                new_base = _apply_base_transform(intermediate_objs[-1], func)
            except InvalidTransformationError as e:
                continue
            
            obj_list, _ = _extract_from_base(new_base, seg_method)
            if len(obj_list) == 0:
                continue

            transforms, intermediate_objs = _check_triviality(new_base, transforms+[func], intermediate_objs+[new_base])
        
    else:
        num_transforms = random.randint(1, 5) if depth == None else depth
        while len(transforms) < num_transforms:
            attempts += 1
            func = random.choice(dsl_obj_operations)
            base_obj = deepcopy(intermediate_objs[-1])
            obj_list, _ = _extract_from_base(base_obj, seg_method)
            
            if attempts > max_attempts or len(obj_list) == 0:
                break   # There are a few problems (~10) that will trip the len(obj_list) == 0 condition
            
            try:
                new_obj_list, obj_idx = _apply_obj_transform(base_obj, obj_list, func)
            except InvalidTransformationError as e:
                continue            

            new_base = _flatten_objects(intermediate_objs[-1], new_obj_list)            
            temp_obj_list, _ = _extract_from_base(new_base, seg_method)
            
            # Prevent transformations that lead to single element objects
            if len(temp_obj_list) < 2:
                continue
            
            transforms, intermediate_objs = _check_triviality(new_base, transforms+[func], intermediate_objs+[new_base])
        
    base_obj = intermediate_objs[-1]
    obj_list, _ = _extract_from_base(base_obj, seg_method)
    
    transformed_objs = [base_obj] + obj_list
    return original_objs, transformed_objs, transforms, obj_idx

def _check_triviality(new_base: ARC_Object, transforms: List, intermediate_objs: List[ARC_Object]):
    for i in range(len(intermediate_objs)-1):
        obj = intermediate_objs[i]
        if new_base.grid.shape == obj.grid.shape and np.array_equal(new_base.grid, obj.grid):
            transforms = transforms[:i]
            intermediate_objs = intermediate_objs[:i+1]
            break
    return transforms, intermediate_objs



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
        raise InvalidTransformationError(f"Invalid transformation: {transform.__name__}.")

    return base_obj

def _apply_obj_transform(base_obj: ARC_Object, obj_list: List[ARC_Object], transform: Callable) -> List[ARC_Object]:
    obj_idx = random.randint(0, len(obj_list) - 1)
    transform_obj = obj_list.pop(obj_idx)  # Remove from list to apply transformation

    if transform.__name__ == 'color':
        new_color = random.choice(VALID_COLORS)
        transform_obj = color(transform_obj, new_color)
    
    elif transform.__name__ == 'recolor':
        old_color, new_color = _get_recolor_colors(transform_obj)
        transform_obj = recolor(transform_obj, old_color, new_color)
    
    elif transform.__name__ == 'rotate':
        transform_obj = rotate(transform_obj)
    
    elif transform.__name__ == 'flip':
        transform_obj = flip(transform_obj)
    
    elif transform.__name__ == 'delete':
        if len(obj_list) <= 2:
            raise InvalidTransformationError("Cannot call delete when few objects - leads to trivial problems.")
        transform_obj = None
    
    elif transform.__name__ == 'translate':
        t_bounds_r, t_bounds_c = _get_transform_bounds(transform_obj, base_obj)
        translate_r, translate_c = random.randint(*t_bounds_r), random.randint(*t_bounds_c)
        if translate_r == 0 and translate_c == 0:
            obj_list.append(deepcopy(transform_obj))   # Need to add back to list if no translation
            raise InvalidTransformationError("Translation is trivial.")
        transform_obj = translate(transform_obj, (translate_r, translate_c))
    
    elif transform.__name__ == 'single_copy':
        new_base = _get_background([base_obj], use_padding=True)
        obj_list.append(deepcopy(transform_obj))   # Since we are copying, we need to keep the original
        t_bounds_r, t_bounds_c = _get_transform_bounds(transform_obj, new_base)
        translate_r, translate_c = random.randint(*t_bounds_r), random.randint(*t_bounds_c)
        if translate_r == 0 and translate_c == 0:
            raise InvalidTransformationError("Translation is trivial.")
        transform_obj = single_copy(new_base, transform_obj, direction=(translate_r, translate_c))
    
    elif transform.__name__ == 'copy_translate':
        new_base = _get_background([base_obj], use_padding=True)
        # No need to append back to obj_list since when you do copy_translate you recreate the object
        t_bounds_r, t_bounds_c = _get_transform_bounds(transform_obj, new_base)
        translate_r, translate_c = _get_translation(transform_obj, t_bounds_r, t_bounds_c)
        if translate_r == 0 and translate_c == 0:
            obj_list.append(deepcopy(transform_obj))   # Need to add back to list if no translation
            raise InvalidTransformationError("Translation is trivial.")
        transform_obj = copy_translate(new_base, transform_obj, direction=(translate_r, translate_c), end=0)
    
    elif transform.__name__ == 'draw_line':
        obj_list.append(deepcopy(transform_obj))
        new_base = _get_background([base_obj], use_padding=True)
        start, end = _get_draw_line_bounds(base_obj)
        transform_obj = draw_line(new_base, start, end, random.choice(VALID_COLORS))
    
    else:
        raise InvalidTransformationError(f"Invalid transformation: {transform.__name__}.")
    
    if transform_obj is not None:
        obj_list.append(transform_obj)
    return obj_list, obj_idx


def _extract_from_base(base_obj: ARC_Object, seg_method:int =0):
    if seg_method == 0:
        seg_method = random.randint(1, 2)
    obj_list = extract_objects(base_obj, method=SEG_METHODS[seg_method])
    if len(obj_list) > 12 or len(obj_list) == 0:
        seg_method = 1   # Color segmentation -- has bound of 12 colors and will return at least one object
        obj_list = extract_objects(base_obj, method=SEG_METHODS[seg_method])
    return obj_list, seg_method

def _get_background(obj_list: List[ARC_Object], use_padding: bool =False) -> ARC_Object:
    """ Returns a background object from a list of objects. """
    # background_color = 12 if use_padding else _get_background_color(obj_list)
    background_color = 12 if use_padding else 0
    max_r, max_c = 0, 0
    for obj in obj_list:
        max_r = max(max_r, obj.height + obj.top_left[0])
        max_c = max(max_c, obj.width + obj.top_left[1])
    new_base = ARC_Object(np.full((max_r, max_c), background_color))
    return new_base

# def _get_background_color(obj_list: List[ARC_Object]) -> int:
#     """ Gets the int associated with the most common color across the ARC_Objects (can be 0) """
#     pixel_counts = [0]*13
#     for obj in obj_list:
#         counts = np.bincount(obj.grid.flatten(), minlength=13)
#         pixel_counts = [x + y for x, y in zip(pixel_counts, counts)]
#     pixel_counts = pixel_counts[0:10]  # Exclude border and padding
#     return pixel_counts.index(max(pixel_counts))

def _flatten_objects(base: ARC_Object, obj_list: List[ARC_Object]) -> ARC_Object:
    """ 
    Flattens a list of ARC_Objects into a single ARC_Object.

    Merges the objects in order of list, so the last object in the list will be on top.
    """
    new_base = _get_background([base])
    
    for obj in obj_list:
        for i in range(obj.height):
            r = i + obj.top_left[0]
            for j in range(obj.width):
                c = j + obj.top_left[1]
                if obj.grid[i, j] in range(0, 10) and (r >= 0 and r < new_base.height) and (c >= 0 and c < new_base.width):
                    new_base.grid[r, c] = obj.grid[i, j]

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

def _get_draw_line_bounds(base_obj: ARC_Object) -> tuple[int, int]:
    """ Returns the start and end coordinates for drawing a line on the base object. """
    start_r = random.randint(0, base_obj.height - 1)
    end_r = random.randint(start_r, base_obj.height - 1)
    start_c = random.randint(0, base_obj.width - 1)
    end_c = random.randint(start_c, base_obj.width - 1)
    return [start_r, start_c], [end_r, end_c]

def _get_transform_bounds(translate_object: ARC_Object, base_obj: ARC_Object) -> tuple[(int, int), (int, int)]:
    """ Given a translate object and base object (root object that defines bounds), returns bounds of valid translations. """
    min_r = base_obj.top_left[0]
    max_r = base_obj.top_left[0] + base_obj.height
    min_c = base_obj.top_left[1]
    max_c = base_obj.top_left[1] + base_obj.width

    r_offset_min = min_r - translate_object.top_left[0]
    r_offset_max = max_r - (translate_object.top_left[0] + translate_object.height)
    c_offset_min = min_c - translate_object.top_left[1]
    c_offset_max = max_c - (translate_object.top_left[1] + translate_object.width)
    
    return ((r_offset_min, r_offset_max), (c_offset_min, c_offset_max))

def _get_translation(obj, range_r, range_c, sharpness=5.0):
    """
    Function that takes object dimensions and valid ranges for new coordinates.

    It uses a biased sampling that favors values close to the object's dimensions, so effective for copy_translate.
    """
    if range_r == (0, 0) and range_c == (0, 0):
        return 0, 0   # Will end up voiding this operation

    sampled_sign_r = np.sign(random.randint(*range_r))
    sampled_sign_c = np.sign(random.randint(*range_c))
    range_r = (range_r[0], 0) if sampled_sign_r < 0 else (0, range_r[1])
    range_c = (range_c[0], 0) if sampled_sign_c < 0 else (0, range_c[1])

    r_value = _biased_value(range_r, obj.height, sharpness)
    c_value = _biased_value(range_c, obj.width, sharpness)
    return r_value, c_value

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
