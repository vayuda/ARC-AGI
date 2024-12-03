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


# Defining our global vars
VALID_COLORS = range(1, 10)
BORDER_PADDING_COLORS = {10, 11, 12}

# These come from source - let's keep them split since if on 'base' we only have a few real options to choose from
dsl_base_operations = [
    color,
    recolor,
    rotate, 
    flip,
    draw_line
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


# =================================================
# Core Functions
# =================================================
def apply_transformation(original_obj_list: List[ARC_Object], obj_idx: int, transformation_function) -> ARC_Object:
    """ Given a list of objects (all ARC_Objects from your input) and an idx that is the object being acted upon,
        applies the transformation, checks if it is a valid transformation (or throws error), and if valid, returns the new 
        ARC Object after flattening the object list
    """
    obj_list = deepcopy(original_obj_list)    # Avoid mutating parent list
    if obj_idx == 0:
        # This is being applied on the base - handle accordingly
        # Call get_args / etc.
        pass
    elif obj_idx < len(obj_list):
        # This is being applied to child element
        # Call get_args / etc.
        pass
    else:
        raise InvalidTransformationError(f"Invalid obj_idx provided - provided idx: {obj_idx} for length {len(obj_list)} list.")


    new_base = _flatten_objects(obj_list[0], obj_list[1:])
    if _trivial_transformation(original_obj_list[0], new_base):
        raise InvalidTransformationError("Trivial transformation detected.")
    
    # return new_base, args


# def get_args(TBU)
# Need to define. Maybe we do 'get_args_base' and 'get_args_obj' since different types of moves allowed on each?



# =================================================
# Helper Functions
# =================================================
def _trivial_transformation(prev_base: ARC_Object, new_base: ARC_Object) -> bool:
    return new_base.grid.shape == prev_base.grid.shape and np.array_equal(new_base.grid, prev_base.grid)

# def _extract_from_base(base_obj: ARC_Object, seg_method:int =0):
#     """ Extraction function that returns a list of child objects split using segmentation method """
#     if seg_method == 0:
#         seg_method = random.randint(1, 2)
#     obj_list = extract_objects(base_obj, method=SEG_METHODS[seg_method])
#     if len(obj_list) > 12 or len(obj_list) == 0:
#         seg_method = 1   # Color segmentation -- has bound of 12 colors and will return at least one object
#         obj_list = extract_objects(base_obj, method=SEG_METHODS[seg_method])
#     return obj_list, seg_method

def _get_background(base: ARC_Object, use_padding: bool=False) -> ARC_Object:
    """ Given a base element, returns a new background object. """
    background_color = 12 if use_padding else 0
    max_r = base.height + base.top_left[0]
    max_c = base.width + base.top_left[1]
    new_base = ARC_Object(np.full((max_r, max_c), background_color))
    return new_base

def _flatten_objects(base: ARC_Object, obj_list: List[ARC_Object]) -> ARC_Object:
    """ 
    Flattens a list of ARC_Objects into a single ARC_Object.

    Merges the objects in order of list, so the last object in the list will be on top.
    """
    new_base = _get_background(base)
    
    for obj in obj_list:
        for i in range(obj.height):
            r = i + obj.top_left[0]
            for j in range(obj.width):
                c = j + obj.top_left[1]
                if obj.grid[i, j] in range(0, 10) and (r >= 0 and r < new_base.height) and (c >= 0 and c < new_base.width):
                    new_base.grid[r, c] = obj.grid[i, j]

    return new_base