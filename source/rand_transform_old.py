import inspect
import random
from typing import List, Tuple, Callable
from collections import defaultdict
from .dsl import *
from .objects import ARC_Object
from .segmentation import extract_objects, SEG_METHODS
from .util import Color

# def generate_rand_param(param_type, base_obj: ARC_Object, obj_lst: list[ARC_Object]):
#     if param_type == Color:
#         return random.choice(list(Color))
#     elif param_type == int:
#         return random.randint(0, 16)
#     elif param_type == Tuple[int, int]:
#         return (random.randint(-16, 16), random.randint(-16, 16))
#     elif param_type == List[int]:
#         return [random.randint(0, base_obj.width), random.randint(0, base_obj.height)]
#     elif param_type == ARC_Object:
#         return random.choice(obj_lst)
#     elif param_type == list[ARC_Object]:
#         return random.choices(obj_lst)
#     else:
#         print('Unhandled type!')
#         return None

# def get_args(func: Callable, base_obj: ARC_Object, obj_lst: list[ARC_Object]) -> list:
#     sig = inspect.signature(func)
#     args = []
#     params = list(sig.parameters.values())
#     if params[0].annotation == ARC_Object:
#         args.append(base_obj)
#         for p in params[1:]:
#             param_type = p.annotation
#             args.append(generate_rand_param(param_type, base_obj, obj_lst))
#     else:
#         for p in params:
#             param_type = p.annotation
#             args.append(generate_rand_param(param_type, base_obj, obj_lst))
#     return args

# def rand_transform(obj: ARC_Object, seg_method:int =0, depth:int =6, include_base:bool =True, print_seg_method:bool =False):
#     '''
#     Input:
#     - obj: An ARC Object
#     - seg_method: 0 = random, 1 = color, 2 = contour
#     - depth: Number of transforms to chain
#     - include_base: Whether to include the base input object in the transform
#     - print_seg_method: Whether to print the segmentation method used
#     '''
#     if seg_method == 0:
#         seg_method = random.randint(1, 3)
    
#     if print_seg_method:
#         print(f'Using segmentation method: {SEG_METHODS[seg_method]}')
    
#     obj_lst = extract_objects(obj, method=SEG_METHODS[seg_method])
    
#     if include_base:
#         obj_lst.insert(0, obj)

#     transforms = {}
#     transforms = defaultdict(lambda:0, transforms)
#     for d in range(depth):
#         new_lst = []
#         func = random.choice(dsl_operations)
#         transforms[func] += 1
#         for o in obj_lst:
#             args = get_args(func, o, obj_lst)
#             new_obj = func(*args)
#             new_lst.append(new_obj)
#         obj_lst = new_lst
    
#     return obj_lst, transforms



# dsl_operations = [
#     color,
#     recolor,
#     rotate, 
#     flip,
#     delete,
#     translate,
#     tile,
#     draw_line,
# ] 

dsl_base_operations = [
    color,
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

def apply_base_transform(base_obj: ARC_Object, transform: Callable) -> ARC_Object:
    if transform.__name__ == 'color':
        new_color = random.randint(1, 9)
        base_obj = color(base_obj, new_color)
    elif transform.__name__ == 'recolor':
        active_colors = _get_active_colors([base_obj])
        recolor_color = random.choice(active_colors)
        new_color = None
        while new_color is None or new_color == recolor_color:
            new_color = random.randint(1, 9)
        base_obj = recolor(base_obj, recolor_color, new_color)
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

    if transform.__name__ == 'color':
        new_color = random.randint(1, 9)
        t_obj = color(t_obj, new_color)
    elif transform.__name__ == 'recolor':
        active_colors = _get_active_colors([t_obj])
        recolor_color = random.choice(active_colors)
        new_color = None
        while new_color is None or new_color == recolor_color:
            new_color = random.randint(1, 9)
        t_obj = recolor(t_obj, recolor_color, new_color)
    elif transform.__name__ == 'rotate':
        t_obj = rotate(t_obj)
    elif transform.__name__ == 'flip':
        t_obj = flip(t_obj)
    elif transform.__name__ == 'delete':
        t_obj = None
    elif transform.__name__ == 'translate':
        transform_x, transform_y = _get_transform_bounds(t_obj, obj_list)
        t_obj = translate(t_obj, (random.randint(transform_x), random.randint(transform_y)))
    elif transform.__name__ == 'single_copy':
        new_base = _get_background([obj_list, t_obj])
        transform_x, transform_y = _get_transform_bounds(t_obj, obj_list)
        t_obj = single_copy(new_base, t_obj, direction=(random.randint(transform_x), random.randint(transform_y)), end=1)
    elif transform.__name__ == 'copy_translate':
        new_base = _get_background([obj_list, t_obj])
        transform_x, transform_y = _get_transform_bounds(t_obj, obj_list)
        translate_x, translate_y = _get_translation(t_obj.width, t_obj.height, transform_x, transform_y)
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


def rand_transform(base_obj: ARC_Object, seg_method:int =0, depth:int =6, use_base:bool =True, print_seg_method:bool =False):
    '''
    Input:
    - base_obj: An ARC Object (base object)
    - seg_method: 0 = random, 1 = color, 2 = contour
    - depth: Number of transforms to chain
    - use_base: Whether to only use base for your transforms or individual objects
    - print_seg_method: Whether to print the segmentation method used
    '''
    if seg_method == 0:
        seg_method = random.randint(1, 3)
    if print_seg_method:
        print(f'Using segmentation method: {SEG_METHODS[seg_method]}')
    obj_list = extract_objects(base_obj, method=SEG_METHODS[seg_method])
    transforms = []

    if use_base:
        num_transforms = random.randint(1, int(depth/2))
        for _ in range(num_transforms):
            func = random.choice(dsl_base_operations)
            transforms.append(func)
            base_obj = apply_base_transform(base_obj, func)
    else:
        num_transforms = random.randint(1, depth)
        for _ in range(num_transforms):
            func = None
            while func is None and (func.__name__ is not 'delete' and len(obj_list) <= 1):
                func = random.choice(dsl_obj_operations)
            
            transforms.append(func)
            obj_list = _apply_obj_transform(obj_list, func)

    return obj_list, transforms

def _get_background_color(obj_list):
    """ Gets the int associated with the most common color across the ARC_Objects (can be 0) """
    pixel_counts = [0]*13
    for obj in obj_list:
        counts = np.bincount(obj.grid.flatten(), minlength=13)
        pixel_counts = [x + y for x, y in zip(pixel_counts, counts)]
    pixel_counts = pixel_counts[0:10]  # Exclude border and padding
    return pixel_counts.index(max(pixel_counts))

def _flatten_objects(obj_list):
    """ 
    Flattens a list of ARC_Objects into a single ARC_Object.

    Merges the objects in order of list, so the last object in the list will be on top.
    """
    new_base = _get_background(obj_list)
    for obj in obj_list:
        new_base = tile(new_base, obj, direction=(0,0), end=1)
    return new_base

def _get_active_colors(obj_list: List[ARC_Object]) -> List[int]:
    """ Returns a list of all the colors used in the objects in the list. """
    active_colors = set()
    for obj in obj_list:
        active_colors.update(set(obj.grid.flatten()))
    active_colors.remove([10, 11, 12])  # Remove border and padding pixels
    return list(active_colors)

def _get_transform_bounds(base_obj: ARC_Object, obj_list: List[ARC_Object]) -> tuple[(int, int), (int, int)]:
    """ Given a base object (object to be translated) and list of objects, returns bounds of valid translations. """
    min_x, min_y = 32, 32
    max_x, max_y = 0, 0
    obj_list.append(base_obj)
    for obj in obj_list:
        min_x = min(min_x, obj.x)
        min_y = min(min_y, obj.y)
        max_x = max(max_x, obj.x + obj.width)
        max_y = max(max_y, obj.y + obj.height)
    
    return ((min_x - base_obj.x, min_y - base_obj.y), (max_x - (base_obj.x + base_obj.width), max_y - (base_obj.y + base_obj.height)))

def _get_background(obj_list: List[ARC_Object]) -> ARC_Object:
    """ Returns a background object from a list of objects. """
    background_color = _get_background_color(obj_list)
    max_x, max_y = 0, 0
    for obj in obj_list:
        max_x = max(max_x, obj.width + obj.x)
        max_y = max(max_y, obj.height + obj.y)
    new_base = ARC_Object(np.full((max_x, max_y), background_color))
    return new_base

def _get_translation(obj_width, obj_height, range_x, range_y, sharpness=5.0):
    """
    Function that takes object dimensions and valid ranges for new coordinates.

    It uses a biased sampling that favors values close to the object's dimensions, so effective for copy_translate.
    """
    if range_x == (0, 0) and range_y == (0, 0):
        raise ValueError("Both ranges cannot be (0, 0).")

    sampled_width = np.random.uniform(*range_x)
    sampled_height = np.random.uniform(*range_y)
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
