import inspect
import random
from typing import List, Tuple, Callable
from collections import defaultdict
from .dsl import *
from .objects import ARC_Object
from .segmentation import extract_objects, SEG_METHODS
from .util import Color

def generate_rand_param(param_type, base_obj: ARC_Object, obj_lst: list[ARC_Object]):
    if param_type == Color:
        return random.choice(list(Color))
    elif param_type == int:
        return random.randint(0, 16)
    elif param_type == Tuple[int, int]:
        return (random.randint(-16, 16), random.randint(-16, 16))
    elif param_type == List[int]:
        return [random.randint(0, base_obj.width), random.randint(0, base_obj.height)]
    elif param_type == ARC_Object:
        return random.choice(obj_lst)
    elif param_type == list[ARC_Object]:
        return random.choices(obj_lst)
    else:
        print('Unhandled type!')
        return None

def get_args(func: Callable, base_obj: ARC_Object, obj_lst: list[ARC_Object]) -> list:
    sig = inspect.signature(func)
    args = []
    params = list(sig.parameters.values())
    if params[0].annotation == ARC_Object:
        args.append(base_obj)
        for p in params[1:]:
            param_type = p.annotation
            args.append(generate_rand_param(param_type, base_obj, obj_lst))
    else:
        for p in params:
            param_type = p.annotation
            args.append(generate_rand_param(param_type, base_obj, obj_lst))
    return args

def rand_transform(obj: ARC_Object, seg_method:int =0, depth:int =6, include_base:bool =True):
    '''
    Input:
    - obj: An ARC Object
    - seg_method: 0 = random, 1 = color, 2 = contour
    - depth: Number of transforms to chain
    - include_base: Whether to include the base input object in the transform
    '''
    if seg_method == 0:
        seg_method = random.randint(1, 4)
    
    print(f'Using segmentation method: {SEG_METHODS[seg_method]}')
    obj_lst = extract_objects(obj, method=SEG_METHODS[seg_method])
    
    if include_base:
        obj_lst.insert(0, obj)

    transforms = {}
    transforms = defaultdict(lambda:0, transforms)
    for d in range(depth):
        new_lst = []
        func = random.choice(dsl_operations)
        transforms[func] += 1
        for o in obj_lst:
            args = get_args(func, o, obj_lst)
            new_obj = func(*args)
            new_lst.append(new_obj)
        obj_lst = new_lst
    
    return obj_lst, transforms
