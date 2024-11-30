import numpy as np
from copy import deepcopy
from scipy.stats import mode
from typing import List, Tuple
from .objects import ARC_Object
from itertools import permutations
from .util import Color

'''
All DSL operations return a new copy of object
'''

def color(obj: ARC_Object, color: Color) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid != 0] = int(color)
    return new_obj

def recolor(obj: ARC_Object, orig_color: Color, new_color: Color) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid == int(orig_color)] = int(new_color)
    return new_obj

def rotate(obj: ARC_Object, num_rotations: int =None) -> ARC_Object:
    new_obj = deepcopy(obj)
    if num_rotations is None:
        num_rotations = np.random.randint(1, 3)
    new_obj.grid = np.rot90(new_obj.grid, num_rotations)
    new_obj.height, new_obj.width = new_obj.grid.shape
    return new_obj

# flips left to right, if we need up/down, then rotate first
def flip(obj: ARC_Object) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid = np.fliplr(new_obj.grid)
    return new_obj

def delete(obj: ARC_Object) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid != 0] = 0
    return new_obj

def translate(obj: ARC_Object, direction: Tuple[int, int]) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_grid = np.zeros_like(new_obj.grid)

    new_top_left_x = new_obj.top_left[0] + direction[0]
    new_top_left_y = new_obj.top_left[1] + direction[1]

    if new_top_left_x + obj.height <= 0:
        new_top_left_x = 1 - obj.height
    elif new_top_left_x >= obj.height:
        new_top_left_x = obj.height - 1

    if new_top_left_y + obj.width <= 0:
        new_top_left_y = 1 - obj.width
    elif new_top_left_y >= obj.width:
        new_top_left_y = obj.width - 1

    src_x_start = max(0, -new_top_left_x)
    src_y_start = max(0, -new_top_left_y)
    src_x_end = min(obj.height, new_obj.height - new_top_left_x)
    src_y_end = min(obj.width, new_obj.width - new_top_left_y)

    dest_x_start = max(0, new_top_left_x)
    dest_y_start = max(0, new_top_left_y)
    dest_x_end = dest_x_start + (src_x_end - src_x_start)
    dest_y_end = dest_y_start + (src_y_end - src_y_start)

    if src_x_start < src_x_end and src_y_start < src_y_end:
        new_grid[dest_x_start:dest_x_end, dest_y_start:dest_y_end] = obj.grid[
            src_x_start:src_x_end, src_y_start:src_y_end
        ]

    new_obj.grid = new_grid
    new_obj.top_left = (max(0, new_top_left_x), max(0, new_top_left_y))
    return new_obj

def single_copy(base: ARC_Object, tile: ARC_Object, direction: Tuple[int, int]) -> ARC_Object:
    return tile(base, tile, direction, 1)

def copy_translate(base: ARC_Object, tile: ARC_Object, direction: Tuple[int, int], end: int) -> ARC_Object:
    return tile(base, tile, direction, end)

def tile(base: ARC_Object, tile: ARC_Object, direction: Tuple[int, int], end: int) -> ARC_Object:
    """
    Tile the object in the given direction end times. If end = 0, tile until the edge of the grid.
    """
    new_obj = deepcopy(base)
    cur_pos = tile.top_left

    iterations = 0
    while end == 0 or iterations < end:
        if (cur_pos[0] < 0 or cur_pos[1] < 0 or
            cur_pos[0] + tile.height > new_obj.height or
            cur_pos[1] + tile.width > new_obj.width):
            break

        for y in range(tile.height):
            for x in range(tile.width):
                if tile.grid[y, x] in range(-1, 10):
                    new_obj.grid[cur_pos[0] + y, cur_pos[1] + x] = tile.grid[y, x]

        cur_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
        iterations += 1

    return new_obj

def draw_line(base: ARC_Object, start: List[int], end: List[int], color: Color) -> ARC_Object:
    """
    Draw a line on the base object from start to end with the given color.
    
    """
    new_obj = deepcopy(base)
    start_x = max(0, min(new_obj.height - 1, start[0]))
    start_y = max(0, min(new_obj.width - 1, start[1]))
    end_x = max(0, min(new_obj.height - 1, end[0]))
    end_y = max(0, min(new_obj.width - 1, end[1]))

    if start_x == end_x:
        y_start = min(start_y, end_y)
        y_end = max(start_y, end_y)
        new_obj.grid[start_x, y_start:y_end + 1] = int(color)
    elif start_y == end_y:
        x_start = min(start_x, end_x)
        x_end = max(start_x, end_x)
        new_obj.grid[x_start:x_end + 1, start_y] = int(color)
    elif start_y - end_y == start_x - end_x:
        x_range = range(start_x, end_x + 1) if start_x < end_x else range(start_x, end_x - 1, -1)
        y_range = range(start_y, end_y + 1) if start_y < end_y else range(start_y, end_y - 1, -1)
        for x, y in zip(x_range, y_range):
            new_obj.grid[x, y] = int(color)

    return new_obj

dsl_operations = [
    color,
    recolor,
    rotate, 
    flip,
    delete,
    translate,
    tile,
    draw_line,
]   



# def transpose(obj: ARC_Object) -> ARC_Object:
#     new_obj = deepcopy(obj)
#     new_obj.grid = new_obj.grid.T
#     return new_obj

# top_left and size are (width, height) 
# def crop(obj: ARC_Object, top_left: Tuple[int, int], size: Tuple[int, int]) -> ARC_Object:
#     image = obj.grid[top_left[1] : top_left[1] + size[1], top_left[0] : top_left[0] + size[0]]
#     return ARC_Object(image, np.ones_like(image))

# def remove_loose(obj: ARC_Object) -> ARC_Object:
#     '''
#     For cleaning up grids with several clusters and random loose pixels.
#     Check for all 2-by-2 sub-grids a pixel belongs to, if any one of them is fully colored;
#     if none, remove the pixel.
#     Still leaves some loose ends, but 2-by-2 seems to work the best overall.
#     '''
#     mask = obj.grid != 0
#     color = np.argwhere(mask)
#     for x, y in color:
#         retain = False
#         for i in range(max(0, x - 1), min(mask.shape[0] - 1, x) + 1):
#             for j in range(max(0, y - 1), min(mask.shape[1] - 1, y) + 1):
#                 if i + 1 < mask.shape[0] and j + 1 < mask.shape[1]:
#                     if np.all(mask[i : i + 2, j : j + 2]):
#                         retain = True
#                         break
#             if retain:
#                 break
#         if not retain:
#             mask[x][y] = False
#     image = np.where(mask, obj.grid, 0)
#     return ARC_Object(image, np.ones_like(image))

# def or_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
#     # color of obj1 takes precedence in new object
#     mask1 = obj1.grid != 0
#     mask2 = obj2.grid != 0
#     image = np.where(mask1 | mask2, np.where(obj1.grid == 0, obj2.grid, obj1.grid), 0)
#     return ARC_Object(image, np.ones_like(image))

# def and_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
#     # color of obj1 takes precedence in new object
#     mask1 = obj1.grid != 0
#     mask2 = obj2.grid != 0
#     image = np.where(mask1 & mask2, obj1.grid, 0)
#     return ARC_Object(image, np.ones_like(image))

# def xor_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
#     # color of obj1 takes precedence in new object
#     mask1 = obj1.grid != 0
#     mask2 = obj2.grid != 0
#     image = np.where(mask1 ^ mask2, obj1.grid, 0)
#     return ARC_Object(image, np.ones_like(image))

# def majority(objs: list[ARC_Object]) -> ARC_Object:
#     height, h_count = np.unique([o.height for o in objs], axis=0, return_counts=True)
#     width, w_count = np.unique([o.width for o in objs], axis=0, return_counts=True)
#     h = height[np.argmax(h_count)]
#     w = width[np.argmax(w_count)]
#     grids = []
#     # Standardize sizes of all inputs. Only handles cases when outlier is larger than others,
#     # which appears to be the most common case
#     for o in objs:
#         grid = o.grid
#         while grid.shape[0] > h:
#             count = np.count_nonzero(grid, axis=1)
#             if count[0] >= count[-1]:
#                 grid = np.delete(grid, -1, 0)
#             else:
#                 grid = np.delete(grid, 0, 0)
#         while grid.shape[1] > w:
#             count = np.count_nonzero(grid, axis=0)
#             if count[0] >= count[-1]:
#                 grid = np.delete(grid, -1, 1)
#             else:
#                 grid = np.delete(grid, 0, 1)
#         grids.append(grid)
#     stacked = np.stack(grids, axis=0)
#     majority, _ = mode(stacked, axis=0)
#     image = majority.squeeze()
#     return ARC_Object(image, np.ones_like(image)) 


# Non-DSL Functions
def draw(base: ARC_Object, tile: ARC_Object, position: Tuple[int, int]) -> ARC_Object:
    """
    Draw the tile on the base object at the given position.
    
    """
    base.grid[position[0] : position[0] + tile.height, position[1] : position[1] + tile.width] = tile.grid
