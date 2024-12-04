import random
import numpy as np
from copy import deepcopy
from typing import List, Tuple
from .objects import ARC_Object
from .util import Color

'''
All DSL operations return a new copy of object
'''

def color(obj: ARC_Object, color: Color) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[(new_obj.grid != 0) & (new_obj.grid < 10)] = int(color)
    return new_obj

def recolor(obj: ARC_Object, orig_color: Color, new_color: Color) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid == int(orig_color)] = int(new_color)
    return new_obj

def rotate(obj: ARC_Object, num_rotations: int =None) -> ARC_Object:
    new_obj = deepcopy(obj)
    if num_rotations is None:
        num_rotations = 1
    new_obj.grid = np.rot90(new_obj.grid, num_rotations)
    new_obj.height, new_obj.width = new_obj.grid.shape
    return new_obj

def flip(obj: ARC_Object) -> ARC_Object:
    """ Mirrors over the y-axis. """
    new_obj = deepcopy(obj)
    new_obj.grid = np.fliplr(new_obj.grid)
    return new_obj

def delete(obj: ARC_Object) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid != 0] = 0
    return new_obj

def translate(obj: ARC_Object, direction: Tuple[int, int]) -> ARC_Object:
    """
    Assumes direction is a tuple of (r, c) to move the object and that direction does not take object off the base grid.
    """
    new_obj = deepcopy(obj)
    new_top_left_r = new_obj.top_left[0] + direction[0]
    new_top_left_c = new_obj.top_left[1] + direction[1]
    new_obj.top_left = (new_top_left_r, new_top_left_c)
    return new_obj

def single_copy(base: ARC_Object, tile_obj: ARC_Object, direction: Tuple[int, int]) -> ARC_Object:
    new_tile_obj = deepcopy(tile_obj)
    new_tile_obj.top_left = (tile_obj.top_left[0]+direction[0], tile_obj.top_left[1]+direction[1])
    return tile(base, new_tile_obj, direction, 1)

def copy_translate(base: ARC_Object, tile_obj: ARC_Object, direction: Tuple[int, int], end: int) -> ARC_Object:
    return tile(base, tile_obj, direction, end)

def tile(base: ARC_Object, tile_obj: ARC_Object, direction: Tuple[int, int], end: int) -> ARC_Object:
    """
    Tile the object in the given direction (r, c) 'end' number of times. If end = 0, tile until the edge of the grid.
    """
    new_obj = deepcopy(base)
    cur_pos = tile_obj.top_left     # Cur pos in form (r, c)
    if direction[0] == 0 and direction[1] == 0:
        raise ValueError("Direction cannot be (0, 0) - may cause infinite loop.")

    iterations = 0
    while end == 0 or iterations < end:
        if (cur_pos[0] + tile_obj.height < 0 or cur_pos[0] >= new_obj.height or 
            cur_pos[1] + tile_obj.width < 0 or cur_pos[1] >= new_obj.width):
            break

        for i in range(tile_obj.height):
            r = cur_pos[0] + i
            for j in range(tile_obj.width):
                c = cur_pos[1] + j
                if tile_obj.grid[i, j] in range(0, 10) and (r >= 0 and r < new_obj.height and c >= 0 and c < new_obj.width):
                    new_obj.grid[r, c] = tile_obj.grid[i, j]

        cur_pos = (cur_pos[0] + direction[0], cur_pos[1] + direction[1])
        iterations += 1

    return new_obj

def draw_line(base: ARC_Object, start: List[int], end: List[int], color: Color) -> ARC_Object:
    """
    Draw a line on the base object from start to end with the given color.
    
    Start should be in the form of (r, c) and end should be in the form of (r, c).
    """
    new_obj = deepcopy(base)
    start_r = max(0, min(new_obj.height - 1, start[0]))
    start_c = max(0, min(new_obj.width - 1, start[1]))
    end_r = max(0, min(new_obj.height - 1, end[0]))
    end_c = max(0, min(new_obj.width - 1, end[1]))

    if start_r == end_r:   # Draw horizontal line
        c_start = min(start_c, end_c)
        c_end = max(start_c, end_c)
        new_obj.grid[start_r, c_start:c_end + 1] = int(color)
    elif start_c == end_c:  # Draw vertical line
        r_start = min(start_r, end_r)
        r_end = max(start_r, end_r)
        new_obj.grid[r_start:r_end + 1, start_c] = int(color)
    elif start_r == end_r and start_c == end_c:   # Single point
        new_obj.grid[start_r, start_c] = int(color)
    else:    # Diagonal line
        r_range = range(start_r, end_r + 1) if start_r < end_r else range(start_r, end_r - 1, -1)
        c_range = range(start_c, end_c + 1) if start_c < end_c else range(start_c, end_c - 1, -1)
        # If x_range and y_range are different lengths, zip truncates to the shorter length
        for x, y in zip(r_range, c_range):
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

def majority(objs: list[ARC_Object]) -> ARC_Object:
    try:
        height, h_count = np.unique([o.height for o in objs], axis=0, return_counts=True)
        width, w_count = np.unique([o.width for o in objs], axis=0, return_counts=True)
        h = height[np.argmax(h_count)]
        w = width[np.argmax(w_count)]
        grids = []
        # Standardize sizes of all inputs. Only handles cases when outlier is larger than others,
        # which appears to be the most common case
        for o in objs:
            grid = o.grid
            while grid.shape[0] > h:
                count = np.count_nonzero(grid, axis=1)
                if count[0] >= count[-1]:
                    grid = np.delete(grid, -1, 0)
                else:
                    grid = np.delete(grid, 0, 0)
            while grid.shape[1] > w:
                count = np.count_nonzero(grid, axis=0)
                if count[0] >= count[-1]:
                    grid = np.delete(grid, -1, 1)
                else:
                    grid = np.delete(grid, 0, 1)
            grids.append(grid)
        stacked = np.stack(grids, axis=0)
        majority, _ = mode(stacked, axis=0)
        image = majority.squeeze()
        return ARC_Object(image, np.ones_like(image)) 
    except:
        return None


# Non-DSL Functions
def draw(base: ARC_Object, tile: ARC_Object, position: Tuple[int, int]) -> ARC_Object:
    """
    Draw the tile on the base object at the given position.
    
    """
    base.grid[position[0] : position[0] + tile.height, position[1] : position[1] + tile.width] = tile.grid

def and_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
    # Determine the overlapping region dimensions
    height = min(obj1.grid.shape[0], obj2.grid.shape[0])
    width = min(obj1.grid.shape[1], obj2.grid.shape[1])
    
    # Crop both grids to the overlapping region
    cropped_grid1 = obj1.grid[:height, :width]
    cropped_grid2 = obj2.grid[:height, :width]
    
    # Perform the 'and' operation on the overlapping region
    mask1 = cropped_grid1 != 0
    mask2 = cropped_grid2 != 0
    image = np.where(mask1 & mask2, cropped_grid1, 0)
    
    # Create a new ARC_Object for the result
    return ARC_Object(image, np.ones_like(image))

def or_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
    # Determine the overlapping region dimensions
    height = min(obj1.grid.shape[0], obj2.grid.shape[0])
    width = min(obj1.grid.shape[1], obj2.grid.shape[1])
    
    # Crop both grids to the overlapping region
    cropped_grid1 = obj1.grid[:height, :width]
    cropped_grid2 = obj2.grid[:height, :width]
    
    # Perform the 'and' operation on the overlapping region
    mask1 = cropped_grid1 != 0
    mask2 = cropped_grid2 != 0
    image = np.where(mask1 | mask2, np.where(cropped_grid1 == 0, cropped_grid2, cropped_grid1), 0)
    
    # Create a new ARC_Object for the result
    return ARC_Object(image, np.ones_like(image))

def xor_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
    # Determine the overlapping region dimensions
    height = min(obj1.grid.shape[0], obj2.grid.shape[0])
    width = min(obj1.grid.shape[1], obj2.grid.shape[1])
    
    # Crop both grids to the overlapping region
    cropped_grid1 = obj1.grid[:height, :width]
    cropped_grid2 = obj2.grid[:height, :width]
    
    # Perform the 'and' operation on the overlapping region
    mask1 = cropped_grid1 != 0
    mask2 = cropped_grid2 != 0
    image = np.where(mask1 ^ mask2, cropped_grid1, 0)
    
    # Create a new ARC_Object for the result
    return ARC_Object(image, np.ones_like(image))