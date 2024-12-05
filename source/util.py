import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw
import json
from enum import IntEnum

class Color(IntEnum):
# Commented some colors out so they cannot be randomly chosen as parameters for dsl operations
    # BLACK = 0
    BLUE = 1
    ORANGE = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    LIGHT_ORANGE = 7
    CYAN = 8
    RED = 9
    # BORDER = 10
    # IMAGEBORDER = 11
    # PAD = 12

BORDER = 10

COLOR_TO_HEX = {
    -1: '#FF6700',  # blaze orange
    0:  '#000000',  # black
    1:  '#1E93FF',  # blue
    2:  '#F93C31',  # orange
    3:  '#4FCC30',  # green
    4:  '#FFDC00',  # yellow
    5:  '#999999',  # grey
    6:  '#E53AA3',  # pink
    7:  '#FF851B',  # light orange
    8:  '#87D8F1',  # cyan
    9:  '#921231',  # red
    10: '#FFFFFF',  # white
    11: '#FF6700',  # active grid border
    12: '#D2B48C',  # image padding
}

def hex_to_rgb(hex_color):
    """ Convert a hex color to an RGB tuple with values in the range [0, 1]. """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def plot_tensors_with_colors(tensors):
    """ Provide as an iterable of 2D tensors. """
    
    num_examples = len(tensors)
    fig, axes = plt.subplots(1, num_examples, figsize=(num_examples * 3, 3))
    for i, tensor in enumerate(tensors):
        tensor_np = tensor.numpy()
        img_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in tensor_np])
        axes[i].imshow(img_rgb, interpolation='nearest')
        axes[i].axis('off')  # Hide axes    
    plt.show()


def plot_image_and_mask(image, mask=None, title=""):
    """
    Plot an image tensor with the corresponding mask.

    Args:
        image (torch.Tensor): A 2D tensor representing the image, with integer values corresponding to keys in COLOR_TO_HEX.
        mask (torch.Tensor, optional): A 2D tensor representing the mask, where '1' indicates masked areas.
        title (str, optional): The title for the plot.

    Returns:
        None. The function displays the image with the mask applied (if provided).
    """
    result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    
    for key, hex_value in COLOR_TO_HEX.items():
        rgb_value = hex_to_rgb(hex_value)
        result_image[image == key] = rgb_value

    if mask is not None:
        yellow_tint = np.array([255, 255, 153]) / 255.0
        result_image[mask == 1] = result_image[mask == 1] * 0.6 + yellow_tint * 0.4

    plt.figure(figsize=(3,3))
    plt.imshow(result_image)
    plt.title(title)
    plt.axis('off')
    plt.show()
    

def plot_grayscale(grid):
    """
    Displays a NumPy grid of values (0-255) in grayscale.
    
    Args:
        grid (numpy.ndarray): A 2D NumPy array with values from 0 to 255.
    """
    if not isinstance(grid, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    if grid.ndim != 2:
        raise ValueError("Input grid must be a 2D NumPy array.")
    if np.min(grid) < 0 or np.max(grid) > 255:
        raise ValueError("Grid values must be in the range 0-255.")
    
    plt.imshow(grid, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Turn off the axes for better display
    plt.show()
    
def visualize_object(obj):
    if obj.parent is not None:
        grid = np.zeros_like(obj.parent.grid)
        
        grid[obj.top_left[0]:obj.top_left[0] + obj.height, obj.top_left[1]:obj.top_left[1] + obj.width] = obj.grid
    else:
        grid = obj.grid
    cell_size = 13
    border_size = 1
    # Create a new grid to hold the object
    img_width = grid.shape[1] * cell_size + (grid.shape[1] + 1) * border_size
    img_height = grid.shape[0] * cell_size + (grid.shape[0] + 1) * border_size

    img = Image.new('RGB', (img_width, img_height), COLOR_TO_HEX[BORDER])
    draw = ImageDraw.Draw(img)

    # Draw colored rectangles for each cell
    for i, row in enumerate(grid):
        for j, color in enumerate(row):
            x = j * (cell_size + border_size) + border_size
            y = i * (cell_size + border_size) + border_size
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=COLOR_TO_HEX[color], outline=COLOR_TO_HEX[BORDER])
    
    return img

    
def visualize_problem(puzzle_id: str):
    with open(f"data/training/{puzzle_id}.json", 'r') as f:
        data = json.load(f)
    examples = data['train']
    
    input_grids = [(np.array(example['input']), np.array(example['output'])) for example in examples]
    test_grid = (np.array(data['test'][0]['input']), np.array(data['test'][0]['output']))
    # add borders to all grids
    input_grids = [(np.pad(input_grid, 1, constant_values=BORDER), np.pad(output_grid, 1, constant_values=BORDER)) for input_grid, output_grid in input_grids]
    test_grid = (np.pad(test_grid[0], 1, constant_values=BORDER), np.pad(test_grid[1], 1, constant_values=BORDER))
    
    #find the max dimensions of all the grids: input, output, and test
    max_height = max([grid.shape[0] for grid, _ in input_grids] + [test_grid[0].shape[0], test_grid[1].shape[0]])
    max_width = max([grid.shape[1] for grid, _ in input_grids] + [test_grid[0].shape[1], test_grid[1].shape[1]])
                    
    # Create a new grid to hold all the grids
    combined_height = (1+ max_height) * (len(input_grids )+1)+3
    combined_width = 2 * max_width + 1
    combined_grid = np.zeros((combined_height, combined_width), dtype=int)

    # Paste all input grids into the combined grid
    current_y = 0
    for input_grid, output_grid in input_grids:
        combined_grid[current_y:current_y + input_grid.shape[0], 0:input_grid.shape[1]] = input_grid
        combined_grid[current_y:current_y + output_grid.shape[0], 
                      input_grid.shape[1] + 1:input_grid.shape[1] + 1 + output_grid.shape[1]] = output_grid
        current_y += input_grid.shape[0] + 1
        
    # draw a line to separate the input and output grids
    combined_grid[current_y, :] = BORDER
    current_y += 2
    # Paste the test grid into the combined grid
    combined_grid[current_y:current_y + test_grid[0].shape[0], 0:test_grid[0].shape[1]] = test_grid[0]
    combined_grid[current_y:current_y + test_grid[1].shape[0], 
                  test_grid[0].shape[1] + 1:test_grid[0].shape[1] + 1 + test_grid[1].shape[1]] = test_grid[1]
    current_y += max_height
    # remove extra space at the bottom
    combined_grid = combined_grid[:current_y]
    combined_height = current_y
    # draw this grid
    border_size = 1
    cell_size = 13

    # Calculate image dimensions
    img_width = combined_width * cell_size + (combined_width + 1) * border_size
    img_height = combined_height * cell_size + (combined_height + 1) * border_size

    img = Image.new('RGB', (img_width, img_height), COLOR_TO_HEX[BORDER])
    draw = ImageDraw.Draw(img)

    # Draw colored rectangles for each cell
    for i, row in enumerate(combined_grid):
        for j, color in enumerate(row):
            x = j * (cell_size + border_size) + border_size
            y = i * (cell_size + border_size) + border_size
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=COLOR_TO_HEX[color], outline=COLOR_TO_HEX[BORDER])
    
    return img

def visualize_set(objects):
    max_height = max([obj.grid.shape[0] for obj  in objects])
    max_width = max([obj.grid.shape[1] for obj  in objects])
    # number of objects to render in each row
    closest_square = int(np.ceil(np.sqrt(len(objects))))
    
    combined_height = (len(objects) // closest_square) * (max_height+1)+1
    combined_width = closest_square * (max_width+1)+1
    
    combined_grid = np.zeros((combined_height, combined_width), dtype=int)
    current_y = 0
    current_x = 0
    for obj in objects:
        combined_grid[current_y:current_y + obj.grid.shape[0], current_x:current_x + obj.grid.shape[1]] = obj.grid
        current_x += max_width
        # draw a vertical line seperating the objects
        combined_grid[current_y:current_y + max_height, current_x+max_width] = BORDER
        current_x += 1
        if current_x >= combined_width:
            current_x = 0
            current_y += max_height+1


# =============================================
# Helper Functions for Objects
# =============================================
def upsize_image(image, scale_factor):
    """
    Resizes a torch tensor of integers based on the eligibility for scaling up or down.
    
    Args:
        image (torch.Tensor): A 2D tensor of integers with shape (H, W).
    
    Returns Resized image (2-d torch.Tensor of ints).
    """
    scaled_image = image.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
    return scaled_image

def pad_to_32x32(image, center_image=True):
    """ Pads the input image tensor to 32x32 with custom padding rules. """
    # Create a 1-pixel border of '11' around the image
    bordered_image = torch.nn.functional.pad(image, (1, 1, 1, 1), value=11)
    cropped_image = bordered_image[:32, :32]
    
    h, w = cropped_image.shape
    if center_image:
        x, y = (32 - h) // 2, (32 - w) // 2
    else:
        x, y = 0, 0

    pad_bottom = max(0, 32 - h)
    pad_right = max(0, 32 - w)
    
    padded_image = torch.nn.functional.pad(
        cropped_image, 
        (x, pad_right - x, y, pad_bottom - y), 
        value=12
    )

    return padded_image

def matrix_to_rgb_tensor(matrix):
    """
    Converts a 2D NumPy matrix of integers (0-12) into a 3D RGB tensor (H x W x 3).
    
    Args:
        matrix (np.ndarray): 2D NumPy array with integers ranging from 0 to 12.
    
    Returns torch.Tensor: A tensor of shape (3, H, W) with RGB values normalized to [0, 1].
    """
    height, width = matrix.shape
    rgb_array = np.zeros((height, width, 3), dtype=np.float32)  # Initialize RGB array

    # Convert each integer in the matrix to its corresponding RGB values
    for y in range(height):
        for x in range(width):
            rgb_array[y, x] = hex_to_rgb(COLOR_TO_HEX[matrix[y, x]])
    
    # Convert to PyTorch tensor and permute to (3, H, W)
    rgb_tensor = torch.tensor(rgb_array).permute(2, 0, 1)  # Convert (H x W x 3) to (3 x H x W)
    return rgb_tensor