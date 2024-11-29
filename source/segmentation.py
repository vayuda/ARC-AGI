import cv2
import numpy as np
from collections import deque

from .objects import ARC_Object

SEG_METHODS = {
    1: 'color',
    2: 'contour',
    3: 'contour_scale',
    4: 'monochrome_contour',
}

def extract_objects(source_object, method='color', print_on_init=False, embedding_model=None):
    """
        Given an ARC_Object and extraction method, return a list of sub-objects for that ARC_Object.

        Args:
            object (ARC_Object): The input image.
            method (str): The method to use for object extraction. Options are 'color', 'contour', 'contour_scale'.
            freq_scale (bool): If true, convert the image to grayscale and scale based on freqs 
            print_on_init (bool): If True, print the grid upon initialization of the object.
            embedding_model (torch.nn.Module): If provided, use this model to generate embeddings.    
    """
    objects = []
    if type(source_object) == ARC_Object:
        image = source_object.get_grid()
    padding = np.where(image == 12, 1, 0)

    if method == 'color':
        color_masks = get_color_masks(image)
        for mask in color_masks:
            if ((mask + padding).all() == 1) or np.array_equal(mask, padding):
                continue   # Skip since same as before / just extracting padding
            new_object = ARC_Object(image=image, mask=mask, parent=source_object, embedding_model=embedding_model)
            if print_on_init:
                new_object.plot_grid()
            
            objects.append(new_object)
            source_object.add_child(new_object)
    elif method == 'contour':
        contour_masks, hierarchy = get_contour_masks(image, False)
        for mask in contour_masks:
            if ((mask + padding).all() == 1) or np.array_equal(mask, padding):
                continue   # Skip since same as before / just extracting padding
            new_object = ARC_Object(image=image, mask=mask, parent=source_object, embedding_model=embedding_model)
            if print_on_init:
                new_object.plot_grid() 
            
            objects.append(new_object)
            source_object.add_child(new_object)
    elif method == 'contour_scale':
        contour_masks, hierarchy = get_contour_masks(image, True)
        for mask in contour_masks:
            if ((mask + padding).all() == 1) or np.array_equal(mask, padding):
                continue   # Skip since same as before / just extracting padding
            new_object = ARC_Object(image=image, mask=mask, parent=source_object, embedding_model=embedding_model)
            if print_on_init:
                new_object.plot_grid() 
            
            objects.append(new_object)
            source_object.add_child(new_object)
    elif method == 'monochrome_contour':
        if print_on_init:
            print('Source Object')
            source_object.plot_grid()
        arc_objects = get_monochrome_contour(image)
        for object in arc_objects:
            if print_on_init:
                print('New Object')
                object.plot_grid()
            source_object.add_child(object)
        return arc_objects
    else:
        raise ValueError(f"Invalid method: {method}")

    return objects


def get_color_masks(image):
    """
    Generate masks for each color in a 2D NumPy array image where pixel values are integers from 0 to 9.

    Args:
        image (numpy.ndarray): A 2D NumPy array representing the image with integer values between 0 and 9.

    Returns:
        list of numpy.ndarray: A list of binary masks where '1' represents the presence of a specific color.
                               Masks are only returned for colors present in the image.
    """
    masks = []
    
    # Iterate over the possible values (0 to 9)
    for color in range(10):
        # Create a binary mask where the pixel value matches the current color
        mask = (image == color).astype(np.uint8)
        
        # Only append the mask if it contains any pixels (i.e., if the color is present)
        if np.any(mask):
            masks.append(mask)
    
    return masks


def get_contour_masks(image, freq_scale):
    """
    Apply contour detection to the input image and return the contour masks and hierarchy.

    Args:
        image (numpy.ndarray): A 2D numpy array representing the input image, with values expected to be between 0 and 9.
        freq_scale (bool): If true, convert the image to scaled grayscale (based on frequency of pixel values)

    Returns:
        tuple: A tuple containing:
            - list: A list of 2D numpy arrays (mask) where '1' represents contour areas for each contour.
            - numpy.ndarray: A hierarchy array describing parent-child contour relationships.
    """
    # Scale image values from 0-9 to 0-255
    if freq_scale:
        counts = [np.sum(image == i).item() for i in range(13)]
        twelve_count = (counts[12], 12)
        counts = sorted([(c, i) for i, c in enumerate(counts[:12]) if c != 0], reverse=True)
        counts.append(twelve_count)
        mapped_pxls = [0] * 13
        running_count = 0
        for count, pxl in counts:
            mapped_pxls[pxl] = running_count
            running_count += count
        mapped_pxls = np.array(mapped_pxls) / (image.shape[0] * image.shape[1]) * 255
        mapped_pxls = mapped_pxls.astype(np.uint8)
        scaled_image = mapped_pxls[image]
    else:
        scaled_image = (image * 255 / 9).astype(np.uint8)
    
    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(scaled_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a list of masks for each contour
    contour_masks = []
    for i in range(len(contours)):
        mask = np.zeros_like(scaled_image, dtype=np.uint8)
        cv2.drawContours(mask, contours, i, 1, thickness=cv2.FILLED)
        contour_masks.append(mask)
    
    return contour_masks, hierarchy

def get_monochrome_contour(image):
    """
    Detect objects for each color in an integer array using flood fill with 8-connectivity.
    
    Args:
        image: 2D numpy array where integers represent different colors
        
    Returns:
        A list of Arc Objects, each representing an object of a specific color
    """
    # Create a copy of the image to avoid modifying the original
    visited = np.zeros_like(image)
    # Dictionary to store loops for each color
    objects = []
    h,w = image.shape
                
    for i in range(h):
        for j in range(w):
            if visited[i, j] == 0 and image[i,j] != 0:  # Unvisited non black pixel
                color = image[i, j]
                loop_mask = np.zeros_like(image)
                loop_coords = cbfs(image, i, j)
                for y, x in loop_coords:
                    loop_mask[y, x] = 1
                    visited[y, x] = 1
                    
                objects.append(ARC_Object(image, loop_mask,color=color,start=(i,j)))
    return objects


def cbfs(image: np.array, y: int, x: int):
    '''
    Returns the coordinates of all pixels connected to the starting pixel (x, y) with the same color as the start color.
    
    Args:
        image: 2D numpy array representing the image
        x: x-coordinate of the starting pixel
        y: y-coordinate of the starting pixel
    '''
    frontier = deque([(y, x)])
    visited = set()
    target_color = image[y, x]
    while frontier:
        y, x = frontier.popleft()
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            ny, nx = y + dy, x + dx
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and image[ny, nx] == target_color and (ny, nx) not in visited:
                frontier.append((ny, nx))
                visited.add((ny, nx))
    return visited