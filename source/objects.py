import torch
import numpy as np
import matplotlib.pyplot as plt

from .util import plot_image_and_mask, upsize_image, pad_to_32x32, matrix_to_rgb_tensor

class ARC_Object:
    def __init__(self, image, mask=None, start=None, color=None, parent=None, embedding_model=None):
        """
            ARC_Object class to store the grid and other information of the object in the image.
            Note: All pixels with value of '12' are 'padding' pixels.

            Args:
                image (numpy.ndarray): A 2D numpy array representing the image.
                mask (numpy.ndarray): A 2D numpy array representing the mask of the object.
                parent (ARC_Object): If provided, assign a pointer to your parent object.
                embedding_model (torch.nn.Module): If provided, use this model to generate embeddings.
        """
        # Get our positional information and num active pixels
        self.active_pixels = np.sum(mask)
        # Compute our positions
        if mask is not None:
            if mask.shape != image.shape:
                raise ValueError("Mask and image must have the same shape.")
            x_nonzeros = np.nonzero(np.sum(mask, axis=0))[0]  # Columns with non-zero values
            y_nonzeros = np.nonzero(np.sum(mask, axis=1))[0]  # Rows with non-zero values
            self.N = int(y_nonzeros[0])
            self.S = int(y_nonzeros[-1])
            self.W =  int(x_nonzeros[0])
            self.E =  int(x_nonzeros[-1])
            self.top_left = (int(y_nonzeros[0]), int(x_nonzeros[0])) if len(x_nonzeros) > 0 and len(y_nonzeros) > 0 else (0, 0)
            self.width = int(x_nonzeros[-1] - x_nonzeros[0] + 1)
            self.height = int(y_nonzeros[-1] - y_nonzeros[0] + 1)
            image = np.where(mask == 0, 12, image)
        # no mask just treat the whole image as the object
        else:
            self.top_left = (0, 0)
            self.width = int(image.shape[1])
            self.height = int(image.shape[0])
                
        self.start = start # coordinate of the pixel in the mask with the smallest flattened index (y * width + x)
        self.parent = parent
        self.color = color
        self.children = set()
        
        self.grid = image[self.top_left[0]:self.top_left[0] + self.height,
                          self.top_left[1]:self.top_left[1] + self.width]
        self.shape = self.grid != 12
        if embedding_model is not None:
            self.set_embedding(embedding_model)
        else:
            self.embedding = None

    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, child):
        self.children.add(child)

    def remove_child(self, child):
        self.children.discard(child)

    def get_grid(self):
        return self.grid
    
    def get_center(self):
        return (self.top_left[0] + self.height // 2, self.top_left[1] + self.width // 2)

    def set_embedding(self, embedding_model, pad_to_square=True, square_size=32, use_grads=False, send_to_cpu=False):
        grid_tensor = torch.tensor(self.grid)
        if pad_to_square:
            max_r, max_c = self.height, self.width
            scale_factor = int(32 / max(max_r, max_c))
            grid_tensor = upsize_image(grid_tensor, scale_factor)   # Scale up image - makes it easier to train on
            grid_tensor = pad_to_32x32(grid_tensor, center_image=True)   # Pad to 32x32

            if square_size != 32:
                # Since we'll be using external models, map colors
                color_map = {11: 10, 12: 10}
                grid_tensor = [[color_map[p] if p in color_map else p for p in row] for row in grid_tensor.numpy()]
                grid_tensor = matrix_to_rgb_tensor(grid_tensor)                
                
                grid_tensor = torch.nn.functional.interpolate(
                    grid_tensor.unsqueeze(0).float(),
                    size=(square_size, square_size),
                    mode='nearest'
                ).squeeze(0)
                
                # Plot to check work
                plt.imshow(grid_tensor.permute(1, 2, 0).cpu().numpy())
                plt.axis('off')
                plt.show()
           

        # Embed our grid now
        grid_tensor = grid_tensor.to(embedding_model.device).unsqueeze(0)
        if use_grads:
            cls_logits, _, _ = embedding_model(grid_tensor, save_attn=False, temperature=1)
        else:
            with torch.no_grad():
                cls_logits, _, _ = embedding_model(grid_tensor, save_attn=False, temperature=1)
    
        cls_logits = cls_logits / cls_logits.norm(dim=-1)
        self.embedding = cls_logits.squeeze(0).squeeze(0)
        if send_to_cpu:
            self.embedding.to('cpu')

    def plot_grid(self):
        plot_image_and_mask(self.grid)