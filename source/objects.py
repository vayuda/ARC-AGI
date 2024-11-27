import torch
import numpy as np

from util import plot_image_and_mask

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
            x_nonzeros = np.nonzero(np.sum(mask, axis=0))[0]  # Columns with non-zero values
            y_nonzeros = np.nonzero(np.sum(mask, axis=1))[0]  # Rows with non-zero values
            self.top_left = (int(y_nonzeros[0]), int(x_nonzeros[0])) if len(x_nonzeros) > 0 and len(y_nonzeros) > 0 else (0, 0)
            self.width = x_nonzeros[-1] - x_nonzeros[0] + 1
            self.height = y_nonzeros[-1] - y_nonzeros[0] + 1
            image = np.where(mask == 0, 12, image)
        # no mask just treat the whole image as the object
        else:
            self.top_left = (0, 0)
            self.width = image.shape[1]
            self.height = image.shape[0]
                
        self.start = start # coordinate of the pixel in the mask with the smallest flattened index (y * width + x)
        self.parent = parent
        self.color = color
        self.children = set()
        
        self.grid = image[self.top_left[0]:self.top_left[0] + self.height,
                          self.top_left[1]:self.top_left[1] + self.width]
        self.shape = self.grid != 0
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
    

    def set_embedding(self, embedding_model):
        grid_tensor = torch.tensor(self.grid).unsqueeze(0).to(embedding_model.device)
        with torch.no_grad():
            cls_logits, _, _ = embedding_model(grid_tensor, save_attn=False, temperature=1)
        self.embedding = cls_logits.squeeze(0).squeeze(0).cpu()


    def plot_grid(self):
        plot_image_and_mask(self.grid)