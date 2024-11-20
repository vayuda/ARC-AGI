import torch
from .vision_transformer import VisionTransformer


def get_encoder(filename, print_statements=True, device='cpu'):
    """
        Given a filename in the 'trained_models' folder, this loads and returns a vision transformer trained using the iBOT objective

    Args:
        filename (str): Example 'trained_models/vit_20241110_75k.pth'
    """  
    ViT = VisionTransformer.load_model(filename, print_statements, device)
    return ViT.to(device)