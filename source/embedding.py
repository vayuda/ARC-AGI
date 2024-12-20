import os
import sys

import torch
import image_encoder.transformers as transformers
import image_encoder.v2_trainer as v2

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)


def load_mobilenet_v2():
    # Load pretrained mobilenet
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(pretrained=True)
    model = torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((1, 1)))
    return model

def load_resnet50():
    # Load the pretrained ResNet
    from torchvision.models import resnet50
    resnet50 = resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return resnet50

def load_ViT(filename='vit_11-21-24_400k_vF.pth', device='cpu'):
    file = f'image_encoder/trained_models/{filename}'
    filepath = os.path.join(root, file)
    model, _ = transformers.VisionTransformer.load_model(filepath, print_statements=True, device=device)
    model = model.to(device)
    return model

def load_v2_ViT(filename='vit_12-3-24_100k_v2.pth', device='cpu'):
    file = f'image_encoder/v2_trainer/trained_models_v2/{filename}'
    filepath = os.path.join(root, file)
    model = v2.VisionTransformer.load_model(filepath, print_statements=True, device=device)
    model = model.to(device)
    return model