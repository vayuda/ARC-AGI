import torch
import image_encoder.transformers as transformers



def load_resnet50():
    # Load the pretrained ResNet
    from torchvision.models import resnet50
    resnet50 = resnet50(pretrained=True)
    resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1])
    return resnet50

def load_ViT(device='cpu'):
    file = 'image_encoder/trained_models/vit_11-21-24_400k_vF.pth'
    model = transformers.VisionTransformer.load_model(file, print_statements=True, device=device)
    model = model.to(device)
    return model