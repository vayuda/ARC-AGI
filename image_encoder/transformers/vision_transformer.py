# Code borrowed from DINOv2
# https://github.com/facebookresearch/dinov2/tree/main
import os
import math
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F


class TransformationEmbeddings(nn.Module):
    def __init__(self, dim, transformations, device='cpu'):
        """
        Given an embedding dimension (dim) and a list of strings (transformations), 
        creates an nn.Embedding associated with each transformation.
        
        Args:
            dim (int): Embedding dimension.
            transformations (list of str): List of transformation names.
            device (str): Device for the embedding layer.
        """
        super().__init__()
        self.transformations = transformations
        self.dim = dim
        self.device = device
        self.ttoi = {t: i for i, t in enumerate(transformations)}
        self.itot = {i: t for i, t in enumerate(transformations)}
        self.embs = nn.Embedding(num_embeddings=len(transformations), embedding_dim=dim, device=device)

    def get_embedding(self, transformations):
        """
        Fetches the embedding vector associated with a given transformation.

        Args:
            transformations (dict): Dictionary of transformations.

        Returns:
            torch.Tensor: The embedding vector for the transformation.
        """
        vec = torch.zeros(self.dim, device=self.device)
        for k, v in transformations.items():
            if k in self.ttoi:
                vec += self.embs(torch.tensor(self.ttoi[k], device=self.device)) * v
        return vec

    def normalize(self):
        with torch.no_grad():
            self.embs.weight.data = F.normalize(self.embs.weight.data, p=2, dim=1)
    
    @classmethod
    def load_embeddings(cls, dim, transformations, device, embeddings):
        instance = cls(dim, transformations, device=device)
        instance.embs.load_state_dict(embeddings)
        return instance

    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, temperature, pad_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # Store q, k, v as one matrix and unpack on fwd pass
        q, k, v = qkv.unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale) / temperature
        if pad_mask is not None:
            pad_mask_flat = pad_mask.squeeze(-1).to(attn.device)
            expanded_mask = pad_mask_flat.unsqueeze(1).unsqueeze(2)
            expanded_mask = torch.transpose(expanded_mask, -1, -2) @ expanded_mask
            attn = attn.masked_fill(expanded_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(torch.nan_to_num(attn))
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0, bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        pos_enc='Sinusoidal'
    ):
        super().__init__()
        if pos_enc == 'Sinusoidal':
            self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        elif pos_enc == 'RoPE':
            self.attn = RoPEAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        else:
            raise ValueError(f"'pos_enc' must be either 'Sinusoidal' or 'RoPE' - instead is '{pos_enc}'")
        
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, temperature, pad_mask=None):
        x_temp, attn = self.attn(self.norm1(x), temperature, pad_mask)
        x = x + x_temp
        x_temp = self.mlp(self.norm2(x))
        x = x + x_temp
        return x, attn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        max_img_size=32,
        unique_patches=13,
        embed_dim=32,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        pos_enc='Sinusoidal',
        sinusoidal_theta=10000,
        head=False,
        output_classes=13,
        print_statements=True,
        device='cpu',
        **kwargs
    ):
        """
        Args:
            max_img_size (int): max input image size
            unique_patches (int): unique ints in input grids
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            pos_enc (string): 'RoPE' or 'Sinusoidal'
            sinusoidal_theta (int): Theta value for our sinusoidal positional encodings
            head (bool): if true, add final linear layer to output logits for classification
            output_classes (int): output dimension of your classification head
        """
        super().__init__()
        self.model_params = {
            key: value for key, value in locals().items()
            if key in ["max_img_size", "unique_patches", "embed_dim", "depth", "num_heads", "mlp_ratio", "qkv_bias", "drop_rate", "attn_drop_rate", "pos_enc", "sinusoidal_theta", "device"]
        }   # Store model params for easy reloading

        self.device = device
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.head = head
        self.pos_enc = pos_enc

        self.patch_embeddings = nn.Embedding(num_embeddings=unique_patches+1, embedding_dim=embed_dim, device=device)
        self.mask_token = unique_patches    # Token id for the masking token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim, device=device) * 0.02)
        if pos_enc == "Sinusoidal":
            self.register_buffer('sinusoidal_enc', self.get_sinusoidal_positional_encodings(max_img_size, embed_dim, theta=sinusoidal_theta).to(device))
        
        self.blocks = nn.Sequential(*[Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate, pos_enc=pos_enc) for i in range(depth)])
            
        if self.head:
            self.classification_head = nn.Linear(embed_dim, output_classes, bias=False)

        if print_statements:
            print(f"Vision Transformer instantiated with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters using {pos_enc} encodings.")

    def forward(self, x, mask=None, save_attn=False, temperature=1):
        B, H, W = x.shape
        if mask is not None:
            x = torch.where(mask == 1, self.mask_token, x)
        
        pad_mask = torch.where(x >= 11, 0, torch.ones(x.shape, device=self.device)).to(self.device)
        pad_mask = torch.cat((torch.ones((B, 1, 1), device=self.device), pad_mask.view(B, H*W, -1)), dim=1)
                              
        x = self.patch_embeddings(x)
        if self.pos_enc == 'Sinusoidal':
            x += self.sinusoidal_enc[:H, :W, :]
        
        x = x.view(B, H * W, -1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        attns = []
        for blk in self.blocks:
            x, attn = blk(x, temperature, pad_mask)
            if save_attn:
                attns.append(attn)
        x = self.norm(x)    # B x N x C
        
        cls_logits = x[:, :1, :]         # B x 1 x C   # Just the class token
        patch_logits = x[:, 1:, :]       # B x (N-1) x C    # Just the patch values
        if self.head:
            patch_logits = self.classification_head(patch_logits)     # B x (N-1) x output_classes
        return cls_logits, patch_logits, attns

    @staticmethod
    def get_sinusoidal_positional_encodings(max_img_size, embed_dim, theta=10000):
        """ 
        Given max_img_size, computes positional encodings using a sinusoidal method on two channels and returns a tensor of size 'max_img_size x max_img_size x embed_dim'
        """
        def _positional_encoding_2d(x, y, d):
            x, y = torch.tensor(float(x)), torch.tensor(float(y))            
            pe_x = torch.tensor([torch.sin(x / (theta ** (2 * (k // 2) / d))) if k % 2 == 0 else torch.cos(x / (theta ** (2 * (k // 2) / d)))
                                 for k in range(d // 2)])
            pe_y = torch.tensor([torch.sin(y / (theta ** (2 * (k // 2) / d))) if k % 2 == 0 else torch.cos(y / (theta ** (2 * (k // 2) / d)))
                                 for k in range(d // 2)])
            pe = torch.cat([pe_x, pe_y])
            return pe

        positional_grid = torch.zeros((max_img_size, max_img_size, embed_dim))
        for x in range(max_img_size):
            for y in range(max_img_size):
                positional_grid[x, y, :] = _positional_encoding_2d(x, y, embed_dim)        
        return positional_grid

    def save_model(self, transformation_embeddings):
        """Save the model, its parameters, and the transformation embeddings to a file."""
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"vit_{timestamp}.pth"
        save_path = os.path.join(save_dir, save_name)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_params': self.model_params,
            'transformations': transformation_embeddings.transformations,
            'embedding_state_dict': transformation_embeddings.embs.state_dict(),
            'embedding_dim': transformation_embeddings.dim
        }, save_path)
        print(f"Model and transformation embeddings saved to {save_path}")
    
    @classmethod
    def load_model(cls, path, print_statements=False, device='cpu'):
        """Load the model and transformation embeddings from a file."""
        checkpoint = torch.load(path, map_location=torch.device(device))
        checkpoint['model_params']['device'] = device        
        model = cls(**checkpoint['model_params'], print_statements=print_statements)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        transformation_embeddings = TransformationEmbeddings.load_embeddings(
            dim = checkpoint['embedding_dim'],
            transformations = checkpoint['transformations'],
            device = device,
            embeddings = checkpoint['embedding_state_dict']
        )
        
        return model.to(device), transformation_embeddings



# =====================================================
# RoPE Attention
# =====================================================
def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    else:
        freqs_cis = freqs_cis[..., :x.shape[-2], :x.shape[-1]]
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class RoPEAttention(Attention):
    """Multi-head Attention block with rotary position embeddings."""
    def __init__(self, *args, rope_theta=3.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
        freqs = init_2d_freqs(dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta, rotate=True).view(2, -1)
        self.freqs = nn.Parameter(freqs, requires_grad=True)
        
        t_x, t_y = init_t_xy(end_x=32, end_y=32)
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)
        
    def forward(self, x, temperature, pad_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        w = h = math.sqrt(x.shape[1] - 1)
        t_x, t_y = self.freqs_t_x, self.freqs_t_y
        if self.freqs_t_x.shape[0] != x.shape[1] - 1:
            t_x, t_y = init_t_xy(end_x=w, end_y=h)
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
        freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        
        q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)        
        attn = ((q * self.scale) @ k.transpose(-2, -1)) / temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn