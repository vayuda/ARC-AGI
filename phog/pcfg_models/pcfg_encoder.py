import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

import source as source
import image_encoder as image_encoder


class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout, device):
        """
            One head of attention.

            Args:
                n_embd: Embedding dimensionality
                head_size: Internal head dimension - this is also our output dim
                head_num: Relevant for if you're using AliBi - tells which head number you are for defining 'm'
                device: pytorch device to compute on
        """
        super().__init__()
        self.device = device
        self.K = nn.Linear(n_embd, head_size, bias=False)    # [C x head_size]
        self.Q = nn.Linear(n_embd, head_size, bias=False)    # [C x head_size]
        self.V = nn.Linear(n_embd, head_size, bias=False)    # [C x head_size]
        self.dropout = nn.Dropout(dropout)            
        
    def forward(self, x):
        """ 
            Input:  [B x T x C] 
            Output: [B x T x head_size] 
        """
        k = self.K(x)       # [T x head_size]
        q = self.Q(x)       # [T x head_size]
        v = self.V(x)       # [T x head_size]
        
        affinities = q @ k.T * k.shape[-1]**(-0.5)   # [T x T]
        obj_affinities = affinities[1:2, :].clone()  # [1 x T]
        affinities = F.softmax(affinities, dim=-1)   # [T x T]
        affinities = self.dropout(affinities)
        
        out = affinities @ v    # [T x head_size]
        return out, obj_affinities  # [T x head_size]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, device):
        """ Multiple heads of attention in parallel. """
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([AttentionHead(n_embd, head_size, dropout, device) for _ in range(n_head)])
        self.proj  = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """ x is of shape [T x C] """
        outputs, affinities = [], []
        for head in self.heads:
            out, obj_affinity = head(x)
            outputs.append(out)
            affinities.append(obj_affinity)
        
        out = torch.cat(outputs, dim=-1)   # [T x head_size*n_head]
        out = self.dropout(self.proj(out))   # [T x C]
        return out, affinities
                 

class FeedForward(nn.Module):
    def __init__(self, n_embd, ff_hd, dropout):
        """ Simple 2-layer feed-forward network. """
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(n_embd, ff_hd),
            nn.ReLU(),
            nn.Linear(ff_hd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """ x is of shape [T x C] """
        return self.nn(x)   # [T x C]


class AttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head, ff_hd, dropout, device):
        """ Encoder Transformer Block. """
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, dropout, device)
        self.ffwd = FeedForward(n_embd, ff_hd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_maps):
        """ Input and output both of shape [T x C] """
        x_res = x
        x, att_map = self.attention(self.ln1(x))
        x = x_res + x
        x = x + self.ffwd(self.ln2(x))
        if attention_maps is None:
            attention_maps = att_map
        else:
            attention_maps += att_map
        return x, attention_maps


class PCFG_Encoder(nn.Module):
    def __init__(self, n_embd, n_head, n_layer, ff_hd, dropout, block_size, dsl_mapping, embedding_model_fp, freeze_emb_model=True, print_statements=True, device='cpu', **kwargs):
        """
        Encoder Model from Attention is All You Need 

        Args:
            n_embd: Embedding dimensions
            n_head: Number of attention heads
            n_layer: Number of transformer layers
            ff_hd: Feedforward hidden dimensionality
            dropout: Dropout rate to be used across model
            block_size: Max context length you'll see (does not include the 2 special CLS tokens)
            dsl_mapping: Dict of 'dsl_func' to 'idx' that you use for labeling with this specific model. Not used here, but store for future use
            device: torch.device for computation (e.g., 'cuda' or 'cpu')
        """
        super().__init__()
        self.device = device
        self.model_params = {
            key: value for key, value in locals().items()
            if key in ["n_embd", "n_head", "n_layer", "ff_hd", "dropout", "block_size", "dsl_mapping", "embedding_model_fp", "freeze_emb_model"]
        }

        # Instantiate special tokens
        self.special_tokens = nn.Parameter(torch.randn(4, n_embd, device=device) * 0.02)   # cls_dsl, cls_obj, pad, sep
        self.dsl_mapping = dsl_mapping
        
        self.pos_embd_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd, device=device)
        self.blocks = nn.ModuleList([AttentionBlock(n_embd, n_head, ff_hd, dropout, device) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.dsl_class_head = nn.Linear(n_embd, len(dsl_mapping), bias=False)
        self.apply(self._init_weights)
        self.load_embedding_model(embedding_model_fp, freeze_emb_model)
        
        if print_statements:
            print(f"PCFG encoder instantiated with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """ Input list of ARC Objects and strings """
        T, C = x.shape
        x += self.pos_embd_table(torch.arange(T, device=self.device))
        
        attention_maps = None
        for i, block in enumerate(self.blocks):
            if i == len(self.blocks)-1:
                x, attention_maps = block(x, attention_maps)
            else:
                x, _ = block(x, attention_maps)
        x = self.ln_f(x)         # [T x C]
        dsl_cls = self.dsl_class_head(x[0, :])
        obj_att = torch.cat(attention_maps, dim=0)   # num_heads x T
        obj_att = obj_att.sum(dim=0)        
        return dsl_cls, obj_att
    
    def get_special_tokens(self, device):
        return self.special_tokens.to(device)

    def load_embedding_model(self, embedding_model_fp, freeze_emb_model):
        self.embedding_model_fp = embedding_model_fp
        self.freeze_emb_model = freeze_emb_model
        
        if self.embedding_model_fp == "mobilenet_v2":
            self.embedding_model = source.embedding.load_mobilenet_v2().to(self.device)
        else:
            self.embedding_model = source.embedding.load_v2_ViT(embedding_model_fp, device=self.device)
            if freeze_emb_model:
                for param in self.embedding_model.parameters():
                    param.requires_grad = False

    def save_model(self):
        """Save the model to a file."""
        save_dir = "trained_pcfg_models"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"pcfg_encoder_{timestamp}.pth"
        save_path = os.path.join(save_dir, save_name)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_params': self.model_params,
        }, save_path)
        print(f"Model and embedding model saved to {save_path}")

    @classmethod
    def load_model(cls, path, print_statements=False, device='cpu'):
        """Load the model and embedding model from a file."""
        checkpoint = torch.load(path, map_location=torch.device(device))
        checkpoint['model_params']['device'] = device                
        model = cls(**checkpoint['model_params'], print_statements=print_statements)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model.to(device)
