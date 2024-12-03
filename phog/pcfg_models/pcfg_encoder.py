import os
import sys
import datetime

import torch
import torch.nn as nn
import torch.functional as F

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

import source as source





# Need an encoder model with a few different 'CLS' tokens that we'll use
# One will be used to get the last layer of attention and try to use that to predict the objects that we apply this to
# One will be used to classify the correct DSL operation that shoud occur next

# WIll need it to be able to accept two different types of inputs
# One with 'attended obj' and the other without.
# For the one with attended obj, you compute the two part loss plus classification
# For the one without, you will end up doing multi-class classification using only the classification head



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
        B,T,C = x.shape
        k = self.K(x)       # [B x T x head_size]
        q = self.Q(x)       # [B x T x head_size]
        v = self.V(x)       # [B x T x head_size]
        
        affinities = q @ k.transpose(-2, -1) * k.shape[-1]**(-0.5)   # [B x T x T]
        affinities = F.softmax(affinities, dim=-1)                   # [B x T x T]
        affinities = self.dropout(affinities)
        
        out = affinities @ v    # [B x T x head_size]
        return out, affinities  # [B x T x head_size]


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, device):
        """ Multiple heads of attention in parallel. """
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([AttentionHead(n_embd, head_size, dropout, device) for _ in range(n_head)])
        self.proj  = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """ x is of shape [B x T x C] """
        outputs, affinities = [], []
        for head in self.heads:
            out, affinity = head(x)
            outputs.append(out)
            affinities.append(affinity)
        
        out = torch.cat(outputs, dim=-1)   # [B x T x head_size*n_head]
        out = self.dropout(self.proj(out))   # [B x T x C]
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
        """ x is of shape [B x T x C] """
        return self.nn(x)   # [B x T x C]


class AttentionBlock(nn.Module):
    def __init__(self, n_embd, n_head, ff_hd, dropout, device):
        """ Encoder Transformer Block. """
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, dropout, device)
        self.ffwd = FeedForward(n_embd, ff_hd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, attention_maps):
        """ Input and output both of shape [B x T x C] """
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
    def __init__(self, n_embd, n_head, n_layer, ff_hd, dropout, block_size, dsl_mapping, embedding_model=None, print_statements=True, device='cpu', **kwargs):
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
            if key in ["n_embd", "n_head", "n_layer", "ff_hd", "dropout", "block_size", "dsl_mapping"]
        }

        # Instantiate special tokens
        self.pad_token = nn.Parameter(torch.randn(1, 1, n_embd, device=device) * 0.02)
        self.sep_token = nn.Parameter(torch.randn(1, 1, n_embd, device=device) * 0.02)
        self.cls_dsl_token = nn.Parameter(torch.randn(1, 1, n_embd, device=device) * 0.02)
        self.cls_obj_token = nn.Parameter(torch.randn(1, 1, n_embd, device=device) * 0.02)
        self.dsl_mapping = dsl_mapping
        
        self.pos_embd_table = nn.Embedding(num_embeddings=block_size, embedding_dim=n_embd)
        self.blocks = nn.ModuleList([AttentionBlock(n_embd, n_head, ff_hd, dropout, device) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)
        self.set_embedding_model(embedding_model)

        if print_statements:
            print(f"Vision Transformer instantiated with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input):
        """ Input list of ARC Objects and strings """
        x = self.embed_input(input)     # [T x C]
        T, C = x.shape
        x += self.pos_embd_table(torch.arange(T, device=self.device))
        
        attention_maps = None
        for block in self.blocks:
            x, attention_maps = block(x, attention_maps)
        x = self.ln_f(x)         # [B x T x C]

        return x, attention_maps

    def embed_input(self, input):
        if self.embedding_model is None:
            raise RuntimeError("self.embedding_model cannot be None.")
        x = torch.zeros((len(input), self.model_params['n_embd']))
        for i, obj in enumerate(input):
            if obj == "<PAD>":
                x[:, i] = self.pad_token
            elif obj == "<SEP>":
                x[:, i] = self.sep_token
            elif obj.isinstance(source.ARC_Object):
                obj.set_embedding(self.embedding_model)
                x[:, i] = obj.embedding
            else:
                raise NameError("Input contains object that is not '<PAD>', '<SEP>', or an ARC_Object.")
        x = torch.cat((self.cls_dsl_token, self.cls_obj_token, x), dim=1)
        
        return x

    def set_embedding_model(self, embedding_model):
        if embedding_model is None:
            self.set_embedding_model = None
        else:
            self.embedding_model = embedding_model
            # Need to turn off gradient updates for the embedding model.
            self.embedding_model.requires_grad = False

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
        print(f"Model and transformation embeddings saved to {save_path}")
    
    @classmethod
    def load_model(cls, path, print_statements=False, device='cpu'):
        """Load the model and transformation embeddings from a file."""
        checkpoint = torch.load(path, map_location=torch.device(device))
        checkpoint['model_params']['device'] = device        
        model = cls(**checkpoint['model_params'], print_statements=print_statements)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return model.to(device)