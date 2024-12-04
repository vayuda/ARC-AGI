import os
import sys

import torch
import torch.nn.functional as F

# Need to add root to sys.path to import source and image_encoder
current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

import source as source



# Defining our helper functions

def embed_input(input, pcfg_encoder, use_grads, device):
    x = torch.zeros((len(input), pcfg_encoder.model_params['n_embd']), device=device)
    special_tokens = pcfg_encoder.get_special_tokens(device)   # cls_dsl, cls_obj, pad, sep
    
    try:
        first_pad = input[:16].index('<PAD>')
    except ValueError:
        first_pad = 16
    
    for i, obj in enumerate(input):
        if obj == "<PAD>":
            x[i, :] = special_tokens[2, :]
        elif obj == "<SEP>":
            x[i, :] = special_tokens[3, :]
        elif isinstance(obj, source.ARC_Object):
            # x[i, :] = torch.zeros((1, 64), device=device)
            obj.set_embedding(pcfg_encoder.embedding_model, use_grads=use_grads)
            x[i, :] = obj.embedding.to(device)
        else:
            raise NameError("Input contains object that is not '<PAD>', '<SEP>', or an ARC_Object.")
    x = torch.cat((special_tokens[:2, :], x), dim=0)
    return x, first_pad


def compute_kl_loss(dsl_cls, obj_att, dsl_label, obj_label, temp=0.2):
    """
    Compute multi-label multi-class loss using KL divergence.

    Args:
        dsl_cls: Tensor of size (len(dsl)) - predicted logits for DSL classes.
        obj_att: Tensor of size (len(obj_att)) - predicted logits for object attention.
        dsl_label: Tensor of integers up to len(dsl), representing ground truth labels for DSL classes.
        obj_label: Tensor of integers up to len(obj_att), representing ground truth labels for object attention.

    Returns:
        Total loss as a scalar tensor.
    """
    # One-hot encode labels for multi-label setup
    dsl_target = torch.zeros_like(dsl_cls, dtype=torch.float)
    dsl_target[dsl_label] = 1.0
    dsl_target = dsl_target / dsl_target.sum(dim=-1, keepdim=True)  # Normalize to probabilities
    obj_target = torch.zeros_like(obj_att, dtype=torch.float)
    obj_target[obj_label] = 1.0
    obj_target = obj_target / obj_target.sum(dim=-1, keepdim=True)  # Normalize to probabilities

    # Convert predictions to log-probabilities
    dsl_log_prob = F.log_softmax(dsl_cls, dim=-1)
    obj_log_prob = F.log_softmax(obj_att, dim=-1)

    # Compute KL divergence loss
    dsl_loss = F.kl_div(dsl_log_prob, dsl_target, reduction='batchmean')  # log_target=False by default
    obj_loss = F.kl_div(obj_log_prob, obj_target, reduction='batchmean')

    return dsl_loss, obj_loss