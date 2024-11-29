import os
import csv
from datetime import datetime

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



# ========================================================================
# Loss Functions
# ========================================================================
def compute_loss(u, u_masks, v, v_masks, student_net, teacher_net, transformations, transformation_embeddings):
    B, H, W = u.shape
    
    # Forward pass
    u_s_class, u_s_patch, _ = student_net(u, u_masks)
    u_t_class, u_t_patch, _ = teacher_net(u)
    v_s_class, v_s_patch, _ = student_net(v, v_masks)
    v_t_class, v_t_patch, _ = teacher_net(v)
    
    # Compute loss
    u_s_class, u_t_class = u_s_class.squeeze(1), u_t_class.squeeze(1)   # B x dim
    v_s_class, v_t_class = v_s_class.squeeze(1), v_t_class.squeeze(1)
    t_delta = get_transformation_vector(transformations, transformation_embeddings)
    loss_1 = _compute_infoNCE_loss(u_s_class, v_t_class-t_delta)
    loss_2 = _compute_infoNCE_loss(u_t_class, v_s_class-t_delta)
    return loss_1 + loss_2


def get_transformation_vector(transformations, transformation_embeddings):
    vecs = []
    transformation_embeddings.normalize()
    for t in transformations:
        vecs.append(transformation_embeddings.get_embedding(t))
    return torch.stack(vecs).to(transformation_embeddings.device)


def _compute_infoNCE_loss(embeddings_1, embeddings_2, temperature=1):
    """Compute InfoNCE loss between two sets of embeddings.
       Each input should have shape [B, C], where matching indices are positive pairs."""
    B = embeddings_1.shape[0]    
    embeddings_1 = F.normalize(embeddings_1, dim=1)  # [B, C]
    embeddings_2 = F.normalize(embeddings_2, dim=1)  # [B, C]
    similarity_matrix = torch.matmul(embeddings_1, embeddings_2.T) / temperature  # [B, B]
    labels = torch.arange(B, device=embeddings_1.device)
    loss = F.cross_entropy(similarity_matrix, labels)    
    return loss


def get_eval_loss(student_net, teacher_net, loader, transformation_embeddings, device, max_eval_iters=None):
    with torch.no_grad():
        eval_loss, loss_iters = 0, 0
        for _, (ids, u, u_masks, v, v_masks, transformations) in enumerate(loader):
            u, u_masks, v, v_masks = u.to(device), u_masks.to(device), v.to(device), v_masks.to(device)
            loss_iters += 1            
            loss = compute_loss(u, u_masks, v, v_masks, student_net, teacher_net, transformations, transformation_embeddings)
            eval_loss += loss.cpu().item()
            if max_eval_iters is not None and loss_iters == max_eval_iters:
                break
    
    return (eval_loss/loss_iters)



# ========================================================================
# Plotting Functions
# ========================================================================
def update_teacher_weights(student_net, teacher_net, ema_alpha):
    """Updates the teacher network parameters using EMA from the student network."""
    with torch.no_grad():
        for student_param, teacher_param in zip(student_net.parameters(), teacher_net.parameters()):
            if isinstance(student_param, torch.nn.Embedding):
                teacher_param.data.copy_(student_param.data)
            else:
                teacher_param.data.mul_(1 - ema_alpha).add_(student_param.data, alpha=ema_alpha)



# ========================================================================
# Plotting Functions
# ========================================================================
COLOR_TO_HEX = {
    -1: '#FF6700',  # blaze orange
    0:  '#000000',  # black
    1:  '#1E93FF',  # blue
    2:  '#F93C31',  # orange
    3:  '#4FCC30',  # green
    4:  '#FFDC00',  # yellow
    5:  '#999999',  # grey
    6:  '#E53AA3',  # pink
    7:  '#FF851B',  # light orange
    8:  '#87D8F1',  # cyan
    9:  '#921231',  # red
    10: '#555555',  # masked token
    11: '#FF6700',  # border
    12: '#000000',  # outside image
}


def hex_to_rgb(hex_color):
    """ Convert a hex color to an RGB tuple with values in the range [0, 1]. """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def get_idx(x, y, CLS=False, size=(32,32)):
    """
    Get the index for a given (x, y) position in a 32x32 grid with an optional CLS token.
    If CLS is True, returns 0. Otherwise, calculates and returns the unrolled index + 1.
    """
    if CLS:
        return 0
    return y * size[1] + x + 1  # +1 to account for CLS token


def plot_tensors_with_colors(tensors, title=None):
    """
    Plot an iterable of 2D tensors with optional title.

    Args:
        tensors (iterable): Iterable of 2D tensors to plot.
        title (str, optional): Title for the chart. Defaults to None.
    """
    num_examples = len(tensors)
    fig, axes = plt.subplots(1, num_examples, figsize=(num_examples * 3, 3))
    if num_examples == 1:
        axes = [axes]
    if title is not None:
        fig.suptitle(title)
    for i, tensor in enumerate(tensors):
        tensor_np = tensor.numpy()
        img_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in tensor_np])
        axes[i].imshow(img_rgb, interpolation='nearest')
        axes[i].axis('off')    
    plt.show()


def plot_attention_map(attention_matrix, idx, size=(32,32)):
    """
    Given an attention matrix of shape (1025, 1025) and an idx, 
    plot the attention map for that idx as a 32x32 grid and print the CLS token attention score.
    """
    attention_map = attention_matrix[idx, 1:].reshape(size)  # Skip CLS token in the reshaping

    # Plotting our attention map
    plt.figure(figsize=(6, 6))
    plt.imshow(attention_map, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Attention Score')
    plt.title(f'Attention Map for Index {idx}')
    plt.axis('off')
    plt.show()


def plot_tensor_with_highlight(tensor, idx=None):
    """
    Plots a single 2D tensor with a grid and highlights a specific index with a white border.
    
    Parameters:
    - tensor: a 2D tensor to plot
    - idx: the index to highlight in the 32x32 grid (ignores if idx == 0)
    """
    fig, ax = plt.subplots(figsize=(5, 5))  # Set default size to 6x6
    height, width = tensor.shape
    
    # Calculate (x, y) for the given idx (ignoring CLS token)
    if idx and idx > 0:
        x, y = (idx - 1) % width, (idx - 1) // width  # Convert idx to (x, y) in 32x32 grid
    
    # Convert the tensor to RGB using COLOR_TO_HEX mapping
    tensor_np = tensor.numpy().astype(int)  # Ensure 2D and integer type
    img_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in tensor_np])
    
    # Plot the tensor with grid lines
    ax.imshow(img_rgb, interpolation='nearest')
    
    # Add grid lines
    ax.grid(which="major", color="#777777", linestyle="-", linewidth=0.5)
    ax.set_xticks(np.arange(-.5, 32, 1))
    ax.set_yticks(np.arange(-.5, 32, 1))
    
    # Highlight the selected position if valid
    if idx and idx > 0:
        rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=2, edgecolor="white", facecolor="none")
        ax.add_patch(rect)
    
    # Remove axes for a clean look
    ax.axis('off')
    
    plt.show()



# ========================================================================
# Misc
# ========================================================================
def top_k_cosine_similarity(tensor, idx, k, delta_vec=None, largest=True):
    """
    Compute the cosine similarity of a specified vector (indexed by `idx`) 
    against all other vectors in `tensor` and return the indices and similarity values 
    of the top `k` most similar vectors (or least similar if `largest=False`).

    Args:
        tensor (torch.Tensor): An n x d tensor where n is the number of items and d is the dimensionality.
        idx (int): Index of the vector to compare against.
        k (int): Number of top (or bottom) similar items to return.
        largest (bool): If True, return indices of the top k largest similarities. If False, return smallest.

    Returns:
        torch.Tensor: 1D tensor of indices for the top k most (or least) similar vectors.
        torch.Tensor: 1D tensor of similarity values for the top k most (or least) similar vectors.
    """
    normalized_tensor = torch.nn.functional.normalize(tensor, dim=1)
    reference_vector = normalized_tensor[idx].unsqueeze(0).clone()  # Shape: 1 x d
    if delta_vec is not None:
        reference_vector -= delta_vec
        reference_vector = torch.nn.functional.normalize(reference_vector, dim=1)
    cosine_similarities = torch.matmul(normalized_tensor, reference_vector.T).squeeze()  # Shape: n
    top_k_values, top_k_indices = torch.topk(cosine_similarities, k=k, largest=largest)
    return top_k_indices, top_k_values


def setup_csv_logger(log_dir='logs'):
    """
    Set up the CSV logger by creating the logs directory and initializing a CSV file.

    Args:
        log_dir (str): Directory to store the log file.

    Returns:
        str: Path to the CSV file.
    """
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_file = os.path.join(log_dir, f'training_log_{current_time}.csv')

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sample_Num', 'Train_Loss', 'Eval_Loss', 'Samples_per_Second'])

    return csv_file


def log_to_csv(csv_file, sample_num, train_loss, eval_loss, samples_per_second):
    """ Append training and evaluation metrics to the CSV file. """
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([sample_num, train_loss, eval_loss, samples_per_second])