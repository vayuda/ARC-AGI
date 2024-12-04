import os
import csv
from datetime import datetime

import torch
import torch.nn.functional as F



# ========================================================================
# Loss Functions
# ========================================================================
def compute_loss(u, u_masks, v, v_masks, student_net, teacher_net):
    B, H, W = u.shape
    
    # Forward pass
    u_s_class, u_s_patch, _ = student_net(u, u_masks)
    u_t_class, u_t_patch, _ = teacher_net(u)
    v_s_class, v_s_patch, _ = student_net(v, v_masks)
    v_t_class, v_t_patch, _ = teacher_net(v)
    
    # Compute loss
    u_s_class, u_t_class = u_s_class.squeeze(1), u_t_class.squeeze(1)   # B x dim
    v_s_class, v_t_class = v_s_class.squeeze(1), v_t_class.squeeze(1)
    loss_1 = _compute_infoNCE_loss(u_s_class, v_t_class)
    loss_2 = _compute_infoNCE_loss(u_t_class, v_s_class)
    return loss_1 + loss_2

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


# ========================================================================
# Image manipulation helpers
# ========================================================================
def upsize_image(image, scale_factor):
    """
    Resizes a torch tensor of integers based on the eligibility for scaling up or down.
    
    Args:
        image (torch.Tensor): A 2D tensor of integers with shape (H, W).
    
    Returns Resized image (2-d torch.Tensor of ints).
    """
    scaled_image = image.repeat_interleave(scale_factor, dim=0).repeat_interleave(scale_factor, dim=1)
    return scaled_image


# ========================================================================
# Optim Function
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
# Logging
# ========================================================================
def setup_csv_logger(log_dir='logs'):
    """
    Set up the CSV logger by creating the logs directory and initializing a CSV file.

    Args:
        log_dir (str): Directory to store the log file.

    Returns:
        tuple: Paths to the unique log file and the 'latest' log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_csv_file = os.path.join(log_dir, f'training_log_{current_time}.csv')
    latest_csv_file = os.path.join(log_dir, 'training_log_latest.csv')

    # Initialize both unique and latest CSV files
    for file in [unique_csv_file, latest_csv_file]:
        with open(file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample_Num', 'Train_Loss', 'Samples_per_Second'])

    return unique_csv_file, latest_csv_file


def log_to_csv(csv_files, sample_num, train_loss, samples_per_second):
    """
    Append training and evaluation metrics to both the unique and 'latest' CSV files.

    Args:
        csv_files (tuple): Paths to the unique log file and the 'latest' log file.
        sample_num (int): Current sample number.
        train_loss (float): Current training loss.
        samples_per_second (float): Samples processed per second.
    """
    unique_csv_file, latest_csv_file = csv_files

    # Log to both files
    for file in [unique_csv_file, latest_csv_file]:
        with open(file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([sample_num, train_loss, samples_per_second])