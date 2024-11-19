import time
import torch
from torch.cuda.amp import GradScaler, autocast

import loader
import utility
import transformers

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
scaler = GradScaler()   # Train w/ Mixed Precision for faster training

# Hyperparams
loader_params = {
    "batch_size": 4,
    "pad_images": True,
    "percent_mask": 0.2,
    "shuffle": True,
    "place_central": False
}

train_params = {
    "epochs": 25,
    "lr": 1e-4,
    "print_frequency": 100,
    "max_eval_iters": None,
    "ema_alpha": 0.01,
    "nce_weight": 0.4
}

model_params = {
    "max_img_size": 32,
    "unique_patches": 13,
    "embed_dim": 16,
    "depth": 6,
    "num_heads": 2,
    "mlp_ratio": 2,
    "qkv_bias": False,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "pos_enc": "Sinusoidal",
    "head": False,
    "print_statements": True,
    "output_classes": 13,
    "device": device
}

def train_iBOT(student_ViT, teacher_ViT, loader_params, train_params, device):
    csv_log_file = utility.setup_csv_logger()
    train_dataloader = loader.get_dataloader('dict_traindata.txt', loader_params)
    eval_dataloader = loader.get_dataloader('dict_evaldata.txt', loader_params)
    optim = torch.optim.AdamW(student_ViT.parameters(), lr=train_params['lr'])
    
    for epoch in range(train_params['epochs']):
        print(f"\nEpoch {epoch + 1}/{train_params['epochs']}")
        print(f"{'Iter':>9} || {'Train Patch':>10} | {'Train Class':>10} | {'Train Total':>10} || {'Eval Patch':>10} | {'Eval Class':>10} | {'Eval Total':>10} || {'Samples/s':>10}")
        train_patch, train_class, last_iter, train_samples = 0, 0, 0, 0
        start_time = time.time()
        
        for i, (ids, u, u_masks, v, v_masks) in enumerate(train_dataloader):
            u, u_masks, v, v_masks = u.to(device), u_masks.to(device), v.to(device), v_masks.to(device)
            train_samples += u.shape[0]

            with torch.amp.autocast('cuda'):  # Enable mixed precision
                patch_loss, class_loss = utility.compute_loss(
                    u, u_masks, v, v_masks, student_ViT, teacher_ViT,
                    train_params['nce_weight'], device
                )
                loss = patch_loss + class_loss

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            
            utility.update_teacher_weights(student_ViT, teacher_ViT, train_params['ema_alpha'])
    
            train_patch += patch_loss.cpu().item()
            train_class += class_loss.cpu().item()
            if (i+1) % train_params['print_frequency'] == 0 or (i+1) == len(train_dataloader):
                iter_count = (i+1) - last_iter
                elapsed_time = time.time() - start_time
                
                eval_patch, eval_class = utility.get_eval_loss(
                    student_ViT, teacher_ViT, eval_dataloader,
                    train_params['nce_weight'], device,
                    max_eval_iters=train_params['max_eval_iters']
                )
                train_patch /= iter_count
                train_class /= iter_count
                total_train_loss = train_patch + train_class
                total_eval_loss = eval_patch + eval_class
                samples_per_second = ((iter_count * loader_params['batch_size']) / elapsed_time)
                print(f"{(i + 1):>4}/{len(train_dataloader):>4} || {train_patch:>11.3f} | {train_class:>11.3f} | {total_train_loss:>11.3f} || {eval_patch:>10.3f} | {eval_class:>10.3f} | {total_eval_loss:>10.3f} || {samples_per_second:>9.2f}")

                last_iter = (i+1)
                start_time = time.time()
                sample_num = train_samples + epoch * len(train_dataloader) * loader_params['batch_size']
                utility.log_to_csv(csv_log_file, sample_num, train_patch, train_class, total_train_loss, eval_patch, eval_class, total_eval_loss, samples_per_second)
                train_patch, train_class = 0, 0
        
        
        student_ViT.save_model()


# Instantiate and initialize teacher network to match the student
student_ViT = transformers.VisionTransformer(**model_params).to(device)
teacher_ViT = transformers.VisionTransformer(**model_params).to(device)
teacher_ViT.load_state_dict(student_ViT.state_dict())

# Train model
train_iBOT(student_ViT, teacher_ViT, loader_params, train_params, device)
