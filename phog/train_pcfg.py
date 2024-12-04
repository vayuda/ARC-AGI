import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

current_file_dir = os.path.abspath(os.path.dirname(__file__))
root = os.path.abspath(os.path.join(current_file_dir, ".."))
if root not in sys.path:
    sys.path.append(root)

import pcfg_models as pcfg_models
import pcfg_loader as loader
import pcfg_util as pcfg_util
import source as source

scaler = GradScaler()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


# Defining hyperparams
loader_params = {
    'datafolder': 'training',
    'batch_size': 1,
    'shuffle': True,
    'only_inputs': True,
    'print_steps': False,
    'moves_per_step': 1,
    'max_steps': 1,
    'p_use_base': 0.1,
}
data_yielder = loader.get_pcfg_datayielder(**loader_params)

# Instantiate our Encoder classifier
dsl_list = data_yielder.label_ops
embedding_model_fp = 'vit_11-21-24_100k_vF.pth'
em = source.embedding.load_ViT(embedding_model_fp, device='cpu')

pcfg_encoder = pcfg_models.PCFG_Encoder(
    n_embd = em.embed_dim, 
    n_head = 4, 
    n_layer = 6, 
    ff_hd = int(em.embed_dim * 2), 
    dropout = 0,
    block_size = 35, 
    dsl_mapping = dsl_list, 
    embedding_model_fp = embedding_model_fp,
    freeze_emb_model=True, 
    device = device
)
pcfg_encoder = pcfg_encoder.to(device)
em = None   # Clear the model from memory

train_params = {
    "epochs": 1,
    "lr": 1e-4,
    "print_frequency": 100,
}


# Define our training loop
optim = torch.optim.AdamW(pcfg_encoder.parameters(), lr=train_params['lr'])
avg_losses = []
seen_labels = [0] * len(dsl_list)

for epoch in range(train_params['epochs']):
    active = True
    step, dsl_loss_sum, obj_loss_sum, total_loss_sum = 0, 0, 0, 0

    while active:
        try:
            key, input, label, obj_idx = next(data_yielder)
        except StopIteration:
            active = False
            break
        except loader.SampleExhausted:
            continue

        with torch.amp.autocast('cuda'):
            x, first_pad = pcfg_util.embed_input(input, pcfg_encoder, use_grads=not pcfg_encoder.freeze_emb_model, device=device)
            dsl_cls, obj_att = pcfg_encoder(x)
            obj_att = obj_att[2:first_pad + 2]  # Since we have 2 special cls tokens at the start

            # Compute individual losses
            dsl_loss, obj_loss = pcfg_util.compute_kl_loss(
                dsl_cls, 
                obj_att, 
                torch.tensor(label, device=device), 
                torch.tensor(obj_idx, device=device)
            )

            optim.zero_grad(set_to_none=True)
            # total_loss = dsl_loss + obj_loss
            total_loss = dsl_loss
            scaler.scale(total_loss).backward()
            scaler.step(optim)
            scaler.update()

        # Accumulate losses
        total_loss_sum += total_loss.item()
        dsl_loss_sum += dsl_loss.item()
        obj_loss_sum += obj_loss.item()
        step += 1

        if step % train_params['print_frequency'] == 0:
            avg_total_loss = total_loss_sum / train_params['print_frequency']
            avg_dsl_loss = dsl_loss_sum / train_params['print_frequency']
            avg_obj_loss = obj_loss_sum / train_params['print_frequency']

            avg_losses.append((avg_total_loss, avg_dsl_loss, avg_obj_loss))

            print(f"[{(epoch+1):>2}/{train_params['epochs']:>2}] - {step:>4}: Total Loss = {avg_total_loss:.4f}, DSL Loss = {avg_dsl_loss:.4f}, Obj Loss = {avg_obj_loss:.4f}")

            total_loss_sum = 0
            dsl_loss_sum = 0
            obj_loss_sum = 0
        
        if step > 200:
            print("Breaking early...")