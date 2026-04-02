"""
eval/make_top5_figure.py
========================
Standalone script that ONLY generates the top-1 vs top-5 accuracy figure.
Does not touch Figure 1, Figure 3, or any other part of the pipeline.

Usage:
    python -m eval.make_top5_figure
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.dataset import generate_dataset
from ml.train import load_trained_model

# ── Load config ───────────────────────────────────────────────
CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml')
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

Nt       = CFG['system']['Nt']
Nr       = CFG['system']['Nr']
L        = CFG['system']['L']
SEED     = CFG['system']['seed']
F_SIZE   = CFG['codebook']['F_size']
W_SIZE   = CFG['codebook']['W_size']
T        = CFG['probing']['T_default']
SNR_TEST = CFG['snr']['test_snr_values']

CKPT_DIR = 'outputs/checkpoints'
FIG_DIR  = 'outputs/figures'
os.makedirs(FIG_DIR, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'

print(f"Generating Top-1 vs Top-5 accuracy figure (T={T})...")

ckpt_path = os.path.join(CKPT_DIR, f'model_T{T}.pt')
assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

model, z_mean, z_std = load_trained_model(ckpt_path, DEVICE)
model.eval()

top1_list = []
top5_list = []

for snr_db in SNR_TEST:
    print(f"  SNR={snr_db}dB...")

    ds = generate_dataset(
        n_samples=2000,
        Nt=Nt, Nr=Nr, L=L,
        F_size=F_SIZE, W_size=W_SIZE, T=T,
        snr_db_min=snr_db, snr_db_max=snr_db,
        seed=SEED + 5000 + int(snr_db * 10 + 200),
    )

    z      = ds['z']
    labels = ds['label']

    # Normalize — same way as training
    z_norm = (z - z_mean) / z_std          # may broadcast to (1, n) or (n, n)
    z_norm = z_norm.reshape(len(z), -1)    # force to (n_samples, input_dim)
    z_t    = torch.tensor(z_norm, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(z_t)                                        # (n, 1024)
        top1_preds  = logits.argmax(dim=1).cpu().numpy()           # (n,)
        top5_preds  = torch.topk(logits, k=5, dim=1).indices.cpu().numpy()  # (n, 5)

    top1_acc = float((top1_preds == labels).mean())
    top5_acc = float((top5_preds == labels.reshape(-1, 1)).any(axis=1).mean())

    top1_list.append(top1_acc * 100)
    top5_list.append(top5_acc * 100)
    print(f"    Top-1: {top1_acc*100:.1f}%   Top-5: {top5_acc*100:.1f}%")

# ── Plot ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(SNR_TEST, top1_list, 'o-',  color='#d7191c', linewidth=2, markersize=7,
        label='Top-1 Accuracy (exact beam match)')
ax.plot(SNR_TEST, top5_list, 's--', color='#2c7bb6', linewidth=2, markersize=7,
        label='Top-5 Accuracy (true beam in top 5)')

ax.set_xlabel('SNR (dB)', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=13)
ax.set_title('Beam Prediction Accuracy vs SNR', fontsize=13)
ax.set_ylim([-2, 102])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.4)
fig.tight_layout()

out_path = os.path.join(FIG_DIR, 'fig2_acc_vs_snr_top5.pdf')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out_path}")

# ── Also save numbers to JSON ─────────────────────────────────
results = {snr: {'top1': t1/100, 'top5': t5/100}
           for snr, t1, t5 in zip(SNR_TEST, top1_list, top5_list)}
json_path = os.path.join('outputs', 'top5_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {json_path}")
print("Done.")