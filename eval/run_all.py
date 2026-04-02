"""
eval/run_all.py
===============
ONE COMMAND to reproduce all results and figures.

Usage:
    python -m eval.run_all

This script:
  1. Generates training/validation/test datasets.
  2. Trains the ML model (once for each T value, once for SNR mismatch).
  3. Evaluates baselines and ML model.
  4. Saves all three required figures to outputs/figures/.

Estimated runtime: ~30-60 minutes on CPU, ~10-15 minutes with GPU.
(Most time is dataset generation and training.)
"""

import os
import sys
import json
import yaml
import numpy as np
import torch

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.dataset import generate_dataset, save_dataset, load_dataset
from codebook.codebook import build_dft_codebook, snr_db_to_linear, achievable_rate
from channel.channel import generate_channel
from baseline.baselines import (
    random_subsampling, oracle_rate, exhaustive_sweep
)
from ml.model import BeamMLP
from ml.train import BeamDataset, train_model, load_trained_model
from eval.evaluate import compute_acc_vs_snr, compute_snr_mismatch
from eval.plots import (
    plot_rate_vs_T, plot_acc_vs_snr, plot_snr_mismatch, plot_training_curves
)
from channel.dataset import measure_pilots, build_probing_design


# ── Load config ───────────────────────────────────────────────
CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'default.yaml')
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

# ── System params ─────────────────────────────────────────────
Nt       = CFG['system']['Nt']
Nr       = CFG['system']['Nr']
L        = CFG['system']['L']
SEED     = CFG['system']['seed']
F_SIZE   = CFG['codebook']['F_size']
W_SIZE   = CFG['codebook']['W_size']
N_CLASSES = F_SIZE * W_SIZE

T_VALUES  = CFG['probing']['T_values']
T_DEFAULT = CFG['probing']['T_default']
RX_BEAMS  = CFG['probing']['Rx_beams_probed']

N_TRAIN   = CFG['dataset']['n_train']
N_VAL     = CFG['dataset']['n_val']
N_TEST    = CFG['dataset']['n_test']

SNR_MIN   = CFG['snr']['train_snr_db_min']
SNR_MAX   = CFG['snr']['train_snr_db_max']
SNR_TEST  = CFG['snr']['test_snr_values']

MM_SNR_MIN = CFG['snr']['mismatch_train_snr_db_min']
MM_SNR_MAX = CFG['snr']['mismatch_train_snr_db_max']

DATA_DIR  = 'outputs/datasets'
CKPT_DIR  = 'outputs/checkpoints'
FIG_DIR   = 'outputs/figures'
RESULTS_PATH = 'outputs/results.json'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Auto-detect device
if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = 'cpu'
print(f"\n{'='*60}")
print(f"ECE 508 mmWave Beam Management — Run All")
print(f"Device: {DEVICE} | Seed: {SEED}")
print(f"{'='*60}\n")

torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================
# STEP 1: Generate datasets for each T value
# ================================================================
print("\n[STEP 1] Generating datasets...")

def get_or_generate_dataset(tag, n, snr_min, snr_max, T, seed_offset=0):
    path = os.path.join(DATA_DIR, f'{tag}_T{T}.npz')
    if os.path.exists(path):
        print(f"  Loading cached: {path}")
        return load_dataset(path)
    ds = generate_dataset(
        n_samples=n, Nt=Nt, Nr=Nr, L=L,
        F_size=F_SIZE, W_size=W_SIZE, T=T,
        snr_db_min=snr_min, snr_db_max=snr_max,
        Rx_beams_probed=RX_BEAMS,
        seed=SEED + seed_offset,
    )
    save_dataset(ds, path)
    return ds


# ================================================================
# STEP 2: Train models for each T (matched SNR)
# ================================================================
print("\n[STEP 2] Training ML models for each T value...")

histories = {}
for T in T_VALUES:
    print(f"\n  --- T = {T} ---")
    ckpt_path = os.path.join(CKPT_DIR, f'model_T{T}.pt')

    if os.path.exists(ckpt_path):
        print(f"  Checkpoint exists, skipping training: {ckpt_path}")
        continue

    train_ds = get_or_generate_dataset('train', N_TRAIN, SNR_MIN, SNR_MAX, T, seed_offset=0)
    val_ds   = get_or_generate_dataset('val',   N_VAL,   SNR_MIN, SNR_MAX, T, seed_offset=1000)

    input_dim = train_ds['z'].shape[1]
    train_torch = BeamDataset(train_ds['z'], train_ds['label'])
    val_torch   = BeamDataset(val_ds['z'],   val_ds['label'])

    # Store normalization stats alongside the model
    history = train_model(
        train_torch, val_torch, input_dim, N_CLASSES,
        CFG, ckpt_path, device=DEVICE
    )
    histories[T] = history
    plot_training_curves(history, tag=f'T{T}', save=True)


# ================================================================
# STEP 3: Train mismatched model (high SNR only) for Add-on
# ================================================================
print("\n[STEP 3] Training SNR-mismatched model (Add-on 1)...")

T = T_DEFAULT
mismatch_ckpt = os.path.join(CKPT_DIR, f'model_mismatch_T{T}.pt')

if not os.path.exists(mismatch_ckpt):
    train_mm = get_or_generate_dataset(
        'train_mismatch', N_TRAIN, MM_SNR_MIN, MM_SNR_MAX, T, seed_offset=5000
    )
    val_mm = get_or_generate_dataset(
        'val_mismatch', N_VAL, MM_SNR_MIN, MM_SNR_MAX, T, seed_offset=6000
    )
    input_dim = train_mm['z'].shape[1]
    train_torch_mm = BeamDataset(train_mm['z'], train_mm['label'])
    val_torch_mm   = BeamDataset(val_mm['z'],   val_mm['label'])
    h = train_model(
        train_torch_mm, val_torch_mm, input_dim, N_CLASSES,
        CFG, mismatch_ckpt, device=DEVICE
    )
    plot_training_curves(h, tag='mismatch', save=True)
else:
    print("  Mismatch model checkpoint exists, skipping.")


# ================================================================
# STEP 4: Evaluate baselines + ML for Figure 1 (Rate vs T)
# ================================================================
print("\n[STEP 4] Computing Rate vs Overhead T (Figure 1)...")

EVAL_SNR = 10.0  # dB — fixed SNR for the rate vs T plot
N_EVAL_BASELINE = 500  # channels to average over

rng_eval = np.random.default_rng(SEED + 99)
F = build_dft_codebook(Nt, F_SIZE)
W = build_dft_codebook(Nr, W_SIZE)
P_eval, sigma2_eval = snr_db_to_linear(EVAL_SNR)

rate_results = {}

for T in T_VALUES:
    print(f"  T={T}...")
    ckpt_path = os.path.join(CKPT_DIR, f'model_T{T}.pt')
    ml_available = os.path.exists(ckpt_path)

    # Reload model fresh for each T — each T has a different input_dim
    model_T, z_mean_T, z_std_T = None, None, None
    if ml_available:
        model_T, z_mean_T, z_std_T = load_trained_model(ckpt_path, DEVICE)
        probe_tx, probe_rx = build_probing_design(F_SIZE, W_SIZE, T, RX_BEAMS)

    oracle_rates, exhaust_rates, rand_rates, ml_rates = [], [], [], []

    for _ in range(N_EVAL_BASELINE):
        H = generate_channel(Nt, Nr, L, rng_eval)

        _, _, r_or = oracle_rate(H, F, W, P_eval, sigma2_eval)
        _, _, r_ex = exhaustive_sweep(H, F, W, P_eval, sigma2_eval, rng_eval)
        _, _, r_ra = random_subsampling(H, F, W, P_eval, sigma2_eval, T, rng_eval)

        oracle_rates.append(r_or)
        exhaust_rates.append(r_ex)
        rand_rates.append(r_ra)

        if ml_available:
            z = measure_pilots(H, F, W, probe_tx, probe_rx, P_eval, sigma2_eval, rng_eval)
            z_norm = (z - z_mean_T) / z_std_T
            z_t = torch.tensor(z_norm.flatten(), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = model_T(z_t)
                pred = int(logits.argmax(dim=1)[0])
            pi, pj = pred // W_SIZE, pred % W_SIZE
            eff = np.conj(W[pj]) @ H @ F[pi]
            snr_ml = P_eval * np.abs(eff) ** 2 / sigma2_eval
            ml_rates.append(achievable_rate(snr_ml))

    rate_results[T] = {
        'oracle':     float(np.mean(oracle_rates)),
        'exhaustive': float(np.mean(exhaust_rates)),
        'random':     float(np.mean(rand_rates)),
        'ml':         float(np.mean(ml_rates)) if ml_rates else None,
    }

fig1 = plot_rate_vs_T(rate_results, save=True)
print("  Figure 1 saved.")


# ================================================================
# STEP 5: Figure 2 — Accuracy vs SNR
# ================================================================
print("\n[STEP 5] Computing Accuracy vs SNR (Figure 2)...")

T = T_DEFAULT
ckpt_path = os.path.join(CKPT_DIR, f'model_T{T}.pt')


if os.path.exists(ckpt_path):
    model_default, z_mean_def, z_std_def = load_trained_model(ckpt_path, DEVICE)

    acc_dict = compute_acc_vs_snr(
        model=model_default,
        z_mean=z_mean_def,
        z_std=z_std_def,
        Nt=Nt, Nr=Nr, L=L,
        F_size=F_SIZE, W_size=W_SIZE, T=T,
        snr_values=SNR_TEST,
        n_eval=2000,
        device=DEVICE,
    )

    fig2 = plot_acc_vs_snr(acc_dict, save=True)
    print("  Figure 2 saved.") 
else:
    print("  Default model not found, skipping Figure 2.")
    acc_dict = {}


# ================================================================
# STEP 6: Figure 3 — SNR Mismatch (Add-on 1)
# ================================================================
print("\n[STEP 6] Computing SNR Mismatch (Figure 3, Add-on 1)...")

mismatch_test_snrs = [-10, -8, -6, -4, -2, 0]

if os.path.exists(ckpt_path) and os.path.exists(mismatch_ckpt):
    model_matched, z_mean_mat, z_std_mat = load_trained_model(ckpt_path, DEVICE)
    model_mismatch, z_mean_mis, z_std_mis = load_trained_model(mismatch_ckpt, DEVICE)

    mismatch_results = compute_snr_mismatch(
        model_matched=model_matched,
        model_mismatched=model_mismatch,
        z_mean_matched=z_mean_mat,
        z_std_matched=z_std_mat,
        z_mean_mismatched=z_mean_mis,
        z_std_mismatched=z_std_mis,
        Nt=Nt, Nr=Nr, L=L,
        F_size=F_SIZE, W_size=W_SIZE, T=T,
        low_snr_values=mismatch_test_snrs,
        n_eval=2000,
        device=DEVICE,
    )

    fig3 = plot_snr_mismatch(mismatch_results, save=True)
    print("  Figure 3 saved.")
else:
    print("  Models not found, skipping Figure 3.")
    mismatch_results = {}


# ================================================================
# STEP 7: Save all numeric results to JSON (for the report)
# ================================================================
print("\n[STEP 7] Saving numeric results...")

all_results = {
    'config': {
        'Nt': Nt, 'Nr': Nr, 'L': L,
        'F_size': F_SIZE, 'W_size': W_SIZE,
        'T_values': T_VALUES, 'T_default': T_DEFAULT,
        'train_snr': [SNR_MIN, SNR_MAX],
        'mismatch_train_snr': [MM_SNR_MIN, MM_SNR_MAX],
        'n_train': N_TRAIN, 'n_val': N_VAL, 'n_test': N_TEST,
    },
    'rate_vs_T': {str(k): v for k, v in rate_results.items()},
    'acc_vs_snr': {str(k): v for k, v in acc_dict.items()},
    'snr_mismatch': {
        'matched':    {str(k): v for k, v in mismatch_results.get('matched', {}).items()},
        'mismatched': {str(k): v for k, v in mismatch_results.get('mismatched', {}).items()},
    },
}

os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"  Results saved to {RESULTS_PATH}")

print(f"\n{'='*60}")
print("ALL DONE! Outputs:")
print(f"  Figures:  {FIG_DIR}/")
print(f"  Results:  {RESULTS_PATH}")
print(f"  Models:   {CKPT_DIR}/")
print(f"{'='*60}\n")

# ================================================================
# STEP 8: Figure 2b — Top-1 vs Top-5 Accuracy (Add-on)
# ================================================================
print("\n[STEP 8] Generating Top-1 vs Top-5 accuracy figure...")

import runpy
runpy.run_module('eval.make_top5_figure', run_name='__main__')