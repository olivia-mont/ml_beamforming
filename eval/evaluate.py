"""
eval/evaluate.py
================
All evaluation logic: computing accuracy, achievable rate, and the
SNR mismatch robustness add-on (Section 5.1 of the spec).

What this file does in plain English:
  - Given a trained model and test data, measure:
      1. Top-1 accuracy: did we pick the exact right beam?
      2. Achievable rate: how fast is the link at the predicted beam?
      3. Rate ratio vs oracle: what fraction of the best possible rate do we get?
  - For the SNR Mismatch Add-on:
      - Train the model on HIGH SNR data only.
      - Test it on LOW SNR data.
      - Compare its performance vs a model trained on all SNRs.
      - This shows how well the model "generalizes" to bad conditions.

FIX (channel mismatch):
  The original code regenerated H using np.random.default_rng(idx + 1000000),
  producing a DIFFERENT channel than the one used during dataset generation.
  This meant best_i/best_j (the oracle label) was optimal for H1 but we were
  evaluating it against H2 — making the rate ratio completely wrong.

  Fix: dataset.py now stores a 'channel_seed' array. We use that seed here to
  regenerate the exact same H that produced each sample's label.
"""

import os
import numpy as np
import torch
from codebook.codebook import build_dft_codebook, snr_db_to_linear, achievable_rate
from channel.channel import generate_channel
from channel.dataset import generate_dataset, measure_pilots, build_probing_design
from ml.train import BeamDataset, load_trained_model


# ============================================================
# Core evaluation function
# ============================================================
def evaluate_model_on_dataset(
    model,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    test_dataset: dict,
    Nt: int,
    Nr: int,
    device: str = 'cpu',
    snr_db_override: float = None,
) -> dict:
    """
    Evaluate a trained model on a test dataset.

    For each test sample:
      - Run the model to get predicted beam pair (i_hat, j_hat).
      - Compute achievable rate at predicted beam.
      - Compare to oracle rate (rate at true best beam).

    Args:
        model:           Trained BeamMLP.
        z_mean, z_std:   Normalization stats from training.
        test_dataset:    Dict from generate_dataset().
        Nt, Nr:          System parameters (to rebuild codebook).
        snr_db_override: If given, evaluate at this fixed SNR instead of
                         the per-sample SNR stored in the dataset.

    Returns:
        dict with: top1_acc, avg_rate_ml, avg_rate_oracle, rate_ratio,
                   per_sample arrays.
    """
    model.eval()

    z = test_dataset['z']
    labels = test_dataset['label']
    best_i = test_dataset['best_i']
    best_j = test_dataset['best_j']
    snr_db_arr = test_dataset['snr_db']
    # FIX: load the per-sample channel seeds stored by generate_dataset
    channel_seeds = test_dataset['channel_seed']
    F_size = int(test_dataset['F_size'])
    W_size = int(test_dataset['W_size'])
    L = int(test_dataset['L'])

    # Rebuild codebook
    F = build_dft_codebook(Nt, F_size)
    W = build_dft_codebook(Nr, W_size)

    # Normalize inputs the same way as during training
    z_norm = (z - z_mean) / z_std
    z_tensor = torch.tensor(z_norm.reshape(len(z), -1), dtype=torch.float32).to(device)

    # Get predictions
    with torch.no_grad():
        logits = model(z_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    # Decode predicted beam pair
    pred_i = preds // W_size
    pred_j = preds % W_size

    # Top-1 accuracy
    correct = (preds == labels)
    top1_acc = correct.mean()

    # Achievable rate at predicted vs oracle beam
    n = len(z)
    rates_ml = np.zeros(n)
    rates_oracle = np.zeros(n)

    for idx in range(n):
        snr_db = snr_db_override if snr_db_override is not None else float(snr_db_arr[idx])
        P, sigma2 = snr_db_to_linear(snr_db)

        # FIX: Replay the exact channel H used during dataset generation.
        # channel_seeds[idx] was stored by generate_dataset so we can do this
        # exactly — no approximation, no different-channel confusion.
        H = generate_channel(Nt, Nr, L, np.random.default_rng(int(channel_seeds[idx])))

        # ML predicted beam rate
        eff_ml = np.conj(W[pred_j[idx]]) @ H @ F[pred_i[idx]]
        snr_ml = P * np.abs(eff_ml) ** 2 / sigma2
        rates_ml[idx] = achievable_rate(snr_ml)

        # Oracle beam rate (using stored best_i, best_j — valid because same H)
        eff_or = np.conj(W[best_j[idx]]) @ H @ F[best_i[idx]]
        snr_or = P * np.abs(eff_or) ** 2 / sigma2
        rates_oracle[idx] = achievable_rate(snr_or)

    return {
        'top1_acc': float(top1_acc),
        'avg_rate_ml': float(rates_ml.mean()),
        'avg_rate_oracle': float(rates_oracle.mean()),
        'rate_ratio': float(rates_ml.mean() / (rates_oracle.mean() + 1e-9)),
        'rates_ml': rates_ml,
        'rates_oracle': rates_oracle,
        'pred_i': pred_i,
        'pred_j': pred_j,
    }


# ============================================================
# Figure 1 data: Rate vs Overhead T
# ============================================================
def compute_rate_vs_T(
    model_dir: str,
    Nt: int, Nr: int, L: int,
    F_size: int, W_size: int,
    T_values: list,
    snr_db: float,
    n_eval: int,
    device: str = 'cpu',
) -> dict:
    """
    For each T in T_values, load the model trained with that T and
    compute its average achievable rate. Also compute baselines.

    Returns dict: {T -> {'ml': rate, 'random': rate, 'oracle': rate, 'exhaustive': rate}}
    """
    from baseline.baselines import random_subsampling, oracle_rate, exhaustive_sweep

    rng = np.random.default_rng(888)
    F = build_dft_codebook(Nt, F_size)
    W = build_dft_codebook(Nr, W_size)
    P, sigma2 = snr_db_to_linear(snr_db)

    results = {}

    for T in T_values:
        print(f"  Evaluating T={T}...")
        ckpt_path = os.path.join(model_dir, f"model_T{T}.pt")

        ml_rates, rand_rates, oracle_rates, exhaust_rates = [], [], [], []

        for _ in range(n_eval):
            H = generate_channel(Nt, Nr, L, rng)

            _, _, r_oracle = oracle_rate(H, F, W, P, sigma2)
            _, _, r_exhaust = exhaustive_sweep(H, F, W, P, sigma2, rng)
            _, _, r_rand = random_subsampling(H, F, W, P, sigma2, T, rng)
            oracle_rates.append(r_oracle)
            exhaust_rates.append(r_exhaust)
            rand_rates.append(r_rand)

            # ML: generate measurement for this H
            if os.path.exists(ckpt_path):
                model, z_mean, z_std = load_trained_model(ckpt_path, device)
                probe_tx, probe_rx = build_probing_design(F_size, W_size, T, Rx_beams_probed=2)
                z = measure_pilots(H, F, W, probe_tx, probe_rx, P, sigma2, rng)
                z_norm = (z - z_mean) / z_std
                z_t = torch.tensor(z_norm.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = model.predict(z_t).item()
                pi, pj = pred // W_size, pred % W_size
                eff = np.conj(W[pj]) @ H @ F[pi]
                snr_ml = P * np.abs(eff) ** 2 / sigma2
                ml_rates.append(achievable_rate(snr_ml))

        results[T] = {
            'oracle':     float(np.mean(oracle_rates)),
            'exhaustive': float(np.mean(exhaust_rates)),
            'random':     float(np.mean(rand_rates)),
            'ml':         float(np.mean(ml_rates)) if ml_rates else None,
        }

    return results


# ============================================================
# Figure 2 data: Top-1 Accuracy vs SNR
# ============================================================
def compute_acc_vs_snr(
    model,
    z_mean: np.ndarray,
    z_std: np.ndarray,
    Nt: int, Nr: int, L: int,
    F_size: int, W_size: int,
    T: int,
    snr_values: list,
    n_eval: int,
    device: str = 'cpu',
) -> dict:
    """
    Evaluate top-1 accuracy at different SNR values.
    Used for Figure 2 (Accuracy vs SNR plot).

    Returns: {snr_db -> top1_acc}
    """
    acc_dict = {}
    for snr_db in snr_values:
        print(f"  Evaluating accuracy at SNR={snr_db}dB...")
        # Generate fresh test set at this specific SNR
        ds = generate_dataset(
            n_samples=n_eval,
            Nt=Nt, Nr=Nr, L=L,
            F_size=F_size, W_size=W_size, T=T,
            snr_db_min=snr_db, snr_db_max=snr_db,  # fixed SNR
            seed=9999 + int(snr_db * 10),
        )

        z = ds['z']
        labels = ds['label']
        z_norm = (z - z_mean) / z_std
        z_t = torch.tensor(
            z_norm.reshape(z_norm.shape[0], -1),
            dtype=torch.float32,
        ).to(device)

        with torch.no_grad():
            preds = model.predict(z_t).cpu().numpy()

        acc_dict[snr_db] = float((preds == labels).mean())

    return acc_dict


# ============================================================
# Figure 3 (Add-on): SNR Mismatch
# ============================================================
def compute_snr_mismatch(
    model_matched,        # Model trained on full SNR range
    model_mismatched,     # Model trained on HIGH SNR only
    z_mean_matched: np.ndarray,
    z_std_matched: np.ndarray,
    z_mean_mismatched: np.ndarray,
    z_std_mismatched: np.ndarray,
    Nt: int, Nr: int, L: int,
    F_size: int, W_size: int,
    T: int,
    low_snr_values: list,
    n_eval: int,
    device: str = 'cpu',
) -> dict:
    """
    Evaluate both models on LOW SNR test data to show mismatch degradation.

    This is the SNR Mismatch robustness slice (Add-on 1, Section 5.1).

    - model_matched:    trained on SNR in [-10, 20] dB (knows low SNR).
    - model_mismatched: trained on SNR in [5, 20] dB (never saw low SNR).
    - Test both on SNR in [-10, 0] dB.
    """
    results = {'matched': {}, 'mismatched': {}}

    for snr_db in low_snr_values:
        print(f"  SNR mismatch test at SNR={snr_db}dB...")
        ds = generate_dataset(
            n_samples=n_eval,
            Nt=Nt, Nr=Nr, L=L,
            F_size=F_size, W_size=W_size, T=T,
            snr_db_min=snr_db, snr_db_max=snr_db,
            seed=7777 + int(snr_db * 10 + 100),
        )
        z = ds['z']
        labels = ds['label']

        for tag, model, z_mean, z_std in [
            ('matched',    model_matched,    z_mean_matched,    z_std_matched),
            ('mismatched', model_mismatched, z_mean_mismatched, z_std_mismatched),
        ]:
            z_norm = (z - z_mean) / z_std
            z_t = torch.tensor(
                z_norm.reshape(z_norm.shape[0], -1),
                dtype=torch.float32,
            ).to(device)
            with torch.no_grad():
                preds = model.predict(z_t).cpu().numpy()
            results[tag][snr_db] = float((preds == labels).mean())

    return results