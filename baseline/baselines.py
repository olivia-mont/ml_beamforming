"""
baseline/baselines.py
=====================
Implements the two required baselines from Section 4.3 of the spec.

Baseline 1 — Exhaustive Sweep (Upper Bound):
  Test ALL beam pairs, pick the one with the highest measured power.
  This is the best we can possibly do; it sets the ceiling.

Baseline 2 — Random Subsampling:
  Pick T random beam pairs, measure them, take the best.
  This is a "dumb" baseline — no learning involved.

Both baselines return: predicted (i, j), and the achievable rate.
"""

import numpy as np
from codebook.codebook import snr_db_to_linear, achievable_rate
from channel.channel import generate_channel
from codebook.codebook import build_dft_codebook


# ============================================================
# Helper: simulate a single noisy power measurement
# ============================================================
def _measure_one(H, f, w, P, sigma2, rng):
    """
    Measure received power for one (f, w) beam pair.
    Returns z = |sqrt(P)*w^H H f + noise|^2

    Args:
        H:      Channel matrix (Nr, Nt).
        f:      Tx beam vector (Nt,) — NOT conjugated; H @ f gives (Nr,).
        w:      Rx combiner vector (Nr,) — NOT conjugated; w^H = conj(w).
        P:      Transmit power.
        sigma2: Noise variance.
        rng:    Random generator.
    """
    # BUG FIX: f and w must NOT be pre-conjugated here.
    # The signal model is: w^H H f, i.e. conj(w) @ H @ f.
    # F[i] and W[j] store the beam vectors directly — pass them as-is.
    signal = np.sqrt(P) * (np.conj(w) @ H @ f)   # scalar complex
    noise = (rng.standard_normal() + 1j * rng.standard_normal()) * np.sqrt(sigma2 / 2)
    return np.abs(signal + noise) ** 2


# ============================================================
# Baseline 1: Exhaustive Sweep
# ============================================================
def exhaustive_sweep(H, F, W, P, sigma2, rng):
    """
    Probe ALL |F|*|W| = 1024 beam pairs and pick the best.

    This is the upper bound baseline — it uses the most overhead (T=1024)
    but achieves near-oracle performance.

    Returns:
        best_i, best_j: predicted best beam pair indices.
        rate:           achievable rate at the selected beam pair.
    """
    F_size, W_size = F.shape[0], W.shape[0]
    best_z = -np.inf
    best_i, best_j = 0, 0

    for i in range(F_size):
        for j in range(W_size):
            # BUG FIX: pass F[i] and W[j] directly — no .conj() here.
            z = _measure_one(H, F[i], W[j], P, sigma2, rng)
            if z > best_z:
                best_z = z
                best_i, best_j = i, j

    # Compute true (noiseless) SNR at selected beam
    # BUG FIX: conj(W[best_j]) @ H @ F[best_i], not F[best_i].conj()
    eff = np.conj(W[best_j]) @ H @ F[best_i]
    snr = P * np.abs(eff) ** 2 / sigma2
    return best_i, best_j, achievable_rate(snr)


# ============================================================
# Baseline 2: Random Subsampling
# ============================================================
def random_subsampling(H, F, W, P, sigma2, T, rng):
    """
    Pick T random beam pairs, measure them, return the best.

    Args:
        T: Number of beam pairs to probe (the "overhead budget").

    Returns:
        best_i, best_j: predicted best beam pair indices.
        rate:           achievable rate at the selected beam pair.
    """
    F_size, W_size = F.shape[0], W.shape[0]

    # Sample T random (tx, rx) pairs without replacement
    all_pairs = [(i, j) for i in range(F_size) for j in range(W_size)]
    chosen = [all_pairs[k] for k in rng.choice(len(all_pairs), size=min(T, len(all_pairs)), replace=False)]

    best_z = -np.inf
    best_i, best_j = chosen[0]

    for i, j in chosen:
        # BUG FIX: pass F[i] and W[j] directly — no .conj() here.
        z = _measure_one(H, F[i], W[j], P, sigma2, rng)
        if z > best_z:
            best_z = z
            best_i, best_j = i, j

    # BUG FIX: conj(W[best_j]) @ H @ F[best_i], not F[best_i].conj()
    eff = np.conj(W[best_j]) @ H @ F[best_i]
    snr = P * np.abs(eff) ** 2 / sigma2
    return best_i, best_j, achievable_rate(snr)


# ============================================================
# Oracle (true best — used as ceiling reference)
# ============================================================
def oracle_rate(H, F, W, P, sigma2):
    """
    Find the true best beam pair (no noise, no budget limit).
    Used as the absolute ceiling in plots.
    """
    F_size, W_size = F.shape[0], W.shape[0]
    best_snr = -np.inf
    best_i, best_j = 0, 0

    for i in range(F_size):
        for j in range(W_size):
            # Consistent with compute_snr in codebook.py: conj(w) @ H @ f
            eff = np.conj(W[j]) @ H @ F[i]
            snr = P * np.abs(eff) ** 2 / sigma2
            if snr > best_snr:
                best_snr = snr
                best_i, best_j = i, j

    return best_i, best_j, achievable_rate(best_snr)


# ============================================================
# Evaluate baselines over many channel realizations
# ============================================================
def evaluate_baselines(
    n_eval: int,
    Nt: int, Nr: int, L: int,
    F_size: int, W_size: int,
    snr_db: float,
    T_values: list,
    seed: int = 99,
) -> dict:
    """
    Run both baselines over n_eval random channels and average the results.

    Returns a dict with average rates for each method and T value.
    This is used to generate the "rate vs overhead T" plot.
    """
    rng = np.random.default_rng(seed)
    F = build_dft_codebook(Nt, F_size)
    W = build_dft_codebook(Nr, W_size)
    P, sigma2 = snr_db_to_linear(snr_db)

    results = {
        'oracle': [],
        'exhaustive': [],
        'random': {T: [] for T in T_values},
    }

    print(f"  Evaluating baselines over {n_eval} channels at SNR={snr_db}dB...")
    for n in range(n_eval):
        if n % 500 == 0 and n > 0:
            print(f"    {n}/{n_eval}...")
        H = generate_channel(Nt, Nr, L, rng)

        _, _, r_oracle = oracle_rate(H, F, W, P, sigma2)
        results['oracle'].append(r_oracle)

        _, _, r_exhaust = exhaustive_sweep(H, F, W, P, sigma2, rng)
        results['exhaustive'].append(r_exhaust)

        for T in T_values:
            _, _, r_rand = random_subsampling(H, F, W, P, sigma2, T, rng)
            results['random'][T].append(r_rand)

    # Average over all channels
    avg = {
        'oracle': float(np.mean(results['oracle'])),
        'exhaustive': float(np.mean(results['exhaustive'])),
        'random': {T: float(np.mean(results['random'][T])) for T in T_values},
    }
    return avg


if __name__ == "__main__":
    print("Quick baseline test (50 channels)...")
    results = evaluate_baselines(
        n_eval=50, Nt=64, Nr=16, L=3,
        F_size=64, W_size=16, snr_db=10,
        T_values=[8, 16, 32],
        seed=99,
    )
    print(f"  Oracle avg rate:      {results['oracle']:.3f} bps/Hz")
    print(f"  Exhaustive avg rate:  {results['exhaustive']:.3f} bps/Hz")
    for T, r in results['random'].items():
        print(f"  Random T={T} avg rate: {r:.3f} bps/Hz")
    print("Baselines OK.")