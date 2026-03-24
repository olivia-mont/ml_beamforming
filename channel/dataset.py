"""
channel/dataset.py
==================
Generates the full dataset of (measurement vector z, best beam label)
pairs that we use to train and evaluate our neural network.

What this file does in plain English:
  - For each channel H, we:
      1. Figure out the oracle best beam pair (i*, j*) — this is the label.
      2. Probe T beam pairs and collect noisy power measurements z.
      3. Store (z, label) for training.
  - The "probing design" is fixed and uniform — we pick T evenly-spaced
    Tx beams and use a small fixed set of Rx beams.

FIX (channel mismatch):
  - We now store a per-sample seed alongside every sample.
  - evaluate.py uses that seed to regenerate the EXACT same channel H
    that produced the label, so oracle and ML rates are computed on the
    same channel. Without this, the stored best_i/best_j label is optimal
    for a different H than the one used at evaluation time — making the
    rate ratio metric meaningless.
"""

import numpy as np
import os
from channel.channel import generate_channel
from codebook.codebook import (
    build_dft_codebook,
    oracle_best_beam_batch,
    snr_db_to_linear,
)


def build_probing_design(F_size: int, W_size: int, T: int, Rx_beams_probed: int = 2):
    """
    Build the fixed probing set: which (Tx beam, Rx beam) pairs to measure.

    Strategy:
      - Pick T Tx beam indices evenly spaced across [0, F_size).
      - Fix Rx beams to the first `Rx_beams_probed` beams.
      - All combinations: T * Rx_beams_probed total measurements.

    This is the "fixed measurement design" required by Section 3 of the spec.

    Returns:
        probe_tx: (n_probes,) int array of Tx beam indices.
        probe_rx: (n_probes,) int array of Rx beam indices.
    """
    # Evenly spaced Tx beam indices
    tx_indices = np.linspace(0, F_size - 1, T, dtype=int)
    rx_indices = np.arange(Rx_beams_probed)

    # All (tx, rx) combinations
    probe_tx = np.repeat(tx_indices, Rx_beams_probed)       # T*Rx_beams_probed
    probe_rx = np.tile(rx_indices, T)                        # T*Rx_beams_probed

    return probe_tx, probe_rx


def measure_pilots(
    H: np.ndarray,
    F: np.ndarray,
    W: np.ndarray,
    probe_tx: np.ndarray,
    probe_rx: np.ndarray,
    P: float,
    sigma2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate pilot measurements for ONE channel H. Equation (6).

    For each probed (tx, rx) pair, we compute:
        z_t = |sqrt(P) * w_j^H H f_i + noise|^2

    Args:
        H:         Channel matrix (Nr, Nt).
        F:         Tx codebook (F_size, Nt).
        W:         Rx codebook (W_size, Nr).
        probe_tx:  Tx beam indices to probe.
        probe_rx:  Rx beam indices to probe.
        P:         Transmit power.
        sigma2:    Noise variance.
        rng:       Random generator for noise.

    Returns:
        z: (n_probes,) float array of received power measurements.
    """
    n_probes = len(probe_tx)
    z = np.zeros(n_probes, dtype=float)

    for t in range(n_probes):
        f = F[probe_tx[t]]   # (Nt,) Tx beam
        w = W[probe_rx[t]]   # (Nr,) Rx combiner

        # Noiseless received signal: w^H H f = conj(w) @ H @ f
        signal = np.sqrt(P) * (np.conj(w) @ H @ f)  # scalar complex

        # Add complex Gaussian noise
        noise = (rng.standard_normal() + 1j * rng.standard_normal()) * np.sqrt(sigma2 / 2)

        # Received power measurement
        z[t] = np.abs(signal + noise) ** 2

    return z


def generate_dataset(
    n_samples: int,
    Nt: int,
    Nr: int,
    L: int,
    F_size: int,
    W_size: int,
    T: int,
    snr_db_min: float,
    snr_db_max: float,
    Rx_beams_probed: int = 2,
    seed: int = 42,
) -> dict:
    """
    Generate a complete dataset.

    For each sample:
      1. Draw a random SNR from [snr_db_min, snr_db_max].
      2. Derive a deterministic per-sample channel seed from the master seed.
      3. Generate a random channel H using that per-sample seed.
      4. Find oracle best beam pair (i*, j*) — this becomes the label.
      5. Simulate T pilot measurements z (with a separate noise RNG).
      6. Store the per-sample channel seed so evaluation can replay H exactly.

    FIX: We derive per-sample channel seeds deterministically from the master
    seed instead of drawing channels from a shared sequential RNG. This lets
    evaluate.py call:
        H = generate_channel(Nt, Nr, L, np.random.default_rng(channel_seed[idx]))
    and get exactly the same H that produced best_i[idx] / best_j[idx].

    Args:
        n_samples:      How many (z, label) pairs to create.
        Nt, Nr, L:      System parameters.
        F_size, W_size: Codebook sizes.
        T:              Number of Tx beams to probe.
        snr_db_min/max: SNR range for this dataset.
        Rx_beams_probed: How many Rx beams to fix during probing.
        seed:           Master random seed.

    Returns:
        dict with keys:
          'z':             (n_samples, T*Rx_beams_probed) float array.
          'label':         (n_samples,) int array — flattened index i*W_size+j.
          'best_i':        (n_samples,) int array — best Tx beam index.
          'best_j':        (n_samples,) int array — best Rx beam index.
          'snr_db':        (n_samples,) float array — per-sample SNR used.
          'channel_seed':  (n_samples,) int array — seed to replay channel H.
          'probe_tx', 'probe_rx': the fixed probing design used.
          'F_size', 'W_size', 'T', 'L': stored for downstream use.
    """
    # Master RNG — used only for SNR draws and noise; NOT for channels.
    master_rng = np.random.default_rng(seed)

    # Build codebooks (same for all samples)
    F = build_dft_codebook(Nt, F_size)
    W = build_dft_codebook(Nr, W_size)

    # Fixed probing design (same for all samples — fair comparison)
    probe_tx, probe_rx = build_probing_design(F_size, W_size, T, Rx_beams_probed)
    n_probes = len(probe_tx)

    # Derive deterministic per-sample channel seeds from master seed.
    # Using a separate seed sequence ensures channel generation is
    # independent of how many noise samples the master RNG has drawn.
    seed_rng = np.random.default_rng(seed + 1_000_000)
    channel_seeds = seed_rng.integers(0, 2**31, size=n_samples)

    # Allocate output arrays
    Z = np.zeros((n_samples, n_probes), dtype=float)
    labels = np.zeros(n_samples, dtype=np.int32)
    best_i_arr = np.zeros(n_samples, dtype=np.int32)
    best_j_arr = np.zeros(n_samples, dtype=np.int32)
    snr_db_arr = np.zeros(n_samples, dtype=float)

    print(f"  Generating {n_samples} samples (T={T}, L={L}, "
          f"SNR=[{snr_db_min},{snr_db_max}]dB)...")

    for n in range(n_samples):
        if n % 10000 == 0 and n > 0:
            print(f"    {n}/{n_samples} done...")

        # Random SNR for this sample (drawn from master RNG)
        snr_db = master_rng.uniform(snr_db_min, snr_db_max)
        P, sigma2 = snr_db_to_linear(snr_db)

        # FIX: Generate channel from its own dedicated seed — fully reproducible.
        ch_rng = np.random.default_rng(int(channel_seeds[n]))
        H = generate_channel(Nt, Nr, L, ch_rng)

        # Oracle best beam pair (noiseless, exhaustive)
        best_i, best_j, _ = _oracle_single(H, F, W, P, sigma2)

        # Pilot measurements — use master RNG for noise only
        z = measure_pilots(H, F, W, probe_tx, probe_rx, P, sigma2, master_rng)

        # Store
        Z[n] = z
        best_i_arr[n] = best_i
        best_j_arr[n] = best_j
        labels[n] = best_i * W_size + best_j   # Flattened beam pair index
        snr_db_arr[n] = snr_db

    return {
        'z': Z,
        'label': labels,
        'best_i': best_i_arr,
        'best_j': best_j_arr,
        'snr_db': snr_db_arr,
        'channel_seed': channel_seeds,   # FIX: stored so evaluation can replay H
        'probe_tx': probe_tx,
        'probe_rx': probe_rx,
        'F_size': F_size,
        'W_size': W_size,
        'T': T,
        'L': L,
    }


def _oracle_single(H, F, W, P, sigma2):
    """Oracle for a single channel (slow loop, used in dataset generation)."""
    F_size, W_size = F.shape[0], W.shape[0]
    best_snr = -np.inf
    best_i, best_j = 0, 0
    for i in range(F_size):
        for j in range(W_size):
            # conj(w) @ H @ f — consistent with compute_snr in codebook.py
            eff = np.conj(W[j]) @ H @ F[i]
            snr = P * np.abs(eff) ** 2 / sigma2
            if snr > best_snr:
                best_snr = snr
                best_i, best_j = i, j
    return best_i, best_j, best_snr


def save_dataset(dataset: dict, path: str):
    """Save dataset to a .npz file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **dataset)
    print(f"  Saved dataset to {path}")


def load_dataset(path: str) -> dict:
    """Load dataset from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    print("Quick dataset generation test (100 samples)...")
    ds = generate_dataset(
        n_samples=100, Nt=64, Nr=16, L=3,
        F_size=64, W_size=16, T=32,
        snr_db_min=-10, snr_db_max=20,
        seed=42
    )
    print(f"  z shape:          {ds['z'].shape}")
    print(f"  label shape:      {ds['label'].shape}")
    print(f"  label range:      [{ds['label'].min()}, {ds['label'].max()}]  (max should be <1024)")
    print(f"  channel_seed[0]:  {ds['channel_seed'][0]}  (should be a large int)")

    # Verify replay: regenerate channel 0 and check oracle matches stored label
    from codebook.codebook import build_dft_codebook, snr_db_to_linear
    F = build_dft_codebook(64, 64)
    W = build_dft_codebook(16, 16)
    H0 = generate_channel(64, 16, 3, np.random.default_rng(int(ds['channel_seed'][0])))
    P, sigma2 = snr_db_to_linear(float(ds['snr_db'][0]))
    bi, bj, _ = _oracle_single(H0, F, W, P, sigma2)
    stored_label = int(ds['label'][0])
    replayed_label = bi * 16 + bj
    match = "✓ MATCH" if stored_label == replayed_label else "✗ MISMATCH"
    print(f"  Channel replay check: stored={stored_label}, replayed={replayed_label}  {match}")
    print("Dataset generator OK.")