"""
codebook/codebook.py
====================
Builds the DFT codebook and finds the oracle best beam pair.

What this file does in plain English:
  - A "codebook" is just a list of antenna directions (beams).
  - We use DFT (Discrete Fourier Transform) beams, which evenly
    cover all possible directions — think of it like dividing a
    compass into equal slices.
  - The "oracle" function cheats and tests EVERY possible beam pair
    to find the best one. This is our upper-bound baseline.
"""

import numpy as np
from channel.channel import ula_steering_vector


def build_dft_codebook(n_antennas: int, codebook_size: int) -> np.ndarray:
    """
    Build a DFT codebook: a set of beams evenly covering all directions.

    Each beam f_k points in a slightly different direction. Together they
    tile the entire angular space.

    Args:
        n_antennas:    Number of antennas (Nt for Tx, Nr for Rx).
        codebook_size: Number of beams (|F| or |W|).

    Returns:
        Codebook matrix of shape (codebook_size, n_antennas), complex.
        Each row is one unit-norm beam vector.
    """
    codebook = np.zeros((codebook_size, n_antennas), dtype=complex)

    # Evenly spaced angles mapped through arcsin to cover [-pi/2, pi/2]
    # The DFT grid: spatial frequencies u_k = 2k/N - 1, k=0,...,N-1
    for k in range(codebook_size):
        # Spatial frequency in [-1, 1]
        u = 2 * k / codebook_size - 1
        # Clamp to avoid arcsin domain errors at exact boundaries
        u = np.clip(u, -1 + 1e-9, 1 - 1e-9)
        angle = np.arcsin(u)
        codebook[k] = ula_steering_vector(angle, n_antennas)

    return codebook   # Shape: (codebook_size, n_antennas)


def compute_snr(
    H: np.ndarray,
    f: np.ndarray,
    w: np.ndarray,
    P: float,
    sigma2: float
) -> float:
    """
    Compute received SNR for a given channel, Tx beam f, Rx combiner w.

    This is Equation (2) from the spec:
        gamma(f, w) = P * |w^H H f|^2 / sigma^2

    Args:
        H:      Channel matrix (Nr, Nt), complex.
        f:      Tx beam vector (Nt,), complex, unit norm.
        w:      Rx combiner vector (Nr,), complex, unit norm.
        P:      Transmit power (linear scale).
        sigma2: Noise variance (linear scale).

    Returns:
        SNR as a float (linear, not dB).
    """
    # w^H H f is a scalar: the effective channel gain
    effective_gain = np.conj(w) @ H @ f   # scalar complex
    return P * np.abs(effective_gain) ** 2 / sigma2


def oracle_best_beam(
    H: np.ndarray,
    F: np.ndarray,
    W: np.ndarray,
    P: float,
    sigma2: float
) -> tuple[int, int, float]:
    """
    Exhaustively find the best (Tx beam, Rx beam) pair. Equation (5).

    This is the "cheating" upper bound — in real life you can't test
    all 1024 combinations instantly, but this tells us the best possible.

    Args:
        H:      Channel matrix (Nr, Nt).
        F:      Tx codebook (F_size, Nt).
        W:      Rx codebook (W_size, Nr).
        P:      Transmit power.
        sigma2: Noise variance.

    Returns:
        (best_i, best_j, best_snr): indices into F and W, and the SNR.
    """
    F_size = F.shape[0]
    W_size = W.shape[0]

    best_snr = -np.inf
    best_i, best_j = 0, 0

    for i in range(F_size):
        for j in range(W_size):
            snr = compute_snr(H, F[i], W[j], P, sigma2)
            if snr > best_snr:
                best_snr = snr
                best_i, best_j = i, j

    return best_i, best_j, best_snr


def oracle_best_beam_batch(
    H_batch: np.ndarray,
    F: np.ndarray,
    W: np.ndarray,
    P: float,
    sigma2: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized oracle for a batch of channels.

    Uses matrix operations to find the best beam pair for each channel
    without slow Python loops over the batch dimension.

    Args:
        H_batch: (n_samples, Nr, Nt) complex array.
        F:       (F_size, Nt) Tx codebook.
        W:       (W_size, Nr) Rx codebook.
        P, sigma2: Scalar power and noise.

    Returns:
        best_i: (n_samples,) int array — best Tx beam index per sample.
        best_j: (n_samples,) int array — best Rx beam index per sample.
    """
    n_samples = H_batch.shape[0]
    F_size = F.shape[0]
    W_size = W.shape[0]

    # Compute all effective channel gains: W^H @ H @ F^H
    # H_batch shape: (n, Nr, Nt)
    # F.T shape: (Nt, F_size) -> H @ F.T: (n, Nr, F_size)
    HF = H_batch @ F.conj().T           # (n, Nr, F_size)
    WHF = W.conj() @ HF.transpose(0, 2, 1)  # (n, F_size, W_size) ... let me do it step by step

    # Actually: for each sample, we want SNR[i,j] = P/sigma2 * |w_j^H H f_i|^2
    # Let's do: gains[n, i, j] = w_j^H (H_n f_i)
    # H_n f_i has shape (Nr,) for each n,i
    # HF[n, :, i] = H_n @ f_i^* (shape Nr) -- note F stores rows as beams

    # HF[n, :, i] = H_batch[n] @ F[i].conj() ... but we want H @ f not H @ f*
    # F[i] is the beam vector; H f = H_batch[n] @ F[i]
    HF = np.einsum('nri,fi->nrf', H_batch, F.conj())  # (n, Nr, F_size)

    # Now gains[n, i, j] = W[j].conj() @ HF[n, :, i]
    gains = np.einsum('nrf,jr->njf', HF, W.conj())    # wait, let me be careful
    # gains[n, f_idx, w_idx] = sum_r W[w_idx, r].conj() * HF[n, r, f_idx]
    gains = np.einsum('nrf,wr->nfw', HF, W.conj())    # (n, F_size, W_size)

    snr_matrix = P * np.abs(gains) ** 2 / sigma2      # (n, F_size, W_size)

    # Flatten over (F_size, W_size) and find argmax
    flat_idx = np.argmax(snr_matrix.reshape(n_samples, -1), axis=1)  # (n,)
    best_i = flat_idx // W_size
    best_j = flat_idx % W_size

    return best_i.astype(np.int32), best_j.astype(np.int32)


def snr_db_to_linear(snr_db: float) -> tuple[float, float]:
    """
    Convert SNR in dB to (P, sigma2) with P=1 fixed.

    We fix P=1 and set sigma2 = 1 / 10^(snr_db/10).
    This is a standard convention.
    """
    P = 1.0
    sigma2 = P / (10 ** (snr_db / 10))
    return P, sigma2


def achievable_rate(snr_linear: float) -> float:
    """Compute log2(1 + SNR) — the Shannon rate proxy. Equation (2)."""
    return np.log2(1 + snr_linear)


# ----------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Testing codebook...")
    F = build_dft_codebook(n_antennas=64, codebook_size=64)
    W = build_dft_codebook(n_antennas=16, codebook_size=16)
    print(f"  Tx codebook shape: {F.shape}  (expected: (64, 64))")
    print(f"  Rx codebook shape: {W.shape}  (expected: (16, 16))")

    # Check each beam is unit norm
    norms_F = np.linalg.norm(F, axis=1)
    norms_W = np.linalg.norm(W, axis=1)
    print(f"  Tx beam norms min/max: {norms_F.min():.4f} / {norms_F.max():.4f}  (should be ~1.0)")
    print(f"  Rx beam norms min/max: {norms_W.min():.4f} / {norms_W.max():.4f}  (should be ~1.0)")
    print("Codebook OK.")
