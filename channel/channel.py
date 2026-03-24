"""
channel/channel.py
==================
Generates geometric sparse mmWave MIMO channels using the
Saleh-Valenzuela model described in the project spec (Equation 3).

What this file does in plain English:
  - A mmWave channel H describes how a signal travels from the
    transmitter (Nt antennas) to the receiver (Nr antennas).
  - The signal bounces off L "scatterers" (walls, cars, buildings).
  - Each bounce has a random strength (alpha), a random departure
    angle (phi), and a random arrival angle (theta).
  - We combine all L bounces into the matrix H.
"""

import numpy as np


def ula_steering_vector(angle_rad: float, n_antennas: int) -> np.ndarray:
    """
    Compute the ULA (Uniform Linear Array) steering vector.

    This is equation (4) in the spec. Think of it as a "direction pointer"
    — it encodes which direction the antenna array is pointing.

    Args:
        angle_rad:   Angle of departure or arrival in radians.
        n_antennas:  Number of antennas in the array (Nt or Nr).

    Returns:
        Complex numpy array of shape (n_antennas,), unit norm.
    """
    indices = np.arange(n_antennas)
    # Half-wavelength spacing: d = lambda/2, so d/lambda = 0.5
    # Phase shift between adjacent antennas = pi * sin(angle)
    steering = np.exp(1j * np.pi * np.sin(angle_rad) * indices)
    return steering / np.sqrt(n_antennas)


def generate_channel(Nt: int, Nr: int, L: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one random mmWave MIMO channel matrix H.

    This implements Equation (3) from the spec:
        H = sqrt(Nt*Nr/L) * sum_{l=1}^{L} alpha_l * ar(theta_l) * at(phi_l)^H

    Args:
        Nt:   Number of transmit antennas (default 64).
        Nr:   Number of receive antennas (default 16).
        L:    Number of scattering paths (default 3).
        rng:  Numpy random generator (pass in for reproducibility).

    Returns:
        H: Complex numpy array of shape (Nr, Nt).
    """
    normalization = np.sqrt(Nt * Nr / L)

    H = np.zeros((Nr, Nt), dtype=complex)

    for _ in range(L):
        # Path gain: complex Gaussian, mean 0, variance 1
        alpha = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)

        # Departure angle (phi): uniform over [-pi/2, pi/2]
        phi = rng.uniform(-np.pi / 2, np.pi / 2)

        # Arrival angle (theta): uniform over [-pi/2, pi/2]
        theta = rng.uniform(-np.pi / 2, np.pi / 2)

        # Steering vectors
        at = ula_steering_vector(phi, Nt)    # Tx steering, shape (Nt,)
        ar = ula_steering_vector(theta, Nr)  # Rx steering, shape (Nr,)

        # Rank-1 outer product: ar * at^H, shape (Nr, Nt)
        H += alpha * np.outer(ar, at.conj())

    return normalization * H


def generate_channel_batch(
    n_samples: int,
    Nt: int,
    Nr: int,
    L: int,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a batch of N independent channel realizations.

    Args:
        n_samples: How many channels to generate.
        Nt, Nr, L: System parameters.
        seed:      Random seed for reproducibility.

    Returns:
        H_batch: Complex array of shape (n_samples, Nr, Nt).
    """
    rng = np.random.default_rng(seed)
    H_batch = np.array([generate_channel(Nt, Nr, L, rng) for _ in range(n_samples)])
    return H_batch


# ----------------------------------------------------------------
# Quick sanity check — run this file directly to verify it works:
#   python channel/channel.py
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("Testing channel generator...")
    Nt, Nr, L = 64, 16, 3
    rng = np.random.default_rng(42)
    H = generate_channel(Nt, Nr, L, rng)

    print(f"  H shape:     {H.shape}   (expected: ({Nr}, {Nt}))")
    print(f"  H dtype:     {H.dtype}   (expected: complex128)")
    print(f"  ||H||_F:     {np.linalg.norm(H):.2f}   (typical: ~30-40 for these params)")

    H_batch = generate_channel_batch(100, Nt, Nr, L, seed=42)
    print(f"  Batch shape: {H_batch.shape}  (expected: (100, {Nr}, {Nt}))")
    print("Channel generator OK.")
