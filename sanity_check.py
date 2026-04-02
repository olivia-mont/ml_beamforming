"""
sanity_check.py
===============
Run this FIRST after cloning the repo to verify your environment is set up correctly.

Usage:
    python sanity_check.py

Expected output: all checks print OK with no errors.
Takes < 30 seconds.
"""

import sys
import numpy as np

print("=" * 55)
print("ECE 508 mmWave Project — Environment Sanity Check")
print("=" * 55)

# ── 1. Python version ────────────────────────────────────────
v = sys.version_info
assert v.major == 3 and v.minor >= 10, f"Need Python 3.10+, got {v}"
print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")

# ── 2. Required packages ─────────────────────────────────────
try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"     CUDA available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print(f"     Apple MPS available")
    else:
        print(f"     CPU only (training will be slower but still works)")
except ImportError:
    print("[FAIL] PyTorch not found — run: pip install torch")
    sys.exit(1)

try:
    import matplotlib
    print(f"[OK] Matplotlib {matplotlib.__version__}")
except ImportError:
    print("[FAIL] Matplotlib not found — run: pip install matplotlib")
    sys.exit(1)

try:
    import yaml
    print(f"[OK] PyYAML found")
except ImportError:
    print("[FAIL] PyYAML not found — run: pip install pyyaml")
    sys.exit(1)

# ── 3. Project imports ───────────────────────────────────────
print()
print("Checking project modules...")

try:
    from channel.channel import generate_channel
    print("[OK] channel.channel")
except Exception as e:
    print(f"[FAIL] channel.channel: {e}")
    sys.exit(1)

try:
    from codebook.codebook import build_dft_codebook, snr_db_to_linear
    print("[OK] codebook.codebook")
except Exception as e:
    print(f"[FAIL] codebook.codebook: {e}")
    sys.exit(1)

try:
    from ECE508.ml_beamforming.mmwave_project.channel.dataset_old import generate_dataset
    print("[OK] channel.dataset")
except Exception as e:
    print(f"[FAIL] channel.dataset: {e}")
    sys.exit(1)

try:
    from ECE508.ml_beamforming.mmwave_project.baseline.baselines_old import oracle_rate, random_subsampling
    print("[OK] baseline.baselines")
except Exception as e:
    print(f"[FAIL] baseline.baselines: {e}")
    sys.exit(1)

try:
    from ml.model import BeamMLP
    print("[OK] ml.model")
except Exception as e:
    print(f"[FAIL] ml.model: {e}")
    sys.exit(1)

# ── 4. Quick functional checks ───────────────────────────────
print()
print("Running quick functional checks...")

rng = np.random.default_rng(42)
H = generate_channel(64, 16, 3, rng)
assert H.shape == (16, 64), f"Bad H shape: {H.shape}"
print(f"[OK] Channel generation: H shape={H.shape}, ||H||_F={np.linalg.norm(H):.1f}")

F = build_dft_codebook(64, 64)
W = build_dft_codebook(16, 16)
assert np.allclose(np.linalg.norm(F, axis=1), 1.0), "Beams not unit norm"
print(f"[OK] Codebook: F={F.shape}, W={W.shape}, all beams unit-norm")

ds = generate_dataset(20, 64, 16, 3, 64, 16, 16, -10, 20, seed=42)
assert ds['z'].shape == (20, 32), f"Bad z shape: {ds['z'].shape}"  # T=16, Rx=2 → 32 probes
assert ds['label'].max() < 1024
print(f"[OK] Dataset: z={ds['z'].shape}, labels in [0,{ds['label'].max()}]")

model = BeamMLP(input_dim=32, n_classes=1024)
z_t = torch.randn(4, 32)
out = model(z_t)
assert out.shape == (4, 1024)
print(f"[OK] BeamMLP: input=32 → output shape={out.shape}")

# ── 5. Config file ───────────────────────────────────────────
import os
import yaml
cfg_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
assert os.path.exists(cfg_path), f"Config not found: {cfg_path}"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
assert cfg['system']['Nt'] == 64
print(f"[OK] Config: Nt={cfg['system']['Nt']}, Nr={cfg['system']['Nr']}, L={cfg['system']['L']}")

print()
print("=" * 55)
print("ALL CHECKS PASSED — ready to run:")
print("  python -m eval.run_all")
print("=" * 55)
