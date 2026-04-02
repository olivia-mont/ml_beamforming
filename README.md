# ECE 508 Term Project — mmWave Beam Management

**Learning-Aided mmWave Beam Management with Geometric Sparse Channels and Codebooks**

Team members: Daniel Best and Olivia Monteiro
Add-on: Robustness Slice — SNR Mismatch

---

## What this project does

We train a neural network to predict the best antenna beam pair for a mmWave link,
using only a small number of noisy pilot power measurements.
We compare against exhaustive sweep and random subsampling baselines,
and test robustness when the model is evaluated on SNR conditions it wasn't trained on.

---

## Repository Structure

```
mmwave_project/
├── configs/
│   └── default.yaml          # All hyperparameters live here
├── channel/
│   ├── channel.py            # Channel simulator (Saleh-Valenzuela model)
│   └── dataset.py            # Dataset generation (pilot measurements + labels)
├── codebook/
│   └── codebook.py           # DFT codebook + oracle beam selection
├── baseline/
│   └── baselines.py          # Exhaustive sweep + random subsampling
├── ml/
│   ├── model.py              # MLP neural network definition
│   └── train.py              # Training loop with early stopping
├── eval/
│   ├── evaluate.py           # Accuracy, rate, SNR mismatch evaluation
│   ├── plots.py              # All figure generation
│   └── run_all.py            # ONE command to reproduce everything
├── outputs/                  # Created automatically when you run
│   ├── figures/              # PDF figures for the report
│   ├── checkpoints/          # Saved model weights
│   ├── datasets/             # Cached datasets (.npz files)
│   └── results.json          # All numeric results
└── requirements.txt
```

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd mmwave_project

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Reproduce All Results

```bash
python -m eval.run_all
```

This single command:
1. Generates training/validation/test datasets for all T values
2. Trains the ML model (matched + mismatched SNR)
3. Evaluates baselines (exhaustive sweep, random subsampling)
4. Saves all three required figures to `outputs/figures/`
5. Saves all numeric results to `outputs/results.json`

**Expected runtime:** ~30-60 min on CPU | ~10-15 min with GPU

---

## Sanity checks (run these first to verify setup)

```bash
python channel/channel.py      # Test channel generator
python codebook/codebook.py    # Test DFT codebook
python channel/dataset.py      # Test dataset generator (100 samples)
python baseline/baselines.py   # Test baselines (50 channels)
python ml/model.py             # Test neural network
```

---

## System Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Nt | 64 | Number of Tx antennas |
| Nr | 16 | Number of Rx antennas |
| L | 3 | Number of multipath components |
| \|F\| | 64 | Tx codebook size (DFT beams) |
| \|W\| | 16 | Rx codebook size (DFT beams) |
| Total classes | 1024 | F_size × W_size |
| T (default) | 32 | Pilot measurements used |
| Training SNR | [-10, 20] dB | Matched model |
| Training SNR | [5, 20] dB | Mismatched model (Add-on) |

---

## Probing Design (Fixed, per Section 3 of spec)

- **Tx beams:** T evenly-spaced indices from the 64-beam DFT codebook
- **Rx beams:** First 2 beams fixed (Rx_beams_probed = 2)
- **Total measurements per channel:** T × 2
- Design is identical across all methods for fair comparison

---

## Output Figures

| File | Plot | Spec Reference |
|------|------|----------------|
| `fig1_rate_vs_T.pdf` | Achievable rate vs overhead T | Section 4.4 Plot 1 |
| `fig2_acc_vs_snr.pdf` | Top-1 accuracy vs SNR | Section 4.4 Plot 2 |
| `fig3_snr_mismatch.pdf` | SNR mismatch robustness | Add-on 1 / Section 4.4 Plot 3 |

---

## Changing Parameters

Edit `configs/default.yaml` and re-run. Delete `outputs/datasets/` and `outputs/checkpoints/`
if you want to force regeneration from scratch.
