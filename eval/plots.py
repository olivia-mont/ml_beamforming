"""
eval/plots.py
=============
Generates all required figures from the spec.

Figure 1: Achievable Rate vs Overhead T  (Section 4.4, Plot 1)
Figure 2: Top-1 Accuracy vs SNR          (Section 4.4, Plot 2)
Figure 3: SNR Mismatch Robustness        (Section 4.4 / Add-on 1)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (works on servers too)
import matplotlib.pyplot as plt


# ── Consistent style across all figures ──────────────────────
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'grid.alpha': 0.4,
})

COLORS = {
    'oracle':      '#2c7bb6',   # Blue
    'exhaustive':  '#4dac26',   # Green
    'ml':          '#d7191c',   # Red
    'random':      '#fdae61',   # Orange
    'matched':     '#d7191c',   # Red
    'mismatched':  '#2c7bb6',   # Blue
}

OUT_DIR = 'outputs/figures'


def _save(fig, filename):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ============================================================
# Figure 1: Rate vs Overhead T
# ============================================================
def plot_rate_vs_T(results: dict, save: bool = True):
    """
    Plot achievable rate vs probing overhead T.

    Args:
        results: dict from compute_rate_vs_T():
                 {T -> {'ml': rate, 'random': rate, 'oracle': rate, 'exhaustive': rate}}
    """
    T_values = sorted(results.keys())

    oracle_rates    = [results[T]['oracle']     for T in T_values]
    exhaustive_rates = [results[T]['exhaustive'] for T in T_values]
    ml_rates        = [results[T]['ml']         for T in T_values if results[T]['ml'] is not None]
    T_ml            = [T                        for T in T_values if results[T]['ml'] is not None]
    random_rates    = [results[T]['random']     for T in T_values]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Oracle is a flat horizontal line (doesn't depend on T)
    ax.axhline(oracle_rates[0], color=COLORS['oracle'], linestyle='--',
               label='Oracle (exhaustive, noiseless)')

    ax.plot(T_values, exhaustive_rates, 's--', color=COLORS['exhaustive'],
            label='Exhaustive sweep (upper bound, T=1024)')
    ax.plot(T_ml,     ml_rates,         'o-',  color=COLORS['ml'],
            label='ML (our model)')
    ax.plot(T_values, random_rates,     '^:',  color=COLORS['random'],
            label='Random subsampling')

    ax.set_xlabel('Probing Overhead T (# beam pairs measured)')
    ax.set_ylabel('Average Achievable Rate (bps/Hz)')
    ax.set_title('Rate vs Probing Overhead')
    ax.set_xscale('log', base=2)
    ax.set_xticks(T_values)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    if save:
        _save(fig, 'fig1_rate_vs_T.pdf')
    return fig


# ============================================================
# Figure 2: Top-1 Accuracy vs SNR
# ============================================================
def plot_acc_vs_snr(acc_dict: dict, save: bool = True):
    """
    Plot top-1 beam prediction accuracy vs SNR.

    Args:
        acc_dict: {snr_db -> top1_acc} from compute_acc_vs_snr().
    """
    snr_values = sorted(acc_dict.keys())
    accs = [acc_dict[snr] * 100 for snr in snr_values]  # convert to %

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(snr_values, accs, 'o-', color=COLORS['ml'], label='ML model')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Beam Prediction Accuracy vs SNR')
    ax.set_ylim([-2, 102])
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    if save:
        _save(fig, 'fig2_acc_vs_snr.pdf')
    return fig


# ============================================================
# Figure 3: SNR Mismatch Robustness (Add-on 1)
# ============================================================
def plot_snr_mismatch(mismatch_results: dict, save: bool = True):
    """
    Plot accuracy of matched vs mismatched model at low SNR.

    Args:
        mismatch_results: from compute_snr_mismatch():
          {'matched': {snr_db -> acc}, 'mismatched': {snr_db -> acc}}
    """
    snr_values = sorted(mismatch_results['matched'].keys())

    acc_matched    = [mismatch_results['matched'][s]    * 100 for s in snr_values]
    acc_mismatched = [mismatch_results['mismatched'][s] * 100 for s in snr_values]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(snr_values, acc_matched,    'o-', color=COLORS['matched'],
            label='Model trained on full SNR [-10, 20] dB')
    ax.plot(snr_values, acc_mismatched, 's--', color=COLORS['mismatched'],
            label='Model trained on high SNR [5, 20] dB only')

    ax.set_xlabel('Test SNR (dB)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('Add-on 1: SNR Mismatch Robustness\n'
                 'Both models tested at low SNR (out-of-distribution for mismatched model)')
    ax.set_ylim([-2, 102])
    ax.legend()
    ax.grid(True)

    # Annotate the degradation
    for i, snr in enumerate(snr_values):
        gap = acc_matched[i] - acc_mismatched[i]
        if gap > 1:
            ax.annotate(f'↓{gap:.1f}%',
                        xy=(snr, acc_mismatched[i]),
                        xytext=(snr + 0.3, acc_mismatched[i] + 3),
                        fontsize=9, color='gray')

    fig.tight_layout()
    if save:
        _save(fig, 'fig3_snr_mismatch.pdf')
    return fig


# ============================================================
# Bonus: Training curves (useful for debugging)
# ============================================================
def plot_training_curves(history: dict, tag: str = '', save: bool = True):
    """Plot train/val loss and val accuracy over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'],   label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Curves')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [a * 100 for a in history['val_acc']], color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True)

    fig.tight_layout()
    if save:
        _save(fig, f'training_curves_{tag}.pdf')
    return fig
