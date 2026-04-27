"""
Generate theory and robustness figures for the sigma_c vs CFI paper.

Figure 4: (a) Single-rate theorem G(f) monotonic decay
           (b) Two-rate model CFI with interior peak
Figure 5: (a) Leave-one-out stability
           (b) Power-law fit with regression line
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(r"D:\code\qfi_sigma_c")
sys.path.insert(0, str(OUT))
from comprehensive_qfi_analysis import (
    load_all_circuit_data_fast, CIRCUIT_SHORT, TAU,
)

with open(OUT / "validation_results.json") as f:
    vr = json.load(f)

with open(OUT / "theoretical_results.json") as f:
    tr = json.load(f)

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

# ============================================================
# FIGURE 4: Theory
# ============================================================
print("Generating fig4_theory.pdf ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

# --- Panel (a): Single-rate theorem ---
# Show G(f) for a few representative circuits
circuits_to_show = [1, 6, 9, 15]  # Random-2, Random-1, Kitaev, XXZ
colors = ['#E53935', '#757575', '#7B1FA2', '#2E7D32']
labels = ['C01 Random-2', 'C06 Random-1', 'C09 Kitaev', 'C15 XXZ']

fs = np.linspace(0.001, 0.999, 500)

for cidx, col, lab in zip(circuits_to_show, colors, labels):
    depths, probs, raw_meas, cnots = load_all_circuit_data_fast(cidx)
    q = probs[1].copy()  # depth 1 distribution
    u = 1.0 / 64
    s = q - u

    Gs = []
    for f in fs:
        d = np.maximum((1 - f) * q + f * u, 1e-30)
        G_val = (1 - f) ** 2 * np.sum(s ** 2 / d)
        Gs.append(G_val)
    Gs = np.array(Gs)
    # Normalize to G(0+)
    Gs_norm = Gs / Gs[0]
    ax1.plot(fs, Gs_norm, color=col, linewidth=1.5, label=lab)

ax1.set_xlabel(r'Depolarizing fraction $f$')
ax1.set_ylabel(r'$G(f) / G(0^+)$')
ax1.set_title(r'(a) Single-rate: $G(f)$ monotonically decreasing')
ax1.legend(loc='upper right', framealpha=0.9)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1.05)
ax1.axhline(1/np.e, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.text(0.7, 1/np.e + 0.03, r'$1/e$', color='gray', fontsize=8)

# --- Panel (b): Two-rate model CFI with peak ---
K = 64
u = 1.0 / K

# Mode 1 (fast): concentrates at x=0
a1 = np.zeros(K)
a1[0] = 0.30
a1[1:] = -0.30 / (K - 1)

# Mode 2 (slow): competing
a2 = np.zeros(K)
a2[0] = -0.08
a2[1] = 0.12
a2[2] = 0.06
a2[3] = 0.04
a2[4:] = -0.14 / (K - 4)

ratios_to_plot = [5.0, 10.0, 20.0]
ratio_colors = ['#1E88E5', '#E53935', '#2E7D32']

for tau_ratio, col in zip(ratios_to_plot, ratio_colors):
    tau1 = 1.0
    tau2 = tau1 * tau_ratio
    n_pts = 40000
    eps = np.linspace(0, tau2 * 8, n_pts)

    p_all = np.zeros((n_pts, K))
    dp_all = np.zeros((n_pts, K))
    for i, e in enumerate(eps):
        e1 = np.exp(-e / tau1)
        e2 = np.exp(-e / tau2)
        p_all[i] = u + a1 * e1 + a2 * e2
        dp_all[i] = -(a1 / tau1) * e1 - (a2 / tau2) * e2

    p_all = np.maximum(p_all, 1e-15)
    cfi = np.sum(dp_all ** 2 / p_all, axis=1)

    # Performance
    r = np.max(p_all, axis=1)
    r0 = r[0]
    thr = r0 * TAU

    # Normalize CFI
    cfi_norm = cfi / cfi.max()
    ax2.plot(eps / tau1, cfi_norm, color=col, linewidth=1.5,
             label=rf'$\tau_2/\tau_1 = {int(tau_ratio)}$')

    # Mark peak
    pk_idx = np.argmax(cfi)
    ax2.plot(eps[pk_idx] / tau1, 1.0, 'v', color=col, markersize=6)

    # Mark sigma_c
    cross = np.where((r[:-1] >= thr) & (r[1:] < thr))[0]
    if len(cross) > 0:
        idx = cross[0]
        sc = eps[idx] + (eps[idx + 1] - eps[idx]) * (thr - r[idx]) / (r[idx + 1] - r[idx])
        ax2.axvline(sc / tau1, color=col, linestyle=':', linewidth=0.8, alpha=0.6)

ax2.set_xlabel(r'Noise $\epsilon / \tau_1$')
ax2.set_ylabel(r'Normalized CFI')
ax2.set_title(r'(b) Two-rate: competing modes $\rightarrow$ CFI peak')
ax2.legend(loc='upper right', framealpha=0.9)
ax2.set_xlim(0, 50)
ax2.set_ylim(0, 1.15)
# Add annotation for sigma_c vs eps*
ax2.annotate(r'$\epsilon^*$', xy=(0.45, 0.95), xycoords='axes fraction',
             fontsize=9, ha='center', color='#555555')
ax2.annotate(r'$\sigma_c$', xy=(0.55, 0.60), xycoords='axes fraction',
             fontsize=9, ha='center', color='#555555')

plt.tight_layout()
fig.savefig(OUT / "fig4_theory.pdf", bbox_inches='tight')
plt.close()
print("  -> fig4_theory.pdf saved")


# ============================================================
# FIGURE 5: Robustness
# ============================================================
print("Generating fig5_robustness.pdf ...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2))

circuits = vr["circuits"]
ckeys = sorted(circuits.keys())
sc_vals = np.array([circuits[k]["sigma_c"] for k in ckeys])
cp_vals = np.array([circuits[k]["cfi_peak"] for k in ckeys])
N = len(ckeys)

# --- Panel (a): Leave-one-out stability ---
full_r = vr['log_pearson_r']
loo_r = []
for i in range(N):
    sc_loo = np.delete(sc_vals, i)
    cp_loo = np.delete(cp_vals, i)
    r_lp, _ = sp_stats.pearsonr(np.log(sc_loo), np.log(cp_loo))
    loo_r.append(r_lp)

xlabels = [ck for ck in ckeys]
x_pos = np.arange(N)
bar_colors = ['#E53935' if r < full_r else '#2E7D32' for r in loo_r]

ax1.barh(x_pos, loo_r, color=bar_colors, height=0.7, alpha=0.8)
ax1.axvline(full_r, color='k', linestyle='--', linewidth=1.2, label=f'Full ($r = {full_r:.3f}$)')
ax1.set_yticks(x_pos)
ax1.set_yticklabels(xlabels, fontsize=8)
ax1.set_xlabel(r'Log-scale Pearson $r$')
ax1.set_title('(a) Leave-one-out stability')
ax1.set_xlim(0.6, 1.0)
ax1.legend(loc='lower right', fontsize=8)
ax1.invert_yaxis()

# --- Panel (b): Power-law fit ---
log_sc = np.log(sc_vals)
log_cp = np.log(cp_vals)
slope, intercept, r_val, _, se = sp_stats.linregress(log_cp, log_sc)

# Regression line
x_fit = np.linspace(log_cp.min() - 0.3, log_cp.max() + 0.3, 100)
y_fit = intercept + slope * x_fit

# Confidence band
n_df = N - 2
t_crit = sp_stats.t.ppf(0.975, n_df)
x_mean = np.mean(log_cp)
se_fit = se * np.sqrt(1 / N + (x_fit - x_mean) ** 2 / np.sum((log_cp - x_mean) ** 2))
y_upper = y_fit + t_crit * se_fit
y_lower = y_fit - t_crit * se_fit

# Identity line
ax2.plot(x_fit, x_fit, 'k--', linewidth=0.8, alpha=0.5, label=r'$\beta = 1$')

# Regression
ax2.fill_between(x_fit, y_lower, y_upper, color='#1E88E5', alpha=0.15)
ax2.plot(x_fit, y_fit, '-', color='#1E88E5', linewidth=1.5,
         label=rf'$\beta = {slope:.2f}$ [{slope - t_crit*se:.2f}, {slope + t_crit*se:.2f}]')

# Category colors
CAT_COLORS = {
    "structured":    "#E53935",
    "topological":   "#7B1FA2",
    "gap_protected": "#2E7D32",
    "null_random":   "#757575",
    "edge_case":     "#F57C00",
}

for i, ck in enumerate(ckeys):
    cat = circuits[ck]["category"]
    col = CAT_COLORS.get(cat, '#333333')
    ax2.scatter(log_cp[i], log_sc[i], c=col, s=50, zorder=5, edgecolors='k', linewidths=0.5)
    ax2.annotate(ck, (log_cp[i], log_sc[i]), fontsize=6,
                 xytext=(4, 4), textcoords='offset points')

ax2.set_xlabel(r'$\ln\,\epsilon^*$ (CFI peak)')
ax2.set_ylabel(r'$\ln\,\sigma_c$')
ax2.set_title(rf'(b) Power-law fit: $R^2 = {r_val**2:.2f}$')
ax2.legend(loc='upper left', fontsize=7)

plt.tight_layout()
fig.savefig(OUT / "fig5_robustness.pdf", bbox_inches='tight')
plt.close()
print("  -> fig5_robustness.pdf saved")

print("Done.")
