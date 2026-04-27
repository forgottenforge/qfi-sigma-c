"""
Generate publication figures for the sigma_c vs CFI paper.

Figure 1: Log-scale scatter plot (main result)
Figure 2: Individual CFI curves (4x4 grid)
Figure 3: Ratio kappa = sigma_c / CFI_peak by category + systematic offset
"""

import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUT = Path(r"D:\code\qfi_sigma_c")

# Load results
with open(OUT / "validation_results.json") as f:
    vr = json.load(f)

with open(OUT / "comprehensive_results.json") as f:
    cr = json.load(f)

# ============================================================
# Style
# ============================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
})

CAT_COLORS = {
    "structured":    "#E53935",    # red
    "topological":   "#7B1FA2",    # purple
    "gap_protected": "#2E7D32",    # green
    "null_random":   "#757575",    # gray
    "edge_case":     "#F57C00",    # orange
}

CAT_LABELS = {
    "structured":    "Structured (SU(2)/U(1))",
    "topological":   "Topological",
    "gap_protected": "Gap-protected",
    "null_random":   "Null / Random",
    "edge_case":     "Edge case",
}

# ============================================================
# Data extraction
# ============================================================
circuits = vr["circuits"]
names = []
sigma_cs = []
cfi_peaks = []
kappas = []
categories = []
delta_cis = []
sc_cis = []
cp_cis = []
zero_in_ci = []

for ckey in sorted(circuits):
    c = circuits[ckey]
    names.append(ckey)
    sigma_cs.append(c["sigma_c"])
    cfi_peaks.append(c["cfi_peak"])
    kappas.append(c["sigma_c"] / c["cfi_peak"])
    categories.append(c["category"])
    delta_cis.append(c["delta_ci"])
    sc_cis.append(c["sigma_c_ci"])
    cp_cis.append(c["cfi_peak_ci"])
    zero_in_ci.append(c["zero_in_ci"])

sigma_cs = np.array(sigma_cs)
cfi_peaks = np.array(cfi_peaks)
kappas = np.array(kappas)


# ============================================================
# FIGURE 1: Main result — log-scale scatter
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Panel (a): Log-log scatter
ax = axes[0]
for i, name in enumerate(names):
    cat = categories[i]
    color = CAT_COLORS[cat]
    marker = '*' if zero_in_ci[i] else 'o'
    ms = 14 if zero_in_ci[i] else 8

    ax.scatter(sigma_cs[i], cfi_peaks[i], c=color, marker=marker,
               s=ms**2, edgecolors='black', linewidths=0.5, zorder=5)

    # Error bars from bootstrap CIs
    sc_ci = sc_cis[i]
    cp_ci = cp_cis[i]
    ax.errorbar(sigma_cs[i], cfi_peaks[i],
                xerr=[[sigma_cs[i] - sc_ci[0]], [sc_ci[1] - sigma_cs[i]]],
                yerr=[[cfi_peaks[i] - cp_ci[0]], [cp_ci[1] - cfi_peaks[i]]],
                fmt='none', color=color, alpha=0.4, linewidth=1.0, capsize=2)

    ax.annotate(name, (sigma_cs[i], cfi_peaks[i]),
                fontsize=7, ha='left', va='bottom',
                xytext=(4, 4), textcoords='offset points')

# Reference lines
lim = [4, 400]
ax.plot(lim, lim, 'k--', alpha=0.3, linewidth=1, label='$\\sigma_c = \\epsilon^*$')
ax.plot(lim, [l/1.3 for l in lim], 'k:', alpha=0.3, linewidth=1,
        label='$\\sigma_c = 1.3\\,\\epsilon^*$')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('$\\sigma_c$ (CNOT count)')
ax.set_ylabel('CFI peak $\\epsilon^*$ (CNOT count)')
ax.set_title(f'(a) Log-scale scatter (Pearson $r = {vr["log_pearson_r"]:.2f}$, '
             f'$p = {vr["log_pearson_p"]:.4f}$)')
ax.set_aspect('equal')

# Legend
handles = []
for cat in ["structured", "topological", "gap_protected", "null_random", "edge_case"]:
    if cat in set(categories):
        handles.append(mpatches.Patch(color=CAT_COLORS[cat], label=CAT_LABELS[cat]))
handles.append(plt.Line2D([0], [0], marker='*', color='gray', markersize=10,
                           linestyle='None', label='$0 \\in \\Delta_{\\mathrm{CI}}$'))
handles.append(plt.Line2D([0], [0], marker='o', color='gray', markersize=6,
                           linestyle='None', label='$0 \\notin \\Delta_{\\mathrm{CI}}$'))
ax.legend(handles=handles, fontsize=8, loc='upper left')

# Panel (b): Ratio kappa by circuit
ax2 = axes[1]
x_pos = np.arange(len(names))
colors = [CAT_COLORS[c] for c in categories]
bars = ax2.bar(x_pos, kappas, color=colors, edgecolor='black', linewidth=0.5)

# Highlight matches
for i, z in enumerate(zero_in_ci):
    if z:
        bars[i].set_hatch('//')

ax2.axhline(1.0, color='black', linestyle='--', alpha=0.3, linewidth=1)
ax2.axhline(1.33, color='black', linestyle=':', alpha=0.3, linewidth=1)
ax2.axhline(2.0, color='red', linestyle='-', alpha=0.3, linewidth=1)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('$\\kappa = \\sigma_c / \\epsilon^*$')
ax2.set_title('(b) Ratio by circuit')

# Annotate zones
ax2.text(8.5, 1.15, 'perfect coincidence', fontsize=7, ha='right', alpha=0.5)
ax2.text(8.5, 1.5, 'generic offset ($\\kappa \\approx 1.3$)',
         fontsize=7, ha='right', alpha=0.5)
ax2.text(8.5, 2.7, 'symmetry-protected', fontsize=7, ha='right', color='red', alpha=0.7)

ax2.set_ylim(0, 4.2)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(OUT / f"fig1_main_result.{ext}", bbox_inches='tight')
plt.close()
print("Saved fig1_main_result.png/pdf")


# ============================================================
# FIGURE 2: 4x4 individual circuit CFI curves
# ============================================================
import sys
sys.path.insert(0, str(OUT))
from comprehensive_qfi_analysis import (
    get_category, CIRCUIT_LABELS, CIRCUIT_SHORT,
    load_all_circuit_data_fast, classical_fisher_information,
    compute_performance, find_sigma_c, TAU,
)

fig = plt.figure(figsize=(16, 16))
gs = GridSpec(4, 4, hspace=0.35, wspace=0.30,
              left=0.06, right=0.97, top=0.96, bottom=0.04)

for cidx in range(16):
    row = cidx // 4
    col = cidx % 4
    ax = fig.add_subplot(gs[row, col])

    cat = get_category(cidx)
    color = CAT_COLORS.get(cat, "#333333")

    # Load and compute
    depths, probs, raw_meas, cnots = load_all_circuit_data_fast(cidx)
    param_evo = cnots[1:].astype(float)
    probs_evo = probs[1:]

    cfi_evo = classical_fisher_information(param_evo, probs_evo)
    perf_evo = compute_performance(probs_evo)
    sc, _ = find_sigma_c(param_evo, perf_evo)

    # CFI peak
    if len(cfi_evo) > 2:
        pk_idx = np.argmax(cfi_evo[1:-1]) + 1
        cfi_pk = param_evo[pk_idx]
    else:
        pk_idx = np.argmax(cfi_evo)
        cfi_pk = param_evo[pk_idx]

    # Plot CFI
    ax.plot(param_evo, cfi_evo, 'o-', color=color, markersize=3, linewidth=1.2)
    ax.axvline(cfi_pk, color=color, linestyle='--', alpha=0.6, linewidth=0.8)

    # Plot sigma_c
    if sc is not None:
        ax.axvline(sc, color='red', linestyle='-', alpha=0.8, linewidth=1.5)

    # Performance on twin axis
    ax2 = ax.twinx()
    ax2.plot(param_evo, perf_evo, 's-', color='gray', markersize=2,
             alpha=0.4, linewidth=0.8)
    if sc is not None:
        threshold = TAU * perf_evo[0]
        ax2.axhline(threshold, color='red', linestyle=':', alpha=0.3, linewidth=0.5)
    ax2.tick_params(axis='y', labelsize=6, colors='gray')

    # Title with match indicator
    ckey = f"C{cidx:02d}"
    match_str = ""
    if ckey in vr["circuits"] and vr["circuits"][ckey]["zero_in_ci"]:
        match_str = " [M]"
    elif ckey not in vr["circuits"]:
        if sc is None:
            match_str = " [NT]"

    ax.set_title(f'{ckey}: {CIRCUIT_SHORT[cidx]}{match_str}',
                 fontsize=9, fontweight='bold', color=color)
    if row == 3:
        ax.set_xlabel('CNOT count', fontsize=8)
    if col == 0:
        ax.set_ylabel('CFI', fontsize=8)

for ext in ['png', 'pdf']:
    fig.savefig(OUT / f"fig2_circuit_panels.{ext}", bbox_inches='tight')
plt.close()
print("Saved fig2_circuit_panels.png/pdf")


# ============================================================
# FIGURE 3: Diagnostic — delta distribution + detection rate
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# (a) Delta distribution
ax = axes[0]
deltas = [c["delta"] for c in circuits.values()]
delta_rels = [c["delta_rel"] for c in circuits.values()]
cats_list = [c["category"] for c in circuits.values()]
colors_list = [CAT_COLORS[c] for c in cats_list]

ax.bar(range(len(deltas)), deltas, color=colors_list, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('$|\\Delta|$ = $|\\sigma_c - \\epsilon^*|$ (CNOTs)')
ax.set_title('(a) Absolute delta by circuit')
ax.axhline(10, color='gray', linestyle=':', alpha=0.5, label='1 grid step')
ax.axhline(20, color='gray', linestyle='--', alpha=0.5, label='2 grid steps')
ax.legend(fontsize=7)

# (b) Relative delta
ax = axes[1]
ax.bar(range(len(delta_rels)), delta_rels, color=colors_list,
       edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('$|\\Delta| / \\sigma_c$')
ax.set_title('(b) Relative delta by circuit')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.axhline(1.0, color='red', linestyle='-', alpha=0.3)

# (c) Detection rate for all 16 circuits
ax = axes[2]
det_rates = []
det_names = []
det_colors = []
for res in cr["blind_kappa"]:
    cidx = res["circuit"]
    cat = get_category(cidx)
    det_rate = res["bootstrap_evo"]["sigma_c_detection_rate"]
    det_rates.append(det_rate)
    det_names.append(f"C{cidx:02d}")
    det_colors.append(CAT_COLORS.get(cat, "#333"))

ax.bar(range(16), det_rates, color=det_colors, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(16))
ax.set_xticklabels(det_names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('$\\sigma_c$ detection rate')
ax.set_title('(c) Bootstrap detection rate (B=2000)')
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='threshold')
ax.legend(fontsize=7)
ax.set_ylim(0, 1.05)

plt.tight_layout()
for ext in ['png', 'pdf']:
    fig.savefig(OUT / f"fig3_diagnostics.{ext}", bbox_inches='tight')
plt.close()
print("Saved fig3_diagnostics.png/pdf")

print("\nAll figures generated successfully.")
