"""
Generate publication figures for R1 revision.

Figure 1: Combined log-log scatter (Ankaa-3 + Cepheus-1 + IonQ)
Figure 2: Symmetry-breaking kappa series (Block A)
Figure 3: N=8 vs N=6 kappa comparison (Block B)

Author: M. C. Wurm
Date: April 2026
"""

import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
})

OUT = Path(r"d:\code\qfi_sigma_c")

# Load critical analysis results
with open(OUT / "critical_analyses_results.json") as f:
    data = json.load(f)

combined = data['combined_results']
table_data = data['table_data']


# =====================================================================
# FIGURE 1: Combined scatter plot (log-log)
# =====================================================================

fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))

platform_markers = {'Ankaa-3': 'o', 'Cepheus-1': 's'}
platform_colors_base = {'Ankaa-3': '#333333', 'Cepheus-1': '#1976D2'}

# Category colors
cat_colors = {
    'structured': '#2196F3',
    'topological': '#9C27B0',
    'gap_protected': '#4CAF50',
    'null_random': '#9E9E9E',
    'edge_case': '#FF9800',
    'symmetry_breaking': '#E91E63',
    'null': '#9E9E9E',
}

# Map circuit categories from table_data
table_cats = {}
for t in table_data:
    cat_map = {'S': 'structured', 'T': 'topological', 'GP': 'gap_protected',
               'NR': 'null_random', 'EC': 'edge_case', 'SB': 'symmetry_breaking'}
    table_cats[t['label']] = cat_map.get(t['category'], 'structured')

# Original circuit categories
orig_cats = {
    'C01_Rand-2': 'null_random', 'C02_Heis': 'structured',
    'C04_ATA': 'edge_case', 'C06_Rand-1': 'null_random',
    'C09_Kitaev': 'topological', 'C11_Clust': 'topological',
    'C12_BCS': 'gap_protected', 'C13_TB': 'structured',
    'C15_XXZ': 'gap_protected',
}

for r in combined:
    sc = r['sigma_c_cnot']
    ep = r['eps_star_cnot']
    if sc is None or ep is None or sc <= 0 or ep <= 0:
        continue

    label = r['label']
    platform = r['platform']
    marker = platform_markers.get(platform, 'D')

    # Get category
    if label in orig_cats:
        cat = orig_cats[label]
    elif label in table_cats:
        cat = table_cats[label]
    else:
        cat = 'structured'

    color = cat_colors.get(cat, '#333333')
    edge = 'black' if platform == 'Ankaa-3' else '#555555'
    size = 60 if platform == 'Ankaa-3' else 45

    ax1.scatter(sc, ep, c=color, marker=marker, s=size,
                edgecolors=edge, linewidths=0.6, zorder=5, alpha=0.85)

    # Annotate key points
    short = label.replace('C0', 'C').replace('C1', 'C1')
    if platform == 'Ankaa-3':
        short_map = {
            'C01_Rand-2': 'C01', 'C02_Heis': 'C02', 'C04_ATA': 'C04',
            'C06_Rand-1': 'C06', 'C09_Kitaev': 'C09', 'C11_Clust': 'C11',
            'C12_BCS': 'C12', 'C13_TB': 'C13', 'C15_XXZ': 'C15',
        }
        short = short_map.get(label, label[:6])
    else:
        # Only annotate selected Cepheus-1 points
        annotate_list = ['heis_aniso_20', 'tb_vz05', 'heisenberg_n8',
                         'tfim_ordered_n8', 'all_to_all_fine',
                         'tfim_h20', 'compass_model', 'alt_bond_ising']
        if label not in annotate_list:
            continue
        short_map = {
            'heis_aniso_20': 'H100%', 'tb_vz05': 'TB0.5',
            'heisenberg_n8': 'H-N8', 'tfim_ordered_n8': 'TO-N8',
            'all_to_all_fine': 'ATA-f', 'tfim_h20': 'TFIM2',
            'compass_model': 'Comp', 'alt_bond_ising': 'ABI',
        }
        short = short_map.get(label, label[:5])

    ax1.annotate(short, (sc, ep), fontsize=5.5, ha='left', va='bottom',
                 xytext=(3, 3), textcoords='offset points', alpha=0.8)

# Identity line
lim_lo = 5
lim_hi = 300
ax1.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.3, linewidth=0.8,
         label=r'$\kappa = 1$')
# kappa = 1.3 line
ax1.plot([lim_lo, lim_hi], [lim_lo/1.3, lim_hi/1.3], 'r--', alpha=0.25,
         linewidth=0.8, label=r'$\kappa = 1.3$')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlim(lim_lo, lim_hi)
ax1.set_ylim(lim_lo, lim_hi)
ax1.set_xlabel(r'$\sigma_c$ (CNOT count)')
ax1.set_ylabel(r'$\epsilon^*$ (CFI peak, CNOT count)')
ax1.set_title(r'Combined $\sigma_c$ vs $\epsilon^*$ ($n = 33$, $r_{\log} = 0.80$)',
              fontweight='bold', fontsize=9)

# Legend
handles = []
for cat, color in [('Structured', '#2196F3'), ('Topological', '#9C27B0'),
                    ('Gap-protected', '#4CAF50'), ('Null/random', '#9E9E9E'),
                    ('Edge case', '#FF9800'), ('Symm.-breaking', '#E91E63')]:
    handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=6, label=cat))
handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                      markeredgecolor='black', markersize=6, label='Ankaa-3'))
handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
                      markeredgecolor='#555', markersize=6, label='Cepheus-1'))
ax1.legend(handles=handles, loc='upper left', fontsize=6.5, ncol=2,
           framealpha=0.9)

ax1.set_aspect('equal')
fig1.tight_layout()
for ext in ['pdf', 'png']:
    fig1.savefig(OUT / f"fig_combined_scatter.{ext}", bbox_inches='tight')
plt.close(fig1)
print("Figure 1 saved: fig_combined_scatter.pdf/png")


# =====================================================================
# FIGURE 2: Symmetry-breaking kappa series
# =====================================================================

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(7, 3.2))

# Heisenberg series
aniso_pct = [10, 50, 100]
heis_kappas = []
heis_ci_lo = []
heis_ci_hi = []
for t in table_data:
    if t['label'] in ['heis_aniso_11', 'heis_aniso_15', 'heis_aniso_20']:
        k = t['kappa_cnot']
        kci = t['bootstrap_cnot']['kappa_ci']
        heis_kappas.append(k)
        heis_ci_lo.append(k - kci[0] if not np.isnan(kci[0]) else 0)
        heis_ci_hi.append(kci[1] - k if not np.isnan(kci[1]) else 0)

ax2a.errorbar(aniso_pct, heis_kappas,
              yerr=[heis_ci_lo, heis_ci_hi],
              fmt='o-', color='#E91E63', capsize=4, markersize=7,
              linewidth=1.5, label='Cepheus-1')
ax2a.axhline(2.32, color='#2196F3', linestyle='--', linewidth=1,
             alpha=0.6, label=r'C02 Ankaa-3 ($\kappa = 2.32$)')
ax2a.axhspan(1.1, 1.3, alpha=0.1, color='green', label='Generic range')
ax2a.set_xlabel('ZZ anisotropy (%)')
ax2a.set_ylabel(r'$\kappa$')
ax2a.set_title('(a) Heisenberg SU(2) breaking', fontweight='bold', fontsize=9)
ax2a.set_xticks(aniso_pct)
ax2a.set_ylim(0.5, 3.0)
ax2a.legend(fontsize=6.5, loc='upper right')

# TB series
vz_vals = [0.1, 0.3, 0.5]
tb_kappas = []
tb_ci_lo = []
tb_ci_hi = []
for t in table_data:
    if t['label'] in ['tb_vz01', 'tb_vz03', 'tb_vz05']:
        k = t['kappa_cnot']
        kci = t['bootstrap_cnot']['kappa_ci']
        tb_kappas.append(k)
        tb_ci_lo.append(k - kci[0] if not np.isnan(kci[0]) else 0)
        tb_ci_hi.append(kci[1] - k if not np.isnan(kci[1]) else 0)

ax2b.errorbar(vz_vals, tb_kappas,
              yerr=[tb_ci_lo, tb_ci_hi],
              fmt='s-', color='#E91E63', capsize=4, markersize=7,
              linewidth=1.5, label='Cepheus-1')
ax2b.axhline(3.60, color='#2196F3', linestyle='--', linewidth=1,
             alpha=0.6, label=r'C13 Ankaa-3 ($\kappa = 3.60$)')
ax2b.axhspan(1.1, 1.3, alpha=0.1, color='green', label='Generic range')
ax2b.set_xlabel(r'$V_z$ (U(1) breaking strength)')
ax2b.set_ylabel(r'$\kappa$')
ax2b.set_title('(b) Tight binding U(1) breaking', fontweight='bold', fontsize=9)
ax2b.set_xticks(vz_vals)
ax2b.set_ylim(0.5, 4.0)
ax2b.legend(fontsize=6.5, loc='upper right')

fig2.tight_layout()
for ext in ['pdf', 'png']:
    fig2.savefig(OUT / f"fig_symmetry_breaking.{ext}", bbox_inches='tight')
plt.close(fig2)
print("Figure 2 saved: fig_symmetry_breaking.pdf/png")


# =====================================================================
# FIGURE 3: N=6 vs N=8 kappa comparison
# =====================================================================

fig3, ax3 = plt.subplots(figsize=(5, 3.5))

# Pair N=6 (original Ankaa-3) with N=8 (Cepheus-1)
pairs = [
    ('Heisenberg', 2.32, 1.23),  # C02 original, heisenberg_n8
    ('Tight binding', 3.60, 1.44),  # C13, tight_binding_n8
    ('XXZ', 1.16, 0.86),  # C15, xxz_aniso_n8
    ('Kitaev', 1.72, 1.49),  # C09, kitaev_detuned_n8
    ('BCS', 1.18, 0.93),  # C12, bcs_8qubit
    ('Cluster SPT', 1.11, 1.22),  # C11, cluster_spt_n8
]

labels = [p[0] for p in pairs]
k6 = [p[1] for p in pairs]
k8 = [p[2] for p in pairs]

x = np.arange(len(labels))
width = 0.35

bars6 = ax3.bar(x - width/2, k6, width, label='$N = 6$ (Ankaa-3)',
                color='#2196F3', alpha=0.8, edgecolor='black', linewidth=0.5)
bars8 = ax3.bar(x + width/2, k8, width, label='$N = 8$ (Cepheus-1)',
                color='#E91E63', alpha=0.8, edgecolor='black', linewidth=0.5)

ax3.axhline(1.3, color='green', linestyle='--', linewidth=1, alpha=0.5,
            label='Generic $\\kappa \\approx 1.3$')
ax3.set_ylabel(r'$\kappa$')
ax3.set_xticks(x)
ax3.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
ax3.set_title(r'$\kappa$ at $N = 6$ vs $N = 8$', fontweight='bold', fontsize=9)
ax3.legend(fontsize=7, loc='upper right')
ax3.set_ylim(0, 4.0)

fig3.tight_layout()
for ext in ['pdf', 'png']:
    fig3.savefig(OUT / f"fig_n_scaling.{ext}", bbox_inches='tight')
plt.close(fig3)
print("Figure 3 saved: fig_n_scaling.pdf/png")


# =====================================================================
# FIGURE 4: TFIM ordered N=8 becomes testable
# =====================================================================

fig4, ax4 = plt.subplots(figsize=(4, 3))

# Load TFIM ordered N=8 performance curve
with open(OUT / "r1_supplement_block_b.json") as f:
    block_b = json.load(f)

tfim_data = block_b['circuits']['tfim_ordered_n8']
depths_tfim = []
perf_tfim = []
for d_str, d_data in sorted(tfim_data['depths'].items(), key=lambda x: int(x[0])):
    bs = d_data.get('bitstrings', [])
    if len(bs) == 0:
        continue
    d = int(d_str)
    if d == 0:
        continue
    counts = {}
    for b in bs:
        counts[b] = counts.get(b, 0) + 1
    r = max(counts.values()) / len(bs)
    depths_tfim.append(d)
    perf_tfim.append(r)

depths_tfim = np.array(depths_tfim)
perf_tfim = np.array(perf_tfim)
tau = 1.0 / np.e
threshold = tau * perf_tfim[0]

ax4.plot(depths_tfim, perf_tfim, 'o-', color='#E91E63', markersize=5, linewidth=1.5)
ax4.axhline(threshold, color='gray', linestyle='--', linewidth=1,
            label=f'$\\tau \\cdot r(1) = {threshold:.3f}$')
ax4.axhline(1/256, color='#9E9E9E', linestyle=':', linewidth=0.8,
            label='$1/K = 1/256$')
ax4.set_xlabel('Trotter depth')
ax4.set_ylabel('$r(d) = \\max_x p(x|d)$')
ax4.set_title('TFIM ordered $N=8$: testable ($\\kappa = 0.60$)',
              fontweight='bold', fontsize=9)
ax4.legend(fontsize=7)

fig4.tight_layout()
for ext in ['pdf', 'png']:
    fig4.savefig(OUT / f"fig_tfim_n8.{ext}", bbox_inches='tight')
plt.close(fig4)
print("Figure 4 saved: fig_tfim_n8.pdf/png")

print("\nAll figures generated.")
