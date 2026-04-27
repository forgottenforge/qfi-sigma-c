"""Quick re-run of Phase 5 from cached results."""
import json
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import sys
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_qfi_analysis import (
    CIRCUIT_LABELS, CIRCUIT_SHORT, get_category, find_sigma_c,
)

# Load validation results
with open(Path(__file__).parent / "validation_results.json") as f:
    vr = json.load(f)

circuits = vr["circuits"]

print("=" * 80)
print("[5/5] CATEGORY ANALYSIS AND DIAGNOSTICS")
print("=" * 80)

# Category breakdown
print("\nCategory breakdown (reliable circuits only):")
cats = {}
for ckey, d in circuits.items():
    cat = d["category"]
    if cat not in cats:
        cats[cat] = {"match": 0, "fail": 0}
    if d["zero_in_ci"]:
        cats[cat]["match"] += 1
    else:
        cats[cat]["fail"] += 1

for cat in sorted(cats):
    m = cats[cat]["match"]
    f = cats[cat]["fail"]
    print(f"  {cat:<14}: {m} match, {f} fail -> {100*m/(m+f):.0f}% match")

# Symmetry analysis
symmetry_circuits = {"C02", "C13"}
print(f"\nSymmetry-protected circuits:")
for ckey in sorted(circuits):
    d = circuits[ckey]
    sym = ckey in symmetry_circuits
    match = d["zero_in_ci"]
    print(f"  {ckey} {d['label']:<20}: symmetry={'YES' if sym else 'no ':>3}, "
          f"match={'YES' if match else 'no ':>3}, delta_rel={d['delta_rel']:.3f}")

sym_match = sum(1 for c in circuits if c in symmetry_circuits and circuits[c]["zero_in_ci"])
sym_total = sum(1 for c in circuits if c in symmetry_circuits)
nosym_match = vr["paired_delta_overlap"] - sym_match
nosym_total = vr["n_reliable"] - sym_total
print(f"\n  With symmetry:    {sym_match}/{sym_total} match")
print(f"  Without symmetry: {nosym_match}/{nosym_total} match "
      f"({100*nosym_match/nosym_total:.0f}%)")

# Fisher exact test
if sym_total > 0 and nosym_total > 0:
    table = [[nosym_match, nosym_total - nosym_match],
             [sym_match, sym_total - sym_match]]
    odds, fisher_p = sp_stats.fisher_exact(table)
    print(f"  Fisher exact test (symmetry vs match): p = {fisher_p:.4f}")

print("\n" + "=" * 80)
print("FINAL STATISTICAL SUMMARY")
print("=" * 80)

print(f"""
Dataset: 16 Hamiltonians x 12 Trotter depths x 500 shots x 6 qubits
        (Rigetti Ankaa-3, blind kappa experiment, 2026-02-24)

Classification:
  Reliable testable circuits:     {vr['n_reliable']}
  Borderline (det < 50%):         {vr['n_borderline']}
  Non-testable (no sigma_c):      {vr['n_nontestable']}

Primary Result (Paired Bootstrap Delta, B=5000):
  0 in 95% CI of delta:           {vr['paired_delta_overlap']}/{vr['n_reliable']} = {100*vr['paired_delta_rate']:.0f}%
  Median |delta|:                  {vr['median_delta_abs']:.1f} CNOTs
  Median delta_rel:                {vr['median_delta_rel']:.3f}

Correlation:
  Spearman rho:                    {vr['spearman_rho']:.4f} (p = {vr['spearman_p']:.4f})
  Pearson r:                       {vr['pearson_r']:.4f} (p = {vr['pearson_p']:.4f})
  Log-scale Pearson r:             {vr['log_pearson_r']:.4f} (p = {vr['log_pearson_p']:.4f})
  Permutation p-value:             {vr['permutation_p']:.4f}

Binomial test (vs chance):
  Observed overlaps:               {vr['paired_delta_overlap']}/{vr['n_reliable']}
  Estimated chance rate:           {vr['chance_rate']:.3f}
  Binomial p-value:                {vr['binomial_p']:.6f}

Category Analysis:
  Generic (no symmetry):           {nosym_match}/{nosym_total} match = {100*nosym_match/nosym_total:.0f}%
  Symmetry-protected:              {sym_match}/{sym_total} match = {100*sym_match/sym_total:.0f}%

INTERPRETATION:
  sigma_c and CFI_peak are SIGNIFICANTLY CORRELATED (rho=0.69, p=0.039).
  The correlation is especially strong on log scale (r=0.84, p=0.004),
  suggesting a power-law relationship: CFI_peak ~ sigma_c^alpha.

  However, sigma_c != CFI_peak exactly. Per-circuit t-tests show most
  have sigma_c SYSTEMATICALLY LARGER than CFI_peak (sigma_c/CFI_peak > 1
  for 8/9 circuits). This is consistent with the multi-rate decay model:
  CFI peaks at the fast-mode crossover, sigma_c comes later when total
  performance has decayed to 1/e.

  The 5/9 paired-delta coincidence rate is conservative. With grid
  resolution of 10 CNOTs, 6/9 circuits have |delta| < 20 CNOTs.
""")
print("=" * 80)
