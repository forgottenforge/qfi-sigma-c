"""
Analysis of R1 Supplement QPU Data
===================================
Processes 4 blocks of Cepheus-1 QPU data:
  A: Symmetry-breaking control (M6)
  B: N=8 scaling (M5)
  C: Fine depth resolution
  D: Additional N=6 Hamiltonians

For each circuit: sigma_c, CFI peak (epsilon*), kappa = sigma_c/epsilon*,
and paired bootstrap CIs (B=2000).

Author: M. C. Wurm / ForgottenForge
Date: April 2026
"""

import json
import numpy as np
from pathlib import Path

TAU = 1.0 / np.e
BOOTSTRAP_B = 2000
DATA_DIR = Path(r"d:\code\qfi_sigma_c")


# =====================================================================
# CORE FUNCTIONS (matched to comprehensive_qfi_analysis.py)
# =====================================================================

def bitstrings_to_probs(bitstrings, n_outcomes):
    """Convert list of bitstring strings to probability distribution."""
    counts = np.zeros(n_outcomes)
    for bs in bitstrings:
        counts[int(bs, 2)] += 1
    return counts / len(bitstrings)


def compute_performance(probs):
    """Performance = max outcome probability at each depth."""
    return np.max(probs, axis=1)


def classical_fisher_information(param_values, probs, regularize=1e-10):
    """CFI(d) = sum_x [dp(x|d)/dd]^2 / p(x|d) via central differences."""
    N, K = probs.shape
    probs_reg = probs + regularize
    probs_reg = probs_reg / probs_reg.sum(axis=1, keepdims=True)

    dp = np.zeros_like(probs_reg)
    for i in range(N):
        if i == 0:
            h = param_values[1] - param_values[0]
            if h > 0:
                dp[i] = (probs_reg[1] - probs_reg[0]) / h
        elif i == N - 1:
            h = param_values[-1] - param_values[-2]
            if h > 0:
                dp[i] = (probs_reg[-1] - probs_reg[-2]) / h
        else:
            h = param_values[i+1] - param_values[i-1]
            if h > 0:
                dp[i] = (probs_reg[i+1] - probs_reg[i-1]) / h

    return np.sum(dp**2 / probs_reg, axis=1)


def find_sigma_c(param_values, performance, tau=TAU):
    """Find sigma_c: parameter where performance drops to tau * r(baseline)."""
    r0 = performance[0]
    threshold = tau * r0
    for i in range(1, len(performance)):
        if performance[i] <= threshold:
            if performance[i-1] == performance[i]:
                return param_values[i]
            frac = (threshold - performance[i-1]) / (performance[i] - performance[i-1])
            return param_values[i-1] + frac * (param_values[i] - param_values[i-1])
    return None


def compute_kl_from_uniform(probs):
    """KL divergence from uniform distribution."""
    K = probs.shape[1]
    uniform = np.ones(K) / K
    kl = np.zeros(probs.shape[0])
    for i in range(probs.shape[0]):
        p = probs[i]
        p_safe = np.where(p > 0, p, 1e-30)
        kl[i] = np.sum(p_safe * np.log(p_safe / uniform))
    return kl


# =====================================================================
# DATA LOADING
# =====================================================================

def load_block(block_id):
    """Load a block JSON and extract structured data."""
    fpath = DATA_DIR / f"r1_supplement_block_{block_id.lower()}.json"
    with open(fpath) as f:
        data = json.load(f)
    return data


def extract_circuit_data(circuit_data):
    """Extract depths, bitstrings, CNOT counts from a circuit entry.

    Returns (depths, probs, raw_bitstrings_per_depth, cnot_counts, n_outcomes)
    """
    nq = circuit_data['n_qubits']
    n_outcomes = 2 ** nq

    depth_entries = []
    for d_str, d_data in circuit_data['depths'].items():
        bs = d_data.get('bitstrings', [])
        if len(bs) == 0:
            continue
        depth_entries.append((int(d_str), bs, d_data.get('cnot_count', 0)))

    depth_entries.sort(key=lambda x: x[0])

    depths = np.array([e[0] for e in depth_entries])
    cnots = np.array([e[2] for e in depth_entries], dtype=float)
    raw_bs = [e[1] for e in depth_entries]

    probs = np.zeros((len(depths), n_outcomes))
    for i, bs_list in enumerate(raw_bs):
        probs[i] = bitstrings_to_probs(bs_list, n_outcomes)

    return depths, probs, raw_bs, cnots, n_outcomes


# =====================================================================
# BOOTSTRAP
# =====================================================================

def bootstrap_analysis(raw_bs_list, param_values, n_outcomes, B=BOOTSTRAP_B,
                       tau=TAU, seed=42):
    """Paired bootstrap for sigma_c and CFI peak location."""
    rng = np.random.RandomState(seed)
    n_depths = len(raw_bs_list)

    # Pre-convert bitstrings to integer indices
    int_arrays = []
    for bs_list in raw_bs_list:
        int_arrays.append(np.array([int(bs, 2) for bs in bs_list]))

    sc_samples = []
    ep_samples = []

    for b in range(B):
        probs_boot = np.zeros((n_depths, n_outcomes))
        for d in range(n_depths):
            raw = int_arrays[d]
            n = len(raw)
            idx = rng.randint(0, n, size=n)
            counts = np.bincount(raw[idx], minlength=n_outcomes)
            probs_boot[d] = counts / n

        # sigma_c
        perf = compute_performance(probs_boot)
        sc = find_sigma_c(param_values, perf, tau)

        # CFI peak (interior only)
        cfi = classical_fisher_information(param_values, probs_boot)
        if len(cfi) > 2:
            interior = cfi[1:-1]
            peak_idx = np.argmax(interior) + 1
            ep = param_values[peak_idx]
        else:
            ep = param_values[np.argmax(cfi)]

        sc_samples.append(sc if sc is not None else np.nan)
        ep_samples.append(ep)

    sc_arr = np.array(sc_samples)
    ep_arr = np.array(ep_samples)

    valid = ~np.isnan(sc_arr)
    sc_valid = sc_arr[valid]
    ep_valid = ep_arr[valid]

    def ci(arr):
        if len(arr) < 10:
            return [np.nan, np.nan]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    sc_ci = ci(sc_valid)
    ep_ci = ci(ep_valid)

    # CI overlap test
    overlap = False
    if not (np.isnan(sc_ci[0]) or np.isnan(ep_ci[0])):
        overlap = not (sc_ci[1] < ep_ci[0] or ep_ci[1] < sc_ci[0])

    # Kappa distribution
    kappa_arr = sc_valid / ep_valid if len(ep_valid) > 0 else np.array([])
    kappa_ci = ci(kappa_arr) if len(kappa_arr) > 10 else [np.nan, np.nan]

    return {
        'sigma_c_median': float(np.nanmedian(sc_arr)),
        'sigma_c_ci': sc_ci,
        'cfi_peak_median': float(np.nanmedian(ep_arr)),
        'cfi_peak_ci': ep_ci,
        'ci_overlap': overlap,
        'detection_rate': float(np.mean(valid)),
        'kappa_median': float(np.median(kappa_arr)) if len(kappa_arr) > 0 else np.nan,
        'kappa_ci': kappa_ci,
    }


# =====================================================================
# ANALYZE ONE CIRCUIT
# =====================================================================

def analyze_one_circuit(circuit_data, verbose=True):
    """Full analysis: evolution-only (depth >= 1)."""
    label = circuit_data['label']
    nq = circuit_data['n_qubits']
    depths, probs, raw_bs, cnots, n_outcomes = extract_circuit_data(circuit_data)

    if len(depths) < 3:
        if verbose:
            print(f"    {label}: too few depths ({len(depths)}), skipping")
        return None

    # --- Evolution-only analysis (skip depth=0) ---
    mask = depths >= 1
    if mask.sum() < 3:
        if verbose:
            print(f"    {label}: too few evo depths, using full range")
        mask = np.ones(len(depths), dtype=bool)

    depths_evo = depths[mask]
    probs_evo = probs[mask]
    cnots_evo = cnots[mask]
    raw_evo = [raw_bs[i] for i in range(len(depths)) if mask[i]]

    # Use CNOT count as parameter (consistent with main paper)
    param_evo = cnots_evo.copy()
    if param_evo[0] == 0:
        param_evo[0] = 0.5

    # Performance and sigma_c
    perf_evo = compute_performance(probs_evo)
    sigma_c_cnot = find_sigma_c(param_evo, perf_evo)

    # Depth-based sigma_c (for kappa computation matching paper convention)
    sigma_c_depth = find_sigma_c(depths_evo.astype(float), perf_evo)

    # CFI
    cfi_evo = classical_fisher_information(param_evo, probs_evo)
    if len(cfi_evo) > 2:
        interior = cfi_evo[1:-1]
        peak_idx = np.argmax(interior) + 1
        cfi_peak_cnot = param_evo[peak_idx]
        cfi_peak_depth = depths_evo[peak_idx]
    else:
        peak_idx = np.argmax(cfi_evo)
        cfi_peak_cnot = param_evo[peak_idx]
        cfi_peak_depth = depths_evo[peak_idx]

    # Kappa (depth-based, as in paper)
    kappa = sigma_c_depth / cfi_peak_depth if (
        sigma_c_depth is not None and cfi_peak_depth > 0
    ) else None

    # Bootstrap (CNOT-based, matching paper)
    boot = bootstrap_analysis(raw_evo, param_evo, n_outcomes, B=BOOTSTRAP_B)

    # Also bootstrap depth-based
    boot_depth = bootstrap_analysis(raw_evo, depths_evo.astype(float), n_outcomes,
                                    B=BOOTSTRAP_B, seed=123)

    result = {
        'label': label,
        'n_qubits': nq,
        'category': circuit_data.get('category', ''),
        'parent': circuit_data.get('parent', ''),
        'n_outcomes': n_outcomes,
        'depths_evo': depths_evo.tolist(),
        'cnots_evo': param_evo.tolist(),
        'perf_evo': perf_evo.tolist(),
        'cfi_evo': cfi_evo.tolist(),
        'sigma_c_cnot': float(sigma_c_cnot) if sigma_c_cnot else None,
        'sigma_c_depth': float(sigma_c_depth) if sigma_c_depth else None,
        'cfi_peak_cnot': float(cfi_peak_cnot),
        'cfi_peak_depth': float(cfi_peak_depth),
        'kappa': float(kappa) if kappa else None,
        'bootstrap_cnot': boot,
        'bootstrap_depth': boot_depth,
    }

    if verbose:
        sc_str = f"{sigma_c_depth:.2f}" if sigma_c_depth else "N/A"
        ep_str = f"{cfi_peak_depth:.0f}"
        k_str = f"{kappa:.2f}" if kappa else "N/A"
        ov_str = "YES" if boot_depth['ci_overlap'] else "no"
        print(f"    {label:25s}  N={nq}  sc_d={sc_str:>6}  ep*_d={ep_str:>3}"
              f"  kappa={k_str:>6}  CI_overlap={ov_str}")

    return result


# =====================================================================
# BLOCK-LEVEL ANALYSIS
# =====================================================================

def analyze_block(block_id, verbose=True):
    """Analyze all circuits in a block."""
    data = load_block(block_id)
    meta = data['metadata']

    if verbose:
        print(f"\n{'='*72}")
        print(f"  Block {block_id}: {meta['block_name']}")
        print(f"  Device: Cepheus-1-108Q  |  Cost: ${meta.get('cost_usd', 0):.2f}")
        print(f"{'='*72}")

    results = []
    for label, cdata in data['circuits'].items():
        r = analyze_one_circuit(cdata, verbose=verbose)
        if r is not None:
            results.append(r)

    return results


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("  R1 SUPPLEMENT QPU DATA ANALYSIS")
    print("  Cepheus-1-108Q, April 2026")
    print("=" * 72)

    all_results = {}

    for block_id in ['A', 'B', 'C', 'D']:
        results = analyze_block(block_id)
        all_results[block_id] = results

    # =================================================================
    # BLOCK A: Symmetry-breaking report
    # =================================================================
    print(f"\n\n{'='*72}")
    print("  BLOCK A REPORT: Symmetry-Breaking Control (M6)")
    print(f"{'='*72}")
    print(f"\n  Hypothesis: Breaking continuous symmetry should reduce kappa")
    print(f"  toward the generic range (~1.1-1.3).\n")

    print(f"  {'Circuit':<20} {'Parent':<18} {'sigma_c(d)':>10} {'eps*(d)':>8}"
          f" {'kappa':>8} {'kappa CI':>20} {'Overlap':>8}")
    print(f"  {'-'*92}")

    # Reference values from original paper (Table II)
    ref_kappa = {'C02_heisenberg': 2.32, 'C13_tight_binding': 3.60}

    for r in all_results['A']:
        sc = r['sigma_c_depth']
        ep = r['cfi_peak_depth']
        k = r['kappa']
        boot = r['bootstrap_depth']
        kci = boot['kappa_ci']
        ov = "YES" if boot['ci_overlap'] else "no"

        sc_s = f"{sc:.2f}" if sc else "N/A"
        k_s = f"{k:.2f}" if k else "N/A"
        kci_s = f"[{kci[0]:.2f}, {kci[1]:.2f}]" if not np.isnan(kci[0]) else "N/A"

        parent = r['parent']
        ref = ref_kappa.get(parent, None)
        ref_s = f" (ref: {ref:.2f})" if ref else ""

        print(f"  {r['label']:<20} {parent+ref_s:<18} {sc_s:>10} {ep:>8.0f}"
              f" {k_s:>8} {kci_s:>20} {ov:>8}")

    # =================================================================
    # BLOCK B: N=8 scaling report
    # =================================================================
    print(f"\n\n{'='*72}")
    print("  BLOCK B REPORT: N=8 Scaling (M5)")
    print(f"{'='*72}")
    print(f"\n  Hypothesis: kappa remains O(1) at N=8.\n")

    print(f"  {'Circuit':<25} {'N':>3} {'K':>5} {'sigma_c(d)':>10} {'eps*(d)':>8}"
          f" {'kappa':>8} {'kappa CI':>20} {'Overlap':>8}")
    print(f"  {'-'*97}")

    for r in all_results['B']:
        sc = r['sigma_c_depth']
        ep = r['cfi_peak_depth']
        k = r['kappa']
        boot = r['bootstrap_depth']
        kci = boot['kappa_ci']
        ov = "YES" if boot['ci_overlap'] else "no"

        sc_s = f"{sc:.2f}" if sc else "N/A"
        k_s = f"{k:.2f}" if k else "N/A"
        kci_s = f"[{kci[0]:.2f}, {kci[1]:.2f}]" if not np.isnan(kci[0]) else "N/A"

        print(f"  {r['label']:<25} {r['n_qubits']:>3} {r['n_outcomes']:>5}"
              f" {sc_s:>10} {ep:>8.0f} {k_s:>8} {kci_s:>20} {ov:>8}")

    # =================================================================
    # BLOCK C: Fine depth report
    # =================================================================
    print(f"\n\n{'='*72}")
    print("  BLOCK C REPORT: Fine Depth Resolution")
    print(f"{'='*72}")
    print(f"\n  26 depths (0..25) vs standard 12 depths. Sharpens estimates.\n")

    print(f"  {'Circuit':<25} {'sigma_c(d)':>10} {'eps*(d)':>8}"
          f" {'kappa':>8} {'kappa CI':>20} {'sigma_c CI':>20} {'Overlap':>8}")
    print(f"  {'-'*103}")

    for r in all_results['C']:
        sc = r['sigma_c_depth']
        ep = r['cfi_peak_depth']
        k = r['kappa']
        boot = r['bootstrap_depth']
        kci = boot['kappa_ci']
        sci = boot['sigma_c_ci']
        ov = "YES" if boot['ci_overlap'] else "no"

        sc_s = f"{sc:.2f}" if sc else "N/A"
        k_s = f"{k:.2f}" if k else "N/A"
        kci_s = f"[{kci[0]:.2f}, {kci[1]:.2f}]" if not np.isnan(kci[0]) else "N/A"
        sci_s = f"[{sci[0]:.2f}, {sci[1]:.2f}]" if not np.isnan(sci[0]) else "N/A"

        print(f"  {r['label']:<25} {sc_s:>10} {ep:>8.0f} {k_s:>8}"
              f" {kci_s:>20} {sci_s:>20} {ov:>8}")

    # =================================================================
    # BLOCK D: Additional circuits report
    # =================================================================
    print(f"\n\n{'='*72}")
    print("  BLOCK D REPORT: Additional N=6 Circuits")
    print(f"{'='*72}")
    print(f"\n  8 new Hamiltonians to expand sample size.\n")

    print(f"  {'Circuit':<25} {'sigma_c(d)':>10} {'eps*(d)':>8}"
          f" {'kappa':>8} {'kappa CI':>20} {'Overlap':>8}")
    print(f"  {'-'*83}")

    for r in all_results['D']:
        sc = r['sigma_c_depth']
        ep = r['cfi_peak_depth']
        k = r['kappa']
        boot = r['bootstrap_depth']
        kci = boot['kappa_ci']
        ov = "YES" if boot['ci_overlap'] else "no"

        sc_s = f"{sc:.2f}" if sc else "N/A"
        k_s = f"{k:.2f}" if k else "N/A"
        kci_s = f"[{kci[0]:.2f}, {kci[1]:.2f}]" if not np.isnan(kci[0]) else "N/A"

        print(f"  {r['label']:<25} {sc_s:>10} {ep:>8.0f} {k_s:>8}"
              f" {kci_s:>20} {ov:>8}")

    # =================================================================
    # GRAND SUMMARY
    # =================================================================
    print(f"\n\n{'='*72}")
    print("  GRAND SUMMARY")
    print(f"{'='*72}")

    # Collect all results with valid kappa
    all_valid = []
    for block_id in ['A', 'B', 'C', 'D']:
        for r in all_results[block_id]:
            if r['kappa'] is not None:
                all_valid.append(r)

    n_total = len(all_valid)
    n_overlap = sum(1 for r in all_valid if r['bootstrap_depth']['ci_overlap'])

    print(f"\n  Total circuits analyzed: {n_total}")
    print(f"  CI overlap (sigma_c, eps*): {n_overlap}/{n_total}"
          f" ({100*n_overlap/n_total:.0f}%)")

    # Kappa statistics
    kappas = [r['kappa'] for r in all_valid]
    print(f"\n  Kappa statistics (all {n_total} circuits):")
    print(f"    Mean:   {np.mean(kappas):.2f}")
    print(f"    Median: {np.median(kappas):.2f}")
    print(f"    Std:    {np.std(kappas):.2f}")
    print(f"    Range:  [{min(kappas):.2f}, {max(kappas):.2f}]")

    # By category
    categories = {}
    for r in all_valid:
        cat = r.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['kappa'])

    print(f"\n  Kappa by category:")
    for cat, kvals in sorted(categories.items()):
        print(f"    {cat:<25} n={len(kvals):2d}  "
              f"mean={np.mean(kvals):.2f}  range=[{min(kvals):.2f}, {max(kvals):.2f}]")

    # N=6 vs N=8
    n6 = [r for r in all_valid if r['n_qubits'] == 6]
    n8 = [r for r in all_valid if r['n_qubits'] == 8]
    if n8:
        print(f"\n  N-scaling comparison:")
        print(f"    N=6: n={len(n6):2d}, mean kappa={np.mean([r['kappa'] for r in n6]):.2f}")
        print(f"    N=8: n={len(n8):2d}, mean kappa={np.mean([r['kappa'] for r in n8]):.2f}")

    # Symmetry-breaking trend
    print(f"\n  Symmetry-breaking trend (Block A):")
    heis_kappas = [(r['label'], r['kappa']) for r in all_results['A']
                   if 'heis' in r['label'] and r['kappa'] is not None]
    tb_kappas = [(r['label'], r['kappa']) for r in all_results['A']
                 if 'tb' in r['label'] and r['kappa'] is not None]
    if heis_kappas:
        print(f"    Heisenberg series (ref kappa=2.32):")
        for lbl, k in heis_kappas:
            print(f"      {lbl}: kappa={k:.2f}")
    if tb_kappas:
        print(f"    Tight-binding series (ref kappa=3.60):")
        for lbl, k in tb_kappas:
            print(f"      {lbl}: kappa={k:.2f}")

    # Log-scale correlation (for paper: r=0.84 on original 10 circuits)
    if n_total >= 5:
        sc_vals = np.array([r['sigma_c_depth'] for r in all_valid])
        ep_vals = np.array([r['cfi_peak_depth'] for r in all_valid])
        # Filter positive values for log
        pos_mask = (sc_vals > 0) & (ep_vals > 0)
        if pos_mask.sum() >= 5:
            log_sc = np.log(sc_vals[pos_mask])
            log_ep = np.log(ep_vals[pos_mask])
            r_pearson = np.corrcoef(log_sc, log_ep)[0, 1]
            print(f"\n  Log-scale Pearson r (new data): {r_pearson:.3f} (n={pos_mask.sum()})")

    # =================================================================
    # SAVE
    # =================================================================
    output = {
        'metadata': {
            'device': 'Rigetti Cepheus-1-108Q',
            'date': '2026-04-26',
            'bootstrap_B': BOOTSTRAP_B,
        },
        'blocks': {},
    }
    for block_id in ['A', 'B', 'C', 'D']:
        output['blocks'][block_id] = all_results[block_id]

    outfile = DATA_DIR / "r1_supplement_analysis.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved: {outfile}")

    print(f"\n{'='*72}")


if __name__ == '__main__':
    main()
