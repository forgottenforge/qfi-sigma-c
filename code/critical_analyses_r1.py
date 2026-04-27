"""
Critical Analyses for R1 Revision
==================================
Addresses the central questions from the Opus review:

1. CNOT-based log-Pearson r on combined dataset (Original + New)
2. Block C subsampling test (26 depths -> 12 depths)
3. C02 Cepheus-1 baseline (heisenberg_fine = C02 at 26 depths)
4. Clean kappa statistics with pathological circuits flagged
5. Combined scatter data for publication figure

Author: M. C. Wurm
Date: April 2026
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

TAU = 1.0 / np.e
BOOTSTRAP_B = 2000

# =====================================================================
# CORE FUNCTIONS
# =====================================================================

def bitstrings_to_probs(bitstrings, n_outcomes):
    counts = np.zeros(n_outcomes)
    for bs in bitstrings:
        counts[int(bs, 2)] += 1
    return counts / len(bitstrings)


def compute_performance(probs):
    return np.max(probs, axis=1)


def classical_fisher_information(param_values, probs, regularize=1e-10):
    N, K = probs.shape
    probs_reg = probs + regularize
    probs_reg = probs_reg / probs_reg.sum(axis=1, keepdims=True)
    dp = np.zeros_like(probs_reg)
    for i in range(N):
        if i == 0:
            h = param_values[1] - param_values[0]
            if h > 0: dp[i] = (probs_reg[1] - probs_reg[0]) / h
        elif i == N - 1:
            h = param_values[-1] - param_values[-2]
            if h > 0: dp[i] = (probs_reg[-1] - probs_reg[-2]) / h
        else:
            h = param_values[i+1] - param_values[i-1]
            if h > 0: dp[i] = (probs_reg[i+1] - probs_reg[i-1]) / h
    return np.sum(dp**2 / probs_reg, axis=1)


def find_sigma_c(param_values, performance, tau=TAU):
    r0 = performance[0]
    threshold = tau * r0
    for i in range(1, len(performance)):
        if performance[i] <= threshold:
            if performance[i-1] == performance[i]:
                return param_values[i]
            frac = (threshold - performance[i-1]) / (performance[i] - performance[i-1])
            return param_values[i-1] + frac * (param_values[i] - param_values[i-1])
    return None


def bootstrap_analysis(raw_bs_list, param_values, n_outcomes, B=BOOTSTRAP_B,
                       tau=TAU, seed=42):
    rng = np.random.RandomState(seed)
    n_depths = len(raw_bs_list)
    int_arrays = [np.array([int(bs, 2) for bs in bsl]) for bsl in raw_bs_list]

    sc_samples, ep_samples = [], []
    for b in range(B):
        probs_boot = np.zeros((n_depths, n_outcomes))
        for d in range(n_depths):
            raw = int_arrays[d]
            n = len(raw)
            idx = rng.randint(0, n, size=n)
            counts = np.bincount(raw[idx], minlength=n_outcomes)
            probs_boot[d] = counts / n

        perf = compute_performance(probs_boot)
        sc = find_sigma_c(param_values, perf, tau)

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
        if len(arr) < 10: return [np.nan, np.nan]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    sc_ci = ci(sc_valid)
    ep_ci = ci(ep_valid)
    overlap = False
    if not (np.isnan(sc_ci[0]) or np.isnan(ep_ci[0])):
        overlap = not (sc_ci[1] < ep_ci[0] or ep_ci[1] < sc_ci[0])

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
# DATA LOADING
# =====================================================================

DATA_DIR = Path(r"d:\code\qfi_sigma_c")
ORIG_DIR = Path(r"D:\code\onto\particle_plots\braket_archive\2026-02-24_blind_kappa")

DEPTHS_STANDARD = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25]


def load_original_circuit(circuit_idx):
    """Load original Ankaa-3 data for one circuit."""
    with open(ORIG_DIR / "CATALOG.json") as f:
        catalog = json.load(f)

    tasks = sorted(
        [t for t in catalog["tasks"] if t["circuit"] == circuit_idx],
        key=lambda t: t["depth_idx"]
    )

    depths, probs_list, raw_list, cnots = [], [], [], []
    for task in tasks:
        fpath = ORIG_DIR / task["file"]
        with open(fpath) as f:
            data = json.load(f)
        meas = np.array(data["measurements"])
        n_shots = meas.shape[0]
        n_outcomes = 64
        counts = np.zeros(n_outcomes)
        for shot in meas:
            idx = 0
            for b in shot:
                idx = (idx << 1) | int(b)
            counts[idx] += 1
        p = counts / n_shots
        # Also store bitstrings for bootstrap
        bs_list = [''.join(str(int(b)) for b in shot) for shot in meas]

        depths.append(task["depth"])
        probs_list.append(p)
        raw_list.append(bs_list)
        cnots.append(task["n_cnot"])

    return (np.array(depths), np.array(probs_list), raw_list,
            np.array(cnots, dtype=float))


def load_new_circuit(block_id, label):
    """Load new Cepheus-1 data for one circuit."""
    fpath = DATA_DIR / f"r1_supplement_block_{block_id.lower()}.json"
    with open(fpath) as f:
        data = json.load(f)

    cdata = data['circuits'][label]
    nq = cdata['n_qubits']
    n_outcomes = 2 ** nq

    entries = []
    for d_str, d_data in cdata['depths'].items():
        bs = d_data.get('bitstrings', [])
        if len(bs) == 0:
            continue
        entries.append((int(d_str), bs, d_data.get('cnot_count', 0)))

    entries.sort(key=lambda x: x[0])
    depths = np.array([e[0] for e in entries])
    raw = [e[1] for e in entries]
    cnots = np.array([e[2] for e in entries], dtype=float)

    probs = np.zeros((len(depths), n_outcomes))
    for i, bs_list in enumerate(raw):
        probs[i] = bitstrings_to_probs(bs_list, n_outcomes)

    return depths, probs, raw, cnots, n_outcomes


def analyze_evo(depths, probs, raw_bs, cnots, n_outcomes=64):
    """Evolution-only (depth >= 1) analysis with CNOT parameter."""
    mask = depths >= 1
    if mask.sum() < 3:
        mask = np.ones(len(depths), dtype=bool)

    d_evo = depths[mask]
    p_evo = probs[mask]
    c_evo = cnots[mask].copy()
    r_evo = [raw_bs[i] for i in range(len(depths)) if mask[i]]

    if c_evo[0] == 0:
        c_evo[0] = 0.5

    perf = compute_performance(p_evo)
    sc_cnot = find_sigma_c(c_evo, perf)
    sc_depth = find_sigma_c(d_evo.astype(float), perf)

    cfi = classical_fisher_information(c_evo, p_evo)
    if len(cfi) > 2:
        interior = cfi[1:-1]
        peak_idx = np.argmax(interior) + 1
        ep_cnot = c_evo[peak_idx]
        ep_depth = d_evo[peak_idx]
    else:
        peak_idx = np.argmax(cfi)
        ep_cnot = c_evo[peak_idx]
        ep_depth = d_evo[peak_idx]

    kappa_cnot = sc_cnot / ep_cnot if (sc_cnot and ep_cnot > 0) else None
    kappa_depth = sc_depth / ep_depth if (sc_depth and ep_depth > 0) else None

    boot_cnot = bootstrap_analysis(r_evo, c_evo, n_outcomes, B=BOOTSTRAP_B)
    boot_depth = bootstrap_analysis(r_evo, d_evo.astype(float), n_outcomes,
                                     B=BOOTSTRAP_B, seed=123)

    return {
        'sigma_c_cnot': float(sc_cnot) if sc_cnot else None,
        'sigma_c_depth': float(sc_depth) if sc_depth else None,
        'eps_star_cnot': float(ep_cnot),
        'eps_star_depth': float(ep_depth),
        'kappa_cnot': float(kappa_cnot) if kappa_cnot else None,
        'kappa_depth': float(kappa_depth) if kappa_depth else None,
        'bootstrap_cnot': boot_cnot,
        'bootstrap_depth': boot_depth,
        'depths_evo': d_evo.tolist(),
        'cnots_evo': c_evo.tolist(),
        'perf_evo': perf.tolist(),
        'cfi_cnot': cfi.tolist(),
    }


# =====================================================================
# ANALYSIS 1: CNOT-BASED PEARSON r ON COMBINED DATASET
# =====================================================================

def analysis_1_pearson_r():
    """Compute CNOT-based log-Pearson r on original + new data."""
    print("\n" + "=" * 72)
    print("  ANALYSIS 1: CNOT-BASED LOG-PEARSON r (COMBINED DATASET)")
    print("=" * 72)

    # Original 9 testable circuits from Ankaa-3
    ORIGINAL_LABELS = {
        1: "Rand-2", 2: "Heis", 4: "ATA", 6: "Rand-1",
        9: "Kitaev", 11: "Clust", 12: "BCS", 13: "TB", 15: "XXZ"
    }

    results = []

    print("\n  --- Original dataset (Ankaa-3, 12 depths) ---")
    for cidx, short in ORIGINAL_LABELS.items():
        depths, probs, raw, cnots = load_original_circuit(cidx)
        r = analyze_evo(depths, probs, raw, cnots, 64)
        if r['sigma_c_cnot'] is not None:
            results.append({
                'label': f"C{cidx:02d}_{short}",
                'platform': 'Ankaa-3',
                'n_qubits': 6,
                **r
            })
            print(f"    C{cidx:02d} {short:8s}  sc={r['sigma_c_cnot']:7.1f}  "
                  f"ep*={r['eps_star_cnot']:7.1f}  "
                  f"kappa={r['kappa_cnot']:.2f}  "
                  f"overlap={'YES' if r['bootstrap_cnot']['ci_overlap'] else 'no'}")
        else:
            print(f"    C{cidx:02d} {short:8s}  sc=N/A")

    # New circuits from Cepheus-1
    NEW_CIRCUITS = [
        ('A', 'heis_aniso_11'), ('A', 'heis_aniso_15'), ('A', 'heis_aniso_20'),
        ('A', 'tb_vz01'), ('A', 'tb_vz03'), ('A', 'tb_vz05'),
        ('B', 'heisenberg_n8'), ('B', 'tight_binding_n8'), ('B', 'xxz_aniso_n8'),
        ('B', 'tfim_ordered_n8'), ('B', 'kitaev_detuned_n8'), ('B', 'bcs_8qubit'),
        ('B', 'cluster_spt_n8'), ('B', 'random_layered_n8_2'),
        ('C', 'all_to_all_fine'), ('C', 'heisenberg_fine'),
        ('C', 'tight_binding_fine'), ('C', 'kitaev_detuned_fine'),
        # Exclude bcs_6qubit_fine (pathological eps*=21)
        ('D', 'tfim_h05'), ('D', 'tfim_h20'), ('D', 'xxz_moderate'),
        # Exclude j1j2_frustrated (N/A)
        ('D', 'xy_zfield'), ('D', 'alt_bond_ising'), ('D', 'compass_model'),
        # Exclude random_layered_3 (N/A)
    ]

    print(f"\n  --- New dataset (Cepheus-1, Blocks A-D) ---")
    for block_id, label in NEW_CIRCUITS:
        depths, probs, raw, cnots, n_out = load_new_circuit(block_id, label)
        r = analyze_evo(depths, probs, raw, cnots, n_out)

        nq = int(np.log2(n_out))
        if r['sigma_c_cnot'] is not None:
            results.append({
                'label': f"{label}",
                'platform': 'Cepheus-1',
                'n_qubits': nq,
                **r
            })
            print(f"    {label:25s}  sc={r['sigma_c_cnot']:7.1f}  "
                  f"ep*={r['eps_star_cnot']:7.1f}  "
                  f"kappa={r['kappa_cnot']:.2f}  "
                  f"overlap={'YES' if r['bootstrap_cnot']['ci_overlap'] else 'no'}")
        else:
            print(f"    {label:25s}  sc=N/A (excluded)")

    # Compute Pearson r (log-scale)
    sc_vals = np.array([r['sigma_c_cnot'] for r in results])
    ep_vals = np.array([r['eps_star_cnot'] for r in results])

    pos = (sc_vals > 0) & (ep_vals > 0)
    log_sc = np.log(sc_vals[pos])
    log_ep = np.log(ep_vals[pos])

    r_pearson, p_pearson = stats.pearsonr(log_sc, log_ep)
    rho_spearman, p_spearman = stats.spearmanr(sc_vals[pos], ep_vals[pos])

    print(f"\n  --- Combined Correlation (n={pos.sum()}) ---")
    print(f"    Log-Pearson r  = {r_pearson:.3f} (p = {p_pearson:.4f})")
    print(f"    Spearman rho   = {rho_spearman:.3f} (p = {p_spearman:.4f})")

    # Also compute for subsets
    ankaa = [r for r in results if r['platform'] == 'Ankaa-3']
    cepheus = [r for r in results if r['platform'] == 'Cepheus-1']

    if len(ankaa) >= 5:
        sc_a = np.array([r['sigma_c_cnot'] for r in ankaa])
        ep_a = np.array([r['eps_star_cnot'] for r in ankaa])
        r_a, p_a = stats.pearsonr(np.log(sc_a), np.log(ep_a))
        print(f"\n    Ankaa-3 only (n={len(ankaa)}):  r = {r_a:.3f} (p = {p_a:.4f})")

    if len(cepheus) >= 5:
        sc_c = np.array([r['sigma_c_cnot'] for r in cepheus])
        ep_c = np.array([r['eps_star_cnot'] for r in cepheus])
        pos_c = (sc_c > 0) & (ep_c > 0)
        if pos_c.sum() >= 5:
            r_c, p_c = stats.pearsonr(np.log(sc_c[pos_c]), np.log(ep_c[pos_c]))
            rho_c, p_rc = stats.spearmanr(sc_c[pos_c], ep_c[pos_c])
            print(f"    Cepheus-1 only (n={pos_c.sum()}): r = {r_c:.3f} (p = {p_c:.4f})"
                  f"  rho = {rho_c:.3f}")

    # Exclude pathological (ATA, bcs_fine) and broad-CI (cluster_spt_n8)
    clean = [r for r in results
             if r['label'] not in ['C04_ATA', 'bcs_6qubit_fine']
             and not (r['label'] == 'cluster_spt_n8'
                      and r['bootstrap_cnot']['kappa_ci'][1] > 5)]
    if len(clean) >= 5:
        sc_cl = np.array([r['sigma_c_cnot'] for r in clean])
        ep_cl = np.array([r['eps_star_cnot'] for r in clean])
        pos_cl = (sc_cl > 0) & (ep_cl > 0)
        if pos_cl.sum() >= 5:
            r_cl, p_cl = stats.pearsonr(np.log(sc_cl[pos_cl]), np.log(ep_cl[pos_cl]))
            rho_cl, p_rcl = stats.spearmanr(sc_cl[pos_cl], ep_cl[pos_cl])
            print(f"\n    Clean subset (no ATA, bcs_fine, broad-CI; n={pos_cl.sum()}):")
            print(f"      r = {r_cl:.3f} (p = {p_cl:.6f})  rho = {rho_cl:.3f}")

    return results


# =====================================================================
# ANALYSIS 2: BLOCK C SUBSAMPLING TEST
# =====================================================================

def analysis_2_subsampling():
    """Re-analyze Block C at 12 depths (subsampled from 26)."""
    print("\n\n" + "=" * 72)
    print("  ANALYSIS 2: BLOCK C SUBSAMPLING TEST (26 -> 12 depths)")
    print("=" * 72)
    print("\n  Question: Is the kappa shift (e.g. Heis 2.32->0.98) due to")
    print("  finer resolution or platform change?")
    print("  Method: Re-analyze Block C data using only the standard 12 depths.\n")

    BLOCK_C_CIRCUITS = [
        'all_to_all_fine', 'heisenberg_fine', 'tight_binding_fine',
        'kitaev_detuned_fine', 'bcs_6qubit_fine'
    ]

    STANDARD_DEPTHS = set(DEPTHS_STANDARD)

    print(f"  {'Circuit':<25} {'kappa_26d':>10} {'kappa_12d':>10} {'kappa_Ankaa':>12}"
          f" {'Shift_26->12':>12} {'Verdict':>12}")
    print(f"  {'-'*83}")

    ANKAA_REF = {
        'all_to_all_fine': ('C04', 0.31),
        'heisenberg_fine': ('C02', 2.32),
        'tight_binding_fine': ('C13', 3.60),
        'kitaev_detuned_fine': ('C09', 1.72),
        'bcs_6qubit_fine': ('C12', 1.18),
    }

    for label in BLOCK_C_CIRCUITS:
        depths_full, probs_full, raw_full, cnots_full, n_out = load_new_circuit('C', label)

        # Full 26-depth analysis
        r_full = analyze_evo(depths_full, probs_full, raw_full, cnots_full, n_out)

        # Subsample to standard 12 depths
        mask_12 = np.array([d in STANDARD_DEPTHS for d in depths_full])
        if mask_12.sum() < 3:
            print(f"  {label:25s}  too few standard depths, skipping")
            continue

        d_12 = depths_full[mask_12]
        p_12 = probs_full[mask_12]
        c_12 = cnots_full[mask_12]
        r_12_bs = [raw_full[i] for i in range(len(depths_full)) if mask_12[i]]

        r_sub = analyze_evo(d_12, p_12, r_12_bs, c_12, n_out)

        ref_label, ref_kappa = ANKAA_REF.get(label, ('?', None))
        ref_str = f"{ref_kappa:.2f}" if ref_kappa else "N/A"

        k26 = r_full['kappa_cnot']
        k12 = r_sub['kappa_cnot']

        k26_s = f"{k26:.2f}" if k26 else "N/A"
        k12_s = f"{k12:.2f}" if k12 else "N/A"

        if k26 and k12:
            shift = k12 - k26
            shift_s = f"{shift:+.2f}"
            # Verdict: if k12 is much closer to Ankaa ref, it's resolution
            # If k12 ~ k26, it's platform
            if ref_kappa and abs(k12 - ref_kappa) < abs(k26 - ref_kappa):
                verdict = "RESOLUTION"
            else:
                verdict = "PLATFORM"
        else:
            shift_s = "N/A"
            verdict = "N/A"

        print(f"  {label:25s} {k26_s:>10} {k12_s:>10} {ref_str:>12}"
              f" {shift_s:>12} {verdict:>12}")

    print(f"\n  Interpretation:")
    print(f"    RESOLUTION = 12-depth kappa closer to Ankaa-3 ref -> finer grid explains shift")
    print(f"    PLATFORM   = 12-depth kappa still differs from Ankaa-3 -> hardware change dominates")


# =====================================================================
# ANALYSIS 3: C02 CEPHEUS-1 BASELINE
# =====================================================================

def analysis_3_c02_baseline():
    """Check heisenberg_fine as C02 baseline on Cepheus-1."""
    print("\n\n" + "=" * 72)
    print("  ANALYSIS 3: C02 (HEISENBERG) CEPHEUS-1 BASELINE")
    print("=" * 72)
    print("\n  Question: Is Block A internally consistent?")
    print("  heisenberg_fine (Block C) = C02 at anisotropy=0 on Cepheus-1")
    print("  Block A heis_aniso_* = C02 variants on same hardware\n")

    # Load heisenberg_fine as baseline (26 depths)
    d_fine, p_fine, r_fine, c_fine, n_out = load_new_circuit('C', 'heisenberg_fine')
    r_baseline_full = analyze_evo(d_fine, p_fine, r_fine, c_fine, n_out)

    # Subsample to 12 depths for fair comparison with Block A
    mask_12 = np.array([d in set(DEPTHS_STANDARD) for d in d_fine])
    d_12 = d_fine[mask_12]
    p_12 = p_fine[mask_12]
    c_12 = c_fine[mask_12]
    r_12_bs = [r_fine[i] for i in range(len(d_fine)) if mask_12[i]]
    r_baseline_12 = analyze_evo(d_12, p_12, r_12_bs, c_12, n_out)

    print(f"  C02 Heisenberg (SU(2) symmetric, anisotropy=0):")
    print(f"    Ankaa-3 original:       kappa = 2.32")
    print(f"    Cepheus-1, 26 depths:   kappa = {r_baseline_full['kappa_cnot']:.2f}"
          if r_baseline_full['kappa_cnot'] else "    Cepheus-1, 26 depths:   kappa = N/A")
    print(f"    Cepheus-1, 12 depths:   kappa = {r_baseline_12['kappa_cnot']:.2f}"
          if r_baseline_12['kappa_cnot'] else "    Cepheus-1, 12 depths:   kappa = N/A")

    # Now Block A variants
    print(f"\n  Block A symmetry-breaking series (all Cepheus-1, 12 depths):")
    print(f"  {'Circuit':<20} {'Anisotropy':>12} {'kappa_cnot':>12} {'kappa_depth':>12}")
    print(f"  {'-'*58}")

    # Baseline
    k_cnot = r_baseline_12['kappa_cnot']
    k_depth = r_baseline_12['kappa_depth']
    print(f"  {'heis_iso (C02)':20s} {'0%':>12}"
          f" {k_cnot:.2f}" if k_cnot else f" {'N/A':>12}",
          f" {k_depth:.2f}" if k_depth else f" {'N/A':>12}")

    for label, aniso_pct in [('heis_aniso_11', '10%'),
                              ('heis_aniso_15', '50%'),
                              ('heis_aniso_20', '100%')]:
        d, p, r_bs, c, n = load_new_circuit('A', label)
        r = analyze_evo(d, p, r_bs, c, n)
        k_c = r['kappa_cnot']
        k_d = r['kappa_depth']
        print(f"  {label:20s} {aniso_pct:>12}"
              f" {k_c:.2f}" if k_c else f" {'N/A':>12}",
              f" {k_d:.2f}" if k_d else f" {'N/A':>12}")

    # TB series
    print(f"\n  Tight-binding series:")
    # Load tight_binding_fine as TB baseline
    d_tb, p_tb, r_tb, c_tb, n_tb = load_new_circuit('C', 'tight_binding_fine')
    mask_12_tb = np.array([d in set(DEPTHS_STANDARD) for d in d_tb])
    d_tb12 = d_tb[mask_12_tb]
    p_tb12 = p_tb[mask_12_tb]
    c_tb12 = c_tb[mask_12_tb]
    r_tb12 = [r_tb[i] for i in range(len(d_tb)) if mask_12_tb[i]]
    r_tb_base = analyze_evo(d_tb12, p_tb12, r_tb12, c_tb12, n_tb)

    print(f"  {'Circuit':<20} {'Vz':>12} {'kappa_cnot':>12} {'kappa_depth':>12}")
    print(f"  {'-'*58}")
    k_c = r_tb_base['kappa_cnot']
    k_d = r_tb_base['kappa_depth']
    print(f"  {'tb_pure (C13)':20s} {'0':>12}"
          f" {k_c:.2f}" if k_c else f" {'N/A':>12}",
          f" {k_d:.2f}" if k_d else f" {'N/A':>12}")

    for label, vz in [('tb_vz01', '0.1'), ('tb_vz03', '0.3'), ('tb_vz05', '0.5')]:
        d, p, r_bs, c, n = load_new_circuit('A', label)
        r = analyze_evo(d, p, r_bs, c, n)
        k_c = r['kappa_cnot']
        k_d = r['kappa_depth']
        print(f"  {label:20s} {vz:>12}"
              f" {k_c:.2f}" if k_c else f" {'N/A':>12}",
              f" {k_d:.2f}" if k_d else f" {'N/A':>12}")


# =====================================================================
# ANALYSIS 4: COMPLETE TABLE FOR PAPER (TABLE III)
# =====================================================================

def analysis_4_table_iii():
    """Generate Table III data for paper."""
    print("\n\n" + "=" * 72)
    print("  ANALYSIS 4: TABLE III DATA (CEPHEUS-1 EXTENDED DATASET)")
    print("=" * 72)

    ALL_NEW = [
        ('A', 'heis_aniso_11', 'SB', 'Heis 10%'),
        ('A', 'heis_aniso_15', 'SB', 'Heis 50%'),
        ('A', 'heis_aniso_20', 'SB', 'Heis 100%'),
        ('A', 'tb_vz01', 'SB', 'TB Vz=0.1'),
        ('A', 'tb_vz03', 'SB', 'TB Vz=0.3'),
        ('A', 'tb_vz05', 'SB', 'TB Vz=0.5'),
        ('B', 'heisenberg_n8', 'S', 'Heis N=8'),
        ('B', 'tight_binding_n8', 'S', 'TB N=8'),
        ('B', 'xxz_aniso_n8', 'GP', 'XXZ N=8'),
        ('B', 'tfim_ordered_n8', 'GP', 'TFIM-o N=8'),
        ('B', 'kitaev_detuned_n8', 'T', 'Kitaev N=8'),
        ('B', 'bcs_8qubit', 'GP', 'BCS N=8'),
        ('B', 'cluster_spt_n8', 'T', 'Clust N=8'),
        ('B', 'random_layered_n8', 'NR', 'Rand N=8'),
        ('B', 'random_layered_n8_2', 'NR', 'Rand2 N=8'),
        ('C', 'all_to_all_fine', 'EC', 'ATA fine'),
        ('C', 'heisenberg_fine', 'S', 'Heis fine'),
        ('C', 'tight_binding_fine', 'S', 'TB fine'),
        ('C', 'kitaev_detuned_fine', 'T', 'Kitaev fine'),
        ('C', 'bcs_6qubit_fine', 'GP', 'BCS fine'),
        ('D', 'tfim_h05', 'S', 'TFIM h=0.5'),
        ('D', 'tfim_h20', 'S', 'TFIM h=2.0'),
        ('D', 'xxz_moderate', 'S', 'XXZ mod'),
        ('D', 'j1j2_frustrated', 'S', 'J1J2'),
        ('D', 'xy_zfield', 'S', 'XY+Z'),
        ('D', 'alt_bond_ising', 'S', 'ABI'),
        ('D', 'compass_model', 'S', 'Comp'),
        ('D', 'random_layered_3', 'NR', 'Rand3'),
    ]

    print(f"\n  {'Short':<14} {'Blk':>3} {'Cat':>3} {'N':>3} {'#d':>3}"
          f" {'sc_cnot':>9} {'ep*_cnot':>9} {'kappa':>7} {'kappa_CI':>20}"
          f" {'Overlap':>8} {'Status':>12}")
    print(f"  {'-'*100}")

    table_data = []
    for block_id, label, cat, short in ALL_NEW:
        depths, probs, raw, cnots, n_out = load_new_circuit(block_id, label)
        nq = int(np.log2(n_out))
        r = analyze_evo(depths, probs, raw, cnots, n_out)

        sc = r['sigma_c_cnot']
        ep = r['eps_star_cnot']
        k = r['kappa_cnot']
        boot = r['bootstrap_cnot']
        kci = boot['kappa_ci']
        ov = boot['ci_overlap']

        # Status classification
        if sc is None:
            status = "non-testable"
        elif label == 'bcs_6qubit_fine' and r['eps_star_depth'] > 20:
            status = "pathological"
        elif not np.isnan(kci[1]) and kci[1] > 5:
            status = "broad-CI"
        else:
            status = "testable"

        sc_s = f"{sc:.1f}" if sc else "---"
        ep_s = f"{ep:.1f}" if ep else "---"
        k_s = f"{k:.2f}" if k else "---"
        kci_s = (f"[{kci[0]:.2f},{kci[1]:.2f}]"
                 if not np.isnan(kci[0]) else "---")
        ov_s = "yes" if ov else "no"

        nd = len([d for d in depths if d >= 1])

        print(f"  {short:<14} {block_id:>3} {cat:>3} {nq:>3} {nd:>3}"
              f" {sc_s:>9} {ep_s:>9} {k_s:>7} {kci_s:>20}"
              f" {ov_s:>8} {status:>12}")

        table_data.append({
            'label': label, 'short': short, 'block': block_id,
            'category': cat, 'n_qubits': nq, 'n_depths': nd,
            'status': status, **r
        })

    # Summary statistics (testable only)
    testable = [t for t in table_data if t['status'] == 'testable']
    kappas = [t['kappa_cnot'] for t in testable if t['kappa_cnot']]
    print(f"\n  Testable circuits: {len(testable)}")
    print(f"  Overlap rate: {sum(1 for t in testable if t['bootstrap_cnot']['ci_overlap'])}"
          f"/{len(testable)}")
    if kappas:
        print(f"  kappa (CNOT): mean={np.mean(kappas):.2f}, "
              f"median={np.median(kappas):.2f}, "
              f"range=[{min(kappas):.2f}, {max(kappas):.2f}]")

    return table_data


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 72)
    print("  CRITICAL ANALYSES FOR R1 REVISION")
    print("=" * 72)

    # Analysis 1: Pearson r
    combined_results = analysis_1_pearson_r()

    # Analysis 2: Subsampling
    analysis_2_subsampling()

    # Analysis 3: C02 baseline
    analysis_3_c02_baseline()

    # Analysis 4: Table III
    table_data = analysis_4_table_iii()

    # Save all
    output = {
        'combined_results': combined_results,
        'table_data': table_data,
    }
    outfile = DATA_DIR / "critical_analyses_results.json"
    with open(outfile, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  All results saved: {outfile}")


if __name__ == '__main__':
    main()
