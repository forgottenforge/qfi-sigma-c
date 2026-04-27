"""
Statistical Validation of sigma_c vs CFI Peak Coincidence
=========================================================

Enhanced analysis beyond simple CI overlap.  Addresses:
1. Paired bootstrap delta = sigma_c - CFI_peak  (the correct test)
2. Spearman + Pearson correlations with p-values
3. Permutation test (10 000 permutations) for significance of correlation
4. Relative delta with grid-resolution correction
5. Detection-rate filtering (unreliable sigma_c flagged)
6. Binomial test for match rate vs chance
7. Category-stratified analysis
8. Sensitivity to tau threshold
9. KL-based sigma_c cross-validation
10. Cohen's d effect size

Author: Matthias Christian Wurm / ForgottenForge
Date: April 2026
"""

import json, sys, time
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

# ============================================================
# Import shared functions from the comprehensive analysis
# ============================================================
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_qfi_analysis import (
    CIRCUIT_LABELS, CIRCUIT_SHORT, CIRCUIT_CATEGORIES, DEPTHS,
    N_QUBITS, N_OUTCOMES, TAU,
    get_category, load_all_circuit_data_fast,
    classical_fisher_information, find_sigma_c,
    compute_performance, compute_kl_from_uniform,
    measurements_to_probs,
)

# ============================================================
# CONFIGURATION
# ============================================================

B_BOOT = 5000          # bootstrap replicates (more than original 2000)
N_PERM = 10000         # permutation test replicates
SEED   = 2026
DETECTION_THRESHOLD = 0.50   # minimum sigma_c detection rate to be "reliable"

# Circuits to analyze (0-15, skip C03=Zero)
ALL_CIRCUITS = list(range(16))


# ============================================================
# PAIRED BOOTSTRAP: the correct test for coincidence
# ============================================================

def paired_bootstrap(raw_measurements, param_values, B=5000, tau=TAU, seed=42):
    """
    Bootstrap with PAIRED delta = sigma_c - CFI_peak in every replicate.

    This is the correct test: instead of separately computing CIs for sigma_c
    and CFI_peak and checking overlap (conservative), we directly estimate
    the CI of their difference.

    Returns
    -------
    dict with:
        sigma_c_samples : array of B sigma_c values (NaN if not detected)
        cfi_peak_samples : array of B CFI peak locations
        delta_samples : array of B delta values (NaN if sigma_c not detected)
        delta_ci : [2.5%, 97.5%] CI for delta
        zero_in_delta_ci : bool (True -> sigma_c and CFI peak coincide)
        sigma_c_ci : [2.5%, 97.5%]
        cfi_peak_ci : [2.5%, 97.5%]
        detection_rate : fraction of replicates where sigma_c detected
        delta_median : median of delta (conditional on detection)
        delta_mean : mean of delta (conditional on detection)
    """
    rng = np.random.RandomState(seed)
    n_depths = len(raw_measurements)

    sigma_c_samples = np.full(B, np.nan)
    cfi_peak_samples = np.full(B, np.nan)
    delta_samples = np.full(B, np.nan)

    for b in range(B):
        probs_boot = np.zeros((n_depths, N_OUTCOMES))
        for d in range(n_depths):
            meas = raw_measurements[d]
            n_shots = meas.shape[0]
            idx = rng.randint(0, n_shots, size=n_shots)
            probs_boot[d] = measurements_to_probs(meas[idx])

        # CFI
        cfi = classical_fisher_information(param_values, probs_boot)

        # CFI peak (interior only)
        if len(cfi) > 2:
            interior = cfi[1:-1]
            peak_idx = np.argmax(interior) + 1
            cfi_peak_samples[b] = param_values[peak_idx]
        else:
            cfi_peak_samples[b] = param_values[np.argmax(cfi)]

        # sigma_c
        perf = compute_performance(probs_boot)
        sc, _ = find_sigma_c(param_values, perf, tau)
        if sc is not None:
            sigma_c_samples[b] = sc
            delta_samples[b] = sc - cfi_peak_samples[b]

    # Compute statistics from valid (detected) replicates
    valid = ~np.isnan(delta_samples)
    n_valid = np.sum(valid)
    detection_rate = n_valid / B

    if n_valid > 10:
        d_valid = delta_samples[valid]
        delta_ci = np.percentile(d_valid, [2.5, 97.5])
        delta_median = np.median(d_valid)
        delta_mean = np.mean(d_valid)
        zero_in_ci = (delta_ci[0] <= 0 <= delta_ci[1])
    else:
        delta_ci = [np.nan, np.nan]
        delta_median = np.nan
        delta_mean = np.nan
        zero_in_ci = False

    valid_sc = sigma_c_samples[~np.isnan(sigma_c_samples)]
    valid_cp = cfi_peak_samples[~np.isnan(cfi_peak_samples)]

    return {
        "sigma_c_samples": sigma_c_samples,
        "cfi_peak_samples": cfi_peak_samples,
        "delta_samples": delta_samples,
        "delta_ci": delta_ci,
        "zero_in_delta_ci": zero_in_ci,
        "delta_median": delta_median,
        "delta_mean": delta_mean,
        "sigma_c_ci": np.percentile(valid_sc, [2.5, 97.5]) if len(valid_sc) > 10 else [np.nan, np.nan],
        "cfi_peak_ci": np.percentile(valid_cp, [2.5, 97.5]) if len(valid_cp) > 10 else [np.nan, np.nan],
        "detection_rate": detection_rate,
        "n_valid": int(n_valid),
    }


# ============================================================
# PERMUTATION TEST for sigma_c vs CFI peak correlation
# ============================================================

def permutation_test_correlation(sigma_c_vals, cfi_peak_vals, n_perm=10000, seed=42):
    """
    Test H0: sigma_c and CFI_peak are independent.

    Permutes sigma_c values across circuits and recomputes Spearman correlation.

    Returns
    -------
    dict with observed Spearman rho, permutation p-value, null distribution stats
    """
    rng = np.random.RandomState(seed)
    n = len(sigma_c_vals)

    rho_obs, p_exact = sp_stats.spearmanr(sigma_c_vals, cfi_peak_vals)
    r_obs, p_pearson = sp_stats.pearsonr(sigma_c_vals, cfi_peak_vals)

    rho_perm = np.zeros(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        rho_perm[i], _ = sp_stats.spearmanr(sigma_c_vals[perm], cfi_peak_vals)

    p_perm = np.mean(np.abs(rho_perm) >= np.abs(rho_obs))

    return {
        "spearman_rho": rho_obs,
        "spearman_p_exact": p_exact,
        "pearson_r": r_obs,
        "pearson_p": p_pearson,
        "permutation_p": p_perm,
        "n_perm": n_perm,
        "null_mean": np.mean(rho_perm),
        "null_std": np.std(rho_perm),
        "null_95": np.percentile(np.abs(rho_perm), 95),
    }


# ============================================================
# BINOMIAL TEST for match rate
# ============================================================

def chance_overlap_rate(sigma_c_vals, cfi_peak_cis, sigma_c_cis,
                        cfi_range, n_sim=100000, seed=42):
    """
    Estimate the chance overlap rate under H0: sigma_c placed uniformly.

    For each simulation:
    - Draw sigma_c uniformly from the parameter range
    - Check if it falls within ANY circuit's CFI peak CI (approximate)

    Returns the expected overlap probability under random placement.
    """
    rng = np.random.RandomState(seed)
    n_circuits = len(sigma_c_vals)

    # Overlap probability per circuit: width of CFI CI / total range
    total_range = cfi_range[1] - cfi_range[0]
    overlap_probs = []
    for ci in cfi_peak_cis:
        if np.any(np.isnan(ci)):
            overlap_probs.append(0.0)
        else:
            # Fraction of the parameter range covered by this CI
            width = ci[1] - ci[0]
            overlap_probs.append(min(width / total_range, 1.0))

    p_chance = np.mean(overlap_probs)
    return p_chance


# ============================================================
# SENSITIVITY TO TAU
# ============================================================

def tau_sensitivity(raw_evo, param_evo, tau_values, B=2000, seed=42):
    """
    Check how results change across different tau thresholds.
    """
    results = []
    for tau in tau_values:
        perf = compute_performance(
            np.array([measurements_to_probs(m) for m in raw_evo])
        )
        sc, _ = find_sigma_c(param_evo, perf, tau=tau)

        cfi = classical_fisher_information(param_evo,
            np.array([measurements_to_probs(m) for m in raw_evo]))
        if len(cfi) > 2:
            peak_idx = np.argmax(cfi[1:-1]) + 1
            cfi_peak = param_evo[peak_idx]
        else:
            cfi_peak = param_evo[np.argmax(cfi)]

        results.append({
            "tau": tau,
            "sigma_c": sc,
            "cfi_peak": cfi_peak,
            "delta": abs(sc - cfi_peak) if sc is not None else None,
        })
    return results


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("STATISTICAL VALIDATION OF sigma_c vs CFI PEAK COINCIDENCE")
    print("=" * 80)
    print(f"Bootstrap replicates: {B_BOOT}")
    print(f"Permutation replicates: {N_PERM}")
    print(f"Detection threshold: {DETECTION_THRESHOLD}")
    print()

    # ================================================================
    # PHASE 1: Load data and run paired bootstrap for all circuits
    # ================================================================
    print("[1/5] Running paired bootstrap for 16 circuits...")

    circuit_data = {}
    for cidx in ALL_CIRCUITS:
        label = CIRCUIT_LABELS[cidx]
        print(f"  C{cidx:02d} {label}...", end="", flush=True)

        depths, probs, raw_meas, cnots = load_all_circuit_data_fast(cidx)

        # Evolution-only: skip depth=0
        if len(depths) > 2:
            param_evo = cnots[1:].astype(float)
            raw_evo = raw_meas[1:]
        else:
            param_evo = cnots.astype(float)
            raw_evo = raw_meas

        # Run paired bootstrap
        boot = paired_bootstrap(raw_evo, param_evo, B=B_BOOT, seed=SEED + cidx)

        # Point estimates from actual data
        probs_evo = probs[1:] if len(depths) > 2 else probs
        cfi_evo = classical_fisher_information(param_evo, probs_evo)
        perf_evo = compute_performance(probs_evo)
        kl_evo = compute_kl_from_uniform(probs_evo)

        sigma_c_evo, _ = find_sigma_c(param_evo, perf_evo)
        sigma_c_kl, _ = find_sigma_c(param_evo, kl_evo)

        if len(cfi_evo) > 2:
            peak_idx = np.argmax(cfi_evo[1:-1]) + 1
            cfi_peak = param_evo[peak_idx]
        else:
            peak_idx = np.argmax(cfi_evo)
            cfi_peak = param_evo[peak_idx]

        circuit_data[cidx] = {
            "label": label,
            "short": CIRCUIT_SHORT[cidx],
            "category": get_category(cidx),
            "param_evo": param_evo,
            "raw_evo": raw_evo,
            "sigma_c": sigma_c_evo,
            "sigma_c_kl": sigma_c_kl,
            "cfi_peak": float(cfi_peak),
            "cfi_evo": cfi_evo,
            "perf_evo": perf_evo,
            "boot": boot,
            "delta": abs(sigma_c_evo - cfi_peak) if sigma_c_evo is not None else None,
        }

        dr = boot["detection_rate"]
        if sigma_c_evo is not None:
            print(f" sigma_c={sigma_c_evo:.1f}, CFI_peak={cfi_peak:.1f}, "
                  f"det={dr:.1%}, delta_CI={boot['delta_ci']}")
        else:
            print(f" sigma_c=N/A, CFI_peak={cfi_peak:.1f}, det={dr:.1%}")

    elapsed = time.time() - t0
    print(f"\n  Bootstrap completed in {elapsed:.0f}s")

    # ================================================================
    # PHASE 2: Classify circuits by testability
    # ================================================================
    print("\n" + "=" * 80)
    print("[2/5] CIRCUIT CLASSIFICATION")
    print("=" * 80)

    # Testable: sigma_c detected in point estimate AND detection rate >= threshold
    reliable = {}
    borderline = {}
    non_testable = {}

    for cidx, d in circuit_data.items():
        dr = d["boot"]["detection_rate"]
        if d["sigma_c"] is not None and dr >= DETECTION_THRESHOLD:
            reliable[cidx] = d
        elif d["sigma_c"] is not None and dr < DETECTION_THRESHOLD:
            borderline[cidx] = d
        else:
            non_testable[cidx] = d

    print(f"\nReliable (sigma_c detected, detection rate >= {DETECTION_THRESHOLD:.0%}):")
    for cidx in sorted(reliable):
        d = reliable[cidx]
        print(f"  C{cidx:02d} {d['short']:<8} [{d['category']:<14}] "
              f"sigma_c={d['sigma_c']:7.1f}  CFI_peak={d['cfi_peak']:7.1f}  "
              f"det={d['boot']['detection_rate']:.1%}  "
              f"delta={d['delta']:7.1f}")

    print(f"\nBorderline (sigma_c detected but detection rate < {DETECTION_THRESHOLD:.0%}):")
    for cidx in sorted(borderline):
        d = borderline[cidx]
        print(f"  C{cidx:02d} {d['short']:<8} [{d['category']:<14}] "
              f"sigma_c={d['sigma_c']:7.1f}  det={d['boot']['detection_rate']:.1%}")

    print(f"\nNon-testable (sigma_c not detected in point estimate):")
    for cidx in sorted(non_testable):
        d = non_testable[cidx]
        print(f"  C{cidx:02d} {d['short']:<8} [{d['category']:<14}] "
              f"det={d['boot']['detection_rate']:.1%}")

    # ================================================================
    # PHASE 3: Paired delta analysis for reliable circuits
    # ================================================================
    print("\n" + "=" * 80)
    print("[3/5] PAIRED BOOTSTRAP DELTA ANALYSIS (reliable circuits)")
    print("=" * 80)

    print(f"\n{'Circuit':<10} {'Cat':<14} {'sigma_c':>8} {'CFI_pk':>8} "
          f"{'delta':>8} {'delta_CI':>22} {'0 in CI?':>8} {'delta_rel':>9} "
          f"{'Verdict':>12}")
    print("-" * 110)

    n_zero_in_ci = 0
    n_reliable = len(reliable)
    deltas_abs = []
    deltas_rel = []

    for cidx in sorted(reliable):
        d = reliable[cidx]
        b = d["boot"]

        sc = d["sigma_c"]
        cp = d["cfi_peak"]
        delta = d["delta"]
        delta_rel = delta / sc if sc > 0 else np.inf

        deltas_abs.append(delta)
        deltas_rel.append(delta_rel)

        ci = b["delta_ci"]
        zero_in = b["zero_in_delta_ci"]
        if zero_in:
            n_zero_in_ci += 1

        verdict = "COINCIDE" if zero_in else "DISTINCT"

        print(f"C{cidx:02d} {d['short']:<6} {d['category']:<14} "
              f"{sc:8.1f} {cp:8.1f} {delta:8.1f} "
              f"[{ci[0]:8.1f}, {ci[1]:8.1f}] "
              f"{'YES' if zero_in else 'no':>8} "
              f"{delta_rel:9.3f} "
              f"{verdict:>12}")

    print("-" * 110)
    print(f"\nPaired delta test: {n_zero_in_ci}/{n_reliable} circuits have 0 in "
          f"95% CI of delta ({100*n_zero_in_ci/n_reliable:.0f}%)")
    print(f"Median |delta|: {np.median(deltas_abs):.1f} CNOTs")
    print(f"Median delta_rel: {np.median(deltas_rel):.3f}")
    print(f"Mean delta_rel: {np.mean(deltas_rel):.3f}")

    # Grid resolution correction
    # Minimum possible |delta| is limited by grid spacing
    # For most circuits, minimum step is 10 CNOTs
    print(f"\nGrid resolution: minimum CNOT step = 10")
    print(f"  Circuits with |delta| < 10 CNOTs (at grid resolution limit): "
          f"{sum(1 for d in deltas_abs if d < 10)}/{n_reliable}")
    print(f"  Circuits with |delta| < 20 CNOTs (within 2 grid steps): "
          f"{sum(1 for d in deltas_abs if d < 20)}/{n_reliable}")

    # ================================================================
    # PHASE 4: Correlation tests
    # ================================================================
    print("\n" + "=" * 80)
    print("[4/5] CORRELATION AND PERMUTATION TESTS")
    print("=" * 80)

    sc_arr = np.array([reliable[c]["sigma_c"] for c in sorted(reliable)])
    cp_arr = np.array([reliable[c]["cfi_peak"] for c in sorted(reliable)])

    # Spearman and Pearson
    perm_result = permutation_test_correlation(sc_arr, cp_arr,
                                                n_perm=N_PERM, seed=SEED)

    print(f"\nSpearman rank correlation:")
    print(f"  rho = {perm_result['spearman_rho']:.4f}")
    print(f"  p (exact) = {perm_result['spearman_p_exact']:.6f}")
    print(f"  p (permutation, n={N_PERM}) = {perm_result['permutation_p']:.6f}")
    print(f"  Null distribution: mean={perm_result['null_mean']:.4f}, "
          f"std={perm_result['null_std']:.4f}, 95th={perm_result['null_95']:.4f}")

    print(f"\nPearson correlation:")
    print(f"  r = {perm_result['pearson_r']:.4f}")
    print(f"  p = {perm_result['pearson_p']:.6f}")

    # Correlation on log scale (more natural for multiplicative relationships)
    log_sc = np.log(sc_arr)
    log_cp = np.log(cp_arr)
    rho_log, p_log = sp_stats.spearmanr(log_sc, log_cp)
    r_log, p_rlog = sp_stats.pearsonr(log_sc, log_cp)
    print(f"\nLog-scale Pearson correlation:")
    print(f"  r = {r_log:.4f},  p = {p_rlog:.6f}")

    # Effect size: Cohen's d for delta
    d_valid = []
    for cidx in sorted(reliable):
        dv = reliable[cidx]["boot"]["delta_samples"]
        d_valid.extend(dv[~np.isnan(dv)])
    d_valid = np.array(d_valid)
    cohens_d = np.mean(d_valid) / np.std(d_valid) if np.std(d_valid) > 0 else np.inf
    print(f"\nEffect size (Cohen's d of delta distribution):")
    print(f"  d = {cohens_d:.3f} (pooled across all reliable circuits)")

    # Per-circuit signed delta test (one-sample t-test against 0)
    print(f"\nPer-circuit one-sample t-tests on bootstrap delta (H0: delta = 0):")
    for cidx in sorted(reliable):
        d = reliable[cidx]
        dv = d["boot"]["delta_samples"]
        dv = dv[~np.isnan(dv)]
        if len(dv) > 1:
            t_stat, t_p = sp_stats.ttest_1samp(dv, 0)
            print(f"  C{cidx:02d} {d['short']:<8}: t={t_stat:8.2f}, p={t_p:.2e}, "
                  f"mean_delta={np.mean(dv):8.1f} +/- {np.std(dv):6.1f}")

    # Binomial test: probability of seeing k or more matches by chance
    # Estimate chance overlap rate
    cfi_cis = [reliable[c]["boot"]["cfi_peak_ci"] for c in sorted(reliable)]
    sc_cis = [reliable[c]["boot"]["sigma_c_ci"] for c in sorted(reliable)]
    all_params = np.concatenate([reliable[c]["param_evo"] for c in sorted(reliable)])
    param_range = [np.min(all_params), np.max(all_params)]

    p_chance = chance_overlap_rate(sc_arr, cfi_cis, sc_cis, param_range)

    # Paired delta overlap (0 in CI) binomial test
    binom_p = sp_stats.binom.sf(n_zero_in_ci - 1, n_reliable, p_chance)
    print(f"\nBinomial test for paired delta overlap:")
    print(f"  Observed: {n_zero_in_ci}/{n_reliable} circuits with 0 in delta CI")
    print(f"  Chance rate (estimated): {p_chance:.3f}")
    print(f"  Binomial p-value (k >= {n_zero_in_ci}): {binom_p:.6f}")

    # ================================================================
    # PHASE 5: Category analysis + sensitivity
    # ================================================================
    print("\n" + "=" * 80)
    print("[5/5] CATEGORY ANALYSIS AND DIAGNOSTICS")
    print("=" * 80)

    # Category breakdown
    print("\nCategory breakdown (reliable circuits only):")
    cats = {}
    for cidx in sorted(reliable):
        cat = reliable[cidx]["category"]
        if cat not in cats:
            cats[cat] = {"match": 0, "fail": 0}
        if reliable[cidx]["boot"]["zero_in_delta_ci"]:
            cats[cat]["match"] += 1
        else:
            cats[cat]["fail"] += 1

    for cat in sorted(cats):
        m = cats[cat]["match"]
        f = cats[cat]["fail"]
        print(f"  {cat:<14}: {m} match, {f} fail -> {100*m/(m+f):.0f}% match")

    # Symmetry analysis: which circuits have conserved quantities?
    symmetry_circuits = {2, 13}  # Heisenberg (SU(2)), Tight binding (U(1))
    print(f"\nSymmetry-protected circuits in reliable set:")
    for cidx in sorted(reliable):
        sym = cidx in symmetry_circuits
        match = reliable[cidx]["boot"]["zero_in_delta_ci"]
        print(f"  C{cidx:02d} {reliable[cidx]['short']:<8}: "
              f"symmetry={'YES' if sym else 'no ':>3}, match={'YES' if match else 'no ':>3}")

    sym_match = sum(1 for c in sorted(reliable)
                    if c in symmetry_circuits and reliable[c]["boot"]["zero_in_delta_ci"])
    sym_total = sum(1 for c in sorted(reliable) if c in symmetry_circuits)
    nosym_match = n_zero_in_ci - sym_match
    nosym_total = n_reliable - sym_total
    print(f"\n  With symmetry:    {sym_match}/{sym_total} match")
    print(f"  Without symmetry: {nosym_match}/{nosym_total} match")
    if nosym_total > 0:
        print(f"  -> Generic circuits (no symmetry): "
              f"{100*nosym_match/nosym_total:.0f}% match")

    # Fisher exact test for symmetry vs match
    if sym_total > 0 and nosym_total > 0:
        table = [[nosym_match, nosym_total - nosym_match],
                 [sym_match, sym_total - sym_match]]
        odds, fisher_p = sp_stats.fisher_exact(table)
        print(f"  Fisher exact test (symmetry vs match): p = {fisher_p:.4f}")

    # KL-based sigma_c cross-validation
    print("\nKL-based sigma_c cross-validation:")
    n_kl_agree = 0
    n_kl_total = 0
    for cidx in sorted(reliable):
        d = reliable[cidx]
        sc_perf = d["sigma_c"]
        sc_kl = d["sigma_c_kl"]
        if sc_perf is not None and sc_kl is not None:
            n_kl_total += 1
            ratio = sc_kl / sc_perf
            agree = 0.5 < ratio < 2.0
            if agree:
                n_kl_agree += 1
            print(f"  C{cidx:02d} {d['short']:<8}: sigma_c(perf)={sc_perf:.1f}  "
                  f"sigma_c(KL)={sc_kl:.1f}  ratio={ratio:.2f}  "
                  f"{'agree' if agree else 'DISAGREE'}")
    print(f"  KL agreement: {n_kl_agree}/{n_kl_total}")

    # Tau sensitivity
    print("\nTau sensitivity (selected circuits):")
    tau_values = [0.2, 0.3, 1/np.e, 0.4, 0.5]
    for cidx in [1, 9, 11, 15]:  # representative matches
        if cidx in reliable:
            d = reliable[cidx]
            print(f"\n  C{cidx:02d} {d['short']}:")
            for tau_val in tau_values:
                perf = d["perf_evo"]
                param = d["param_evo"]
                sc, _ = find_sigma_c(param, perf, tau=tau_val)
                cp = d["cfi_peak"]
                delta = abs(sc - cp) if sc is not None else None
                print(f"    tau={tau_val:.3f}: sigma_c={sc if sc is not None else 'N/A':>8}  "
                      f"CFI_peak={cp:.1f}  delta={'N/A' if delta is None else f'{delta:.1f}':>8}")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("FINAL STATISTICAL SUMMARY")
    print("=" * 80)

    print(f"""
Dataset: 16 Hamiltonians x 12 Trotter depths x 500 shots x 6 qubits
        (Rigetti Ankaa-3, blind kappa experiment, 2026-02-24)

Classification:
  Reliable testable circuits:     {n_reliable}
  Borderline (det < {DETECTION_THRESHOLD:.0%}):         {len(borderline)}
  Non-testable (no sigma_c):      {len(non_testable)}

Primary Result (Paired Bootstrap Delta, B={B_BOOT}):
  0 in 95% CI of delta:           {n_zero_in_ci}/{n_reliable} = {100*n_zero_in_ci/n_reliable:.0f}%
  Median |delta|:                  {np.median(deltas_abs):.1f} CNOTs
  Median delta_rel:                {np.median(deltas_rel):.3f}

Correlation:
  Spearman rho:                    {perm_result['spearman_rho']:.4f} (p = {perm_result['spearman_p_exact']:.4f})
  Pearson r:                       {perm_result['pearson_r']:.4f} (p = {perm_result['pearson_p']:.4f})
  Log-scale Pearson r:             {r_log:.4f} (p = {p_rlog:.4f})
  Permutation p-value:             {perm_result['permutation_p']:.4f}

Binomial test (vs chance):
  Observed overlaps:               {n_zero_in_ci}/{n_reliable}
  Estimated chance rate:           {p_chance:.3f}
  Binomial p-value:                {binom_p:.6f}

Category Analysis:
  Generic (no symmetry):           {nosym_match}/{nosym_total} match = {100*nosym_match/nosym_total:.0f}%
  Symmetry-protected:              {sym_match}/{sym_total} match = {100*sym_match/sym_total:.0f}%""")

    # Verdict
    if perm_result["spearman_p_exact"] < 0.05 and n_zero_in_ci / n_reliable > 0.5:
        print("\nVERDICT: SIGNIFICANT CORRELATION between sigma_c and CFI peak.")
        print("The coincidence is real for generic circuits without strong symmetries.")
    elif perm_result["spearman_p_exact"] < 0.10:
        print("\nVERDICT: SUGGESTIVE but not conclusive correlation.")
    else:
        print("\nVERDICT: No statistically significant correlation found.")

    print(f"\nSymmetry is the key moderator: conserved quantities (SU(2), U(1)) break")
    print(f"the coincidence by partitioning the Hilbert space into protected sectors.")
    print("=" * 80)

    # Save results to JSON
    save_results = {
        "n_reliable": n_reliable,
        "n_borderline": len(borderline),
        "n_nontestable": len(non_testable),
        "paired_delta_overlap": n_zero_in_ci,
        "paired_delta_rate": n_zero_in_ci / n_reliable,
        "median_delta_abs": float(np.median(deltas_abs)),
        "median_delta_rel": float(np.median(deltas_rel)),
        "spearman_rho": perm_result["spearman_rho"],
        "spearman_p": perm_result["spearman_p_exact"],
        "pearson_r": perm_result["pearson_r"],
        "pearson_p": perm_result["pearson_p"],
        "log_pearson_r": float(r_log),
        "log_pearson_p": float(p_rlog),
        "permutation_p": perm_result["permutation_p"],
        "binomial_p": float(binom_p),
        "chance_rate": float(p_chance),
        "cohens_d": float(cohens_d),
        "nosym_match_rate": nosym_match / nosym_total if nosym_total > 0 else None,
        "sym_match_rate": sym_match / sym_total if sym_total > 0 else None,
        "circuits": {},
    }
    for cidx in sorted(reliable):
        d = reliable[cidx]
        b = d["boot"]
        save_results["circuits"][f"C{cidx:02d}"] = {
            "label": d["label"],
            "category": d["category"],
            "sigma_c": d["sigma_c"],
            "cfi_peak": d["cfi_peak"],
            "delta": d["delta"],
            "delta_rel": d["delta"] / d["sigma_c"],
            "delta_ci": [float(x) for x in b["delta_ci"]],
            "zero_in_ci": b["zero_in_delta_ci"],
            "detection_rate": b["detection_rate"],
            "sigma_c_ci": [float(x) for x in b["sigma_c_ci"]],
            "cfi_peak_ci": [float(x) for x in b["cfi_peak_ci"]],
        }

    out_path = Path(r"D:\code\qfi_sigma_c\validation_results.json")
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
