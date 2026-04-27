"""
Comprehensive QFI vs sigma_c Analysis
======================================
Uses ALL raw Braket data: 16 circuits x 12 Trotter depths x 500 shots x 6 qubits
from Rigetti Ankaa-3 (blind kappa experiment, 2026-02-24).

Central question: Does sigma_c coincide with the peak of Classical Fisher
Information across ALL quantum systems?

Methodology:
1. Extract probability distributions p(x|d) from raw bitstrings (64 outcomes)
2. Compute CFI(d) = sum_x [dp/dd]^2 / p(x|d) with central differences
3. Determine sigma_c from performance decay: r(d_c) = tau * r(0)
4. Bootstrap confidence intervals (B=2000) for both CFI peak and sigma_c
5. Systematic comparison across 16 Hamiltonians

Also incorporates:
- Grover noise sweep (19 noise levels, 4 outcomes)
- Quantum magnetism experiments E3, E5, E6 (chi as QFI proxy)

Author: Matthias Christian Wurm / ForgottenForge
Date: April 2026
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CIRCUIT METADATA
# ============================================================

CIRCUIT_LABELS = {
    0:  "TFIM critical",
    1:  "Random layered 2",
    2:  "Heisenberg XXX",
    3:  "Zero evolution",
    4:  "All-to-all TFIM",
    5:  "Toric code",
    6:  "Random layered 1",
    7:  "TFIM deep",
    8:  "GHZ creating",
    9:  "Kitaev detuned",
    10: "TFIM ordered",
    11: "Cluster SPT",
    12: "BCS 6-qubit",
    13: "Tight binding",
    14: "TFIM paramag",
    15: "XXZ anisotropic",
}

CIRCUIT_SHORT = {
    0: "TFIM-c", 1: "Rand-2", 2: "Heis", 3: "Zero",
    4: "ATA", 5: "Toric", 6: "Rand-1", 7: "TFIM-d",
    8: "GHZ", 9: "Kitaev", 10: "TFIM-o", 11: "Clust",
    12: "BCS", 13: "TB", 14: "TFIM-p", 15: "XXZ",
}

CIRCUIT_CATEGORIES = {
    "structured":    [0, 2, 13, 14],
    "topological":   [5, 9, 11],
    "gap_protected": [10, 12, 15],
    "null_random":   [1, 3, 6],
    "edge_case":     [4, 7, 8],
}

DEPTHS = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25]
N_QUBITS = 6
N_OUTCOMES = 2**N_QUBITS  # 64
TAU = 1.0 / np.e  # ~0.368


def get_category(cidx):
    """Get category string for a circuit index."""
    for cat, circuits in CIRCUIT_CATEGORIES.items():
        if cidx in circuits:
            return cat
    return "unknown"

BASE_DIR = Path(r"D:\code\onto\particle_plots\braket_archive\2026-02-24_blind_kappa")


# ============================================================
# DATA LOADING
# ============================================================

def load_catalog():
    """Load experiment catalog."""
    with open(BASE_DIR / "CATALOG.json") as f:
        return json.load(f)


def load_measurements(circuit_idx, depth_idx):
    """
    Load raw measurement bitstrings for a given circuit and depth.

    Returns
    -------
    measurements : np.ndarray of shape (n_shots, n_qubits)
    """
    catalog = load_catalog()
    # Find the right task
    for task in catalog["tasks"]:
        if task["circuit"] == circuit_idx and task["depth_idx"] == depth_idx:
            fpath = BASE_DIR / task["file"]
            with open(fpath) as f:
                data = json.load(f)
            return np.array(data["measurements"])
    raise ValueError(f"Task not found: circuit={circuit_idx}, depth_idx={depth_idx}")


def bitstring_to_int(bits):
    """Convert a measurement array [0,1,0,1,1,0] to integer index."""
    result = 0
    for b in bits:
        result = (result << 1) | int(b)
    return result


def measurements_to_probs(measurements):
    """
    Convert raw measurement array to probability distribution.

    Parameters
    ----------
    measurements : np.ndarray of shape (n_shots, n_qubits)

    Returns
    -------
    probs : np.ndarray of shape (64,) - probability for each bitstring
    """
    n_shots = measurements.shape[0]
    counts = np.zeros(N_OUTCOMES)
    for shot in measurements:
        idx = bitstring_to_int(shot)
        counts[idx] += 1
    return counts / n_shots


def load_all_circuit_data(circuit_idx):
    """
    Load all 12 depth files for a circuit and compute probability distributions.

    Returns
    -------
    depths : np.ndarray of shape (12,)
    probs : np.ndarray of shape (12, 64)
    raw_measurements : list of 12 arrays, each (500, 6)
    cnot_counts : np.ndarray of shape (12,)
    """
    catalog = load_catalog()

    depths_list = []
    probs_list = []
    raw_list = []
    cnot_list = []

    for task in catalog["tasks"]:
        if task["circuit"] == circuit_idx:
            fpath = BASE_DIR / task["file"]
            with open(fpath) as f:
                data = json.load(f)
            meas = np.array(data["measurements"])
            p = measurements_to_probs(meas)
            depths_list.append(task["depth"])
            probs_list.append(p)
            raw_list.append(meas)
            cnot_list.append(task["n_cnot"])

    # Sort by depth
    order = np.argsort(depths_list)
    depths = np.array(depths_list)[order]
    probs = np.array(probs_list)[order]
    cnots = np.array(cnot_list)[order]
    raw_sorted = [raw_list[i] for i in order]

    return depths, probs, raw_sorted, cnots


# Cache the catalog to avoid re-reading 192 times
_catalog_cache = None
def get_catalog():
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = load_catalog()
    return _catalog_cache


def load_all_circuit_data_fast(circuit_idx):
    """
    Optimized version: loads all depth files for one circuit.
    """
    catalog = get_catalog()

    tasks_for_circuit = sorted(
        [t for t in catalog["tasks"] if t["circuit"] == circuit_idx],
        key=lambda t: t["depth_idx"]
    )

    depths = []
    probs = []
    raw_measurements = []
    cnot_counts = []

    for task in tasks_for_circuit:
        fpath = BASE_DIR / task["file"]
        with open(fpath) as f:
            data = json.load(f)
        meas = np.array(data["measurements"])
        p = measurements_to_probs(meas)
        depths.append(task["depth"])
        probs.append(p)
        raw_measurements.append(meas)
        cnot_counts.append(task["n_cnot"])

    return (np.array(depths), np.array(probs),
            raw_measurements, np.array(cnot_counts))


# ============================================================
# CLASSICAL FISHER INFORMATION
# ============================================================

def classical_fisher_information(param_values, probs, regularize=1e-10):
    """
    Compute Classical Fisher Information F_C(d) from probability distributions.

    F_C(d) = sum_x [dp(x|d)/dd]^2 / p(x|d)

    Uses central differences for derivatives (forward/backward at boundaries).

    Parameters
    ----------
    param_values : np.ndarray of shape (N,) - depth or CNOT count
    probs : np.ndarray of shape (N, K) - probabilities for K outcomes
    regularize : float - prevents division by zero

    Returns
    -------
    cfi : np.ndarray of shape (N,)
    """
    N, K = probs.shape

    # Regularize: add small constant, re-normalize
    probs_reg = probs + regularize
    probs_reg = probs_reg / probs_reg.sum(axis=1, keepdims=True)

    # Numerical derivatives dp/d(param)
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

    # CFI = sum_x (dp/depsilon)^2 / p
    cfi = np.sum(dp**2 / probs_reg, axis=1)

    return cfi


# ============================================================
# SIGMA_C COMPUTATION
# ============================================================

def find_sigma_c(param_values, performance, tau=TAU):
    """
    Find sigma_c: the parameter value where performance drops to tau * r(0).

    Uses linear interpolation between grid points.

    Parameters
    ----------
    param_values : array of parameter values (depth, CNOT count, etc.)
    performance : array of performance metric values
    tau : threshold fraction (default 1/e)

    Returns
    -------
    sigma_c : float or None
    sigma_c_idx : int - nearest grid index
    """
    r0 = performance[0]
    threshold = tau * r0

    # Find crossing point
    for i in range(1, len(performance)):
        if performance[i] <= threshold:
            # Linear interpolation
            if performance[i-1] == performance[i]:
                sigma_c = param_values[i]
            else:
                frac = (threshold - performance[i-1]) / (performance[i] - performance[i-1])
                sigma_c = param_values[i-1] + frac * (param_values[i] - param_values[i-1])
            return sigma_c, i

    # Never crossed threshold
    return None, None


def compute_performance(probs):
    """
    Compute performance metric from probability distributions.

    Performance = max outcome probability (equivalently, success rate for
    the most-likely bitstring). This decays from ~1 to 1/64 as noise increases.
    """
    return np.max(probs, axis=1)


def compute_kl_from_uniform(probs):
    """
    Compute KL divergence from uniform distribution.

    D_KL(p || u) = sum_x p(x) log(p(x) / u(x))

    This is another performance metric: 0 when fully random, large when structured.
    """
    uniform = np.ones(probs.shape[1]) / probs.shape[1]
    kl = np.zeros(probs.shape[0])
    for i in range(probs.shape[0]):
        p = probs[i]
        p_safe = np.where(p > 0, p, 1e-30)
        kl[i] = np.sum(p_safe * np.log(p_safe / uniform))
    return kl


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_cfi_and_sigma_c(raw_measurements, param_values, B=2000, tau=TAU,
                               seed=42):
    """
    Bootstrap analysis for CFI peak location and sigma_c.

    For each bootstrap replicate:
    1. Resample shots with replacement at each depth
    2. Recompute probability distributions
    3. Compute CFI and find its peak
    4. Compute performance and find sigma_c

    Parameters
    ----------
    raw_measurements : list of arrays, each (n_shots, n_qubits)
    param_values : array of parameter values
    B : number of bootstrap replicates
    tau : sigma_c threshold

    Returns
    -------
    dict with bootstrap distributions and confidence intervals
    """
    rng = np.random.RandomState(seed)
    n_depths = len(raw_measurements)

    cfi_peak_locations = []
    cfi_peak_values = []
    sigma_c_values = []
    all_cfi_curves = []

    for b in range(B):
        # Resample shots at each depth
        probs_boot = np.zeros((n_depths, N_OUTCOMES))
        for d in range(n_depths):
            meas = raw_measurements[d]
            n_shots = meas.shape[0]
            idx = rng.randint(0, n_shots, size=n_shots)
            meas_boot = meas[idx]
            probs_boot[d] = measurements_to_probs(meas_boot)

        # Compute CFI
        cfi = classical_fisher_information(param_values, probs_boot)
        all_cfi_curves.append(cfi)

        # Find CFI peak (exclude boundary points d=0 and d=max)
        interior = cfi[1:-1]
        if len(interior) > 0:
            peak_idx = np.argmax(interior) + 1  # offset for removed first point
            cfi_peak_locations.append(param_values[peak_idx])
            cfi_peak_values.append(cfi[peak_idx])
        else:
            cfi_peak_locations.append(np.nan)
            cfi_peak_values.append(np.nan)

        # Compute sigma_c
        perf = compute_performance(probs_boot)
        sc, _ = find_sigma_c(param_values, perf, tau)
        sigma_c_values.append(sc if sc is not None else np.nan)

    cfi_peak_locations = np.array(cfi_peak_locations)
    sigma_c_values = np.array(sigma_c_values)
    cfi_peak_values = np.array(cfi_peak_values)
    all_cfi_curves = np.array(all_cfi_curves)

    # Remove NaN for CI computation
    valid_peaks = cfi_peak_locations[~np.isnan(cfi_peak_locations)]
    valid_sc = sigma_c_values[~np.isnan(sigma_c_values)]

    result = {
        "cfi_peak_mean": np.nanmean(cfi_peak_locations),
        "cfi_peak_median": np.nanmedian(cfi_peak_locations),
        "cfi_peak_ci": np.percentile(valid_peaks, [2.5, 97.5]) if len(valid_peaks) > 10 else [np.nan, np.nan],
        "cfi_peak_std": np.nanstd(cfi_peak_locations),
        "sigma_c_mean": np.nanmean(sigma_c_values),
        "sigma_c_median": np.nanmedian(sigma_c_values),
        "sigma_c_ci": np.percentile(valid_sc, [2.5, 97.5]) if len(valid_sc) > 10 else [np.nan, np.nan],
        "sigma_c_std": np.nanstd(sigma_c_values),
        "sigma_c_detection_rate": np.mean(~np.isnan(sigma_c_values)),
        "cfi_curves_mean": np.nanmean(all_cfi_curves, axis=0),
        "cfi_curves_lo": np.nanpercentile(all_cfi_curves, 2.5, axis=0),
        "cfi_curves_hi": np.nanpercentile(all_cfi_curves, 97.5, axis=0),
        "raw_cfi_peaks": cfi_peak_locations,
        "raw_sigma_c": sigma_c_values,
    }

    return result


# ============================================================
# ANALYZE ONE CIRCUIT
# ============================================================

def compute_shannon_entropy(probs):
    """Compute Shannon entropy H(p) = -sum p log2 p at each depth."""
    h = np.zeros(probs.shape[0])
    for i in range(probs.shape[0]):
        p = probs[i]
        p_safe = p[p > 0]
        h[i] = -np.sum(p_safe * np.log2(p_safe))
    return h


def analyze_circuit(circuit_idx, B=2000):
    """
    Full analysis for one circuit with THREE complementary approaches:

    A) CNOT-based (full range): parameter = CNOT count, includes depth=0
       - sigma_c from max-prob decay vs CNOT count
       - CFI w.r.t. CNOT count

    B) Evolution-only (depth >= 1): parameter = CNOT count, excludes depth=0
       - Baseline = first Trotter step (not vacuum |000000>)
       - Measures noise degradation of the actual quantum state
       - sigma_c from max-prob decay relative to depth-1 baseline

    C) KL-based: parameter = depth, sigma_c from KL divergence decay
       - KL(p||uniform) measures "distance from random"
       - Decays as noise randomizes the state
    """
    print(f"  Loading circuit {circuit_idx:02d} ({CIRCUIT_LABELS[circuit_idx]})...")
    depths, probs, raw_meas, cnots = load_all_circuit_data_fast(circuit_idx)

    # ---- Approach A: Full range, CNOT-based ----
    param_cnot = cnots.astype(float)
    # Avoid zero parameter at depth=0: use small offset
    if param_cnot[0] == 0:
        param_cnot[0] = 0.5  # half a CNOT as placeholder

    cfi_cnot = classical_fisher_information(param_cnot, probs)
    perf = compute_performance(probs)
    kl = compute_kl_from_uniform(probs)
    entropy = compute_shannon_entropy(probs)

    sigma_c_cnot, _ = find_sigma_c(param_cnot, perf)

    # CFI peak (interior only, exclude boundaries)
    interior = cfi_cnot[1:-1]
    if len(interior) > 0:
        peak_idx_A = np.argmax(interior) + 1
        cfi_peak_A = param_cnot[peak_idx_A]
        cfi_val_A = cfi_cnot[peak_idx_A]
    else:
        peak_idx_A = 0
        cfi_peak_A = param_cnot[0]
        cfi_val_A = cfi_cnot[0]

    # ---- Approach B: Evolution-only (depth >= 1) ----
    # Skip depth=0 - analyze noise degradation of the evolved state
    if len(depths) > 2:
        probs_evo = probs[1:]      # depth 1, 2, 3, ...
        param_evo = cnots[1:].astype(float)
        raw_evo = raw_meas[1:]
        depths_evo = depths[1:]

        cfi_evo = classical_fisher_information(param_evo, probs_evo)
        perf_evo = compute_performance(probs_evo)
        kl_evo = compute_kl_from_uniform(probs_evo)

        # sigma_c from evolution baseline (depth=1 is the "ideal" evolved state)
        sigma_c_evo, _ = find_sigma_c(param_evo, perf_evo)
        sigma_c_kl_evo, _ = find_sigma_c(param_evo, kl_evo)

        # CFI peak in evolution range (interior only)
        if len(cfi_evo) > 2:
            interior_evo = cfi_evo[1:-1]
            peak_idx_B = np.argmax(interior_evo) + 1
            cfi_peak_B = param_evo[peak_idx_B]
            cfi_val_B = cfi_evo[peak_idx_B]
        else:
            peak_idx_B = np.argmax(cfi_evo)
            cfi_peak_B = param_evo[peak_idx_B]
            cfi_val_B = cfi_evo[peak_idx_B]
    else:
        probs_evo = probs
        param_evo = param_cnot
        raw_evo = raw_meas
        cfi_evo = cfi_cnot
        perf_evo = perf
        sigma_c_evo = None
        sigma_c_kl_evo = None
        cfi_peak_B = None
        cfi_val_B = None
        depths_evo = depths

    # ---- Approach C: Depth-based with KL ----
    param_depth = depths.astype(float)
    sigma_c_depth, _ = find_sigma_c(param_depth, perf)
    sigma_c_kl, _ = find_sigma_c(param_depth, kl)
    cfi_depth = classical_fisher_information(param_depth, probs)

    # ---- Bootstrap on evolution-only data (most meaningful) ----
    print(f"    Bootstrap (B={B}) on evolution-only data...")
    boot = bootstrap_cfi_and_sigma_c(raw_evo, param_evo, B=B)

    # Also bootstrap full range
    boot_full = bootstrap_cfi_and_sigma_c(raw_meas, param_cnot, B=B, seed=123)

    # Overlap tests
    def ci_overlap(ci1, ci2):
        if np.any(np.isnan(ci1)) or np.any(np.isnan(ci2)):
            return False
        return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])

    overlap_evo = ci_overlap(boot["sigma_c_ci"], boot["cfi_peak_ci"])
    overlap_full = ci_overlap(boot_full["sigma_c_ci"], boot_full["cfi_peak_ci"])

    # Compute deltas
    delta_evo = abs(sigma_c_evo - cfi_peak_B) if (sigma_c_evo is not None and cfi_peak_B is not None) else None
    delta_full = abs(sigma_c_cnot - cfi_peak_A) if sigma_c_cnot is not None else None

    result = {
        "circuit": circuit_idx,
        "label": CIRCUIT_LABELS[circuit_idx],
        "short": CIRCUIT_SHORT[circuit_idx],
        "depths": depths.tolist(),
        "cnots": cnots.tolist(),
        "probs": probs.tolist(),

        # Approach A: Full CNOT range
        "cfi_cnot": cfi_cnot.tolist(),
        "sigma_c_cnot": float(sigma_c_cnot) if sigma_c_cnot is not None else None,
        "cfi_peak_cnot": float(cfi_peak_A),
        "delta_cnot": float(delta_full) if delta_full is not None else None,
        "overlap_cnot": overlap_full,

        # Approach B: Evolution-only (depth >= 1)
        "depths_evo": depths_evo.tolist() if depths_evo is not None else None,
        "cnots_evo": param_evo.tolist(),
        "cfi_evo": cfi_evo.tolist(),
        "perf_evo": perf_evo.tolist(),
        "sigma_c_evo": float(sigma_c_evo) if sigma_c_evo is not None else None,
        "sigma_c_kl_evo": float(sigma_c_kl_evo) if sigma_c_kl_evo is not None else None,
        "cfi_peak_evo": float(cfi_peak_B) if cfi_peak_B is not None else None,
        "delta_evo": float(delta_evo) if delta_evo is not None else None,
        "overlap_evo": overlap_evo,

        # Approach C: KL-based
        "sigma_c_depth": float(sigma_c_depth) if sigma_c_depth is not None else None,
        "sigma_c_kl": float(sigma_c_kl) if sigma_c_kl is not None else None,

        # Observables
        "performance": perf.tolist(),
        "kl_divergence": kl.tolist(),
        "entropy": entropy.tolist(),

        # Bootstrap (evolution-only)
        "bootstrap_evo": {
            "cfi_peak_mean": float(boot["cfi_peak_mean"]),
            "cfi_peak_median": float(boot["cfi_peak_median"]),
            "cfi_peak_ci": [float(x) for x in boot["cfi_peak_ci"]],
            "sigma_c_mean": float(boot["sigma_c_mean"]),
            "sigma_c_median": float(boot["sigma_c_median"]),
            "sigma_c_ci": [float(x) for x in boot["sigma_c_ci"]],
            "sigma_c_detection_rate": float(boot["sigma_c_detection_rate"]),
            "cfi_curves_mean": boot["cfi_curves_mean"].tolist(),
            "cfi_curves_lo": boot["cfi_curves_lo"].tolist(),
            "cfi_curves_hi": boot["cfi_curves_hi"].tolist(),
        },
        "bootstrap_full": {
            "cfi_peak_mean": float(boot_full["cfi_peak_mean"]),
            "cfi_peak_ci": [float(x) for x in boot_full["cfi_peak_ci"]],
            "sigma_c_mean": float(boot_full["sigma_c_mean"]),
            "sigma_c_ci": [float(x) for x in boot_full["sigma_c_ci"]],
            "cfi_curves_mean": boot_full["cfi_curves_mean"].tolist(),
            "cfi_curves_lo": boot_full["cfi_curves_lo"].tolist(),
            "cfi_curves_hi": boot_full["cfi_curves_hi"].tolist(),
        },
    }

    # Summary
    print(f"    [Full]  sigma_c={sigma_c_cnot:.1f} CNOTs, CFI peak={cfi_peak_A:.1f} CNOTs, "
          f"Delta={delta_full:.1f}, CI overlap={overlap_full}" if sigma_c_cnot is not None else
          f"    [Full]  sigma_c=N/A, CFI peak={cfi_peak_A:.1f}")
    if sigma_c_evo is not None and cfi_peak_B is not None:
        print(f"    [Evo]   sigma_c={sigma_c_evo:.1f} CNOTs, CFI peak={cfi_peak_B:.1f} CNOTs, "
              f"Delta={delta_evo:.1f}, CI overlap={overlap_evo}")
    else:
        print(f"    [Evo]   sigma_c=N/A (no threshold crossing in evolution range)")

    return result


# ============================================================
# GROVER + MAGNETISM (from previous analysis)
# ============================================================

def analyze_grover():
    """Load and analyze Grover noise sweep data."""
    path = Path(r"D:\code\clco\noend\grover_stability_data_20250728_211356.json")
    with open(path) as f:
        data = json.load(f)

    noise = np.array(data["noise_levels"])
    orig = data["original_data"]
    outcomes = ["00", "01", "10", "11"]
    probs = np.array([[d["probabilities"][o] for o in outcomes] for d in orig])
    success = np.array([d["success_rate"] for d in orig])

    # CFI
    cfi = classical_fisher_information(noise, probs)

    # sigma_c
    sc, _ = find_sigma_c(noise, success)

    # CFI peak (interior only)
    interior = cfi[1:-1]
    peak_idx = np.argmax(interior) + 1
    peak_loc = noise[peak_idx]

    return {
        "label": "Grover 2-qubit",
        "param": noise.tolist(),
        "cfi": cfi.tolist(),
        "performance": success.tolist(),
        "sigma_c": float(sc) if sc else None,
        "cfi_peak": float(peak_loc),
        "cfi_peak_val": float(cfi[peak_idx]),
    }


def analyze_magnetism():
    """Load and analyze quantum magnetism experiments E1-E6."""
    path = Path(r"D:\code\onto\data\quantum_magnetism_complete_data.json")
    with open(path) as f:
        data = json.load(f)

    results = {}
    for exp_key, exp in data["experiments"].items():
        # Determine sigma_c (different field names per experiment)
        sc = exp.get("sigma_c")
        if sc is None:
            sc = exp.get("sigma_c_field")
        if sc is None:
            continue

        # Determine the observable (chi, witnesses, or correlations)
        if "chi" in exp:
            observable = np.array(exp["chi"])
            obs_name = "chi"
        elif "witnesses" in exp:
            # Entanglement witness: sign flip is the transition
            observable = -np.array(exp["witnesses"])  # flip sign so peak = most entangled
            obs_name = "witness"
        elif "zz_correlations" in exp:
            # Use derivative of correlations as susceptibility proxy
            corr = np.array(exp["zz_correlations"])
            observable = np.abs(np.gradient(corr))
            obs_name = "d_corr"
        else:
            continue

        # Determine parameter axis
        if "noise_levels" in exp:
            param = np.array(exp["noise_levels"])
        elif "fields" in exp:
            param = np.array(exp["fields"])
        elif "damping_rates" in exp:
            param = np.array(exp["damping_rates"])
        elif "distances" in exp:
            param = np.array(exp["distances"])
        else:
            continue

        # Find observable peak
        peak_idx = np.argmax(observable)
        peak_loc = param[peak_idx]

        results[exp_key] = {
            "label": exp_key,
            "sigma_c": float(sc),
            "obs_peak": float(peak_loc),
            "obs_name": obs_name,
            "delta": abs(float(sc) - float(peak_loc)),
            "param": param.tolist(),
            "observable": observable.tolist(),
            "kappa": exp.get("kappa"),
            "interpretation": exp.get("interpretation", ""),
        }

    return results


# ============================================================
# PUBLICATION FIGURES
# ============================================================

def make_comprehensive_figure(all_results, grover, magnetism):
    """Generate publication-quality multi-panel figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches

    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
    })

    category_colors = {
        "structured": "#2196F3",
        "topological": "#9C27B0",
        "gap_protected": "#4CAF50",
        "null_random": "#9E9E9E",
        "edge_case": "#FF9800",
    }

    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(6, 4, hspace=0.38, wspace=0.32,
                  left=0.06, right=0.97, top=0.96, bottom=0.03)

    # ---- Rows 0-3: 16 circuit panels (4x4), evolution-only CFI ----
    for i, r in enumerate(all_results):
        row = i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])

        cat = get_category(r["circuit"])
        color = category_colors.get(cat, "#333333")

        # Plot evolution-only CFI (depth >= 1) with CNOT x-axis
        cnots_evo = np.array(r["cnots_evo"])
        cfi_evo = np.array(r["cfi_evo"])
        boot = r["bootstrap_evo"]

        cfi_mean = np.array(boot["cfi_curves_mean"])
        cfi_lo = np.array(boot["cfi_curves_lo"])
        cfi_hi = np.array(boot["cfi_curves_hi"])

        ax.fill_between(cnots_evo, cfi_lo, cfi_hi, alpha=0.2, color=color)
        ax.plot(cnots_evo, cfi_mean, 'o-', color=color, markersize=3, linewidth=1.2)

        # Mark CFI peak (evolution)
        if r["cfi_peak_evo"] is not None:
            ax.axvline(r["cfi_peak_evo"], color=color, linestyle='--',
                       alpha=0.6, linewidth=0.8)

        # Mark sigma_c (evolution)
        if r["sigma_c_evo"] is not None:
            ax.axvline(r["sigma_c_evo"], color='red', linestyle='-', alpha=0.8,
                       linewidth=1.5)
            sc_ci = boot["sigma_c_ci"]
            if not np.any(np.isnan(sc_ci)):
                ax.axvspan(sc_ci[0], sc_ci[1], alpha=0.1, color='red')

        # Performance on twin axis
        ax2 = ax.twinx()
        perf_evo = np.array(r["perf_evo"])
        ax2.plot(cnots_evo, perf_evo, 's-', color='gray', markersize=2,
                 alpha=0.5, linewidth=0.8)
        ax2.tick_params(axis='y', labelsize=6, colors='gray')
        if col == 3:
            ax2.set_ylabel('P(max)', fontsize=7, color='gray')

        # Title
        match_str = ""
        if r["overlap_evo"]:
            match_str = " *"
        ax.set_title(f'C{r["circuit"]:02d}: {r["short"]}{match_str}',
                     fontsize=8, fontweight='bold', color=color)

        if row == 3:
            ax.set_xlabel('CNOT count')
        if col == 0:
            ax.set_ylabel('CFI')

    # ---- Row 4, left: Evolution-only scatter plot ----
    ax_evo = fig.add_subplot(gs[4, 0:2])

    valid_evo = [r for r in all_results
                 if r["sigma_c_evo"] is not None and r["cfi_peak_evo"] is not None]
    for r in valid_evo:
        cat = get_category(r["circuit"])
        color = category_colors.get(cat, "#333333")
        marker = '*' if r["overlap_evo"] else 'o'
        size = 120 if r["overlap_evo"] else 60

        ax_evo.scatter(r["sigma_c_evo"], r["cfi_peak_evo"],
                      c=color, marker=marker, s=size, edgecolors='black',
                      linewidths=0.5, zorder=5)

        boot = r["bootstrap_evo"]
        sc_ci = boot["sigma_c_ci"]
        cp_ci = boot["cfi_peak_ci"]
        if not np.any(np.isnan(sc_ci)) and not np.any(np.isnan(cp_ci)):
            ax_evo.errorbar(r["sigma_c_evo"], r["cfi_peak_evo"],
                           xerr=[[r["sigma_c_evo"] - sc_ci[0]],
                                 [sc_ci[1] - r["sigma_c_evo"]]],
                           yerr=[[r["cfi_peak_evo"] - cp_ci[0]],
                                 [cp_ci[1] - r["cfi_peak_evo"]]],
                           fmt='none', color=color, alpha=0.4, linewidth=0.8)

        ax_evo.annotate(r["short"], (r["sigma_c_evo"], r["cfi_peak_evo"]),
                       fontsize=6, ha='left', va='bottom',
                       xytext=(3, 3), textcoords='offset points')

    if valid_evo:
        all_sc = [r["sigma_c_evo"] for r in valid_evo]
        all_cp = [r["cfi_peak_evo"] for r in valid_evo]
        lim = max(max(all_sc), max(all_cp)) * 1.15
        ax_evo.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
    ax_evo.set_xlabel('$\\sigma_c$ (CNOT count, evolution-only)')
    ax_evo.set_ylabel('CFI peak (CNOT count)')
    n_match_evo = sum(1 for r in valid_evo if r["overlap_evo"])
    ax_evo.set_title(f'(E) Evolution-Only: $\\sigma_c$ vs CFI Peak '
                     f'({n_match_evo}/{len(valid_evo)} overlap)',
                     fontweight='bold')

    handles = []
    for cat, color in category_colors.items():
        handles.append(mpatches.Patch(color=color,
                       label=cat.replace('_', ' ').title()))
    handles.append(plt.Line2D([0], [0], marker='*', color='gray', markersize=10,
                              label='CI overlap', linestyle='None'))
    ax_evo.legend(handles=handles, fontsize=7, loc='upper left')

    # ---- Row 4, right: Full-range scatter (CNOT-based) ----
    ax_full = fig.add_subplot(gs[4, 2:4])

    valid_full = [r for r in all_results if r["sigma_c_cnot"] is not None]
    for r in valid_full:
        cat = get_category(r["circuit"])
        color = category_colors.get(cat, "#333333")
        marker = '*' if r["overlap_cnot"] else 'o'
        size = 120 if r["overlap_cnot"] else 60

        ax_full.scatter(r["sigma_c_cnot"], r["cfi_peak_cnot"],
                       c=color, marker=marker, s=size, edgecolors='black',
                       linewidths=0.5, zorder=5)
        ax_full.annotate(r["short"], (r["sigma_c_cnot"], r["cfi_peak_cnot"]),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(3, 3), textcoords='offset points')

    if valid_full:
        all_sc = [r["sigma_c_cnot"] for r in valid_full]
        all_cp = [r["cfi_peak_cnot"] for r in valid_full]
        lim = max(max(all_sc), max(all_cp)) * 1.15
        ax_full.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=1)
    n_match_full = sum(1 for r in valid_full if r["overlap_cnot"])
    ax_full.set_xlabel('$\\sigma_c$ (CNOT count, full range)')
    ax_full.set_ylabel('CFI peak (CNOT count)')
    ax_full.set_title(f'(F) Full Range: $\\sigma_c$ vs CFI Peak '
                      f'({n_match_full}/{len(valid_full)} overlap)',
                      fontweight='bold')

    # ---- Row 5, left: Delta distributions ----
    ax_hist = fig.add_subplot(gs[5, 0:2])

    deltas_evo = [r["delta_evo"] for r in valid_evo if r["delta_evo"] is not None]
    deltas_full = [r["delta_cnot"] for r in valid_full if r["delta_cnot"] is not None]

    if deltas_evo:
        ax_hist.hist(deltas_evo, bins=12, color='steelblue', edgecolor='white',
                     alpha=0.7, label='Evolution-only')
    if deltas_full:
        ax_hist.hist(deltas_full, bins=12, color='coral', edgecolor='white',
                     alpha=0.5, label='Full range')
    ax_hist.axvline(0, color='red', linestyle='--', linewidth=1.5)
    ax_hist.set_xlabel('$|\\sigma_c - \\mathrm{CNOT}_{\\mathrm{CFI\\_peak}}|$')
    ax_hist.set_ylabel('Count')
    ax_hist.set_title('(G) $\\Delta$ Distribution', fontweight='bold')
    ax_hist.legend(fontsize=7)

    # ---- Row 5, right: Legacy data ----
    ax_legacy = fig.add_subplot(gs[5, 2:4])

    for key, m in magnetism.items():
        ax_legacy.scatter(m["sigma_c"], m["obs_peak"], c='darkgreen', s=80,
                         marker='D', edgecolors='black', linewidths=0.5, zorder=5)
        ax_legacy.annotate(key.replace('_', ' '), (m["sigma_c"], m["obs_peak"]),
                          fontsize=6, ha='left', va='bottom',
                          xytext=(3, 3), textcoords='offset points')

    if grover["sigma_c"] is not None:
        ax_legacy.scatter(grover["sigma_c"], grover["cfi_peak"],
                         c='darkorange', s=80, marker='^', edgecolors='black',
                         linewidths=0.5, zorder=5)
        ax_legacy.annotate("Grover", (grover["sigma_c"], grover["cfi_peak"]),
                          fontsize=6)

    all_vals = [m["sigma_c"] for m in magnetism.values()] + \
               [m["obs_peak"] for m in magnetism.values()]
    if grover["sigma_c"]:
        all_vals += [grover["sigma_c"], grover["cfi_peak"]]
    if all_vals:
        lim = max(all_vals) * 1.2
        ax_legacy.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax_legacy.set_xlabel('$\\sigma_c$')
    ax_legacy.set_ylabel('$\\chi$ peak / CFI peak')
    ax_legacy.set_title('(H) Legacy Data (Grover + Magnetism)', fontweight='bold')

    # Save
    for ext in ['png', 'pdf']:
        fig.savefig(Path(r"D:\code\qfi_sigma_c") / f"comprehensive_qfi_vs_sigma_c.{ext}",
                    bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: comprehensive_qfi_vs_sigma_c.png/pdf")


# ============================================================
# SUMMARY TABLE
# ============================================================

def print_summary_table(all_results, grover, magnetism):
    """Print publication-ready summary table."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE QFI vs SIGMA_C ANALYSIS - SUMMARY TABLE")
    print("=" * 120)

    # ---- Table 1: Full range (CNOT-based, including depth=0) ----
    print("\nTABLE 1: Full Range Analysis (parameter = CNOT count, baseline = depth 0)")
    print("-" * 110)
    print(f"{'System':<22} {'Category':<14} {'sigma_c':>10} {'CFI peak':>10} "
          f"{'Delta':>8} {'CI overlap':>10} {'sigma_c CI':>18} {'CFI CI':>18}")
    print("-" * 110)

    n_match_full = 0
    n_total_full = 0
    for r in all_results:
        cat = get_category(r["circuit"])
        sc = r["sigma_c_cnot"]
        cp = r["cfi_peak_cnot"]
        boot = r["bootstrap_full"]

        if sc is not None:
            n_total_full += 1
            overlap = r["overlap_cnot"]
            if overlap:
                n_match_full += 1
            flag = "YES" if overlap else "no"
            sc_ci = f"[{boot['sigma_c_ci'][0]:.1f}, {boot['sigma_c_ci'][1]:.1f}]"
            cp_ci = f"[{boot['cfi_peak_ci'][0]:.1f}, {boot['cfi_peak_ci'][1]:.1f}]"
            delta = r["delta_cnot"]
            print(f"C{r['circuit']:02d} {r['label']:<18} {cat:<14} "
                  f"{sc:10.1f} {cp:10.1f} {delta:8.1f} {flag:>10} "
                  f"{sc_ci:>18} {cp_ci:>18}")
        else:
            print(f"C{r['circuit']:02d} {r['label']:<18} {cat:<14} "
                  f"{'N/A':>10} {cp:10.1f} {'N/A':>8} {'N/A':>10}")

    print(f"\nFull range: {n_match_full}/{n_total_full} CI overlaps "
          f"({100*n_match_full/n_total_full:.0f}%)" if n_total_full > 0 else "")

    # ---- Table 2: Evolution-only (depth >= 1) ----
    print("\n\nTABLE 2: Evolution-Only Analysis (parameter = CNOT count, baseline = depth 1)")
    print("-" * 110)
    print(f"{'System':<22} {'Category':<14} {'sigma_c':>10} {'CFI peak':>10} "
          f"{'Delta':>8} {'CI overlap':>10} {'sigma_c CI':>18} {'CFI CI':>18}")
    print("-" * 110)

    n_match_evo = 0
    n_total_evo = 0
    for r in all_results:
        cat = get_category(r["circuit"])
        sc = r["sigma_c_evo"]
        cp = r["cfi_peak_evo"]
        boot = r["bootstrap_evo"]

        if sc is not None and cp is not None:
            n_total_evo += 1
            overlap = r["overlap_evo"]
            if overlap:
                n_match_evo += 1
            flag = "YES" if overlap else "no"
            sc_ci = f"[{boot['sigma_c_ci'][0]:.1f}, {boot['sigma_c_ci'][1]:.1f}]"
            cp_ci = f"[{boot['cfi_peak_ci'][0]:.1f}, {boot['cfi_peak_ci'][1]:.1f}]"
            delta = r["delta_evo"]
            print(f"C{r['circuit']:02d} {r['label']:<18} {cat:<14} "
                  f"{sc:10.1f} {cp:10.1f} {delta:8.1f} {flag:>10} "
                  f"{sc_ci:>18} {cp_ci:>18}")
        else:
            sc_str = f"{sc:.1f}" if sc is not None else "N/A"
            cp_str = f"{cp:.1f}" if cp is not None else "N/A"
            print(f"C{r['circuit']:02d} {r['label']:<18} {cat:<14} "
                  f"{sc_str:>10} {cp_str:>10} {'N/A':>8} {'N/A':>10}")

    print(f"\nEvolution-only: {n_match_evo}/{n_total_evo} CI overlaps "
          f"({100*n_match_evo/n_total_evo:.0f}%)" if n_total_evo > 0 else "")

    # ---- Legacy data ----
    print("\n\nLEGACY DATA (independent noise parameter experiments):")
    print("-" * 70)
    for key, m in magnetism.items():
        print(f"  {key:<30} sigma_c={m['sigma_c']:.4f}  {m['obs_name']}_peak={m['obs_peak']:.4f}  "
              f"Delta={m['delta']:.4f}")

    if grover["sigma_c"]:
        print(f"  {'Grover 2-qubit':<30} sigma_c={grover['sigma_c']:.4f}  "
              f"CFI_peak={grover['cfi_peak']:.4f}  "
              f"Delta={abs(grover['sigma_c'] - grover['cfi_peak']):.4f}")

    print("\n" + "=" * 120)
    print("OVERALL VERDICT:")
    print(f"  Full range:      {n_match_full}/{n_total_full} CI overlaps" if n_total_full > 0 else "")
    print(f"  Evolution-only:  {n_match_evo}/{n_total_evo} CI overlaps" if n_total_evo > 0 else "")
    print("=" * 120)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE QFI vs SIGMA_C ANALYSIS")
    print("16 Hamiltonians x 12 Trotter depths x 500 shots x 6 qubits")
    print("Rigetti Ankaa-3, 2026-02-24")
    print("=" * 70)

    # 1. Analyze all 16 blind kappa circuits
    print("\n[1/3] Analyzing 16 blind kappa circuits...")
    all_results = []
    for cidx in range(16):
        r = analyze_circuit(cidx, B=2000)
        all_results.append(r)

    # 2. Legacy data
    print("\n[2/3] Analyzing legacy data (Grover + Magnetism)...")
    grover = analyze_grover()
    magnetism = analyze_magnetism()

    # 3. Summary
    print_summary_table(all_results, grover, magnetism)

    # 4. Figure
    print("\n[3/3] Generating publication figure...")
    make_comprehensive_figure(all_results, grover, magnetism)

    # 5. Save full results
    output = {
        "blind_kappa": [],
        "grover": grover,
        "magnetism": magnetism,
    }
    for r in all_results:
        # Remove large arrays for JSON
        r_slim = {k: v for k, v in r.items() if k != "probs"}
        output["blind_kappa"].append(r_slim)

    out_path = Path(r"D:\code\qfi_sigma_c\comprehensive_results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Full results saved: {out_path}")


if __name__ == "__main__":
    main()
