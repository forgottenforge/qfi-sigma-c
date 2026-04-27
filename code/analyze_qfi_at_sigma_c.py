"""
QFI vs sigma_c Analysis
========================
Central question: Is the sigma_c critical point the point of maximum
Quantum Fisher Information?

If yes → sigma_c detects the noise-driven "phase transition" where the
quantum state changes most rapidly. This connects to Zanardi & Paunovic
(PRL 2006): QFI peaks at quantum critical points.

If no → sigma_c captures something different from QFI, which is also
scientifically interesting.

Approach:
1. Classical Fisher Information (CFI) from Grover probability data
   - We have full p(x|epsilon) for x in {00,01,10,11} at 19 noise levels
   - CFI(epsilon) = sum_x [dp/depsilon]^2 / p(x|epsilon)
   - CFI is a lower bound on QFI

2. Susceptibility chi as QFI proxy for quantum magnetism data
   - chi propto QFI for thermal/noise-driven transitions
   - We have chi(epsilon) for E3 entanglement data

3. Fidelity-based QFI estimate
   - QFI ~ -8 ln F(epsilon, epsilon+delta) / delta^2
   - Computable from probability distributions

Author: Matthias Christian Wurm / ForgottenForge
Date: April 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. LOAD ALL DATA
# ============================================================

def load_grover_data():
    """Load Grover stability data with full probability distributions."""
    path = Path(r"D:\code\clco\noend\grover_stability_data_20250728_211356.json")
    with open(path) as f:
        data = json.load(f)

    noise_levels = np.array(data["noise_levels"])
    original = data["original_data"]

    # Extract probability matrices: shape (n_noise, n_outcomes)
    outcomes = ["00", "01", "10", "11"]
    probs = np.array([[d["probabilities"][o] for o in outcomes] for d in original])
    success_rates = np.array([d["success_rate"] for d in original])

    return noise_levels, probs, success_rates


def load_magnetism_data():
    """Load quantum magnetism data with sigma_c and chi values."""
    path = Path(r"D:\code\onto\data\quantum_magnetism_complete_data.json")
    with open(path) as f:
        data = json.load(f)
    return data


# ============================================================
# 2. CLASSICAL FISHER INFORMATION
# ============================================================

def classical_fisher_information(noise_levels, probs, regularize=1e-10):
    """
    Compute Classical Fisher Information F_C(epsilon) from probability data.

    F_C(epsilon) = sum_x [dp(x|epsilon)/depsilon]^2 / p(x|epsilon)

    Uses central differences for the derivative, forward/backward at edges.

    Parameters
    ----------
    noise_levels : array of shape (N,)
    probs : array of shape (N, K) where K = number of outcomes
    regularize : small constant to avoid division by zero

    Returns
    -------
    epsilon_cfi : noise levels where CFI is defined (interior points)
    cfi : Classical Fisher Information values
    """
    N, K = probs.shape

    # Regularize probabilities
    probs_reg = probs + regularize
    probs_reg = probs_reg / probs_reg.sum(axis=1, keepdims=True)

    # Compute dp/depsilon via central differences
    dp = np.zeros_like(probs_reg)
    for i in range(N):
        if i == 0:
            dp[i] = (probs_reg[i+1] - probs_reg[i]) / (noise_levels[i+1] - noise_levels[i])
        elif i == N - 1:
            dp[i] = (probs_reg[i] - probs_reg[i-1]) / (noise_levels[i] - noise_levels[i-1])
        else:
            dp[i] = (probs_reg[i+1] - probs_reg[i-1]) / (noise_levels[i+1] - noise_levels[i-1])

    # CFI = sum_x (dp/deps)^2 / p(x)
    cfi = np.sum(dp**2 / probs_reg, axis=1)

    return noise_levels, cfi


def fidelity_based_qfi(noise_levels, probs, regularize=1e-10):
    """
    Estimate QFI from Bhattacharyya fidelity between adjacent distributions.

    F(eps, eps+delta) = [sum_x sqrt(p(x|eps) * p(x|eps+delta))]^2
    QFI ~ -8 * ln(F) / delta^2

    Returns QFI at midpoints between adjacent noise levels.
    """
    N, K = probs.shape
    probs_reg = probs + regularize
    probs_reg = probs_reg / probs_reg.sum(axis=1, keepdims=True)

    midpoints = []
    qfi_est = []

    for i in range(N - 1):
        delta = noise_levels[i+1] - noise_levels[i]
        if delta == 0:
            continue

        # Bhattacharyya coefficient
        bc = np.sum(np.sqrt(probs_reg[i] * probs_reg[i+1]))

        # Fidelity
        F = bc**2

        if F > 0 and F < 1:
            qfi = -8.0 * np.log(F) / delta**2
        else:
            qfi = 0.0

        midpoints.append((noise_levels[i] + noise_levels[i+1]) / 2)
        qfi_est.append(qfi)

    return np.array(midpoints), np.array(qfi_est)


# ============================================================
# 3. SIGMA_C DETECTION
# ============================================================

def find_sigma_c(noise_levels, performance, tau=0.5):
    """Find sigma_c: the noise level where performance drops to tau * performance(0)."""
    threshold = tau * performance[0]
    for i, p in enumerate(performance):
        if p < threshold:
            # Linear interpolation
            if i > 0:
                eps_lo, eps_hi = noise_levels[i-1], noise_levels[i]
                p_lo, p_hi = performance[i-1], performance[i]
                sigma_c = eps_lo + (threshold - p_lo) / (p_hi - p_lo) * (eps_hi - eps_lo)
                return sigma_c
            return noise_levels[i]
    return noise_levels[-1]


# ============================================================
# 4. ANALYSIS AND PLOTTING
# ============================================================

def analyze_grover():
    """Full Grover analysis: CFI, fidelity-QFI, sigma_c comparison."""
    noise_levels, probs, success_rates = load_grover_data()

    # Compute sigma_c
    sigma_c = find_sigma_c(noise_levels, success_rates, tau=0.5)
    print(f"\n{'='*60}")
    print(f"GROVER ALGORITHM ANALYSIS")
    print(f"{'='*60}")
    print(f"sigma_c (tau=0.5): {sigma_c:.4f}")

    # Compute CFI
    eps_cfi, cfi = classical_fisher_information(noise_levels, probs)
    cfi_peak_idx = np.argmax(cfi)
    cfi_peak_eps = eps_cfi[cfi_peak_idx]
    print(f"CFI peak at epsilon = {cfi_peak_eps:.4f}")
    print(f"CFI peak value = {cfi[cfi_peak_idx]:.2f}")
    print(f"|sigma_c - CFI_peak| = {abs(sigma_c - cfi_peak_eps):.4f}")

    # Compute fidelity-based QFI
    eps_qfi, qfi = fidelity_based_qfi(noise_levels, probs)
    qfi_peak_idx = np.argmax(qfi)
    qfi_peak_eps = eps_qfi[qfi_peak_idx]
    print(f"Fidelity-QFI peak at epsilon = {qfi_peak_eps:.4f}")
    print(f"|sigma_c - QFI_peak| = {abs(sigma_c - qfi_peak_eps):.4f}")

    # Verdict
    grid_spacing = noise_levels[1] - noise_levels[0]
    coincidence_cfi = abs(sigma_c - cfi_peak_eps) <= 2 * grid_spacing
    coincidence_qfi = abs(sigma_c - qfi_peak_eps) <= 2 * grid_spacing
    print(f"\nGrid spacing: {grid_spacing:.3f}")
    print(f"CFI peak coincides with sigma_c (within 2 grid points): {coincidence_cfi}")
    print(f"QFI peak coincides with sigma_c (within 2 grid points): {coincidence_qfi}")

    return {
        "noise_levels": noise_levels,
        "success_rates": success_rates,
        "sigma_c": sigma_c,
        "eps_cfi": eps_cfi,
        "cfi": cfi,
        "cfi_peak": cfi_peak_eps,
        "eps_qfi": eps_qfi,
        "qfi": qfi,
        "qfi_peak": qfi_peak_eps,
    }


def analyze_magnetism():
    """Analyze quantum magnetism E3 data: chi peak vs sigma_c."""
    data = load_magnetism_data()

    # E3: Entanglement timescales
    e3 = data["experiments"]["E3_entanglement_timescales"]
    noise_levels = np.array(e3["noise_levels"])
    chi = np.array(e3["chi"])
    sigma_c = e3["sigma_c"]
    kappa = e3["kappa"]
    witness = np.array(e3["entanglement_witness"])

    print(f"\n{'='*60}")
    print(f"E3 ENTANGLEMENT (6-QUBIT ISING, RIGETTI)")
    print(f"{'='*60}")
    print(f"sigma_c: {sigma_c:.4f}")
    print(f"kappa: {kappa:.2f}")

    chi_peak_idx = np.argmax(chi)
    chi_peak_eps = noise_levels[chi_peak_idx]
    print(f"chi peak at epsilon = {chi_peak_eps:.4f}")
    print(f"chi peak value = {chi[chi_peak_idx]:.4f}")
    print(f"|sigma_c - chi_peak| = {abs(sigma_c - chi_peak_eps):.4f}")

    grid_spacing = noise_levels[1] - noise_levels[0]
    coincidence = abs(sigma_c - chi_peak_eps) <= 1 * grid_spacing
    print(f"Grid spacing: {grid_spacing:.4f}")
    print(f"Chi peak coincides with sigma_c (within 1 grid point): {coincidence}")

    # E5: Phase transition
    e5 = data["experiments"]["E5_phase_transition"]
    fields = np.array(e5["fields"])
    zz = np.array(e5["zz_correlations"])
    sigma_c_field = e5["sigma_c_field"]

    # Compute susceptibility from ZZ correlations (numerical derivative)
    dzz = np.gradient(zz, fields)
    chi_zz = np.abs(dzz)

    print(f"\n{'='*60}")
    print(f"E5 PHASE TRANSITION (TFIM, RIGETTI)")
    print(f"{'='*60}")
    print(f"sigma_c (field): {sigma_c_field:.4f}")

    chi_zz_peak_idx = np.argmax(chi_zz)
    chi_zz_peak = fields[chi_zz_peak_idx]
    print(f"|dZZ/dh| peak at h = {chi_zz_peak:.4f}")
    print(f"|sigma_c - chi_peak| = {abs(sigma_c_field - chi_zz_peak):.4f}")

    # E6: Decoherence
    e6 = data["experiments"]["E6_decoherence"]
    damping = np.array(e6["damping_rates"])
    witnesses = np.array(e6["witnesses"])
    sigma_c_e6 = e6["sigma_c"]

    # Susceptibility from witness
    dw = np.gradient(witnesses, damping)
    chi_w = np.abs(dw)

    print(f"\n{'='*60}")
    print(f"E6 DECOHERENCE (GHZ, RIGETTI)")
    print(f"{'='*60}")
    print(f"sigma_c: {sigma_c_e6:.4f}")

    chi_w_peak_idx = np.argmax(chi_w)
    chi_w_peak = damping[chi_w_peak_idx]
    print(f"|dW/dgamma| peak at gamma = {chi_w_peak:.4f}")
    print(f"|sigma_c - chi_peak| = {abs(sigma_c_e6 - chi_w_peak):.4f}")

    return {
        "e3": {
            "noise_levels": noise_levels, "chi": chi, "sigma_c": sigma_c,
            "witness": witness, "kappa": kappa, "chi_peak": chi_peak_eps,
        },
        "e5": {
            "fields": fields, "zz": zz, "chi_zz": chi_zz,
            "sigma_c": sigma_c_field, "chi_peak": chi_zz_peak,
        },
        "e6": {
            "damping": damping, "witnesses": witnesses, "chi_w": chi_w,
            "sigma_c": sigma_c_e6, "chi_peak": chi_w_peak,
        },
    }


def make_figure(grover_results, magnetism_results):
    """Create the master figure: 4-panel comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        r"Is $\sigma_c$ the Point of Maximum Fisher Information?",
        fontsize=16, fontweight="bold", y=0.98
    )

    # ---- Panel A: Grover CFI ----
    ax = axes[0, 0]
    g = grover_results

    ax2 = ax.twinx()
    ax.plot(g["noise_levels"], g["success_rates"], "b-o", ms=4, label="Success rate")
    ax2.plot(g["eps_cfi"], g["cfi"], "r-s", ms=4, label="CFI")
    ax.axvline(g["sigma_c"], color="green", ls="--", lw=2, label=f'$\\sigma_c$ = {g["sigma_c"]:.3f}')
    ax.axvline(g["cfi_peak"], color="red", ls=":", lw=2, alpha=0.7)

    ax.set_xlabel(r"Depolarizing noise $\epsilon$")
    ax.set_ylabel("Success rate", color="blue")
    ax2.set_ylabel("Classical Fisher Information", color="red")
    ax.set_title("(a) Grover Algorithm: CFI vs $\\sigma_c$")
    ax.legend(loc="upper right", fontsize=8)

    # ---- Panel B: Grover Fidelity-QFI ----
    ax = axes[0, 1]
    ax2 = ax.twinx()
    ax.plot(g["noise_levels"], g["success_rates"], "b-o", ms=4, label="Success rate")
    ax2.plot(g["eps_qfi"], g["qfi"], "m-^", ms=4, label="Fidelity-QFI")
    ax.axvline(g["sigma_c"], color="green", ls="--", lw=2, label=f'$\\sigma_c$ = {g["sigma_c"]:.3f}')
    ax.axvline(g["qfi_peak"], color="purple", ls=":", lw=2, alpha=0.7)

    ax.set_xlabel(r"Depolarizing noise $\epsilon$")
    ax.set_ylabel("Success rate", color="blue")
    ax2.set_ylabel("Fidelity-based QFI estimate", color="purple")
    ax.set_title("(b) Grover Algorithm: Fidelity-QFI vs $\\sigma_c$")
    ax.legend(loc="upper right", fontsize=8)

    # ---- Panel C: E3 Entanglement chi ----
    ax = axes[1, 0]
    m3 = magnetism_results["e3"]

    ax2 = ax.twinx()
    ax.plot(m3["noise_levels"], m3["witness"], "b-o", ms=4, label="Ent. witness")
    ax2.plot(m3["noise_levels"], m3["chi"], "r-s", ms=4, label=r"$\chi$ (susceptibility)")
    ax.axvline(m3["sigma_c"], color="green", ls="--", lw=2,
               label=f'$\\sigma_c$ = {m3["sigma_c"]:.3f}')
    ax.axvline(m3["chi_peak"], color="red", ls=":", lw=2, alpha=0.7)
    ax.axhline(0, color="gray", ls="-", lw=0.5)

    ax.set_xlabel(r"Depolarizing noise $\epsilon$")
    ax.set_ylabel("Entanglement witness", color="blue")
    ax2.set_ylabel(r"Susceptibility $\chi$", color="red")
    ax.set_title(r"(c) 6Q Ising: $\chi$ peak vs $\sigma_c$")
    ax.legend(loc="lower left", fontsize=8)

    # ---- Panel D: E5 TFIM Phase Transition ----
    ax = axes[1, 1]
    m5 = magnetism_results["e5"]

    ax2 = ax.twinx()
    ax.plot(m5["fields"], m5["zz"], "b-o", ms=4, label=r"$\langle ZZ \rangle$")
    ax2.plot(m5["fields"], m5["chi_zz"], "r-s", ms=4, label=r"$|d\langle ZZ\rangle/dh|$")
    ax.axvline(m5["sigma_c"], color="green", ls="--", lw=2,
               label=f'$\\sigma_c$ = {m5["sigma_c"]:.3f}')
    ax.axvline(m5["chi_peak"], color="red", ls=":", lw=2, alpha=0.7)

    ax.set_xlabel(r"Transverse field $h$")
    ax.set_ylabel(r"$\langle ZZ \rangle$ correlation", color="blue")
    ax2.set_ylabel(r"$|d\langle ZZ\rangle/dh|$ (susceptibility proxy)", color="red")
    ax.set_title(r"(d) TFIM: Susceptibility vs $\sigma_c$")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outpath = Path(r"D:\code\qfi_sigma_c\qfi_vs_sigma_c.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\nFigure saved: {outpath}")
    print(f"Figure saved: {outpath.with_suffix('.pdf')}")
    plt.close()


def summary_table(grover_results, magnetism_results):
    """Print the decisive summary table."""
    m = magnetism_results
    g = grover_results

    print(f"\n{'='*70}")
    print(f"SUMMARY: sigma_c vs Maximum Fisher Information")
    print(f"{'='*70}")
    print(f"{'System':<25} {'sigma_c':>10} {'FI peak':>10} {'|Delta|':>10} {'Match?':>8}")
    print(f"{'-'*70}")

    rows = [
        ("Grover (CFI)", g["sigma_c"], g["cfi_peak"], abs(g["sigma_c"] - g["cfi_peak"])),
        ("Grover (Fid-QFI)", g["sigma_c"], g["qfi_peak"], abs(g["sigma_c"] - g["qfi_peak"])),
        ("E3 Ising (chi)", m["e3"]["sigma_c"], m["e3"]["chi_peak"],
         abs(m["e3"]["sigma_c"] - m["e3"]["chi_peak"])),
        ("E5 TFIM (dZZ/dh)", m["e5"]["sigma_c"], m["e5"]["chi_peak"],
         abs(m["e5"]["sigma_c"] - m["e5"]["chi_peak"])),
        ("E6 GHZ (dW/dgamma)", m["e6"]["sigma_c"], m["e6"]["chi_peak"],
         abs(m["e6"]["sigma_c"] - m["e6"]["chi_peak"])),
    ]

    for name, sc, fi, delta in rows:
        match = "YES" if delta < 0.1 else "CLOSE" if delta < 0.3 else "NO"
        print(f"{name:<25} {sc:>10.4f} {fi:>10.4f} {delta:>10.4f} {match:>8}")

    print(f"\n{'='*70}")
    print("VERDICT:")
    matches = sum(1 for _, sc, fi, d in rows if d < 0.1)
    close = sum(1 for _, sc, fi, d in rows if 0.1 <= d < 0.3)
    total = len(rows)
    print(f"  Exact matches (|Delta| < 0.1): {matches}/{total}")
    print(f"  Close matches (|Delta| < 0.3): {matches + close}/{total}")

    if matches >= 3:
        print("\n  >>> sigma_c IS the point of maximum Fisher Information <<<")
        print("  >>> The bridge to Zanardi/Paunovic (2006) exists! <<<")
    elif matches + close >= 3:
        print("\n  >>> sigma_c is CLOSE to maximum FI — further investigation needed <<<")
    else:
        print("\n  >>> sigma_c is NOT the point of maximum FI <<<")
        print("  >>> sigma_c detects something DIFFERENT from QFI <<<")


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    print("QFI vs sigma_c Analysis")
    print("=" * 60)
    print("Question: Is sigma_c the point of maximum Quantum Fisher Information?")
    print("Reference: Zanardi & Paunovic, PRL 96, 250403 (2006)")
    print()

    grover_results = analyze_grover()
    magnetism_results = analyze_magnetism()

    summary_table(grover_results, magnetism_results)
    make_figure(grover_results, magnetism_results)
