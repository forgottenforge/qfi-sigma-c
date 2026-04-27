"""
Cross-platform sigma_c vs CFI analysis.

IonQ Forte-1 data:
  R1_sweep: 12 gamma points, 250 shots each (clean uniform grid)
  R23_batch: 2 gamma points (0.0, 0.67), 2000 shots each
  R4_control + R5_communication: supplementary batches

Strategy:
  - sigma_c: use ALL combined data for maximum statistics
  - CFI (entropy gradient): use R1_sweep ONLY (uniform 250 shots)
"""

import json
import numpy as np
from pathlib import Path

TAU = np.exp(-1)
K = 64
N_QUBITS = 6

DATA_DIR = Path(r"d:\code\onto\noise_fingerprints\data\vacuum_telescope_v1")
OUT_DIR = Path(r"d:\code\qfi_sigma_c")


def bitstrings_to_probs(bitstrings, n_outcomes=64):
    counts = np.zeros(n_outcomes)
    for bs in bitstrings:
        counts[int(bs, 2)] += 1
    return counts / len(bitstrings)


def bitstrings_to_hamming_probs(bitstrings, n_qubits=6):
    counts = np.zeros(n_qubits + 1)
    for bs in bitstrings:
        counts[sum(int(c) for c in bs)] += 1
    return counts / len(bitstrings)


def compute_entropy(probs, reg=1e-15):
    p = probs + reg
    p /= p.sum()
    return -np.sum(p * np.log2(p))


def find_sigma_c(gammas, r_vals, tau=TAU):
    threshold = tau * r_vals[0]
    for i in range(len(gammas) - 1):
        if r_vals[i] >= threshold and r_vals[i+1] < threshold:
            frac = (threshold - r_vals[i]) / (r_vals[i+1] - r_vals[i])
            return gammas[i] + frac * (gammas[i+1] - gammas[i])
    return None


def load_ionq_data(filepath):
    """Load IonQ data, separating sweep from batch."""
    with open(filepath) as f:
        data = json.load(f)

    sweep_data = {}  # R1_sweep only (12 gammas, 250 shots each)
    all_data = {}    # all blocks combined

    for block_name, block_data in data["blocks"].items():
        if "measurements" not in block_data:
            continue
        for m in block_data["measurements"]:
            g = m["gamma"]
            bs = m["bitstrings"]

            # Add to all_data
            if g not in all_data:
                all_data[g] = []
            all_data[g].extend(bs)

            # Add to sweep only if R1_sweep
            if block_name == "R1_sweep":
                if g not in sweep_data:
                    sweep_data[g] = []
                sweep_data[g].extend(bs)

    return data["metadata"], sweep_data, all_data


def analyze_ionq(filepath):
    """Full IonQ analysis."""
    metadata, sweep_data, all_data = load_ionq_data(filepath)
    device = metadata["device_label"]

    print(f"\n{'='*60}")
    print(f"Platform: {device}")
    print(f"{'='*60}")

    # --- sigma_c from ALL data ---
    gammas_all = np.array(sorted(all_data.keys()))
    r_all = []
    print(f"\n  Performance curve (all data combined):")
    for g in gammas_all:
        p = bitstrings_to_probs(all_data[g], K)
        r_all.append(np.max(p))
        print(f"    gamma={g:.2f}: {len(all_data[g]):5d} shots, r={np.max(p):.4f}")
    r_all = np.array(r_all)
    sigma_c = find_sigma_c(gammas_all, r_all)
    print(f"  r(0) = {r_all[0]:.4f}, threshold = {TAU*r_all[0]:.4f}")
    print(f"  sigma_c = {sigma_c:.4f}" if sigma_c else "  sigma_c = N/A")

    # --- CFI from R1_sweep only (uniform 250 shots) ---
    gammas_sw = np.array(sorted(sweep_data.keys()))
    print(f"\n  R1_sweep gammas ({len(gammas_sw)} points): {gammas_sw.tolist()}")

    hw_probs = []
    entropies = []
    r_sweep = []
    for g in gammas_sw:
        bs = sweep_data[g]
        hp = bitstrings_to_hamming_probs(bs, N_QUBITS)
        hw_probs.append(hp)
        entropies.append(compute_entropy(hp))
        fp = bitstrings_to_probs(bs, K)
        r_sweep.append(np.max(fp))

    entropies = np.array(entropies)
    r_sweep = np.array(r_sweep)

    # Entropy gradient via central differences
    dH = np.zeros(len(gammas_sw))
    for i in range(len(gammas_sw)):
        if i == 0:
            dH[i] = (entropies[1] - entropies[0]) / (gammas_sw[1] - gammas_sw[0])
        elif i == len(gammas_sw) - 1:
            dH[i] = (entropies[-1] - entropies[-2]) / (gammas_sw[-1] - gammas_sw[-2])
        else:
            dH[i] = (entropies[i+1] - entropies[i-1]) / (gammas_sw[i+1] - gammas_sw[i-1])

    # |dr/dgamma| gradient
    dr = np.zeros(len(gammas_sw))
    for i in range(len(gammas_sw)):
        if i == 0:
            dr[i] = abs(r_sweep[1] - r_sweep[0]) / (gammas_sw[1] - gammas_sw[0])
        elif i == len(gammas_sw) - 1:
            dr[i] = abs(r_sweep[-1] - r_sweep[-2]) / (gammas_sw[-1] - gammas_sw[-2])
        else:
            dr[i] = abs(r_sweep[i+1] - r_sweep[i-1]) / (gammas_sw[i+1] - gammas_sw[i-1])

    # Find peaks (interior only)
    interior_dH = dH[1:-1]
    peak_dH = np.argmax(interior_dH) + 1
    interior_dr = dr[1:-1]
    peak_dr = np.argmax(interior_dr) + 1

    print(f"\n  {'gamma':>6}  {'shots':>5}  {'r':>7}  {'H':>7}  {'dH/dg':>8}  {'|dr/dg|':>8}")
    print(f"  {'-'*52}")
    for i, g in enumerate(gammas_sw):
        markers = ""
        if i == peak_dH: markers += " [dH*]"
        if i == peak_dr: markers += " [dr*]"
        n = len(sweep_data[g])
        print(f"  {g:6.2f}  {n:5d}  {r_sweep[i]:7.4f}  {entropies[i]:7.4f}  "
              f"{dH[i]:8.3f}  {dr[i]:8.4f}{markers}")

    eps_star = gammas_sw[peak_dH]
    kappa = sigma_c / eps_star if (sigma_c and eps_star > 0) else None

    print(f"\n  RESULTS:")
    print(f"  sigma_c       = {sigma_c:.4f}")
    print(f"  eps* (dH/dg)  = {eps_star:.2f}")
    print(f"  eps* (|dr/dg|)= {gammas_sw[peak_dr]:.2f}")
    print(f"  kappa         = {kappa:.3f}" if kappa else "  kappa = N/A")

    return {
        "device": device,
        "n_qubits": metadata["n_qubits"],
        "sigma_c": float(sigma_c) if sigma_c else None,
        "eps_star": float(eps_star),
        "kappa": float(kappa) if kappa else None,
        "gammas_sweep": gammas_sw.tolist(),
        "r_sweep": [float(x) for x in r_sweep],
        "entropies": [float(x) for x in entropies],
        "dH": [float(x) for x in dH],
    }


def bootstrap_ionq(filepath, B=5000, seed=42):
    """Paired bootstrap for sigma_c and entropy-gradient eps*."""
    metadata, sweep_data, all_data = load_ionq_data(filepath)

    gammas_all = np.array(sorted(all_data.keys()))
    gammas_sw = np.array(sorted(sweep_data.keys()))

    # Pre-convert to integer arrays
    all_ints = {g: np.array([int(bs, 2) for bs in all_data[g]]) for g in gammas_all}
    sweep_hws = {g: np.array([sum(int(c) for c in bs) for bs in sweep_data[g]]) for g in gammas_sw}

    rng = np.random.RandomState(seed)
    sc_samples = []
    ep_samples = []

    for b in range(B):
        # sigma_c from all data
        r_vals = []
        for g in gammas_all:
            raw = all_ints[g]
            n = len(raw)
            idx = rng.randint(0, n, size=n)
            counts = np.bincount(raw[idx], minlength=K)
            r_vals.append(np.max(counts) / n)
        r_vals = np.array(r_vals)
        sc = find_sigma_c(gammas_all, r_vals)

        # eps* from sweep entropy gradient
        entropies = []
        for g in gammas_sw:
            raw = sweep_hws[g]
            n = len(raw)
            idx = rng.randint(0, n, size=n)
            hw_counts = np.bincount(raw[idx], minlength=N_QUBITS + 1)
            hp = hw_counts / n + 1e-15
            hp /= hp.sum()
            entropies.append(-np.sum(hp * np.log2(hp)))
        entropies = np.array(entropies)

        dH = np.zeros(len(gammas_sw))
        for i in range(len(gammas_sw)):
            if i == 0:
                dH[i] = (entropies[1] - entropies[0]) / (gammas_sw[1] - gammas_sw[0])
            elif i == len(gammas_sw) - 1:
                dH[i] = (entropies[-1] - entropies[-2]) / (gammas_sw[-1] - gammas_sw[-2])
            else:
                dH[i] = (entropies[i+1] - entropies[i-1]) / (gammas_sw[i+1] - gammas_sw[i-1])

        interior = dH[1:-1]
        ep = gammas_sw[np.argmax(interior) + 1]

        if sc is not None:
            sc_samples.append(sc)
            ep_samples.append(ep)

    sc_samples = np.array(sc_samples)
    ep_samples = np.array(ep_samples)
    delta_samples = sc_samples - ep_samples
    det_rate = len(sc_samples) / B

    delta_ci = [float(np.percentile(delta_samples, 2.5)),
                float(np.percentile(delta_samples, 97.5))]
    sc_ci = [float(np.percentile(sc_samples, 2.5)),
             float(np.percentile(sc_samples, 97.5))]
    ep_ci = [float(np.percentile(ep_samples, 2.5)),
             float(np.percentile(ep_samples, 97.5))]
    kappa_samples = sc_samples / ep_samples
    kappa_ci = [float(np.percentile(kappa_samples, 2.5)),
                float(np.percentile(kappa_samples, 97.5))]
    zero_in_ci = delta_ci[0] <= 0 <= delta_ci[1]

    print(f"\n  Bootstrap (B={B}):")
    print(f"    Detection rate: {det_rate:.3f}")
    print(f"    sigma_c: median={np.median(sc_samples):.4f}, CI {sc_ci}")
    print(f"    eps*:    median={np.median(ep_samples):.4f}, CI {ep_ci}")
    print(f"    Delta CI: {delta_ci}")
    print(f"    Zero in CI: {zero_in_ci}")
    print(f"    kappa: median={np.median(kappa_samples):.3f}, CI {kappa_ci}")

    return {
        "detection_rate": det_rate,
        "sigma_c_ci": sc_ci,
        "eps_star_ci": ep_ci,
        "delta_ci": delta_ci,
        "zero_in_ci": zero_in_ci,
        "median_kappa": float(np.median(kappa_samples)),
        "kappa_ci": kappa_ci,
    }


if __name__ == "__main__":
    results = {}

    # --- IonQ Forte-1 ---
    ionq_file = DATA_DIR / "replication_ionq_qpu_20260201_095758.json"
    if ionq_file.exists():
        r = analyze_ionq(ionq_file)
        if r and r["sigma_c"] is not None:
            results["ionq"] = r
            boot = bootstrap_ionq(ionq_file, B=5000)
            results["ionq"]["bootstrap"] = boot

    # --- Rigetti Ankaa-3 (vacuum telescope) ---
    rigetti_file = DATA_DIR / "replication_rigetti_qpu_20260201_100409.json"
    if rigetti_file.exists():
        with open(rigetti_file) as f:
            rdata = json.load(f)
        measurements = []
        for bname, bdata in rdata["blocks"].items():
            if "measurements" in bdata:
                measurements.extend(bdata["measurements"])
        gdata = {}
        for m in measurements:
            g = m["gamma"]
            if g not in gdata:
                gdata[g] = []
            gdata[g].extend(m["bitstrings"])
        gammas_r = np.array(sorted(gdata.keys()))
        r_vals_r = []
        print(f"\n{'='*60}")
        print(f"Rigetti Ankaa-3 (vacuum telescope)")
        print(f"{'='*60}")
        for g in gammas_r:
            p = bitstrings_to_probs(gdata[g], K)
            r_vals_r.append(np.max(p))
            print(f"  gamma={g:.2f}: {len(gdata[g]):5d} shots, r={np.max(p):.4f}")
        r_vals_r = np.array(r_vals_r)
        sc_r = find_sigma_c(gammas_r, r_vals_r)
        print(f"  sigma_c = {sc_r:.4f}" if sc_r else "  sigma_c = N/A")
        print(f"  (3 gamma points only; CFI not computable)")
        results["rigetti_vt"] = {
            "device": "Rigetti Ankaa-3 (superconducting)",
            "sigma_c": float(sc_r) if sc_r else None,
            "note": "3 gamma points; CFI not computable",
        }

    # --- Save ---
    outfile = OUT_DIR / "cross_platform_results.json"
    with open(outfile, "w") as f:
        json.dump(results, f, indent=2)

    # --- Summary ---
    print(f"\n\n{'='*60}")
    print("FINAL SUMMARY")
    print("="*60)
    if "ionq" in results:
        r = results["ionq"]
        b = r.get("bootstrap", {})
        print(f"\nIonQ Forte-1 (trapped ion, 6 qubits):")
        print(f"  sigma_c = {r['sigma_c']:.4f}")
        print(f"  eps*    = {r['eps_star']:.2f}")
        print(f"  kappa   = {r['kappa']:.3f}")
        if b:
            print(f"  Bootstrap kappa = {b['median_kappa']:.3f} "
                  f"CI [{b['kappa_ci'][0]:.3f}, {b['kappa_ci'][1]:.3f}]")
            print(f"  Zero in delta CI: {b['zero_in_ci']}")

    if "rigetti_vt" in results:
        r = results["rigetti_vt"]
        print(f"\nRigetti Ankaa-3 (vacuum telescope):")
        print(f"  sigma_c = {r['sigma_c']:.4f}" if r["sigma_c"] else "  N/A")

    print(f"\nSaved to {outfile}")
