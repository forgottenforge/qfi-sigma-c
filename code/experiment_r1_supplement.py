#!/usr/bin/env python3
"""
R1 SUPPLEMENTARY EXPERIMENTS FOR PRA PAPER
===========================================
Four experimental blocks addressing reviewer points M5 and M6:

  Block A: Symmetry-breaking control (M6)
           3 C02-Heisenberg variants + 3 C13-tight-binding variants
           ~72 tasks, ~$34

  Block B: N=8 scaling study (M5)
           9 circuits at 8 qubits, 12 depths
           ~108 tasks, ~$51

  Block C: Finer depth resolution
           5 key circuits at 26 depths (every integer 0..25)
           ~130 tasks, ~$62

  Block D: Additional N=6 circuits
           8 new Hamiltonians, 12 depths
           ~96 tasks, ~$46

Usage:
  python experiment_r1_supplement.py --validate             # gate counts + cost
  python experiment_r1_supplement.py --simulate --block A   # local sim, one block
  python experiment_r1_supplement.py --simulate --all       # local sim, all blocks
  python experiment_r1_supplement.py --qpu --block A        # QPU, one block
  python experiment_r1_supplement.py --qpu --all            # QPU, all blocks

Author: M. C. Wurm
Date: April 2026
"""

import argparse
import json
import numpy as np
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from braket.circuits import Circuit
    from braket.aws import AwsDevice
    from braket.devices import LocalSimulator
    HAS_BRAKET = True
except ImportError:
    HAS_BRAKET = False
    print("WARNING: braket not installed. Only --validate (no gate counts) available.")

# =====================================================================
# CONSTANTS
# =====================================================================

DT = 0.2
SHOTS = 500
COST_PER_TASK = 0.30
COST_PER_SHOT = 0.000425      # Apr 2026 Rigetti Cepheus-1 pricing
BUDGET_USD = 250.0
RIGETTI_ARN = "arn:aws:braket:us-west-1::device/qpu/rigetti/Cepheus-1-108Q"

DEPTHS_STANDARD = [0, 1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 25]
DEPTHS_FINE = list(range(26))  # 0, 1, 2, ..., 25

OUTPUT_DIR = Path(r"d:\code\qfi_sigma_c")
CHECKPOINT_FILE = OUTPUT_DIR / "r1_supplement_checkpoint.json"


# =====================================================================
# GATE PRIMITIVES  (from experiment_blind_qpu.py)
# =====================================================================
# All implement exp(-i * theta * P) for Pauli string P.
# Braket: rz(q, angle) = exp(-i * angle/2 * Z)
# So exp(-i * theta * Z) needs rz(q, 2*theta).

def add_zz(circ, q1, q2, theta):
    """exp(-i * theta * Z_q1 Z_q2)"""
    circ.cnot(int(q1), int(q2))
    circ.rz(int(q2), 2.0 * theta)
    circ.cnot(int(q1), int(q2))

def add_xx(circ, q1, q2, theta):
    """exp(-i * theta * X_q1 X_q2)"""
    circ.h(int(q1)); circ.h(int(q2))
    add_zz(circ, q1, q2, theta)
    circ.h(int(q1)); circ.h(int(q2))

def add_yy(circ, q1, q2, theta):
    """exp(-i * theta * Y_q1 Y_q2)"""
    circ.rx(int(q1), -np.pi / 2); circ.rx(int(q2), -np.pi / 2)
    add_zz(circ, q1, q2, theta)
    circ.rx(int(q1), np.pi / 2); circ.rx(int(q2), np.pi / 2)

def add_z(circ, q, theta):
    """exp(-i * theta * Z_q)"""
    circ.rz(int(q), 2.0 * theta)

def add_x(circ, q, theta):
    """exp(-i * theta * X_q)"""
    circ.rx(int(q), 2.0 * theta)

def add_zzz(circ, q1, q2, q3, theta):
    """exp(-i * theta * Z_q1 Z_q2 Z_q3)"""
    circ.cnot(int(q1), int(q3))
    circ.cnot(int(q2), int(q3))
    circ.rz(int(q3), 2.0 * theta)
    circ.cnot(int(q2), int(q3))
    circ.cnot(int(q1), int(q3))

def add_xzx(circ, q1, q2, q3, theta):
    """exp(-i * theta * X_q1 Z_q2 X_q3)"""
    circ.h(int(q1)); circ.h(int(q3))
    add_zzz(circ, q1, q2, q3, theta)
    circ.h(int(q1)); circ.h(int(q3))

def add_zx(circ, q1, q2, theta):
    """exp(-i * theta * Z_q1 X_q2)"""
    circ.h(int(q2))
    add_zz(circ, q1, q2, theta)
    circ.h(int(q2))

def add_xz(circ, q1, q2, theta):
    """exp(-i * theta * X_q1 Z_q2)"""
    circ.h(int(q1))
    add_zz(circ, q1, q2, theta)
    circ.h(int(q1))

def add_xx_chain(circ, qubits, theta):
    """exp(-i * theta * X_{q0}...X_{qn})"""
    target = int(qubits[-1])
    for q in qubits:
        circ.h(int(q))
    for q in qubits[:-1]:
        circ.cnot(int(q), target)
    circ.rz(target, 2.0 * theta)
    for q in reversed(qubits[:-1]):
        circ.cnot(int(q), target)
    for q in qubits:
        circ.h(int(q))


# =====================================================================
# TROTTER STEP BUILDERS — N=6 (existing + new)
# =====================================================================

# --- Existing builders (reused for Block C and as baselines) ---

def trotter_heisenberg(circ, s, n_qubits=6):
    """H = -J/4 sum (XX+YY+ZZ). J=1. SU(2) symmetric."""
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -0.25 * s)
        add_yy(circ, i, i + 1, -0.25 * s)
        add_zz(circ, i, i + 1, -0.25 * s)

def trotter_tight_binding(circ, s, n_qubits=6):
    """H = -t/2 sum (XX+YY). t=1. U(1) symmetric."""
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -0.5 * s)
        add_yy(circ, i, i + 1, -0.5 * s)

def trotter_xxz(circ, s, n_qubits=6):
    """H = -Jz/4 sum ZZ - Jxy/4 sum (XX+YY). Jz=2, Jxy=0.5."""
    for i in range(n_qubits - 1):
        add_zz(circ, i, i + 1, -0.5 * s)
        add_xx(circ, i, i + 1, -0.125 * s)
        add_yy(circ, i, i + 1, -0.125 * s)

def trotter_tfim(circ, s, J, h, n_qubits=6):
    """H = -J sum ZZ - h sum X."""
    for i in range(n_qubits - 1):
        add_zz(circ, i, i + 1, -J * s)
    for i in range(n_qubits):
        add_x(circ, i, -h * s)

def trotter_kitaev_detuned(circ, s, n_qubits=6):
    """H = +0.25*sum Z_i - sum XX (t=1, delta=1, mu=0.5)."""
    for i in range(n_qubits):
        add_z(circ, i, 0.25 * s)
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -1.0 * s)

def trotter_cluster_spt(circ, s, n_qubits=6):
    """H = -sum X_{i-1} Z_i X_{i+1} (bulk) + boundary terms."""
    for i in range(1, n_qubits - 1):
        add_xzx(circ, i - 1, i, i + 1, -1.0 * s)
    add_zx(circ, 0, 1, -1.0 * s)
    add_xz(circ, n_qubits - 2, n_qubits - 1, -1.0 * s)

def trotter_ghz_creating(circ, s, n_qubits=6):
    """H = -J sum ZZ - h X0X1...Xn. J=1, h=0.5."""
    for i in range(n_qubits - 1):
        add_zz(circ, i, i + 1, -1.0 * s)
    add_xx_chain(circ, list(range(n_qubits)), -0.5 * s)

def trotter_all_to_all(circ, s, n_qubits=6):
    """H = -(J/n) sum_{i<j} ZZ - h sum X. J=1, h=0.5."""
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            add_zz(circ, i, j, -(1.0 / n_qubits) * s)
    for i in range(n_qubits):
        add_x(circ, i, -0.5 * s)


# --- Block A: Symmetry-breaking variants ---

def trotter_heis_aniso(circ, s, Jz_over4, n_qubits=6):
    """H = -Jxy/4 sum(XX+YY) - Jz/4 sum ZZ. Jxy/4=0.25 fixed."""
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -0.25 * s)
        add_yy(circ, i, i + 1, -0.25 * s)
        add_zz(circ, i, i + 1, -Jz_over4 * s)

def trotter_tb_with_zz(circ, s, Vz, n_qubits=6):
    """H = -t/2 sum(XX+YY) - Vz sum ZZ. t=1."""
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -0.5 * s)
        add_yy(circ, i, i + 1, -0.5 * s)
        add_zz(circ, i, i + 1, -Vz * s)


# --- Block B: BCS at N=8 ---

def trotter_bcs(circ, s, n_qubits=6):
    """BCS: -(D/2)(XX-YY) pairing + hopping + density-density."""
    n_pairs = n_qubits // 2
    # Pairing within each pair (0,1), (2,3), ...
    for p in range(n_pairs):
        p0, p1 = 2 * p, 2 * p + 1
        add_xx(circ, p0, p1, -0.5 * s)
        add_yy(circ, p0, p1, +0.5 * s)
    # Hopping between adjacent pairs
    for p in range(n_pairs - 1):
        b0, b1 = 2 * p + 1, 2 * (p + 1)
        add_xx(circ, b0, b1, -0.15 * s)
        add_yy(circ, b0, b1, -0.15 * s)
    # Density-density between adjacent pairs
    for p in range(n_pairs - 1):
        for a in [2 * p, 2 * p + 1]:
            for b in [2 * (p + 1), 2 * (p + 1) + 1]:
                add_zz(circ, a, b, +0.125 * s)


# --- Block D: New Hamiltonians ---

def trotter_j1j2(circ, s, J1=1.0, J2=0.5, n_qubits=6):
    """Frustrated J1-J2 Heisenberg chain.
    H = -J1/4 sum_nn(XX+YY+ZZ) - J2/4 sum_nnn(XX+YY+ZZ)."""
    # Nearest-neighbor
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -(J1 / 4) * s)
        add_yy(circ, i, i + 1, -(J1 / 4) * s)
        add_zz(circ, i, i + 1, -(J1 / 4) * s)
    # Next-nearest-neighbor
    for i in range(n_qubits - 2):
        add_xx(circ, i, i + 2, -(J2 / 4) * s)
        add_yy(circ, i, i + 2, -(J2 / 4) * s)
        add_zz(circ, i, i + 2, -(J2 / 4) * s)

def trotter_xy_zfield(circ, s, J=1.0, h=0.5, n_qubits=6):
    """XY model with longitudinal Z-field.
    H = -J/4 sum(XX+YY) - h sum Z."""
    for i in range(n_qubits - 1):
        add_xx(circ, i, i + 1, -(J / 4) * s)
        add_yy(circ, i, i + 1, -(J / 4) * s)
    for i in range(n_qubits):
        add_z(circ, i, -h * s)

def trotter_alt_bond_ising(circ, s, J1=1.0, J2=0.3, h=0.5, n_qubits=6):
    """Alternating-bond transverse-field Ising.
    H = -J1 ZZ_even - J2 ZZ_odd - h X."""
    for i in range(n_qubits - 1):
        J = J1 if (i % 2 == 0) else J2
        add_zz(circ, i, i + 1, -J * s)
    for i in range(n_qubits):
        add_x(circ, i, -h * s)

def trotter_compass(circ, s, Jx=1.0, Jz=1.0, n_qubits=6):
    """Compass model: -Jx XX on even bonds, -Jz ZZ on odd bonds."""
    for i in range(n_qubits - 1):
        if i % 2 == 0:
            add_xx(circ, i, i + 1, -Jx * s)
        else:
            add_zz(circ, i, i + 1, -Jz * s)

def build_random_circuit(depth, n_qubits, seed):
    """Build random layered circuit (no Trotter, no norm scaling)."""
    circ = Circuit()
    if depth > 0:
        rng = np.random.RandomState(seed)
        for _ in range(depth):
            available = list(range(n_qubits))
            rng.shuffle(available)
            for k in range(n_qubits // 2):
                circ.cnot(int(available[2 * k]), int(available[2 * k + 1]))
            for q in range(n_qubits):
                circ.rz(int(q), float(rng.uniform(-np.pi, np.pi)))
                circ.rx(int(q), float(rng.uniform(-np.pi, np.pi)))
    return circ


# =====================================================================
# FROBENIUS NORMS
# =====================================================================
# ||H||_F = 2^(n/2) * sqrt(sum of squared Pauli coefficients)
# For n=6: factor = 8;  for n=8: factor = 16.

def _norm(n_qubits, sum_sq_coeffs):
    return 2 ** (n_qubits / 2) * np.sqrt(sum_sq_coeffs)

# --- N=6 norms ---
NORMS_N6 = {
    # Existing circuits (verified against experiment_blind_qpu.py)
    'heisenberg':     _norm(6, 5 * 3 * 0.25**2),            # 5 bonds, XX+YY+ZZ, coeff 0.25
    'tight_binding':  _norm(6, 5 * 2 * 0.5**2),             # 5 bonds, XX+YY, coeff 0.5
    'xxz_aniso':      _norm(6, 5 * (0.5**2 + 2 * 0.125**2)),  # Jz/4=0.5, Jxy/4=0.125
    'tfim_ordered':   _norm(6, 5 * 1.0**2 + 6 * 0.1**2),    # J=1, h=0.1
    'kitaev_detuned': _norm(6, 5 * 1.0**2 + 6 * 0.25**2),   # XX coeff 1.0, Z coeff 0.25
    'cluster_spt':    _norm(6, 4 * 1.0**2 + 2 * 1.0**2),    # 4 XZX + 2 boundary
    'ghz_creating':   _norm(6, 5 * 1.0**2 + 0.5**2),        # ZZ coeff 1.0, X-chain coeff 0.5
    'all_to_all':     _norm(6, 15 * (1.0/6)**2 + 6 * 0.5**2),  # 15 pairs, 6 X
    'tfim_critical':  _norm(6, 5 * 1.0**2 + 6 * 1.0**2),    # J=1, h=1
    'tfim_paramag':   _norm(6, 5 * 1.0**2 + 6 * 3.0**2),    # J=1, h=3
    'tfim_deep':      _norm(6, 5 * 5.0**2 + 6 * 1.0**2),    # J=5, h=1
    'bcs_6qubit':     _norm(6, 3*(0.5**2+0.5**2) + 2*(0.15**2+0.15**2) + 8*0.125**2),

    # Block A: Symmetry-breaking variants
    'heis_aniso_11':  _norm(6, 5 * (0.25**2 + 0.25**2 + 0.275**2)),
    'heis_aniso_15':  _norm(6, 5 * (0.25**2 + 0.25**2 + 0.375**2)),
    'heis_aniso_20':  _norm(6, 5 * (0.25**2 + 0.25**2 + 0.50**2)),
    'tb_vz01':        _norm(6, 5 * (0.5**2 + 0.5**2 + 0.1**2)),
    'tb_vz03':        _norm(6, 5 * (0.5**2 + 0.5**2 + 0.3**2)),
    'tb_vz05':        _norm(6, 5 * (0.5**2 + 0.5**2 + 0.5**2)),

    # Block D: New Hamiltonians
    'tfim_h05':       _norm(6, 5 * 1.0**2 + 6 * 0.5**2),    # J=1, h=0.5
    'tfim_h20':       _norm(6, 5 * 1.0**2 + 6 * 2.0**2),    # J=1, h=2.0
    'xxz_moderate':   _norm(6, 5 * (0.375**2 + 0.25**2 + 0.25**2)),  # Jz/4=0.375, Jxy/4=0.25
    'j1j2_frustrated': _norm(6, 5*3*0.25**2 + 4*3*0.125**2),  # 5 NN + 4 NNN
    'xy_zfield':      _norm(6, 5 * 2 * 0.25**2 + 6 * 0.5**2),  # J/4=0.25 XX+YY, h=0.5 Z
    'alt_bond_ising': _norm(6, 3*1.0**2 + 2*0.3**2 + 6*0.5**2),  # 3 even + 2 odd + 6 X
    'compass_model':  _norm(6, 3 * 1.0**2 + 2 * 1.0**2),    # 3 XX + 2 ZZ
}

# --- N=8 norms ---
NORMS_N8 = {
    'heisenberg_n8':     _norm(8, 7 * 3 * 0.25**2),
    'tight_binding_n8':  _norm(8, 7 * 2 * 0.5**2),
    'xxz_aniso_n8':      _norm(8, 7 * (0.5**2 + 2 * 0.125**2)),
    'tfim_ordered_n8':   _norm(8, 7 * 1.0**2 + 8 * 0.1**2),
    'kitaev_detuned_n8': _norm(8, 7 * 1.0**2 + 8 * 0.25**2),
    'bcs_8qubit':        _norm(8, 4*(0.5**2+0.5**2) + 3*(0.15**2+0.15**2) + 12*0.125**2),
    'cluster_spt_n8':    _norm(8, 6 * 1.0**2 + 2 * 1.0**2),  # 6 bulk XZX + 2 boundary
}


# =====================================================================
# CIRCUIT REGISTRY
# =====================================================================
# Each entry: (label, n_qubits, category, builder_fn, norm_key, depths, description)

def _make_block_a():
    """M6: Symmetry-breaking control experiments."""
    circuits = []
    # C02 (Heisenberg) variants — break SU(2) by varying ZZ
    for Jz4, label in [(0.275, 'heis_aniso_11'), (0.375, 'heis_aniso_15'),
                        (0.50,  'heis_aniso_20')]:
        circuits.append({
            'label': label,
            'n_qubits': 6,
            'category': 'symmetry_breaking',
            'parent': 'C02_heisenberg',
            'builder': lambda c, s, _jz=Jz4: trotter_heis_aniso(c, s, _jz),
            'norm_key': label,
            'depths': DEPTHS_STANDARD,
            'params': {'Jxy_over4': 0.25, 'Jz_over4': Jz4},
        })
    # C13 (tight binding) variants — break U(1) by adding ZZ
    for Vz, label in [(0.1, 'tb_vz01'), (0.3, 'tb_vz03'), (0.5, 'tb_vz05')]:
        circuits.append({
            'label': label,
            'n_qubits': 6,
            'category': 'symmetry_breaking',
            'parent': 'C13_tight_binding',
            'builder': lambda c, s, _vz=Vz: trotter_tb_with_zz(c, s, _vz),
            'norm_key': label,
            'depths': DEPTHS_STANDARD,
            'params': {'t_over2': 0.5, 'Vz': Vz},
        })
    return circuits

def _make_block_b():
    """M5: N=8 scaling study."""
    circuits = []
    n = 8

    builders = [
        ('heisenberg_n8',     'structured',    lambda c, s: trotter_heisenberg(c, s, n)),
        ('tight_binding_n8',  'structured',    lambda c, s: trotter_tight_binding(c, s, n)),
        ('xxz_aniso_n8',      'gap_protected', lambda c, s: trotter_xxz(c, s, n)),
        ('tfim_ordered_n8',   'gap_protected', lambda c, s: trotter_tfim(c, s, 1.0, 0.1, n)),
        ('kitaev_detuned_n8', 'topological',   lambda c, s: trotter_kitaev_detuned(c, s, n)),
        ('bcs_8qubit',        'gap_protected', lambda c, s: trotter_bcs(c, s, n)),
        ('cluster_spt_n8',    'topological',   lambda c, s: trotter_cluster_spt(c, s, n)),
    ]
    for label, cat, builder in builders:
        circuits.append({
            'label': label,
            'n_qubits': n,
            'category': cat,
            'parent': label.replace('_n8', '').replace('_8qubit', '_6qubit'),
            'builder': builder,
            'norm_key': label,
            'depths': DEPTHS_STANDARD,
            'params': {'n_qubits': n},
        })
    # Random baselines (no norm, no Trotter)
    for seed, label in [(42, 'random_layered_n8'), (137, 'random_layered_n8_2')]:
        circuits.append({
            'label': label,
            'n_qubits': n,
            'category': 'null',
            'parent': 'random',
            'builder': None,  # handled specially in build_circuit
            'norm_key': None,
            'depths': DEPTHS_STANDARD,
            'params': {'seed': seed, 'n_qubits': n},
        })
    return circuits

def _make_block_c():
    """Finer depth resolution for 5 key circuits.

    Selection rationale (per reviewer feedback):
      C04 (all_to_all)      — edge case, widest bootstrap CI [-124,7], resolve ambiguity
      C02 (heisenberg)      — symmetry outlier kappa=2.32, test if robust to resolution
      C13 (tight_binding)   — symmetry outlier kappa=3.60, same test
      C09 (kitaev_detuned)  — generic topological, kappa=1.72
      C12 (bcs_6qubit)      — generic gap-protected, kappa=1.18
    """
    n = 6
    circuits = [
        {
            'label': 'all_to_all_fine',
            'n_qubits': n, 'category': 'edge_case',
            'parent': 'C04_all_to_all',
            'builder': lambda c, s: trotter_all_to_all(c, s, n),
            'norm_key': 'all_to_all',
            'depths': DEPTHS_FINE,
            'params': {},
        },
        {
            'label': 'heisenberg_fine',
            'n_qubits': n, 'category': 'structured',
            'parent': 'C02_heisenberg',
            'builder': lambda c, s: trotter_heisenberg(c, s, n),
            'norm_key': 'heisenberg',
            'depths': DEPTHS_FINE,
            'params': {},
        },
        {
            'label': 'tight_binding_fine',
            'n_qubits': n, 'category': 'structured',
            'parent': 'C13_tight_binding',
            'builder': lambda c, s: trotter_tight_binding(c, s, n),
            'norm_key': 'tight_binding',
            'depths': DEPTHS_FINE,
            'params': {},
        },
        {
            'label': 'kitaev_detuned_fine',
            'n_qubits': n, 'category': 'topological',
            'parent': 'C09_kitaev_detuned',
            'builder': lambda c, s: trotter_kitaev_detuned(c, s, n),
            'norm_key': 'kitaev_detuned',
            'depths': DEPTHS_FINE,
            'params': {},
        },
        {
            'label': 'bcs_6qubit_fine',
            'n_qubits': n, 'category': 'gap_protected',
            'parent': 'C12_bcs_6qubit',
            'builder': lambda c, s: trotter_bcs(c, s, n),
            'norm_key': 'bcs_6qubit',
            'depths': DEPTHS_FINE,
            'params': {},
        },
    ]
    return circuits

def _make_block_d():
    """Additional N=6 Hamiltonians to increase sample size."""
    n = 6
    circuits = [
        {
            'label': 'tfim_h05',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_tfim(c, s, 1.0, 0.5, n),
            'norm_key': 'tfim_h05',
            'depths': DEPTHS_STANDARD,
            'params': {'J': 1.0, 'h': 0.5},
        },
        {
            'label': 'tfim_h20',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_tfim(c, s, 1.0, 2.0, n),
            'norm_key': 'tfim_h20',
            'depths': DEPTHS_STANDARD,
            'params': {'J': 1.0, 'h': 2.0},
        },
        {
            'label': 'xxz_moderate',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_heis_aniso(c, s, 0.375, n),
            'norm_key': 'xxz_moderate',
            'depths': DEPTHS_STANDARD,
            'params': {'Jxy_over4': 0.25, 'Jz_over4': 0.375},
        },
        {
            'label': 'j1j2_frustrated',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_j1j2(c, s, 1.0, 0.5, n),
            'norm_key': 'j1j2_frustrated',
            'depths': DEPTHS_STANDARD,
            'params': {'J1': 1.0, 'J2': 0.5},
        },
        {
            'label': 'xy_zfield',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_xy_zfield(c, s, 1.0, 0.5, n),
            'norm_key': 'xy_zfield',
            'depths': DEPTHS_STANDARD,
            'params': {'J': 1.0, 'h': 0.5},
        },
        {
            'label': 'alt_bond_ising',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_alt_bond_ising(c, s, 1.0, 0.3, 0.5, n),
            'norm_key': 'alt_bond_ising',
            'depths': DEPTHS_STANDARD,
            'params': {'J1': 1.0, 'J2': 0.3, 'h': 0.5},
        },
        {
            'label': 'compass_model',
            'n_qubits': n, 'category': 'structured',
            'parent': 'new',
            'builder': lambda c, s: trotter_compass(c, s, 1.0, 1.0, n),
            'norm_key': 'compass_model',
            'depths': DEPTHS_STANDARD,
            'params': {'Jx': 1.0, 'Jz': 1.0},
        },
        {
            'label': 'random_layered_3',
            'n_qubits': n, 'category': 'null',
            'parent': 'random',
            'builder': None,
            'norm_key': None,
            'depths': DEPTHS_STANDARD,
            'params': {'seed': 314, 'n_qubits': n},
        },
    ]
    return circuits


ALL_BLOCKS = {
    'A': ('M6: Symmetry-breaking control', _make_block_a),
    'B': ('M5: N=8 scaling', _make_block_b),
    'C': ('Finer depth resolution', _make_block_c),
    'D': ('Additional N=6 circuits', _make_block_d),
}


# =====================================================================
# CIRCUIT BUILDER
# =====================================================================

def get_norm(cdef):
    """Look up Frobenius norm for a circuit definition."""
    nk = cdef['norm_key']
    if nk is None:
        return None
    if cdef['n_qubits'] == 8:
        return NORMS_N8.get(nk, NORMS_N6.get(nk))
    return NORMS_N6.get(nk)

def build_circuit(cdef, depth):
    """Build a Braket Circuit for a given circuit definition and depth."""
    nq = cdef['n_qubits']

    if cdef['builder'] is None:
        # Random circuit
        seed = cdef['params']['seed']
        circ = build_random_circuit(depth, nq, seed)
    else:
        circ = Circuit()
        if depth > 0:
            norm = get_norm(cdef)
            s = DT / norm
            for _ in range(depth):
                cdef['builder'](circ, s)

    # Measure all qubits
    for q in range(nq):
        circ.measure(int(q))
    return circ


def count_2q_gates(circ):
    """Count 2-qubit gates in a circuit."""
    return sum(1 for instr in circ.instructions
               if len(instr.target) == 2
               and str(instr.operator) != 'Measure')


# =====================================================================
# VALIDATE MODE
# =====================================================================

def validate_mode(blocks):
    """Print gate counts and cost estimates. No QPU needed."""
    print("=" * 72)
    print("  VALIDATION: Gate counts and cost estimate")
    print("=" * 72)

    total_tasks = 0
    total_cost = 0.0

    for block_id in sorted(blocks):
        block_name, block_fn = ALL_BLOCKS[block_id]
        circuits = block_fn()

        print(f"\n  Block {block_id}: {block_name}")
        print(f"  {'-' * 68}")

        block_tasks = 0
        for cdef in circuits:
            label = cdef['label']
            nq = cdef['n_qubits']
            depths = cdef['depths']
            n_depths = len(depths)
            block_tasks += n_depths

            norm = get_norm(cdef)
            norm_str = f"{norm:.3f}" if norm else "N/A"

            if HAS_BRAKET:
                # Count 2q gates at a few representative depths
                test_depths = [d for d in [1, 4, 12, 25] if d in depths]
                cnots = {}
                for d in test_depths:
                    circ = build_circuit(cdef, d)
                    cnots[d] = count_2q_gates(circ)
                cnot_str = "  ".join(f"d={d}:{cnots[d]}" for d in test_depths)
            else:
                cnot_str = "(braket not installed)"

            print(f"    {label:25s}  N={nq}  ||H||_F={norm_str:>8s}"
                  f"  depths={n_depths:2d}  {cnot_str}")

        block_cost = block_tasks * (COST_PER_TASK + SHOTS * COST_PER_SHOT)
        total_tasks += block_tasks
        total_cost += block_cost
        print(f"    Block {block_id} total: {block_tasks} tasks, ${block_cost:.2f}")

    print(f"\n  {'=' * 68}")
    print(f"  TOTAL: {total_tasks} tasks, ${total_cost:.2f}")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print(f"  Margin: ${BUDGET_USD - total_cost:.2f}")
    print(f"  {'=' * 68}")


# =====================================================================
# SIMULATE MODE
# =====================================================================

def run_block(block_id, device, is_qpu=False):
    """Run a single block on the given device. Returns results dict."""
    block_name, block_fn = ALL_BLOCKS[block_id]
    circuits = block_fn()

    print(f"\n{'=' * 72}")
    print(f"  Block {block_id}: {block_name}")
    mode = "QPU" if is_qpu else "SIMULATOR"
    print(f"  Mode: {mode}  |  {len(circuits)} circuits")
    print(f"{'=' * 72}")

    # Load checkpoint
    checkpoint = {}
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            checkpoint = json.load(f)

    block_results = {
        'metadata': {
            'block': block_id,
            'block_name': block_name,
            'mode': mode.lower(),
            'timestamp_start': datetime.now().isoformat(),
            'shots': SHOTS,
        },
        'circuits': {},
    }

    spent_this_block = 0.0

    for ci, cdef in enumerate(circuits):
        label = cdef['label']
        nq = cdef['n_qubits']
        depths = cdef['depths']
        ck_key = f"{block_id}_{label}"

        if ck_key in checkpoint:
            print(f"  [{ci+1}/{len(circuits)}] {label}: SKIP (checkpointed)")
            block_results['circuits'][label] = checkpoint[ck_key]
            continue

        t0 = time.time()
        circuit_data = {
            'label': label,
            'n_qubits': nq,
            'category': cdef['category'],
            'parent': cdef.get('parent', ''),
            'params': {k: (float(v) if isinstance(v, (int, float, np.floating))
                           else v)
                       for k, v in cdef.get('params', {}).items()},
            'depths': {},
        }

        for di, depth in enumerate(depths):
            # Budget guard (QPU only)
            task_cost = COST_PER_TASK + SHOTS * COST_PER_SHOT
            if is_qpu:
                total_spent = sum(
                    v.get('cost', 0) for v in checkpoint.values()
                ) + spent_this_block
                if total_spent + task_cost > BUDGET_USD * 0.95:
                    print(f"    BUDGET HALT at depth {depth}. "
                          f"Spent: ${total_spent:.2f}")
                    break

            try:
                circ = build_circuit(cdef, depth)
                n_2q = count_2q_gates(circ)
                result = device.run(circ, shots=int(SHOTS)).result()

                # Extract raw bitstrings
                measurements = result.measurements  # numpy array [shots, n_qubits]
                bitstrings = [''.join(str(b) for b in row)
                              for row in measurements.tolist()]

                circuit_data['depths'][str(depth)] = {
                    'bitstrings': bitstrings,
                    'n_shots': len(bitstrings),
                    'cnot_count': n_2q,
                }
                spent_this_block += task_cost

            except Exception as e:
                print(f"    ERROR at {label} depth={depth}: {e}")
                circuit_data['depths'][str(depth)] = {
                    'bitstrings': [],
                    'n_shots': 0,
                    'cnot_count': -1,
                    'error': str(e),
                }
                spent_this_block += task_cost

        elapsed = time.time() - t0

        # Quick analysis: sigma_c from max probability
        probs_at_depth = {}
        for d_str, d_data in circuit_data['depths'].items():
            bs_list = d_data['bitstrings']
            if len(bs_list) == 0:
                continue
            counts = {}
            for bs in bs_list:
                counts[bs] = counts.get(bs, 0) + 1
            probs_at_depth[int(d_str)] = max(counts.values()) / len(bs_list)

        if probs_at_depth:
            sorted_depths = sorted(probs_at_depth.keys())
            r_vals = [probs_at_depth[d] for d in sorted_depths]
            r0 = r_vals[0] if sorted_depths[0] == 0 else r_vals[0]
            # Find sigma_c
            tau = np.exp(-1)
            threshold = tau * r0
            sc = None
            for i in range(len(sorted_depths) - 1):
                if r_vals[i] >= threshold and r_vals[i+1] < threshold:
                    frac = (threshold - r_vals[i]) / (r_vals[i+1] - r_vals[i])
                    sc = sorted_depths[i] + frac * (sorted_depths[i+1] - sorted_depths[i])
                    break
            sc_str = f"sc={sc:.1f}" if sc else "sc=N/A"
        else:
            sc_str = "sc=N/A"

        n_depths_done = len([d for d in circuit_data['depths'].values()
                             if d['n_shots'] > 0])
        print(f"  [{ci+1}/{len(circuits)}] {label:25s}  N={nq}"
              f"  {n_depths_done}/{len(depths)} depths  {sc_str}"
              f"  [{elapsed:.1f}s]")

        block_results['circuits'][label] = circuit_data

        # Checkpoint
        checkpoint[ck_key] = circuit_data
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint, f)

    block_results['metadata']['timestamp_end'] = datetime.now().isoformat()
    block_results['metadata']['cost_usd'] = spent_this_block

    # Save block results
    outfile = OUTPUT_DIR / f"r1_supplement_block_{block_id.lower()}.json"
    with open(outfile, 'w') as f:
        json.dump(block_results, f, indent=2)
    print(f"\n  Block {block_id} saved to {outfile}")
    print(f"  Cost: ${spent_this_block:.2f}")

    return block_results


def simulate_mode(blocks):
    """Run on LocalSimulator."""
    if not HAS_BRAKET:
        print("ERROR: braket not installed.")
        return

    device = LocalSimulator()
    print("=" * 72)
    print("  SIMULATOR MODE")
    print("=" * 72)

    all_results = {}
    for block_id in sorted(blocks):
        result = run_block(block_id, device, is_qpu=False)
        all_results[block_id] = result

    # Summary
    print(f"\n{'=' * 72}")
    print("  SIMULATION SUMMARY")
    print(f"{'=' * 72}")
    for bid, res in all_results.items():
        n_circuits = len(res['circuits'])
        n_total_depths = sum(
            len([d for d in cd['depths'].values() if d['n_shots'] > 0])
            for cd in res['circuits'].values()
        )
        print(f"  Block {bid}: {n_circuits} circuits, "
              f"{n_total_depths} depth-measurements OK")


def qpu_mode(blocks):
    """Run on Rigetti Ankaa-3."""
    if not HAS_BRAKET:
        print("ERROR: braket not installed.")
        return

    print("=" * 72)
    print("  QPU MODE: Rigetti Ankaa-3")
    print(f"  Budget: ${BUDGET_USD:.2f}")
    print("=" * 72)

    try:
        device = AwsDevice(RIGETTI_ARN)
        print(f"  Connected: {device.name}")
    except Exception as e:
        print(f"  ERROR connecting: {e}")
        return

    for block_id in sorted(blocks):
        run_block(block_id, device, is_qpu=True)

    print(f"\n  All requested blocks complete.")


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='R1 supplementary experiments for PRA paper')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--validate', action='store_true',
                       help='Gate counts and cost estimate (no QPU)')
    group.add_argument('--simulate', action='store_true',
                       help='Run on local Braket simulator (free)')
    group.add_argument('--qpu', action='store_true',
                       help='Run on Rigetti Ankaa-3 (costs money!)')
    parser.add_argument('--block', type=str, default=None,
                        help='Block to run: A, B, C, D (comma-separated)')
    parser.add_argument('--all', action='store_true',
                        help='Run all blocks')
    parser.add_argument('--reset-checkpoint', action='store_true',
                        help='Clear checkpoint file before running')
    args = parser.parse_args()

    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("  Checkpoint cleared.")

    # Determine which blocks to run
    if args.all:
        blocks = list(ALL_BLOCKS.keys())
    elif args.block:
        blocks = [b.strip().upper() for b in args.block.split(',')]
        for b in blocks:
            if b not in ALL_BLOCKS:
                print(f"ERROR: Unknown block '{b}'. Choose from A, B, C, D.")
                sys.exit(1)
    else:
        blocks = list(ALL_BLOCKS.keys())

    if args.validate:
        validate_mode(blocks)
    elif args.simulate:
        simulate_mode(blocks)
    elif args.qpu:
        qpu_mode(blocks)


if __name__ == '__main__':
    main()
