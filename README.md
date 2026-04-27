# Critical Noise Threshold and Classical Fisher Information
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Commercial License](https://img.shields.io/badge/License-Commercial-orange.svg)](mailto:nfo@forgottenforge.xyz)

Supplementary Material for the Paper
Correlation and spectral mechanism across quantum circuits.

**Author**: Forgotten Forge

## Summary

We investigate the relationship between the operational noise threshold
sigma_c and the peak of the Classical Fisher Information (CFI) across
quantum circuits. Key findings:

- sigma_c and the CFI peak are strongly correlated (log-scale Pearson
  r = 0.84, p = 0.004) across 16 Trotterized Hamiltonians on the Rigetti
  Ankaa-3 processor (6 qubits, 192 tasks, 500 shots each).
- Generic circuits satisfy sigma_c ~ 1.1-1.3 * epsilon*, while
  Hamiltonians with conserved symmetries show kappa > 2.
- Single-rate depolarizing noise cannot produce interior CFI peaks;
  multi-rate decay is a necessary structural condition.
- An extended dataset of 28 circuits on Rigetti Cepheus-1-108Q confirms
  the correlation (r = 0.84, n = 31) and supports symmetry-breaking and
  N-scaling predictions.
- A cross-platform test on IonQ Forte-1 yields kappa = 1.09.

## Hardware

Experiments were run on three quantum processors via Amazon Braket:

| Processor | Technology | Qubits | Native 2Q gate |
|-----------|-----------|--------|-----------------|
| Rigetti Ankaa-3 | Superconducting (transmon) | 84 | iSWAP (72 ns) |
| Rigetti Cepheus-1-108Q | Superconducting (transmon) | 108 | CZ (60 ns) |
| IonQ Forte-1 | Trapped ion | 36 | MS gate |


## License

Copyright (c) 2026 Forgotten Forge — [forgottenforge.xyz](https://www.forgottenforge.xyz)

Dual-licensed: **AGPL-3.0** for open-source use, **commercial licenses** available.
Contact nfo@forgottenforge.xyz for commercial inquiries.
