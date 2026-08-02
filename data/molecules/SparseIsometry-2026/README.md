# Sparse Isometry 2026 Scripts and Data

This directory contains scripts and data for reproducing the results in the paper "Clifford-efficient sparse state preparation for molecular wavefunctions".

## Prerequisites

```bash
pip install qdk-chemistry[qiskit] qualtran numpy matplotlib qiskit 
```

The `sparse_state_preparation` package under Apache License 2.0 (Rupprecht et al. 2026) is used as a benchmarking
reference. See [Zenodo 18234600](https://zenodo.org/records/18234600) for installation and save it under
`sparse_state_preparation/` in this repository.

## Generating Figures

The paper has three figures, all reproducible from the scripts below.

### Figure 1 — Random matrices + chemical molecules: resources scaled with number of qubits

```bash
python estimate_random_matrix.py
```

This script generates random sparse isometry matrices, loads named molecules from
`data/input_wavefunctions.json`, runs 4 methods, and produces `output/random_matrix_results.png`.

### Figures 2 & 3 — F2 resource scaled with number of configurations

```bash
python estimate_f2.py
```

This script loads the F2 wavefunction from `data/input_wavefunctions.json`, creates prefix subsets
(2–14 configurations), runs 4 methods, and produces `output/f2_matrix_results.png` (line plot) and
`output/f2_matrix_results_stacked.png` (stacked bar chart).

## Sparse State Preparation Benchmarking References

The benchmarking comparisons rely on implementations adapted from the following sources:

1. **Rupprecht et al. 2026**:
   - [Sparse quantum state preparation with improved Toffoli cost](https://arxiv.org/pdf/2601.09388)
   - Code adapted from [Zenodo](https://zenodo.org/records/18234600) under Apache License 2.0.

2. **Ramacciotti et al. 2024**:
   - [A simple quantum algorithm to efficiently prepare sparse states](https://arxiv.org/abs/2310.19309)
   - Code implemented in [Qualtran library](https://github.com/quantumlib/Qualtran/blob/main/qualtran/bloqs/state_preparation/sparse_state_preparation_via_rotations.py)
     under Apache License 2.0.
