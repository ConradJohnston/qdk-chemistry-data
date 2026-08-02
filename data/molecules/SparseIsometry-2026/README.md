# Sparse Isometry 2026 Scripts and Data

Scripts, inputs, resource estimates, and figures for the paper
"Clifford-efficient sparse state preparation for molecular wavefunctions".

The committed results compare four sparse state-preparation methods on random
determinant matrices and molecular wavefunctions.

## Figures

![Random and molecular resource estimates](output/figures/random_matrix_results.png)

![F2 resource estimates](output/figures/f2_matrix_results.png)

![F2 sparse and dense resource breakdown](output/figures/f2_matrix_results_stacked.png)

## Reproducibility

Install the benchmark dependencies:

```bash
pip install qdk-chemistry[qiskit] qualtran numpy matplotlib qiskit
```

The Rupprecht and Wolk reference implementation is distributed under Apache
License 2.0 through [Zenodo record 18234600](https://zenodo.org/records/18234600).
Install it as the importable `sparse_state_preparation` package before running
resource estimation.

Run the full resource estimates from this directory:

```bash
python estimate_random_matrix.py
pythonz estimate_f2.py
```

## Methodology

The random benchmark uses a fixed seed of 42. For each even qubit count, it
constructs a half-filled system, samples unique excitations from the
Hartree-Fock determinant, and sets the number of configurations equal to the
number of qubits. Real coefficients are sampled and normalized. The same
resource estimators are then applied to eight non-F2 molecular wavefunctions.
The plain `gf2x` estimator is skipped above 25 qubits by default; the other
three methods continue across the full random-matrix grid.

The F2 benchmark takes ordered prefixes of the 14-configuration F2
wavefunction, starting with two configurations. Coefficients are renormalized
for every prefix before resource estimation.

The compared methods are:

- `gf2x`: QDK/Chemistry GF2+X sparse isometry.
- `gf2x_binary_encoding`: GF2+X with binary encoding.
- `Rupprecht2026`: batched sparse isometry from Rupprecht and Wolk.
- `Ramacciotti2024`: Qualtran's permutation-based sparse state preparation.

Logical qubits are the maximum required by the sparse-isometry and dense-load
stages. Clifford and non-Clifford counts are summed across those stages.
Non-Clifford count is the sum of Toffoli-family gates and arbitrary rotations.

## File Layout

```text
SparseIsometry-2026/
├── README.md
├── estimate_f2.py                    # Run the F2 configuration scan
├── estimate_random_matrix.py         # Run random and molecular benchmarks
├── generate_random_matrix.py         # Generate random determinant matrices
├── state_preparation_methods.py      # Resource estimators for four methods
├── data/
│   ├── input_wavefunctions.json      # Nine molecular wavefunctions
│   └── structures/                   # Corresponding XYZ structures
└── output/
│   ├── random_matrix_results.json    # Random and molecular benchmark results
│   ├── f2_matrix_results.json        # F2 prefix-scan results
│   └── figures/                      # Three generated PNG figures
```

## Citation

When using these data, cite the accompanying paper,
"Clifford-efficient sparse state preparation for molecular wavefunctions",
and the implementations used in the comparison:

- Rupprecht and Wolk, [Sparse quantum state preparation with improved Toffoli
   cost](https://arxiv.org/abs/2601.09388) (2026).
- Ramacciotti et al., [A simple quantum algorithm to efficiently prepare sparse
   states](https://arxiv.org/abs/2310.19309) (2024).
- [QDK/Chemistry](https://github.com/microsoft/qdk-chemistry).
- [Qualtran](https://github.com/quantumlib/Qualtran).

## License

See the repository [LICENSE](../../../LICENSE).
