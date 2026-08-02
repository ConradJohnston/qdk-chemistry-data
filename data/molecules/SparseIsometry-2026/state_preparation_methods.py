"""State preparation methods for sparse quantum wavefunctions.

All four methods accept a ``qdk_chemistry.data.Wavefunction`` directly:

  - ``gf2x`` — GF2+X elimination-based sparse isometry via qdk_chemistry
    (``sparse_isometry_gf2x``).
  - ``gf2x_binary_encoding`` — GF2+X with batched Toffoli-based binary
    encoding via qdk_chemistry (``sparse_isometry_binary_encoding``).
  - ``Rupprecht2026`` — Batched isometry from Rupprecht & Wolk (2026) via
    Qualtran bloqs.
  - ``Ramacciotti2024`` — Permutation-based sparse state preparation from
    Ramacciotti et al. (2024) via Qualtran.

Also provides helpers shared across methods: ``dense_state_prep``,
``estimate_bloq``, and ``_wavefunction_from_bitstrings``.

Requires: qdk_chemistry, qualtran, qiskit, numpy.
"""

from typing import Any

import numpy as np
from qiskit.compiler import transpile

# Dependency checks
try:
    from qdk_chemistry.algorithms import create
    from qdk_chemistry.algorithms.state_preparation.sparse_isometry import (
        gf2x_with_tracking,
    )
    from qdk_chemistry.data import (
        BasisSet,
        Circuit,
        Configuration,
        Orbitals,
        OrbitalType,
        SciWavefunctionContainer,
        Shell,
    )
    from qdk_chemistry.data import Wavefunction as QDKWavefunction
    from qdk_chemistry.data.circuit import QsharpFactoryData
    from qdk_chemistry.utils.binary_encoding import _dense_qubits_size
    from qdk_chemistry.utils.qsharp import QSHARP_UTILS

except ImportError:
    raise ImportError(
        "ERROR: qdk_chemistry is required. Install the qdk_chemistry package."
    )

try:
    from sparse_state_preparation import SparseStatePreparation
    from sparse_state_preparation.isometry import IsometryToSubspaceViaBatching
except ImportError:
    raise ImportError(
        "ERROR: Rupprecht2026 sparse_state_preparation package is required. "
        "See https://zenodo.org/records/18234600",
    )

try:
    from qualtran import QFxp
    from qualtran.bloqs.state_preparation.sparse_state_preparation_via_rotations import (
        SparseStatePreparationViaRotations,
    )
    from qualtran.resource_counting import QECGatesCost, QubitCount, get_cost_value
except ImportError:
    raise ImportError(
        "ERROR: qualtran is required for sparse isometry resource estimation. "
    )

# Magic numbers required by the reference methods but unused in the benchmarking
PHASE_BITSIZE = 6
NUM_FRAC = 6

# ── Gate counting (Qiskit-based) ──────────────────────────────────────────────
_BASIS_GATES = [
    "x",
    "y",
    "z",
    "cx",
    "swap",
    "cz",
    "ccx",
    "ccz",
    "id",
    "h",
    "s",
    "sdg",
    "rz",
    "cswap",
]
_CLIFFORD_GATES = {"x", "y", "z", "cx", "cz", "h", "s", "sdg", "swap"}
_TOFFOLI_GATES = {"ccx", "ccz", "cswap"}


def _create_test_basis_set(
    num_atomic_orbitals: int, name: str = "test-basis"
) -> BasisSet:
    """Create a minimal basis set with exactly *num_atomic_orbitals* functions.

    Shells are built programmatically (no external data files) so this never
    raises ``ValueError: Tried to generate invalid BasisSet``.

    Args:
        num_atomic_orbitals (int): Number of atomic orbital basis functions to
            include in the basis set.
        name (str): Label for the basis set. Defaults to ``"test-basis"``.

    Returns:
        BasisSet: A minimal ``BasisSet`` containing the requested number of
            basis functions built from S- and P-type shells.
    """
    shells: list[Shell] = []
    atom_index = 0
    functions_created = 0
    while functions_created < num_atomic_orbitals:
        remaining = num_atomic_orbitals - functions_created
        if remaining >= 3:
            shell = Shell(
                atom_index, OrbitalType.P, np.array([1.0, 0.5]), np.array([0.6, 0.4])
            )
            shells.append(shell)
            functions_created += 3
        else:
            for _ in range(remaining):
                shell = Shell(
                    atom_index, OrbitalType.S, np.array([1.0]), np.array([1.0])
                )
                shells.append(shell)
                functions_created += 1
    return BasisSet(name, shells)


def _to_qdk_wavefunction(
    bitstrings: list[str], coeffs: list[complex]
) -> QDKWavefunction:
    """Convert bitstrings and coefficients to a QDK ``Wavefunction``.

    Used internally by ``gf2x`` and ``gf2x_binary_encoding``.  Bitstrings are
    in little-endian qubit order (beta occupancies in the low half, alpha in
    the high half) and the number of qubits must be even.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings representing
            Slater determinants in little-endian qubit order.  All strings must
            have the same even length ``n_qubits``.
        coeffs (list[complex]): Expansion coefficients corresponding to each
            bitstring.  Must have the same length as ``bitstrings``.

    Returns:
        QDKWavefunction: A ``qdk_chemistry`` ``Wavefunction`` object built from
            the supplied determinants and coefficients.
    """
    n_qubits = len(bitstrings[0])
    n_orbitals = n_qubits // 2

    bs = _create_test_basis_set(n_orbitals)
    orbs = Orbitals(np.eye(n_orbitals), None, None, bs, (list(range(n_orbitals)), []))

    dets = []
    for bitstring in bitstrings:
        beta_reversed = bitstring[:n_orbitals]
        alpha_reversed = bitstring[n_orbitals:]
        alpha_str = alpha_reversed[::-1]
        beta_str = beta_reversed[::-1]
        det_chars = []
        for a, b in zip(alpha_str, beta_str):
            if a == "1" and b == "1":
                det_chars.append("2")
            elif a == "1" and b == "0":
                det_chars.append("u")
            elif a == "0" and b == "1":
                det_chars.append("d")
            else:
                det_chars.append("0")
        dets.append(Configuration("".join(det_chars)))

    container = SciWavefunctionContainer(np.array(coeffs), dets, orbs)
    return QDKWavefunction(container)


def estimate_bloq(bloq: Any) -> dict[str, int]:
    """Get resource estimates directly from a qualtran Bloq.

    Args:
        bloq (Any): A qualtran ``Bloq`` supporting ``QubitCount`` and
            ``QECGatesCost`` resource-counting protocols.

    Returns:
        dict[str, int]: Resource estimate with keys:
            - ``logical_qubits``: total logical qubit count.
            - ``toffoli_count``: number of Toffoli / And-bloq gates.
            - ``rotation_count``: number of arbitrary-angle rotation gates.
            - ``non_clifford_count``: ``toffoli_count + rotation_count``.
            - ``clifford_count``: number of Clifford gates.
    """
    qubit_count = get_cost_value(bloq, QubitCount())
    gate_counts = get_cost_value(bloq, QECGatesCost())
    toffoli = int(gate_counts.toffoli + gate_counts.and_bloq)
    rotation = int(gate_counts.rotation)
    return {
        "logical_qubits": int(qubit_count),
        "toffoli_count": toffoli,
        "rotation_count": rotation,
        "non_clifford_count": toffoli + rotation,
        "clifford_count": int(gate_counts.clifford),
    }


def estimate_circuit(circuit: Circuit) -> dict[str, int]:
    """Estimate resources for a qdk_chemistry Circuit via Qiskit gate counting.

    Converts the circuit to a Qiskit ``QuantumCircuit``, transpiles it to the
    benchmark basis gate set (``_BASIS_GATES``), and counts Toffoli, rotation,
    and Clifford gates.

    Args:
        circuit (Circuit): A ``qdk_chemistry`` ``Circuit`` object that exposes
            a ``get_qiskit_circuit()`` method.

    Returns:
        dict[str, int]: Resource estimate with keys:
            - ``logical_qubits``: number of qubits in the transpiled circuit.
            - ``toffoli_count``: number of CCX / CCZ / CSWAP gates.
            - ``rotation_count``: number of RZ (arbitrary-angle) gates.
            - ``non_clifford_count``: ``toffoli_count + rotation_count``.
            - ``clifford_count``: number of Clifford gates.
    """
    qc = circuit.get_qiskit_circuit()
    qc = transpile(qc, basis_gates=_BASIS_GATES, optimization_level=0)
    ops = qc.count_ops()
    toffoli_count = sum(ops.get(g, 0) for g in _TOFFOLI_GATES)
    rotation_count = ops.get("rz", 0)
    clifford_count = sum(ops.get(g, 0) for g in _CLIFFORD_GATES)
    return {
        "logical_qubits": qc.num_qubits,
        "toffoli_count": toffoli_count,
        "rotation_count": rotation_count,
        "non_clifford_count": toffoli_count + rotation_count,
        "clifford_count": clifford_count,
    }


def _dense_state_prep_sv(n_qubits: int, sv: np.ndarray) -> dict[str, int]:
    """Estimate dense state prep resources from a pre-built statevector.

    Normalises ``sv``, wraps it in a Q# ``MakeDenseStatePreparation`` factory,
    and delegates to ``estimate_circuit``.

    Args:
        n_qubits (int): Number of qubits (log2 of the statevector length).
        sv (np.ndarray): Statevector of length ``2**n_qubits``.
            May be complex-valued; imaginary parts must be negligible
            (the Q# backend accepts only real amplitudes).
            Will be L2-normalised before use.

    Returns:
        dict[str, int]: Resource estimate; see ``estimate_circuit`` for keys.
    """
    if np.iscomplexobj(sv):
        if np.max(np.abs(sv.imag)) > 1e-10:
            raise ValueError(
                "Dense state preparation received non-negligible imaginary "
                f"amplitudes (max |imag| = {np.max(np.abs(sv.imag)):.2e}). "
                "The Q# MakeDenseStatePreparation backend supports only real "
                "amplitudes."
            )
        sv = sv.real.copy()
    norm = np.linalg.norm(sv)
    if norm > 0:
        sv = sv / norm
    qsharp_factory = QsharpFactoryData(
        program=QSHARP_UTILS.StatePreparation.MakeDenseStatePreparation,
        parameter={
            "rowMap": list(range(n_qubits)),
            "stateVector": sv.tolist(),
            "numQubits": n_qubits,
        },
    )
    circuit = Circuit(qsharp_factory=qsharp_factory, encoding="jordan-wigner")
    return estimate_circuit(circuit)


def dense_state_prep(bitstrings: list[str], coeffs: list[complex]) -> dict[str, int]:
    """Dense state prep used by all methods. Returns resource estimates.

    Builds a full ``2**n_qubits`` statevector from the supplied bitstrings and
    coefficients, then estimates resources via ``_dense_state_prep_sv``.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings.  All strings
            must have the same length ``n_qubits``.
        coeffs (list[complex]): Expansion coefficients for each bitstring.
            Must have the same length as ``bitstrings``.

    Returns:
        dict[str, int]: Resource estimate; see ``estimate_circuit`` for keys.
    """
    n_qubits = len(bitstrings[0])
    sv = np.zeros(2**n_qubits, dtype=complex)
    for bs, c in zip(bitstrings, coeffs):
        sv[int(bs, 2)] = c
    return _dense_state_prep_sv(n_qubits, sv)


def gf2x(
    bitstrings: list[str], coeffs: list[complex]
) -> tuple[dict[str, int], dict[str, int]]:
    """Run GF2+X sparse isometry via qdk_chemistry.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings representing
            the non-zero amplitudes of the target state.
        coeffs (list[complex]): Expansion coefficients corresponding to
            each bitstring.

    Returns:
        tuple[dict[str, int], dict[str, int]]: A pair
            ``(sparse_est, dense_est)`` where each dict contains resource
            estimates (see ``estimate_circuit`` for keys).
    """
    wfn = _to_qdk_wavefunction(bitstrings, coeffs)
    state_prep = create("state_prep", "sparse_isometry_gf2x")
    params = state_prep._build_qsharp_state_prep_params(wfn)
    dense_est = estimate_circuit(state_prep._create_dense(params))
    sparse_est = estimate_circuit(state_prep._create_isometry(params))
    return sparse_est, dense_est


def gf2x_binary_encoding(
    bitstrings: list[str], coeffs: list[complex]
) -> tuple[dict[str, int], dict[str, int]]:
    """Run GF2+X with binary encoding via qdk_chemistry.

    Applies GF2+X elimination with forward-only tracking to obtain a reduced
    bitstring matrix, then uses batched Toffoli-based binary encoding for the
    sparse isometry.  Falls back to plain ``gf2x`` when binary encoding offers
    no qubit advantage.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings representing
            the non-zero amplitudes of the target state.
        coeffs (list[complex]): Expansion coefficients corresponding to
            each bitstring.

    Returns:
        tuple[dict[str, int], dict[str, int]]: A pair
            ``(sparse_est, dense_est)`` where each dict contains resource
            estimates (see ``estimate_circuit`` for keys).
    """
    n_qubits = len(bitstrings[0])
    bitstring_matrix = np.array(
        [[int(c) for c in bs] for bs in bitstrings], dtype=np.int8
    ).T
    gf2x_result = gf2x_with_tracking(
        bitstring_matrix, skip_diagonal_reduction=True, forward_only=True
    )

    num_rows, num_cols = gf2x_result.reduced_matrix.shape
    if _dense_qubits_size(num_cols) >= num_rows:
        return gf2x(bitstrings, coeffs)

    state_prep = create(
        "state_prep",
        "sparse_isometry_binary_encoding",
        include_negative_controls=True,
        measurement_based_uncompute=True,
    )
    params = state_prep._build_binary_encoding_params(
        gf2x_result, coeffs, n_qubits, bitstrings
    )
    dense_est = estimate_circuit(state_prep._create_dense(params))
    sparse_est = estimate_circuit(state_prep._create_isometry(params))
    return sparse_est, dense_est


def Rupprecht2026(
    bitstrings: list[str],
    coeffs: list[complex],
    phase_bitsize: int = PHASE_BITSIZE,
    num_frac: int = NUM_FRAC,
) -> tuple[dict[str, int], dict[str, int]]:
    """Rupprecht & Wolk 2026 batched isometry method.

    Estimates the isometry cost via qualtran ``IsometryToSubspaceViaBatching``
    and the dense state-preparation cost via ``_dense_state_prep_sv``.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings representing
            the non-zero amplitudes of the target state.
        coeffs (list[complex]): Expansion coefficients corresponding to
            each bitstring.
        phase_bitsize (int): Total bit-size of the fixed-point phase register
            used inside ``SparseStatePreparation``.  Defaults to
            ``PHASE_BITSIZE`` (6).
        num_frac (int): Number of fractional bits in the fixed-point phase
            register.  Defaults to ``NUM_FRAC`` (6).

    Returns:
        tuple[dict[str, int], dict[str, int]]: A pair
            ``(sparse_est, dense_est)`` where ``sparse_est`` comes from
            ``estimate_bloq`` and ``dense_est`` comes from
            ``estimate_circuit``.  See those functions for key descriptions.
    """
    states = np.array([[b == "1" for b in bs] for bs in bitstrings], dtype=bool)
    coeffs_arr = np.array(coeffs)

    sparse_prep = SparseStatePreparation(
        states,
        coeffs_arr,
        QFxp(bitsize=phase_bitsize, num_frac=num_frac),
        uncompute_in_isometry=True,
    )
    dense_bitsize = sparse_prep.isometry.subspace_bitsize
    dense_coeffs = sparse_prep._permuted_coefficients
    # Build the dense statevector directly — the Rupprecht dense register spans
    # all 2^dense_bitsize computational basis states (arbitrary electron counts),
    # which cannot be represented as a QDKWavefunction.  Bypass it entirely.
    sv = np.zeros(2**dense_bitsize, dtype=complex)
    for i, c in enumerate(dense_coeffs):
        sv[i] = c

    isometry_bloq = IsometryToSubspaceViaBatching(
        states=states,
        signs=sparse_prep.isometry.signs,
    )
    sparse_est = estimate_bloq(isometry_bloq)
    dense_est = _dense_state_prep_sv(dense_bitsize, sv)

    return sparse_est, dense_est


def _bitstrings_to_coefficient_map(
    bitstrings: list[str], coeffs: list[complex]
) -> dict[int, complex]:
    """Convert bitstrings to a format compatible with Ramacciotti2024.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings.
        coeffs (list[complex]): Complex expansion coefficients corresponding
            to each bitstring.

    Returns:
        dict[int, complex]: Mapping from integer computational-basis index
            (``int(bitstring, 2)``) to its complex coefficient.
    """
    coef_map: dict[int, complex] = {}
    for bs, cf in zip(bitstrings, coeffs):
        coef_map[int(bs, 2)] = cf
    return coef_map


def Ramacciotti2024(
    bitstrings: list[str], coeffs: list[complex], phase_bitsize: int = PHASE_BITSIZE
) -> tuple[dict[str, int], dict[str, int]]:
    """Ramacciotti et al. 2024 permutation-based sparse state preparation.

    Estimates the basis-permutation isometry cost via qualtran
    ``SparseStatePreparationViaRotations`` and the dense state-preparation
    cost via ``_dense_state_prep_sv``.

    Args:
        bitstrings (list[str]): Computational-basis bitstrings representing
            the non-zero amplitudes of the target state.
        coeffs (list[complex]): Expansion coefficients corresponding to
            each bitstring.
        phase_bitsize (int): Bit-size of the phase register used inside
            ``SparseStatePreparationViaRotations``.  Defaults to
            ``PHASE_BITSIZE`` (6).

    Returns:
        tuple[dict[str, int], dict[str, int]]: A pair
            ``(sparse_est, dense_est)`` where ``sparse_est`` comes from
            ``estimate_bloq`` and ``dense_est`` comes from
            ``estimate_circuit``.  See those functions for key descriptions.
    """
    num_qubits = len(bitstrings[0])

    coef_map = _bitstrings_to_coefficient_map(bitstrings, coeffs)

    sparse_prep = SparseStatePreparationViaRotations.from_coefficient_map(
        N=2**num_qubits, coeff_map=coef_map, phase_bitsize=phase_bitsize
    )
    isometry_bloq = sparse_prep._basis_permutation_bloq
    sparse_est = estimate_bloq(isometry_bloq)

    dense_bitsize = sparse_prep.dense_bitsize
    dense_coeffs = sparse_prep.nonzero_coeffs
    sv = np.zeros(2**dense_bitsize, dtype=complex)
    for i, c in enumerate(dense_coeffs):
        sv[i] = c
    dense_est = _dense_state_prep_sv(dense_bitsize, sv)

    return sparse_est, dense_est


def combine_estimates(sparse: dict[str, int], dense: dict[str, int]) -> dict[str, int]:
    """Combine sparse and dense resource estimates into a single row.

    Args:
        sparse: Resource-estimate dict for the sparse isometry circuit.
        dense: Resource-estimate dict for the dense state preparation circuit.

    Returns:
        Merged dict where qubit count is the max and gate counts are summed.
    """
    return {
        "logical_qubits": max(
            sparse.get("logical_qubits", 0), dense.get("logical_qubits", 0)
        ),
        "toffoli_count": sparse.get("toffoli_count", 0) + dense.get("toffoli_count", 0),
        "rotation_count": sparse.get("rotation_count", 0)
        + dense.get("rotation_count", 0),
        "non_clifford_count": sparse.get("non_clifford_count", 0)
        + dense.get("non_clifford_count", 0),
        "clifford_count": sparse.get("clifford_count", 0)
        + dense.get("clifford_count", 0),
    }
