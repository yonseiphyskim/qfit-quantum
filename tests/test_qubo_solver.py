"""Tests for the QUBO solver components.

These tests verify the QUBO formulation (compute_qubo_coeffs),
the D-Wave integration (solve_qubo with mocked sampler),
and helper utilities (find_redundant_conformers, construct_weights).
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from numpy.typing import NDArray

from qfit.solvers import CVXPYSolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class QUBOTestBase:
    """Shared data for QUBO tests: 3 orthogonal conformer models."""
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([
        [6.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        [0.0, 0.0, 21.0],
    ])

    @staticmethod
    def make_solver(target=None, models=None):
        t = target if target is not None else QUBOTestBase.target
        m = models if models is not None else QUBOTestBase.models
        return CVXPYSolver(t, m)


# ---------------------------------------------------------------------------
# 1. compute_qubo_coeffs tests
# ---------------------------------------------------------------------------

class TestComputeQUBOCoeffs(QUBOTestBase):
    """Test that compute_qubo_coeffs builds a well-formed QUBO matrix."""

    def test_returns_matrix_and_constant(self):
        solver = self.make_solver()
        Q, const = solver.compute_qubo_coeffs()
        assert isinstance(Q, np.ndarray)
        assert isinstance(const, float) or isinstance(const, np.floating)

    def test_matrix_is_square(self):
        solver = self.make_solver()
        Q, _ = solver.compute_qubo_coeffs()
        assert Q.ndim == 2
        assert Q.shape[0] == Q.shape[1]

    def test_matrix_is_symmetric(self):
        solver = self.make_solver()
        Q, _ = solver.compute_qubo_coeffs()
        np.testing.assert_allclose(Q, Q.T, atol=1e-10)

    def test_expected_matrix_size(self):
        """Matrix size should be N*K + K + N*K + N*K + N + K for N conformers."""
        solver = self.make_solver()
        K = 10
        N = 3  # number of valid (non-redundant) conformers
        expected_size = N * K + K + N * K + N * K + N + K
        Q, _ = solver.compute_qubo_coeffs(K=K)
        assert Q.shape == (expected_size, expected_size)

    def test_constant_is_positive(self):
        """The constant includes ρ_obs·ρ_obs which should be positive."""
        solver = self.make_solver()
        _, const = solver.compute_qubo_coeffs()
        assert const > 0

    def test_different_thresholds_produce_different_matrices(self):
        solver = self.make_solver()
        Q1, c1 = solver.compute_qubo_coeffs(tmin=0.1)
        Q2, c2 = solver.compute_qubo_coeffs(tmin=0.4)
        assert not np.allclose(Q1, Q2)

    def test_different_cardinality_produces_different_matrices(self):
        solver = self.make_solver()
        Q1, c1 = solver.compute_qubo_coeffs(cardinality=1)
        Q2, c2 = solver.compute_qubo_coeffs(cardinality=3)
        # At minimum the constants differ due to cardinality^2 term
        assert c1 != c2

    def test_stores_qubo_obj_and_const(self):
        solver = self.make_solver()
        Q, const = solver.compute_qubo_coeffs()
        assert solver.qubo_obj is not None
        assert solver.qubo_const is not None
        np.testing.assert_array_equal(solver.qubo_obj, Q)
        assert solver.qubo_const == const


# ---------------------------------------------------------------------------
# 2. find_redundant_conformers tests
# ---------------------------------------------------------------------------

class TestFindRedundantConformers(QUBOTestBase):

    def test_no_redundant_in_orthogonal_models(self):
        solver = self.make_solver()
        solver.find_redundant_conformers()
        assert len(solver.redundant_indices) == 0
        assert len(solver.valid_indices) == 3

    def test_detects_duplicate_conformer(self):
        models_with_dup = np.array([
            [6.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],  # duplicate of model 0
            [0.0, 0.0, 21.0],
        ])
        solver = self.make_solver(models=models_with_dup)
        solver.find_redundant_conformers()
        assert 1 in solver.redundant_indices
        assert len(solver.valid_indices) == 2

    def test_detects_near_duplicate(self):
        models_near_dup = np.array([
            [6.0, 0.0, 0.0],
            [6.0, 1e-8, 0.0],  # near-duplicate
            [0.0, 0.0, 21.0],
        ])
        solver = self.make_solver(models=models_near_dup)
        solver.find_redundant_conformers(threshold=1e-6)
        assert 1 in solver.redundant_indices

    def test_valid_plus_redundant_equals_total(self):
        models = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        target = np.array([0.5, 0.5])
        solver = self.make_solver(target=target, models=models)
        solver.find_redundant_conformers()
        assert len(solver.valid_indices) + len(solver.redundant_indices) == 5


# ---------------------------------------------------------------------------
# 3. construct_weights tests
# ---------------------------------------------------------------------------

class TestConstructWeights(QUBOTestBase):

    def test_weights_length_matches_nconformers(self):
        solver = self.make_solver()
        solver.find_redundant_conformers()
        solver._weights = np.array([0.3, 0.3, 0.4])
        solver._objective_value = 0.1
        solver.construct_weights()
        assert len(solver.weights) == 3

    def test_redundant_conformer_gets_zero_weight(self):
        models_with_dup = np.array([
            [6.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],  # duplicate
            [0.0, 0.0, 21.0],
        ])
        solver = self.make_solver(models=models_with_dup)
        solver.find_redundant_conformers()
        # _weights only has entries for valid_indices (2 entries)
        solver._weights = np.array([0.5, 0.5])
        solver._objective_value = 0.1
        solver.construct_weights()
        assert solver.weights[1] == 0.0  # redundant index
        assert solver.weights[0] == 0.5
        assert solver.weights[2] == 0.5


# ---------------------------------------------------------------------------
# 4. solve_qubo with mocked D-Wave sampler
# ---------------------------------------------------------------------------

class TestSolveQUBOMocked(QUBOTestBase):
    """Test solve_qubo end-to-end with a mocked D-Wave LeapHybridSampler."""

    @staticmethod
    def _make_mock_sampleset(n_vars, active_bits=None, energy=0.0):
        """Create a mock D-Wave sampleset.

        Parameters
        ----------
        n_vars : int
            Total number of binary variables in the QUBO.
        active_bits : list[int] or None
            Indices of bits to set to 1. If None, all bits are 0.
        energy : float
            The energy value to report.
        """
        sample = {i: 0 for i in range(n_vars)}
        if active_bits is not None:
            for idx in active_bits:
                if idx < n_vars:
                    sample[idx] = 1

        mock_first = MagicMock()
        mock_first.sample = sample
        mock_first.energy = energy

        mock_sampleset = MagicMock()
        mock_sampleset.first = mock_first
        return mock_sampleset

    @patch("qfit.solvers.LeapHybridSampler")
    @patch("qfit.solvers.dimod")
    def test_solve_qubo_runs_without_error(self, mock_dimod, mock_sampler_cls):
        """solve_qubo should complete when D-Wave is mocked."""
        solver = self.make_solver()
        N = 3
        K = 10

        # Compute QUBO to get matrix size
        Q, const = solver.compute_qubo_coeffs(K=K)
        n_vars = Q.shape[0]

        # Mock BQM construction
        mock_bqm = MagicMock()
        mock_dimod.BQM.from_qubo.return_value = mock_bqm

        # Mock sampler - activate some bits in x-block to simulate conformers
        # Activate bits 0-2 for conformer 0, 10-12 for conformer 1, 20-22 for conformer 2
        active = list(range(0, 3)) + list(range(K, K + 3)) + list(range(2 * K, 2 * K + 3))
        mock_sampleset = self._make_mock_sampleset(n_vars, active_bits=active, energy=-0.5)
        mock_sampler_instance = MagicMock()
        mock_sampler_instance.sample.return_value = mock_sampleset
        mock_sampler_cls.return_value = mock_sampler_instance

        solver.solve_qubo(threshold=0.2, cardinality=3)

        # Verify solver produced results
        assert solver.weights is not None
        assert len(solver.weights) == N
        assert solver.objective_value is not None

    @patch("qfit.solvers.LeapHybridSampler")
    @patch("qfit.solvers.dimod")
    def test_solve_qubo_weights_from_unary_encoding(self, mock_dimod, mock_sampler_cls):
        """Weights should equal sum(x[i*K:(i+1)*K]) * Delta for each conformer."""
        solver = self.make_solver()
        N = 3
        K = 10
        Delta = 0.1

        Q, const = solver.compute_qubo_coeffs(K=K)
        n_vars = Q.shape[0]

        mock_bqm = MagicMock()
        mock_dimod.BQM.from_qubo.return_value = mock_bqm

        # Set exactly 5 bits active for conformer 0, 3 for conformer 1, 0 for conformer 2
        active = list(range(0, 5)) + list(range(K, K + 3))
        mock_sampleset = self._make_mock_sampleset(n_vars, active_bits=active, energy=-1.0)
        mock_sampler_instance = MagicMock()
        mock_sampler_instance.sample.return_value = mock_sampleset
        mock_sampler_cls.return_value = mock_sampler_instance

        solver.solve_qubo(threshold=0.2, cardinality=3)

        # conformer 0: 5 bits * 0.1 = 0.5
        # conformer 1: 3 bits * 0.1 = 0.3
        # conformer 2: 0 bits * 0.1 = 0.0
        assert np.isclose(solver.weights[0], 0.5, atol=1e-10)
        assert np.isclose(solver.weights[1], 0.3, atol=1e-10)
        assert np.isclose(solver.weights[2], 0.0, atol=1e-10)

    @patch("qfit.solvers.LeapHybridSampler")
    @patch("qfit.solvers.dimod")
    def test_solve_qubo_objective_value_scaling(self, mock_dimod, mock_sampler_cls):
        """Objective should be energy * scale + const."""
        solver = self.make_solver()
        K = 10

        Q, const = solver.compute_qubo_coeffs(K=K)
        n_vars = Q.shape[0]
        scale = max(abs(Q).max(), 1.0)

        mock_bqm = MagicMock()
        mock_dimod.BQM.from_qubo.return_value = mock_bqm

        test_energy = -2.5
        mock_sampleset = self._make_mock_sampleset(n_vars, active_bits=[], energy=test_energy)
        mock_sampler_instance = MagicMock()
        mock_sampler_instance.sample.return_value = mock_sampleset
        mock_sampler_cls.return_value = mock_sampler_instance

        solver.solve_qubo(threshold=0.2, cardinality=3)

        expected_objective = test_energy * scale + const
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    @patch("qfit.solvers.LeapHybridSampler")
    @patch("qfit.solvers.dimod")
    def test_solve_qubo_calls_dwave_sampler(self, mock_dimod, mock_sampler_cls):
        """Verify that LeapHybridSampler.sample() is actually called."""
        solver = self.make_solver()
        K = 10

        Q, const = solver.compute_qubo_coeffs(K=K)
        n_vars = Q.shape[0]

        mock_bqm = MagicMock()
        mock_dimod.BQM.from_qubo.return_value = mock_bqm

        mock_sampleset = self._make_mock_sampleset(n_vars, energy=0.0)
        mock_sampler_instance = MagicMock()
        mock_sampler_instance.sample.return_value = mock_sampleset
        mock_sampler_cls.return_value = mock_sampler_instance

        solver.solve_qubo(threshold=0.2, cardinality=3)

        # Verify D-Wave was called
        mock_sampler_cls.assert_called_once()
        mock_sampler_instance.sample.assert_called_once()
        call_args = mock_sampler_instance.sample.call_args
        assert call_args[1]["label"] == "qfit-QUBO-hybrid"
