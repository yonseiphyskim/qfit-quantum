import inspect

import numpy as np
import pytest
from numpy.typing import NDArray

import qfit.solvers
from qfit.solvers import (
    available_qubo_solvers,
    available_qp_solvers,
    get_qubo_solver_class,
    get_qp_solver_class,
)


def test_missing_solvers() -> None:
    with pytest.raises(KeyError):
        get_qp_solver_class("NotASolver")
    with pytest.raises(KeyError):
        get_qubo_solver_class("NotASolver")


def test_get_qp_solver() -> None:
    qp_solver_class = get_qp_solver_class(next(iter(available_qp_solvers.keys())))
    assert inspect.isclass(qp_solver_class)
    assert issubclass(qp_solver_class, qfit.solvers.QPSolver)


def test_get_qubo_solver() -> None:
    qubo_solver_class = get_qp_solver_class(next(iter(available_qubo_solvers.keys())))
    assert inspect.isclass(qubo_solver_class)
    assert issubclass(qubo_solver_class, qfit.solvers.QUBOSolver)


@pytest.mark.parametrize("solver_class", available_qp_solvers.values())
class TestQPSolver:
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])

    def test_qp_solver(self, solver_class: type[qfit.solvers.QPSolver]) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qp()

        assert np.allclose(solver.weights, [1 / 3, 1 / 3, 1 / 3], atol=1e-3)
        assert np.isclose(solver.objective_value, 0.0, atol=1e-6)


@pytest.mark.parametrize("solver_class", available_qubo_solvers.values())
class TestQUBOSolver:
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])

    def expected_objective(self, expected_weights: NDArray[np.float_]) -> float:
        return np.sum(np.square(np.inner(self.models, expected_weights) - self.target))

    def test_qubo_solver_with_threshold(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qubo(threshold=0.4)

        expected_weights = np.array([0.0, 0.4, 0.4])
        expected_objective = self.expected_objective(expected_weights)

        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_qubo_solver_with_cardinality_3(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qubo(cardinality=3)

        expected_weights = np.array([1 / 3, 1 / 3, 1 / 3])
        expected_objective = 0.0
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-4)

    def test_qubo_solver_with_cardinality_2(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qubo(cardinality=2)

        expected_weights = np.array([0.0, 1 / 3, 1 / 3])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_qubo_solver_with_cardinality_1(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qubo(cardinality=1)

        expected_weights = np.array([0.0, 0.0, 1 / 3])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_qubo_solver_with_threshold_and_cardinality_1(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qubo(threshold=0.4, cardinality=1)

        expected_weights = np.array([0.0, 0.0, 0.4])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)


@pytest.mark.parametrize("solver_class", available_qubo_solvers.values())
class TestQUBOSolverReuse:
    target = np.array([2.0, 3.0, 7.0])
    models = np.array([[6.0, 0.0, 0.0], [0.0, 9.0, 0.0], [0.0, 0.0, 21.0]])

    @pytest.fixture
    def solver(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> qfit.solvers.QUBOSolver:
        """Instantiate a solver within the scope of this class, to be re-used."""
        return solver_class(self.target, self.models)

    def expected_objective(self, expected_weights: NDArray[np.float_]) -> float:
        return np.sum(np.square(np.inner(self.models, expected_weights) - self.target))

    def test_qubo_solver_with_threshold_and_cardinality_1(
        self, solver_class: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver = solver_class(self.target, self.models)
        solver.solve_qubo(threshold=0.4, cardinality=1)

        expected_weights = np.array([0.0, 0.0, 0.4])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_qubo_solver_with_cardinality_1(
        self, solver: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver.solve_qubo(cardinality=1)

        expected_weights = np.array([0.0, 0.0, 1 / 3])
        expected_objective = self.expected_objective(expected_weights)
        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)

    def test_qubo_solver_with_threshold(
        self, solver: type[qfit.solvers.QUBOSolver]
    ) -> None:
        solver.solve_qubo(threshold=0.4)

        expected_weights = np.array([0.0, 0.4, 0.4])
        expected_objective = self.expected_objective(expected_weights)

        assert np.allclose(solver.weights, expected_weights, atol=1e-3)
        assert np.isclose(solver.objective_value, expected_objective, atol=1e-6)
