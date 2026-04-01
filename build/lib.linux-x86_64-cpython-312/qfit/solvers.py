from __future__ import annotations

import inspect
import logging
import sys
from abc import ABC, abstractmethod
from types import ModuleType
from typing import TYPE_CHECKING, Any, Optional, cast  # pylint: disable=unused-import

import numpy as np
import scipy as sci  # pylint: disable=unused-import
import scipy.sparse  # pylint: disable=unused-import
from numpy.typing import NDArray

import cvxpy as cp

from .utils.optional_lazy_import import lazy_load_module_if_available
import dimod
from dwave.system import LeapHybridSampler

logger = logging.getLogger(__name__)

SolverError: tuple[type[Exception], ...] = RuntimeError

__all__ = [
    "available_qp_solvers",
    "available_qubo_solvers",
    "get_qp_solver_class",
    "get_qubo_solver_class",
]
###############################
# Define solver "interfaces" / Abstractions
#   All solvers in this module must subclass either QPSolver or QUBOSolver, and conform to the interface.
#   The functions in the "Helper Methods" section depend on the subclass membership of these ABCs.
###############################
class GenericSolver(ABC):
    # Class variables
    driver_pkg_name: str
    driver: Optional[ModuleType]

    # Instance variables
    target: NDArray[np.float_]  # 1D array of shape (n_voxels,)
    models: NDArray[np.float_]  # 2D array of shape (n_models, n_voxels,)

    weights: Optional[NDArray[np.float_]] = None
    objective_value: Optional[float] = None
class QPSolver(GenericSolver):
    """Finds the combination of conformer-occupancies that minimizes difference density.

    Problem statement
    -----------------
    We have observed density ρ^o from the user-provided map (target).
    We also have a set of conformers, each with modelled/calculated density ρ^c_i.
    We want find the vector of occupancies ω = <ω_0, ..., ω_n> that minimizes
        the difference between the observed and modelled density --- that minimizes
        a residual sum-of-squares function, rss(ω).
    Mathematically, we wish to minimize:
        min_ω rss(ω) = min_ω || ρ^c ω - ρ^o ||^2

    Expanding & rearranging rss(ω):
        rss(ω) = ( ρ^c ω - ρ^o ).T  ( ρ^c ω - ρ^o )
                = ω.T ρ^c.T ρ^c ω - 2 ρ^o.T ρ^c ω + ρ^o.T ρ^o
    We can rewrite this as
        rss(ω) = ω.T P ω + 2 q.T ω + C
    where
        P =  ρ^c.T ρ^c
        q = -ρ^c.T ρ^o
        C =  ρ^o.T ρ^o

    Noting that the canonical QP objective function is
        g(x) = 1/2 x.T P x + q.T x
    we can use a QP solver to find min_x g(x), which, by equivalence,
        will provide the solution to min_ω rss(ω).

    Solution constraints
    --------------------
    Furthermore, these occupancies are meaningful parameters, so we require
    that their sum is within the unit interval:
        Σ ω_i ≤ 1
    and that each individual occupancy is a positive fractional number:
        0 ≤ ω_i ≤ 1
    """

    @abstractmethod
    def solve_qp(self) -> None: ...
class QUBOSolver(GenericSolver):
    """Finds the combination of conformer-occupancies that minimizes difference density.

    Problem statement
    -----------------
    We have observed density ρ^o from the user-provided map (target).
    We also have a set of conformers, each with modelled/calculated density ρ^c_i.
    We want find the vector of occupancies ω = <ω_0, ..., ω_n> that minimizes
        the difference between the observed and modelled density --- that minimizes
        a residual sum-of-squares function, rss(ω).
    Mathematically, we wish to minimize:
        min_ω rss(ω) = min_ω || ρ^c ω - ρ^o ||^2

    Expanding & rearranging rss(ω):
        rss(ω) = ( ρ^c ω - ρ^o ).T  ( ρ^c ω - ρ^o )
                = ω.T ρ^c.T ρ^c ω - 2 ρ^o.T ρ^c ω + ρ^o.T ρ^o
    We can rewrite this as
        rss(ω) = ω.T P ω + 2 q.T ω + C
    where
        P =  ρ^c.T ρ^c
        q = -ρ^c.T ρ^o
        C =  ρ^o.T ρ^o

    Noting that the canonical QP objective function is
        g(x) = 1/2 x.T P x + q.T x
    we can use a QP solver to find min_x g(x), which, by equivalence,
        will provide the solution to min_ω rss(ω).

    Solution constraints
    --------------------
    Furthermore, these occupancies are meaningful parameters, so we require
    that their sum is within the unit interval:
        Σ ω_i ≤ 1

    We also want to have either:
        (a) a set of conformers of known size (cardinality), or
        (b) a set of conformers with _at least_ threshold occupancy, or else zero (threshold).
    This can be achieved with a mixed-integer linear constraint:
        z_i t_min ≤ ω_i ≤ z_i
    where
        z_i ∈ {0, 1}
        t_min is the minimum-allowable threshold value for ω.
    """

    @abstractmethod
    def solve_qubo(
        self,
        threshold: Optional[float] = None,
        cardinality: Optional[int] = None,
        exact: bool = False,
    ) -> None: ...
###############################
# Define solver implementations
###############################
class CVXPYSolver(QPSolver, QUBOSolver):
    driver_pkg_name = "cvxpy"
    driver = lazy_load_module_if_available(driver_pkg_name)

    def __init__(self, target, models, in_model=None, nthreads=1):
        self.target = target
        self.models = models
        self.in_model = in_model
        self.quad_obj = None
        self.lin_obj = None
        self.qubo_obj = None
        self.qubo_const = None
        
        self.nconformers = models.shape[0]
        self.valid_indices = []
        self.redundant_indices = []

        self._weights = None
        self._objective_value = 0
        self.weights = None

    def find_redundant_conformers(self, threshold=1e-6):

        self.valid_indices = []
        self.redundant_indices = []

        for i in range(self.nconformers):
            if i in self.redundant_indices:
                continue
            self.valid_indices.append(i)
            for j in range(i + 1, self.nconformers):
                if j in self.redundant_indices:
                    continue
                if np.linalg.norm(self.models[i] - self.models[j]) < threshold:
                    self.redundant_indices.append(j)
        assert len(self.valid_indices) + len(self.redundant_indices) == self.nconformers

    def compute_quadratic_coeffs(self):
        # minimize 0.5 x.T P x + q.T x
        #   where P = self.quad_obj =   ρ_model.T ρ_model
        #         q = self.lin_obj  = - ρ_model.T ρ_obs
        # note that ρ_model is the transpose of self.models
        self.find_redundant_conformers()
        self.quad_obj = (
            self.models[self.valid_indices] @ self.models[self.valid_indices].T
        )
        self.lin_obj = -1 * self.models[self.valid_indices] @ self.target
    

    def compute_qubo_coeffs(
        self,
        Delta=0.1,
        lam0=100,
        lam1=100,
        lam2=100,
        lam3=100,
        tmin=None,
        K=10,
        threshold=None,
        cardinality=3,
    ):
        # -----------------------------
        # Handle threshold logic
        # -----------------------------
        if tmin is None:
            tmin = 0.2 if threshold is None else threshold
        # ------------------------------------------------------------
        # Prepare data
        # ------------------------------------------------------------
        self.find_redundant_conformers()
        # Only use valid conformers (non-redundant)
        rho_calc = self.models[self.valid_indices]
        rho_obs = self.target
        N, _ = rho_calc.shape
        # correlation terms
        G = rho_calc @ rho_calc.T        # G_ij = ⟨ρ_calc_i, ρ_calc_j⟩
        f = rho_obs @ rho_calc.T         # f_i = ⟨ρ_obs, ρ_calc_i⟩

        # unary encoding (for block structure)
        M = np.zeros((N, N * K))
        for i in range(N):
            M[i, i*K:(i+1)*K] = 1.0

        # helper vectors
        a = np.ones(N*K)        # for Σ w ≤ 1 constraint
        b = np.ones(K)         # slack for Σ w + s = 1
        S_v = np.zeros((N, N*K))
        for i in range(N):
            S_v[i, i*K:(i+1)*K] = 1.0        # slack (upper)
        S_y = np.zeros((N, N*K))
        for i in range(N):
            S_y[i, i*K:(i+1)*K] = 1.0
        c = np.ones(N)
        d = np.ones(K)

        # ------------------------------------------------------------
        # Core data term (least squares fit)
        # ------------------------------------------------------------
        Q_data = Delta**2 * (M.T @ G @ M)
        c_data = -2 * Delta * (M.T @ f)

        # ------------------------------------------------------------
        # Sum-to-one constraint  (Σ w ≤ 1)
        # ------------------------------------------------------------
        Q_sum1 = lam0 * Delta**2 * np.block([
             [np.outer(a, a), np.outer(a, b)],
             [np.outer(b, a), np.outer(b, b)]
        ])
        c_sum1 = -2 * lam0 * Delta * np.concatenate([a, b])
        const_sum1 = lam0

        # ------------------------------------------------------------
        # Upper bound constraint  (w_i ≤ z_i)
        # ------------------------------------------------------------
        Q_upper = lam1 * np.block([
            [Delta**2*(M.T @ M), Delta**2*(M.T @ S_v), -Delta*M.T],
            [Delta**2*(S_v.T @ M), Delta**2*(S_v.T @ S_v), -Delta*S_v.T],
            [-Delta*M, -Delta*S_v, np.eye(N)]
        ])

        # ------------------------------------------------------------
        # Lower bound constraint  (w_i ≥ t_min z_i)
        # ------------------------------------------------------------
        Q_lower = lam2 * np.block([
            [Delta**2*(M.T @ M), -Delta**2*(M.T @ S_y), -tmin*Delta*M.T],
            [-Delta**2*(S_y.T @ M), Delta**2*(S_y.T @ S_y), tmin*Delta*S_y.T],
            [-tmin*Delta*M, tmin*Delta*S_y, tmin**2*np.eye(N)]
        ])
        
        # ------------------------------------------------------------
        # Sum-to-cardinality constraint  (Σ z ≤ cardinality)
        # ------------------------------------------------------------
        Q_sum2 = lam3 * np.block([
             [np.outer(c, c), Delta*np.outer(c, d)],
             [np.outer(d, c), Delta*np.outer(d, d)]
        ])     
        c_sum2 = -2 * lam3 * cardinality * np.concatenate([c, Delta*d])
        const_sum2 = lam3 * cardinality**2
        # ------------------------------------------------------------
        # Combine into single block matrix
        # ------------------------------------------------------------
        n_x = N*K
        n_u = K
        n_v = N*K
        n_y = N*K
        n_z = N                    
        n_k = K
        n_tot = n_x + n_u + n_v + n_y + n_z + n_k

        Q_total = np.zeros((n_tot, n_tot))
        c_total = np.zeros(n_tot)
        # indices for each variable block
        idx_x = slice(0, n_x)
        idx_u = slice(n_x, n_x+n_u)
        idx_v = slice(n_x+n_u, n_x+n_u+n_v)
        idx_y = slice(n_x+n_u+n_v, n_x+n_u+n_v+n_y)
        idx_z = slice(n_x+n_u+n_v+n_y, n_x+n_u+n_v+n_y+n_z)
        idx_k = slice(n_x+n_u+n_v+n_y+n_z, n_tot)

        # (1) data + sum1 constraint
        Q_total[:n_x, :n_x] += Q_data
        Q_total[:n_x+n_u, :n_x+n_u] += Q_sum1
        c_total[:n_x] += c_data
        c_total[:n_x+n_u] += c_sum1

        # (2) upper constraint
        ix_upper = np.r_[range(idx_x.start, idx_x.stop),
                         range(idx_v.start, idx_v.stop),
                         range(idx_z.start, idx_z.stop)]
        Q_total[np.ix_(ix_upper, ix_upper)] += Q_upper

        # (3) lower constraint
        ix_lower = np.r_[range(idx_x.start, idx_x.stop),
                         range(idx_y.start, idx_y.stop),
                         range(idx_z.start, idx_z.stop)]
        Q_total[np.ix_(ix_lower, ix_lower)] += Q_lower
        
        # (4) cardinality constraint (z + k block)
        start_card = idx_z.start
        end_card = n_tot
        Q_total[start_card:end_card, start_card:end_card] += Q_sum2
        c_total[start_card:end_card] += c_sum2
        
        # symmetrize
        Q_total = 0.5 * (Q_total + Q_total.T)
        Q_total += np.diag(c_total)
        const = (np.dot(rho_obs, rho_obs) + const_sum1 + const_sum2)
    

        # ------------------------------------------------------------
        # Save results for solver
        # ------------------------------------------------------------
        self.qubo_obj = Q_total
        self.qubo_const = const
        

        logger.info(f"Constructed QUBO matrix of size {Q_total.shape}")
        return Q_total, const
        

    def construct_weights(self):
        self.weights = []
        j = 0
        for i in range(self.nconformers):
            if i in self.redundant_indices:
                self.weights.append(0)
            else:
                self.weights.append(self._weights[j])
                j += 1
        self.weights = np.array(self.weights)
        self.objective_value = self._objective_value
        assert len(self.weights) == self.nconformers

    def solve_qubo(self, threshold=0, cardinality=0):
        
        # -----------------------------------------------
        # Step 1. Coefficient preparation
        # -----------------------------------------------
        if self.qubo_obj is None or self.qubo_const is None:
            self.compute_qubo_coeffs()        
        
        self.find_redundant_conformers()
        rho_obs = self.target
        rho_calc = self.models[self.valid_indices]
        N, _ = rho_calc.shape

        # Assemble QUBO with penalty terms
        K = 10
        Delta = 0.1
        lam0, lam1, lam2, lam3 = 100, 100, 100, 100
        tmin = threshold if threshold else 0.2
        Q, const = self.compute_qubo_coeffs(
            Delta=Delta,
            lam0=lam0, lam1=lam1, lam2=lam2, lam3=lam3,
            tmin=tmin, K=K, 
            threshold=threshold,
            cardinality=cardinality,
        )

        # -----------------------------------------------
        # Step 2. Scaling & BQM construction
        # -----------------------------------------------
        scale = max(abs(Q).max(), 1.0)
        Q_scaled = Q / scale

        bqm = dimod.BQM.from_qubo(Q_scaled)

        # -----------------------------------------------
        # Step 3. D-Wave Hybrid Sampling
        # -----------------------------------------------
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm, label="qfit-QUBO-hybrid")

        best_sample = sampleset.first.sample
        objective_value = sampleset.first.energy * scale + const
        sol = np.array([best_sample[i] for i in sorted(best_sample.keys())])

        # -----------------------------------------------
        # Step 4. Extract x-block → continuous w
        # -----------------------------------------------
        n_x = N * K
        x = sol[:n_x]
        w = np.array([np.sum(x[i*K:(i+1)*K]) * Delta for i in range(N)])

        # -----------------------------------------------
        # Step 5. Save results
        # -----------------------------------------------
        self._weights = w
        self.objective_value = objective_value
        self._objective_value = objective_value
        self.construct_weights()

        logger.info("D-Wave hybrid solved QUBO: Objective=%.4f", objective_value)
        
    def rscc_solve_qubo(self, threshold=0, cardinality=0):      
        # -----------------------------------------------
        # Step 1. Coefficient preparation
        # -----------------------------------------------
        if self.qubo_obj is None or self.qubo_const is None:
            self.compute_qubo_coeffs()
   
        rho_obs = self.target
        rho_calc = self.models[self.valid_indices]
        N, _ = rho_calc.shape

        # Assemble QUBO with penalty terms
        K = 10
        Delta = 0.1
        lam0, lam1, lam2, lam3 = 100, 100, 100, 100
        tmin = threshold if threshold else 0.2
        Q, const = self.compute_qubo_coeffs(
            Delta=Delta,
            lam0=lam0, lam1=lam1, lam2=lam2, lam3=lam3,
            tmin=tmin, K=K,
            threshold=threshold,
            cardinality=cardinality,
        )

        # -----------------------------------------------
        # Step 2. Scaling & BQM construction
        # -----------------------------------------------
        scale = max(abs(Q).max(), 1.0)
        Q_scaled = Q / scale

        bqm = dimod.BQM.from_qubo(Q_scaled)

        # -----------------------------------------------
        # Step 3. D-Wave Hybrid Sampling
        # -----------------------------------------------
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm, label="qfit-QUBO-hybrid")

        best_sample = sampleset.first.sample
        objective_value = sampleset.first.energy * scale + const
        sol = np.array([best_sample[i] for i in sorted(best_sample.keys())])

        # -----------------------------------------------
        # Step 4. Extract x-block → continuous w
        # -----------------------------------------------
        n_x = N * K
        x = sol[:n_x]
        w = np.array([np.sum(x[i*K:(i+1)*K]) * Delta for i in range(N)])

        # -----------------------------------------------
        # Step 5. Save results
        # -----------------------------------------------
        self._weights = w
        self.objective_value = objective_value
        self._objective_value = objective_value
        self.construct_weights()
        
        # output the correlation coefficient between the QFIT MODEL density and the target density 
        cutoff=0.002
        filterarray = self._weights >= cutoff
        filtered_weights = self._weights[filterarray]
        filtered_models = self.models[filterarray, :]

        combined_model = np.dot(filtered_weights, filtered_models)
        corr = np.corrcoef(combined_model, self.target)[0, 1]
        print(f"RSCC for model of interest: {corr}")

        # output correlations coefficient between INPUT MODEL density and target density 
        if self.in_model is not None and self.in_model.size > 0:
            input_corr = np.corrcoef(self.in_model, self.target)[0, 1]
            print(f"RSCC for comparision model: {input_corr}")
            
        self.construct_weights()
        
    def solve_qp(self, split_threshold=3000):
        if self.quad_obj is None or self.lin_obj is None:
            self.compute_quadratic_coeffs()

        valid_conformers = len(self.valid_indices)
        self._weights = np.zeros(valid_conformers)
        splits = valid_conformers // split_threshold + 1  # number of splits
        for split in range(splits):
            # take every splits-th element with split as an offset, guaranteeing full coverage
            P = self.quad_obj[split::splits, split::splits]
            q = self.lin_obj[split::splits]
            m = len(P)
            w = cp.Variable(m)
            objective = cp.Minimize(0.5 * cp.quad_form(w, cp.psd_wrap(P)) + q.T @ w)
            constraints = [w >= np.zeros(m), np.ones(m).T @ w <= 1]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            # I'm not sure why objective_values is calculated this way, but doing
            # so to be compatible with the former CPLEXSolver class
            self._objective_value += 2 * prob.value + self.target.T @ self.target
            self._objective_value /= splits
            self._weights[split::splits] = w.value / splits
        self.construct_weights()
###############################
# Helper methods
###############################
def _available_qp_solvers() -> dict[str, type]:
    """List all available QP solver classes in this module."""
    available_solvers = {}

    # Get all classes defined in this module
    #   use module.__dict__ because it preserves order
    #     (unlike dir(module) or inspect.getmembers(module))
    for name, obj in sys.modules[__name__].__dict__.items():
        if inspect.isclass(obj) and obj.__module__ == __name__:
            # Check the class implements QPSolver
            if obj in QPSolver.__subclasses__():
                # Check the driver module is loadable
                if obj.driver is not None:
                    available_solvers[name] = obj
    return available_solvers
def _available_qubo_solvers() -> dict[str, type]:
    """List all available QUBO solver classes in this module."""
    available_solvers = {}

    # Get all classes defined in this module
    #   use module.__dict__ because it preserves order
    #     (unlike dir(module) or inspect.getmembers(module))
    for name, obj in sys.modules[__name__].__dict__.items():
        if inspect.isclass(obj) and obj.__module__ == __name__:
            # Check the class implements QUBOSolver
            if obj in QUBOSolver.__subclasses__():
                # Check the driver module is loadable
                if obj.driver is not None:
                    available_solvers[name] = obj
    return available_solvers
available_qp_solvers = _available_qp_solvers()
available_qubo_solvers = _available_qubo_solvers()
if not available_qp_solvers:
    msg = (
        "Could not find any QP solver engines.\n"
        + "Please ensure that at least one of:\n  "
        + str([solver.driver_pkg_name for solver in QPSolver.__subclasses__()])
        + "\n"
        + "is installed."
    )
    raise ImportError(msg)
if not available_qubo_solvers:
    msg = (
        "Could not find any QUBO solver engines.\n"
        + "Please ensure that at least one of:\n  "
        + str([solver.driver_pkg_name for solver in QUBOSolver.__subclasses__()])
        + "\n"
        + "is installed."
    )
    raise ImportError(msg)
def get_qp_solver_class(solver_type: str) -> type[QPSolver]:
    """Return the class of the requested solver type, or raise a KeyError."""
    return available_qp_solvers[solver_type]
def get_qubo_solver_class(solver_type: str) -> type[QUBOSolver]:
    """Return the class of the requested solver type, or raise a KeyError."""
    return available_qubo_solvers[solver_type]
