# qFit-Quantum

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**qFit-Quantum** extends [qFit](https://github.com/ExcitedStates/qfit-3.0) by replacing the classical Mixed-Integer Quadratic Programming (MIQP) occupancy solver with a Quadratic Unconstrained Binary Optimization (QUBO) formulation solved on D-Wave quantum-classical hybrid hardware.

qFit is a collection of programs for modeling multiconformer protein structures from X-ray crystallography and cryo-EM density maps. In the multiconformer optimization pipeline, selecting which conformers to include and determining their occupancies is a combinatorial optimization problem. qFit-Quantum reformulates this step as a QUBO and solves it via quantum annealing, enabling broader search of the solution space while maintaining or improving model quality.


## What's Different from qFit-3.0

The original qFit-3.0 solves the conformer selection problem using a two-stage **QP → MIQP** pipeline. MIQP uses hard constraints (sum-to-one, minimum occupancy threshold, cardinality limit) that can exclude viable low-occupancy conformers from the search space. Additionally, MIQP is NP-hard and its computational cost grows rapidly with the number of candidate conformers.

qFit-Quantum replaces the MIQP stage with a **QP → QUBO** pipeline:

1. **QUBO formulation** — Continuous occupancy weights are discretized via unary encoding (each w_i is represented as a sum of K binary variables scaled by Δ = 1/K). Inequality constraints (sum-to-one, upper/lower bounds, cardinality) are converted to equality constraints using slack variables, then incorporated into the objective as quadratic penalty terms with Lagrange multipliers λ.

2. **Quantum annealing** — The resulting QUBO matrix is submitted to D-Wave's `LeapHybridSampler`. Quantum tunneling allows the solver to escape local minima more efficiently than classical branch-and-bound methods, enabling faster exploration of the combinatorial search space.

3. **Soft constraints** — Unlike MIQP's hard constraints, the penalty-based approach allows the solver to accept slight constraint violations when the density-fit improvement outweighs the penalty cost. This results in more flexible conformer selection, often recovering additional low-occupancy states supported by the experimental density.

Key code changes are localized to `src/qfit/solvers.py` (QUBO coefficient assembly and D-Wave integration), `src/qfit/qfit.py` (solver dispatch), and CLI options (`--qubo-solver` replaces `--miqp-solver`).


## Citation

> **[Paper title placeholder]**
> [Author list placeholder]
> [Journal/preprint placeholder] ([year])
> [DOI/link placeholder]

Please also cite the original qFit papers:

- [Wankowicz SA et al. Uncovering Protein Ensembles: Automated Multiconformer Model Building for X-ray Crystallography and Cryo-EM. eLife. (2024)](https://doi.org/10.7554/eLife.90606.3)
- [Riley BT et al. qFit 3: Protein and ligand multiconformer modeling for X-ray crystallographic and single-particle cryo-EM density maps. Protein Sci. 30, 270–285 (2021)](https://dx.doi.org/10.1002/pro.4001)
- [Keedy DA et al. Exposing Hidden Alternative Backbone Conformations in X-ray Crystallography Using qFit. PLoS Comput. Biol. 11, e1004507 (2015)](https://dx.doi.org/10.1371/journal.pcbi.1004507)

As this software relies on CVXPY, please also cite:
- [Agrawal et al. A Rewriting System for Convex Optimization Problems. J. Control and Decision. (2018).](https://arxiv.org/abs/1709.04494)
- [Diamond & Boyd. CVXPY: A Python-Embedded Modeling Language for Convex Optimization. JMLR. (2016)](https://www.jmlr.org/papers/volume17/15-408/15-408.pdf)


## Installation

### 1. Clone and create environment

```bash
git clone -b main https://github.com/yonseiphyskim/qfit-quantum.git
cd qfit-quantum
mamba env create -f environment.yml
mamba activate qfit-quantum
pip install .
```

### 2. Configure D-Wave access

A [D-Wave Leap](https://cloud.dwavesys.com/leap/) account is required to run the QUBO solver. After creating an account, configure your API token:

```bash
dwave config create
```

This will prompt for your API token and endpoint. Verify your setup with:

```bash
dwave ping
```

For details, see the [D-Wave Ocean documentation](https://docs.ocean.dwavesys.com/en/stable/overview/install.html).

### Advanced

If you prefer to manage environments manually, qFit-Quantum requires:

- [Python 3.9+](https://python.org)
- [numpy](https://numpy.org), [scipy](https://scipy.org)
- [cvxpy](https://www.cvxpy.org)
- [dimod](https://docs.ocean.dwavesys.com/en/stable/docs_dimod/), [dwave-system](https://docs.ocean.dwavesys.com/en/stable/docs_system/sdk_index.html)


## Usage

### Ligand modeling (X-ray)

To model alternate conformations of ligands, first generate a composite omit map excluding bulk solvent:

```bash
phenix.composite_omit_map input.mtz model.pdb omit-type=refine exclude_bulk_solvent=True
```

Then run qFit-ligand:

```bash
qfit_ligand [COMPOSITE_OMIT_MAP_FILE] [PDB_FILE] -l [LABELS] [SELECTION] -sm [SMILES]
```

Example (PDB: 4MS6):

```bash
qfit_ligand example/qfit_ligand_example/4ms6_composit_map.mtz \
  example/qfit_ligand_example/4ms6.pdb \
  -l 2FOFCWT,PH2FOFCWT A,702 \
  -sm 'C1C[C@H](NC1)C(=O)CCC(=O)N2CCC[C@H]2C(=O)O' -nc 10000
```

The results are output to `multiconformer_ligand_bound_with_protein.pdb` (protein-ligand complex) and `multiconformer_ligand_only.pdb` (ligand alone). After running, perform a final refinement:

```bash
qfit_final_refine_ligand.sh 4ms6.mtz
```

### Cryo-EM

```bash
qfit_protein [MAP_FILE] -r [RES] [PDB_FILE] -em
```

Example (PDB: 7A4M):

```bash
qfit_protein example/qfit_cryoem_example/7A4M_box.ccp4 \
  -r 1.7 \
  example/qfit_cryoem_example/7A4M_box.pdb -em
```

After `multiconformer_model2.pdb` is generated, refine with:

```bash
qfit_final_refine_cryoEM.sh example/qfit_cryoem_example/7A4M_box.ccp4 \
  example/qfit_cryoem_example/multiconformer_model2.pdb \
  example/qfit_cryoem_example/7A4M_box.pdb
```

More options and examples are in the [example](example/README.md) directory.


## Contributing

qFit uses [Black](https://github.com/psf/black) for code formatting. Before committing:

```bash
python3 -m pip install --user black
git config core.hooksPath .githooks/
```


## License

The code is licensed under the MIT licence (see `LICENSE`).

Several modules were taken from the `pymmlib` package, originally licensed under the Artistic License 2.0. See the `licenses` directory for details.
