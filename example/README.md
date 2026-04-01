## Advanced qFit-Quantum features and options

Some of the advanced and specialized options available in qFit-Quantum are demonstrated below. The PDB and map files for each of the examples are placed within their corresponding folders.

> **Note:** This repository focuses on ligand multiconformer modeling (X-ray and cryo-EM). Full protein modeling with `qfit_protein` is supported in the codebase but is not demonstrated here, as it invokes the D-Wave QUBO solver per residue and consumes significant solver time. For protein-level examples, refer to the original [qFit-3.0 repository](https://github.com/ExcitedStates/qfit-3.0).

### 1. Running qFit on cryo-EM structures

qFit can use ccp4 map files as input. To model alternate conformers using
this type of map, it is also necessary to provide the resolution of the data,
which can be achieved by using the flag *-r*.

`qfit_protein [MAP_FILE] [PDB_FILE] -r [RESOLUTION] -em`

#### You also must use the -em flag for cryo-EM structures.

For cryo-EM ccp4 maps, you can use the example from the Apoferritin Chain A (PDB:7A4M).

`qfit_protein qfit_cryoem_example/7A4M_box.ccp4 qfit_cryoem_example/7A4M_box.pdb -r 1.22 -em`

If you would like, you can use [qscore](https://github.com/gregdp/mapq) to determine which residues should be modeled using qFit. After running qscore, run qFit protein using the following command:

`qfit_protein qfit_cryoem_example/7A4M_box.ccp4 qfit_cryoem_example/7A4M_box.pdb -r 1.22 -em --qscore 7A4M.pdb__Q__apoF_chainA.ccp4_All.txt`

After *multiconformer_model2.pdb* has been generated, refine this model using:
`qfit_final_refine_cryoem.sh qfit_cryoem_example/apoF_chainA.ccp4 qfit_cryoem_example/apoF_chainA.pdb multiconformer_model2.pdb`

Note: a pre-generated *multiconformer_model2.pdb* file is placed in the folder for reference.
Bear in mind that this final step currently depends on an existing installation
of the [Phenix software suite](https://phenix-online.org/).

### 2. Modeling alternate conformers of a ligand (X-ray)

To generate a composite omit map for ligands, we recommend running without accounting for bulk solvent.

`phenix.composite_omit_map input.mtz model.pdb omit-type=refine exclude_bulk_solvent=True`

To model alternate conformers of ligands, the command line tool *qfit_ligand*
should be used:

`qfit_ligand [COMPOSITE_OMIT_MAP_FILE] -l [LABEL] [PDB_FILE] [CHAIN,LIGAND] -sm [SMILES]`

Where *LIGAND* corresponds to the numeric identifier of the ligand on the PDB
(aka res. number). The main output file is named *multiconformer_ligand_bound_with_protein.pdb*

If you wish to specify the number of ligand conformers for qFit to sample, use the flag `-nc [NUM_CONFS]`. The default number is set to 5,000 for ligands smaller than 25 heavy atoms, and 7,000 otherwise. In addition, the *qfit_ligand* program can be executed in parallel and the number of concurrent processes can be adjusted using the *-p* flag.

Using the example 4MS6:

`qfit_ligand qfit_ligand_example/4ms6_composite_map.mtz -l 2FOFCWT,PH2FOFCWT qfit_ligand_example/4ms6.pdb A,702 -sm 'C1C[C@H](NC1)C(=O)CCC(=O)N2CCC[C@H]2C(=O)O' -nc 7000`


To refine *multiconformer_ligand_bound_with_protein.pdb*, use the following command

`qfit_final_refine_ligand.sh 4ms6.mtz`

### 3. Modeling alternate conformers of a ligand on an event map

To run *qfit_ligand* on an event map, you must change the labels and include the resolution.

`qfit_ligand [EVENT_MAP_FILE] -l [LABEL] [PDB_FILE] -r [RESOLUTION] [CHAIN,LIGAND] -sm [SMILES]`

Using the example x3200:

`qfit_ligand qfit_ligand_example/x3200_event_map.native.ccp4 -l FWT,PHWT qfit_ligand_example/singl_conf_x3200_pandda_model.pdb -r 1.05 A,201 -sm 'O=C1CCCN1NC2=NC=NC=C2C3=C(F)C=CC=N3'`

### 4. Modeling alternate conformers of a ligand on a cryo-EM map

To run *qfit_ligand* on a cryo-EM map, we suggest using the flag `-em_lig` to reduce the risk of overfitting. You must also include the resolution.

`qfit_ligand [MAP] -l [LABEL] [PDB_FILE] -r [RESOLUTION] [CHAIN,LIGAND] -sm [SMILES] -em_lig`

Using the example 8P70:

`qfit_ligand EMD-17513.map qfit_ligand_example/8P70.pdb -r 2.0 A,201 -sm 'O=C1CCCN1NC2=NC=NC=C2C3=C(F)C=CC=N3' -em_lig`

The link to the map can be found here: https://www.ebi.ac.uk/emdb/EMD-17513

To refine *multiconformer_ligand_bound_with_protein.pdb*, use the following command

`qfit_final_refine_cryoem_ligand.sh EMD-17513.map 2.0`
