#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==================================================
# FORCE XMap.interpolate() TO RETURN SCALAR
# (CRYO-EM VECTOR BUG FIX)
# ==================================================
import qfit.xtal.volume
import numpy as _np

_orig_interpolate = qfit.xtal.volume.XMap.interpolate

def _interpolate_scalar(self, *args, **kwargs):
    v = _orig_interpolate(self, *args, **kwargs)
    if isinstance(v, _np.ndarray):
        return float(v.ravel()[0])
    return float(v)

qfit.xtal.volume.XMap.interpolate = _interpolate_scalar
# ==================================================


from argparse import ArgumentParser
import numpy as np
import pandas as pd
from qfit.xtal.scaler import MapScaler
from qfit.structure import Structure
from qfit.xtal.volume import XMap
from qfit.validator import Validator
import os


# --------------------------------------------------
# ARGUMENTS
# --------------------------------------------------
def build_argparser():
    p = ArgumentParser(description="CRYO-EM EDIAm comparison (deposited vs qFit)")
    p.add_argument("base_structure", type=str, help="Deposited (baseline) PDB")
    p.add_argument("map", type=str, help="Cryo-EM density map (MRC/CCP4/MAP)")
    p.add_argument("--resolution", type=float, required=True,
                   help="Map resolution in Angstrom (required for cryo-EM)")
    p.add_argument("--comp_pdb", type=str, required=True,
                   help="qFit refined multiconformer PDB")
    p.add_argument("--residue", type=str,
                   help="Optional: ChainID,ResidueNumber")
    p.add_argument("--pdb", type=str, required=True,
                   help="PDB ID (used for output filename)")
    p.add_argument("--directory", type=str, default="",
                   help="Output directory")
    return p


# --------------------------------------------------
# SAFETY
# --------------------------------------------------
def atom_count(struct):
    if struct is None:
        return 0
    if getattr(struct, "coor", None) is None:
        return 0
    return struct.coor.shape[0]


# --------------------------------------------------
# EDIA OUTPUT → SCALAR MEAN (SAFE)
# --------------------------------------------------
def edia_mean(v):
    if isinstance(v, (float, int, np.floating)):
        return float(v)
    if isinstance(v, np.ndarray):
        return float(np.nanmean(v))
    if isinstance(v, dict) and "edia_like" in v:
        return edia_mean(v["edia_like"])
    return np.nan


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    options = build_argparser().parse_args()

    # Load structures
    dep_structure = Structure.fromfile(options.base_structure)
    gen_structure = Structure.fromfile(options.comp_pdb)

    # Remove waters
    dep_structure = dep_structure.extract("resn", "HOH", "!=")
    gen_structure = gen_structure.extract("resn", "HOH", "!=")

    # Output storage
    edia_data = {
        "Chain": [],
        "Residue": [],
        "Deposited_EDIAm": [],
        "QFIT_EDIAm": []
    }

    # --------------------------------------------------
    # LOAD CRYO-EM MAP
    # --------------------------------------------------
    xmap = XMap.fromfile(
        options.map,
        resolution=options.resolution
    )
    scaler = MapScaler(xmap)
    xmap = xmap.canonical_unit_cell()

    # --------------------------------------------------
    # SINGLE RESIDUE MODE
    # --------------------------------------------------
    if options.residue:
        chain, resi = options.residue.split(",")

        dep_res = dep_structure.extract(f"resi {resi} and chain {chain}")
        gen_res = gen_structure.extract(f"resi {resi} and chain {chain}")

        if atom_count(dep_res) == 0 or atom_count(gen_res) == 0:
            print(f"[SKIP] empty residue {chain}{resi}")
            return

        combined = gen_res.combine(dep_res)
        scaler.scale(combined, radius=1.5)
        submap = xmap.extract(combined.coor, padding=8)

        validator = Validator(submap, options.resolution, options.directory)

        dep_edia = validator.edia_like_for_atom(dep_res)
        gen_edia = validator.edia_like_for_atom(gen_res)

        edia_data["Chain"].append(chain)
        edia_data["Residue"].append(resi)
        edia_data["Deposited_EDIAm"].append(edia_mean(dep_edia))
        edia_data["QFIT_EDIAm"].append(edia_mean(gen_edia))

    # --------------------------------------------------
    # ALL RESIDUES MODE
    # --------------------------------------------------
    else:
        combined = gen_structure.combine(dep_structure)
        scaler.scale(combined, radius=1.5)
        submap = xmap.extract(combined.coor, padding=8)

        validator = Validator(submap, options.resolution, options.directory)

        for chain in np.unique(dep_structure.chain):
            residues = np.unique(
                dep_structure.extract("chain", chain, "==").resi
            )

            for resi in residues:
                dep_res = dep_structure.extract(f"resi {resi} and chain {chain}")
                gen_res = gen_structure.extract(f"resi {resi} and chain {chain}")

                if atom_count(dep_res) == 0 or atom_count(gen_res) == 0:
                    continue

                dep_edia = validator.edia_like_for_atom(dep_res)
                gen_edia = validator.edia_like_for_atom(gen_res)

                edia_data["Chain"].append(chain)
                edia_data["Residue"].append(resi)
                edia_data["Deposited_EDIAm"].append(edia_mean(dep_edia))
                edia_data["QFIT_EDIAm"].append(edia_mean(gen_edia))

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------
    df = pd.DataFrame(edia_data)
    out_csv = os.path.join(
        options.directory,
        f"{options.pdb}_edia.csv"
    )
    df.to_csv(out_csv, index=False)
    print("Saved ->", out_csv)


if __name__ == "__main__":
    main()

