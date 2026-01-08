#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ==================================================
# FORCE qfit.xtal.volume.XMap.interpolate() TO RETURN SCALAR
# (YOUR qFit ENVIRONMENT SAFE)
# ==================================================
import numpy as _np
import qfit.xtal.volume

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
    p = ArgumentParser(description="EDIA comparison (deposited vs qFit)")
    p.add_argument("base_structure", type=str, help="Base deposited PDB")
    p.add_argument("map", type=str, help="Density map (MTZ/MRC/CCP4)")
    p.add_argument("--comp_pdb", type=str, required=True, help="qFit refined PDB")
    p.add_argument("--residue", type=str, help="ChainID,Residue")
    p.add_argument(
        "-l", "--label",
        default="2FOFCWT,PH2FOFCWT",
        help="MTZ label pair"
    )
    p.add_argument("--pdb", type=str, required=True, help="PDB name")
    p.add_argument("--directory", type=str, default="", help="Output folder")
    return p


# --------------------------------------------------
# SAFETY
# --------------------------------------------------
def safe_atom_count(struct):
    if struct is None:
        return 0
    if getattr(struct, "coor", None) is None:
        return 0
    return struct.coor.shape[0]


# --------------------------------------------------
# SAFE EDIA → ALWAYS FLOAT
# --------------------------------------------------
def safe_edia(validator, ligand):
    try:
        result = validator.edia_like_for_atom(ligand)
    except Exception as e:
        print(f"[WARN] EDIA computation failed: {e}")
        return np.nan

    # float
    if isinstance(result, (float, int, np.floating)):
        return float(result)

    # dict {"edia_like": ...}
    if isinstance(result, dict) and "edia_like" in result:
        v = result["edia_like"]
        if isinstance(v, np.ndarray):
            return float(np.nanmean(v))
        if isinstance(v, (float, int, np.floating)):
            return float(v)

    # list of dicts
    if isinstance(result, list):
        vals = []
        for x in result:
            if isinstance(x, dict) and "edia_like" in x:
                v = x["edia_like"]
                if isinstance(v, np.ndarray):
                    vals.append(np.nanmean(v))
                elif isinstance(v, (float, int, np.floating)):
                    vals.append(v)
        return float(np.nanmean(vals)) if vals else np.nan

    return np.nan


# --------------------------------------------------
# MAP EXTRACTION
# --------------------------------------------------
def extract_map_region(xmap, structure):
    try:
        return xmap.extract(structure.coor, padding=8)
    except Exception as e:
        print(f"[WARN] Map extraction failed: {e}")
        return xmap


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    options = build_argparser().parse_args()

    dep_structure = Structure.fromfile(options.base_structure)
    gen_structure = Structure.fromfile(options.comp_pdb)

    # Remove waters
    dep_structure = dep_structure.extract("resn", "HOH", "!=")
    gen_structure = gen_structure.extract("resn", "HOH", "!=")

    edia_data = {
        "Chain": [],
        "Residue": [],
        "Base_EDIA": [],
        "Comparison_EDIA": []
    }

    # Load map
    xmap = XMap.fromfile(options.map, label=options.label)
    scaler = MapScaler(xmap)
    xmap = xmap.canonical_unit_cell()

    # -----------------------------
    # SINGLE RESIDUE
    # -----------------------------
    if options.residue:
        chain, resi = options.residue.split(",")

        dep_res = dep_structure.extract(f"resi {resi} and chain {chain}")
        gen_res = gen_structure.extract(f"resi {resi} and chain {chain}")

        if safe_atom_count(dep_res) == 0 or safe_atom_count(gen_res) == 0:
            print(f"[SKIP] empty residue {chain}{resi}")
            return

        combined = gen_res.combine(dep_res)
        scaler.scale(combined, radius=1.5)
        submap = extract_map_region(xmap, combined)

        validator = Validator(submap, xmap.resolution, options.directory)

        edia_data["Chain"].append(chain)
        edia_data["Residue"].append(resi)
        edia_data["Base_EDIA"].append(safe_edia(validator, dep_res))
        edia_data["Comparison_EDIA"].append(safe_edia(validator, gen_res))

    # -----------------------------
    # ALL RESIDUES
    # -----------------------------
    else:
        combined = gen_structure.combine(dep_structure)
        scaler.scale(combined, radius=1.5)
        submap = extract_map_region(xmap, combined)

        validator = Validator(submap, xmap.resolution, options.directory)

        for chain in np.unique(dep_structure.chain):
            for resi in np.unique(dep_structure.extract("chain", chain, "==").resi):
                dep_res = dep_structure.extract(f"resi {resi} and chain {chain}")
                gen_res = gen_structure.extract(f"resi {resi} and chain {chain}")

                if safe_atom_count(dep_res) == 0 or safe_atom_count(gen_res) == 0:
                    continue

                edia_data["Chain"].append(chain)
                edia_data["Residue"].append(resi)
                edia_data["Base_EDIA"].append(safe_edia(validator, dep_res))
                edia_data["Comparison_EDIA"].append(safe_edia(validator, gen_res))

    df = pd.DataFrame(edia_data)
    out_csv = os.path.join(options.directory, f"{options.pdb}_edia.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")


if __name__ == "__main__":
    main()

