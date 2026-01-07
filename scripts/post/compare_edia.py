#!/usr/bin/env python
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from qfit.xtal.scaler import MapScaler
from qfit.structure import Structure
from qfit.xtal.volume import XMap
from qfit.validator import Validator
import os


def build_argparser():
    p = ArgumentParser(description=__doc__)
    p.add_argument("base_structure", type=str, help="Base deposited PDB")
    p.add_argument("map", type=str, help="Density map (MTZ/MRC/CCP4)")
    p.add_argument("--comp_pdb", type=str, help="qFit refined PDB")
    p.add_argument("--residue", type=str, help="ChainID,Residue")
    p.add_argument(
        "-l",
        "--label",
        default="2FOFCWT,PH2FOFCWT",
        help="MTZ label pair"
    )
    p.add_argument("--pdb", type=str, help="PDB name")
    p.add_argument("--directory", type=str, default="", help="Output folder")
    return p


def safe_atom_count(struct):
    if struct is None:
        return 0
    if getattr(struct, "coor", None) is None:
        return 0
    if not isinstance(struct.coor, np.ndarray):
        return 0
    return struct.coor.shape[0]


def safe_edia(validator, ligand):
    try:
        result = validator.edia_like_for_atom(ligand)
    except Exception as e:
        print(f"[WARN] EDIA computation failed: {e}")
        return np.nan

    if isinstance(result, float):
        return result

    if isinstance(result, dict) and "edia_like" in result:
        return result["edia_like"]

    if isinstance(result, list) and len(result) > 0:
        try:
            return np.mean([x["edia_like"] for x in result if "edia_like" in x])
        except:
            return np.nan

    return np.nan


def extract_map_region(xmap, structure):
    try:
        return xmap.extract(structure.coor, padding=8)
    except Exception as e:
        print(f"[WARN] Map extraction failed: {e}")
        return xmap


def main():
    p = build_argparser()
    options = p.parse_args()

    # Load PDBs
    dep_structure = Structure.fromfile(options.base_structure)
    gen_structure = Structure.fromfile(options.comp_pdb)

    # ----------------------------------------------------------
    # ⭐ 해결책 2 — canonicalize() 적용
    # ----------------------------------------------------------
    dep_structure = dep_structure.canonicalize()
    gen_structure = gen_structure.canonicalize()
    # ----------------------------------------------------------

    # Remove water
    dep_structure = dep_structure.extract("resn", "HOH", "!=")
    gen_structure = gen_structure.extract("resn", "HOH", "!=")

    edia_data = {"Chain": [], "Residue": [], "Base_EDIA": [], "Comparison_EDIA": []}

    # Load density map
    dep_xmap = XMap.fromfile(options.map, label=options.label)
    dep_scaler = MapScaler(dep_xmap)
    dep_xmap = dep_xmap.canonical_unit_cell()

    # CASE 1 — Compute EDIA for a specific residue
    if options.residue is not None:
        chainid, resi = options.residue.split(",")

        dep_lig = dep_structure.extract(f"resi {resi} and chain {chainid}")
        gen_lig = gen_structure.extract(f"resi {resi} and chain {chainid}")

        if safe_atom_count(dep_lig) == 0 or safe_atom_count(gen_lig) == 0:
            print(f"[SKIP] Empty residue {chainid}{resi}")
            return

        combined = gen_lig.combine(dep_lig)

        dep_scaler.scale(combined, radius=1.5)
        dep_xmap2 = extract_map_region(dep_xmap, combined)

        validator = Validator(dep_xmap2, dep_xmap.resolution, options.directory)

        dep_edia = safe_edia(validator, dep_lig)
        gen_edia = safe_edia(validator, gen_lig)

        edia_data["Chain"].append(chainid)
        edia_data["Residue"].append(resi)
        edia_data["Base_EDIA"].append(dep_edia)
        edia_data["Comparison_EDIA"].append(gen_edia)

    else:
        combined = gen_structure.combine(dep_structure)
        dep_scaler.scale(combined, radius=1.5)
        dep_xmap2 = extract_map_region(dep_xmap, combined)

        validator = Validator(dep_xmap2, dep_xmap.resolution, options.directory)

        for chain in np.unique(dep_structure.chain):
            residues = np.unique(dep_structure.extract("chain", chain, "==").resi)

            for residue in residues:
                dep_lig = dep_structure.extract(f"resi {residue} and chain {chain}")
                gen_lig = gen_structure.extract(f"resi {residue} and chain {chain}")

                if safe_atom_count(dep_lig) == 0 or safe_atom_count(gen_lig) == 0:
                    print(f"[SKIP] Empty or missing ligand {chain}{residue}")
                    continue

                dep_edia = safe_edia(validator, dep_lig)
                gen_edia = safe_edia(validator, gen_lig)

                edia_data["Chain"].append(chain)
                edia_data["Residue"].append(residue)
                edia_data["Base_EDIA"].append(dep_edia)
                edia_data["Comparison_EDIA"].append(gen_edia)

    df = pd.DataFrame(edia_data)
    outname = f"{options.pdb}_edia.csv"
    df.to_csv(outname, index=False)
    print(f"Saved → {outname}")


if __name__ == "__main__":
    main()

