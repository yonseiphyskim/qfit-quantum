#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import numpy as np
import pandas as pd
from qfit.xtal.scaler import MapScaler
from qfit.structure import Structure
from qfit.xtal.volume import XMap
from qfit.validator import Validator
import os

def build_argparser():
    p = ArgumentParser(description="CRYO-EM VOXEL-BASED RSCC")
    p.add_argument("base_pdb")
    p.add_argument("map")
    p.add_argument("--resolution", type=float, required=True)
    p.add_argument("--comp_pdb", required=True)
    p.add_argument("--pdb", required=True)
    p.add_argument("--directory", default="./")
    return p

def main():
    args = build_argparser().parse_args()

    dep = Structure.fromfile(args.base_pdb).extract("resn", "HOH", "!=")
    gen = Structure.fromfile(args.comp_pdb).extract("resn", "HOH", "!=")

    xmap = XMap.fromfile(args.map, resolution=args.resolution)
    scaler = MapScaler(xmap)
    xmap = xmap.canonical_unit_cell()

    combined = gen.combine(dep)
    scaler.scale(combined, radius=1.5)
    submap = xmap.extract(combined.coor, padding=8)

    validator = Validator(submap, args.resolution, args.directory)

    rows = []
    for chain in np.unique(dep.chain):
        chain_sel = dep.extract("chain", chain, "==")
        for resi in np.unique(chain_sel.resi):
            d = dep.extract("chain {} and resi {}".format(chain, resi))
            g = gen.extract("chain {} and resi {}".format(chain, resi))

            rows.append({
                "CHAIN": chain,
                "RESI": resi,
                "RESNAME": d.resn[0],
                "DEPOSITED_RSCC": validator.rscc(d),
                "QFIT_RSCC": validator.rscc(g)
            })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.directory, "{}_rscc.csv".format(args.pdb))
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()

