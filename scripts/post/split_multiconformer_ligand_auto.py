#!/usr/bin/env python

import os
import pandas as pd
from qfit.structure import Structure
from qfit.structure.ligand import Ligand

# =========================
# 설정
# =========================
PDB_LIST_FILE = "LIST.ods"

FILENAME_TEMPLATE = (
    "multiconformer_ligand_bound_with_protein_refine_001_cleaned.pdb"
)

# =========================
# 유틸: ANISOU 제거
# =========================
def remove_anisou_lines(pdb_path):
    cleaned_path = pdb_path + ".noanisou"
    with open(pdb_path) as fin, open(cleaned_path, "w") as fout:
        for line in fin:
            if not line.startswith("ANISOU"):
                fout.write(line)
    return cleaned_path

# =========================
# PDB 리스트 읽기
# =========================
df = pd.read_excel(PDB_LIST_FILE, engine="odf")

required_cols = {"PDB", "Chain", "Residue Number"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"ODS 파일에 다음 컬럼이 필요함: {required_cols}\n"
        f"현재 컬럼: {list(df.columns)}"
    )

# =========================
# 메인 루프
# =========================
for _, row in df.iterrows():
    pdb = str(row["PDB"]).strip()
    chain_id = str(row["Chain"]).strip()
    resi = str(int(row["Residue Number"]))

    pdb_dir = pdb
    pdb_file = os.path.join(pdb_dir, FILENAME_TEMPLATE)

    if not os.path.isfile(pdb_file):
        print(f"[SKIP] File not found: {pdb_file}")
        continue

    print(f"\n=== Processing {pdb} ===")
    print(f"Ligand location: chain {chain_id}, resi {resi}")

    # =========================
    # ANISOU 제거 후 로딩
    # =========================
    pdb_file_clean = remove_anisou_lines(pdb_file)
    structure = Structure.fromfile(pdb_file_clean)

    # =========================
    # Extract ligand
    # =========================
    structure_ligand = structure.extract(
        f"resi {resi} and chain {chain_id}"
    )

    # 🔑 len() → atoms.size
    if structure_ligand.atoms.size == 0:
        print(
            f"[SKIP] No ligand atoms found for {pdb} "
            f"(chain {chain_id}, resi {resi})"
        )
        continue

    altlocs = sorted(set(structure_ligand.altloc) - {""})

    # ==================================================
    # CASE 1: altloc 없는 리간드
    # ==================================================
    if len(altlocs) == 0:
        ligand = Ligand.from_structure(structure_ligand)
        ligand.altloc = ""

        out_path = os.path.join(pdb_dir, f"{pdb}_ligand_A.pdb")
        print(f"Saving: {out_path}")
        ligand.tofile(out_path)
        continue

    # ==================================================
    # CASE 2: altloc 있는 리간드
    # ==================================================
    common_structure = structure_ligand.extract("altloc ''")

    for altloc in altlocs:
        alt_structure = structure_ligand.extract(f"altloc {altloc}")
        occupancies = alt_structure.q

        structure_altloc = common_structure.combine(alt_structure)

        # 🔑 여기 역시 len() → atoms.size
        if structure_altloc.atoms.size == 0:
            print(
                f"[SKIP] Empty structure for {pdb} altloc '{altloc}'"
            )
            continue

        for atom, occupancy in zip(structure_altloc, occupancies):
            atom.q = occupancy

        ligand = Ligand.from_structure(structure_altloc)
        ligand.altloc = ""

        out_path = os.path.join(pdb_dir, f"{pdb}_ligand_{altloc}.pdb")
        print(f"Saving: {out_path}")
        ligand.tofile(out_path)

print("\n✅ All PDBs processed.")

