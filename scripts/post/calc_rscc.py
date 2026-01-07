import os
import subprocess
import pandas as pd
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# --------------------------------------
# 1. Load PDB LIST
# --------------------------------------
ODS_FILE = "PDB LIST.ods"
doc = load(ODS_FILE)
sheets = doc.spreadsheet.getElementsByType(Table)

pdb_list = []

for sheet in sheets:
    for row in sheet.getElementsByType(TableRow):
        cells = row.getElementsByType(TableCell)
        if not cells:
            continue
        texts = cells[0].getElementsByType(P)
        if texts and texts[0].firstChild:
            pdb_code = texts[0].firstChild.data.strip()
            if len(pdb_code) == 4:
                pdb_list.append(pdb_code.upper())

print("Found PDB IDs:", pdb_list)


# --------------------------------------
# 2. RSCC script path
# --------------------------------------
RSCC_SCRIPT = "compare_rscc_voxel.py"


# --------------------------------------
# 3. Process each PDB entry
# --------------------------------------
all_results = []

for pdb in pdb_list:
    folder = pdb.upper()

    if not os.path.isdir(folder):
        print(f"[SKIP] Folder {folder} not found.")
        continue

    print(f"\n[RUN] Processing {folder} ...")

    deposited = os.path.join(folder, f"{pdb.lower()}.pdb")
    qfit = os.path.join(folder, "multiconformer_ligand_bound_with_protein_refine_001.pdb")

    if not os.path.isfile(qfit):
        print(f"[SKIP] No refined qFit model in {folder}")
        continue

    # --------------------------------------
    # Find MTZ — composite omit map FIRST
    # --------------------------------------
    mtz = None
    mtz_files = [f for f in os.listdir(folder) if f.lower().endswith(".mtz")]

    if not mtz_files:
        print(f"[SKIP] No MTZ file found in {folder}")
        continue

    # Priority 1 — composite omit map
    for f in mtz_files:
        lf = f.lower()
        if "omit" in lf or "composite" in lf:
            mtz = os.path.join(folder, f)
            print(f"[INFO] Using COMPOSITE OMIT MTZ: {f}")
            break

    # Priority 2 — refine/map coefficients (map usable MTZ)
    if mtz is None:
        for f in mtz_files:
            lf = f.lower()
            if (
                "refine" in lf or
                "map" in lf or 
                "2fofc" in lf or
                "fwt" in lf or
                "coeff" in lf
            ):
                mtz = os.path.join(folder, f)
                print(f"[INFO] Using MAP-COEFFICIENT MTZ: {f}")
                break

    # Priority 3 — raw structure factor MTZ (sf)
    if mtz is None:
        for f in mtz_files:
            lf = f.lower()
            if "sf" in lf:
                mtz = os.path.join(folder, f)
                print(f"[INFO] Using RAW SF MTZ (low priority): {f}")
                break

    # Priority 4 — fallback
    if mtz is None:
        mtz = os.path.join(folder, mtz_files[0])
        print(f"[INFO] Using FALLBACK MTZ: {mtz_files[0]}")

    # --------------------------------------

    if not (os.path.isfile(deposited) and os.path.isfile(qfit) and mtz):
        print(f"[SKIP] Missing inputs for {folder}")
        continue

    output_csv = os.path.join(folder, f"{pdb}_rscc.csv")

    cmd = [
        "python", RSCC_SCRIPT,
        deposited, mtz,
        "--comp_pdb", qfit,
        "--pdb", pdb,
        "--directory", folder + "/"
    ]

    print("Executing:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] RSCC calculation crashed for {pdb}")
        continue

    if os.path.isfile(output_csv):
        try:
            df = pd.read_csv(output_csv)
            df["PDB"] = pdb
            all_results.append(df)
        except Exception as e:
            print(f"[WARN] Could not read RSCC CSV for {pdb}: {e}")
    else:
        print(f"[WARN] No RSCC output for {pdb}.")


# --------------------------------------
# 5. Merge all RSCC Results
# --------------------------------------
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("RSCC_all_results.csv", index=False)
    print("\n🎉 All RSCC results merged → RSCC_all_results.csv")
else:
    print("\n⚠ No RSCC files generated.")

