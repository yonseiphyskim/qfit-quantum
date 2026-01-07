import os
import subprocess
import pandas as pd
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# --------------------------------------
# 1. Load PDB LIST from ODS
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
# 2. EDIA script path
# --------------------------------------
EDIA_SCRIPT = "compare_edia.py"

# --------------------------------------
# 3. Batch process
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

    if not os.path.isfile(deposited):
        print(f"[SKIP] No deposited PDB in {folder}")
        continue

    if not os.path.isfile(qfit):
        print(f"[SKIP] No qFit refined model in {folder}")
        continue

    # --------------------------------------
    # Choose MTZ (OMIT MAP preferred)
    # --------------------------------------
    mtz_files = [f for f in os.listdir(folder) if f.lower().endswith(".mtz")]

    if not mtz_files:
        print(f"[SKIP] No MTZ found in {folder}")
        continue

    mtz = None

    # Priority 1: composite omit map
    for f in mtz_files:
        lf = f.lower()
        if "omit" in lf or "composite" in lf:
            mtz = os.path.join(folder, f)
            print(f"[INFO] Using OMIT MTZ: {f}")
            break

    # Priority 2: original or refine map
    if mtz is None:
        for f in mtz_files:
            lf = f.lower()
            if (
                "sf" in lf or "map" in lf or "fwt" in lf or "2fofc" in lf or "coeff" in lf
            ):
                mtz = os.path.join(folder, f)
                print(f"[INFO] Using ORIGINAL SF MTZ: {f}")
                break

    # Priority 3: fallback
    if mtz is None:
        mtz = os.path.join(folder, mtz_files[0])
        print(f"[INFO] Using fallback MTZ: {mtz_files[0]}")

    # --------------------------------------
    output_csv = os.path.join(folder, f"{pdb}_edia.csv")

    # Command
    cmd = [
        "python", EDIA_SCRIPT,
        deposited, mtz,
        "--comp_pdb", qfit,
        "--pdb", pdb,
        "--directory", folder + "/"
    ]

    print("Executing:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] EDIA calculation crashed for {pdb}")
        continue

    if os.path.isfile(output_csv):
        try:
            df = pd.read_csv(output_csv)
            df["PDB"] = pdb
            all_results.append(df)
        except Exception as e:
            print(f"[WARN] Cannot read EDIA CSV for {pdb}: {e}")

# --------------------------------------
# 4. Merge All Results
# --------------------------------------
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("EDIA_all_results.csv", index=False)
    print("\n🎉 All EDIA results saved → EDIA_all_results.csv")
else:
    print("\n⚠ No EDIA results were generated.")

