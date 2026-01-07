import os
import subprocess
import pandas as pd
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

# ==================================================
# 1. LOAD PDB LIST FROM ODS
# ==================================================
ODS_FILE = "cryo-em.ods"
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
            pdb = texts[0].firstChild.data.strip()
            if len(pdb) == 4:
                pdb_list.append(pdb.upper())

print("FOUND PDB IDS:", pdb_list)

# ==================================================
# 2. HARD-CODED RESOLUTION MAP (CRYO-EM)
# ==================================================
RESOLUTION_MAP = {
    "8P77": 3.2,
    "8P70": 3.2,
    "8P6X": 3.2,
    "8P78": 3.2,
}

# ==================================================
# 3. EDIA CRYO SCRIPT
# ==================================================
EDIA_SCRIPT = os.path.abspath("compare_edia_cryo.py")

if not os.path.isfile(EDIA_SCRIPT):
    raise RuntimeError(f"EDIA script not found: {EDIA_SCRIPT}")

# ==================================================
# 4. BATCH PROCESS
# ==================================================
all_results = []

for pdb in pdb_list:
    folder = pdb.upper()

    if folder not in RESOLUTION_MAP:
        print(f"[SKIP] No resolution for {pdb}")
        continue

    if not os.path.isdir(folder):
        print(f"[SKIP] Folder {folder} not found")
        continue

    resolution = RESOLUTION_MAP[folder]
    print(f"\n[RUN] {folder} | resolution = {resolution:.2f} Å")

    # --------------------------------------------------
    # FIND DEPOSITED MODEL
    # --------------------------------------------------
    deposited = None
    for f in os.listdir(folder):
        lf = f.lower()
        if (
            lf.endswith(".pdb") and
            "multiconformer" not in lf and
            "qfit" not in lf and
            "refined" not in lf and
            "real_space" not in lf
        ):
            deposited = f
            break

    if deposited is None:
        print(f"[SKIP] Deposited PDB not found in {folder}")
        continue

    print(f"[INFO] Deposited PDB: {deposited}")

    # --------------------------------------------------
    # QFIT MODEL
    # --------------------------------------------------
    qfit = "multiconformer_ligand_bound_with_protein_box_cleaned_real_space_refined_000.pdb"

    if not os.path.isfile(os.path.join(folder, qfit)):
        print(f"[SKIP] qFit refined model not found in {folder}")
        continue

    # --------------------------------------------------
    # CRYO-EM MAP SELECTION
    # --------------------------------------------------
    map_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".map", ".ccp4", ".mrc"))
    ]

    if not map_files:
        print(f"[SKIP] No cryo-EM map found in {folder}")
        continue

    em_map = None
    for f in map_files:
        lf = f.lower()
        if "scaled" in lf or "fixed" in lf or "refine" in lf:
            em_map = f
            break

    if em_map is None:
        em_map = map_files[0]

    print(f"[INFO] Cryo-EM map: {em_map}")

    # --------------------------------------------------
    # RUN EDIA (CRYO-EM)
    # --------------------------------------------------
    output_csv = f"{pdb}_edia.csv"

    cmd = [
        "python",
        EDIA_SCRIPT,
        deposited,
        em_map,
        "--resolution", str(resolution),
        "--comp_pdb", qfit,
        "--pdb", pdb,
        "--directory", "./"
    ]

    print("EXECUTING:", " ".join(cmd))

    try:
        subprocess.run(cmd, cwd=folder, check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] EDIA failed for {pdb}")
        continue

    # --------------------------------------------------
    # COLLECT RESULT
    # --------------------------------------------------
    csv_path = os.path.join(folder, output_csv)

    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df["PDB"] = pdb
        all_results.append(df)
        print(f"[OK] EDIA collected for {pdb}")
    else:
        print(f"[WARN] No EDIA output for {pdb}")

# ==================================================
# 5. MERGE RESULTS
# ==================================================
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("EDIA_all_results.csv", index=False)
    print("\n🎉 All cryo-EM EDIA results saved → EDIA_all_results.csv")
else:
    print("\n⚠ No EDIA results generated")

