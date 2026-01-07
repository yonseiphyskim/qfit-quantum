import os
import subprocess
import pandas as pd

# ==================================================
# HARD-CODED INPUTS (NO EXCEL / NO ODS)
# ==================================================

PDB_LIST = ["8P77", "8P70", "8P6X", "8P78"]

RESOLUTION_MAP = {
    "8P77": 3.2,
    "8P70": 3.2,
    "8P6X": 3.2,
    "8P78": 3.2,
}

# ABSOLUTE PATH TO RSCC SCRIPT
RSCC_SCRIPT = os.path.abspath("compare_rscc_voxel_cryo.py")

if not os.path.isfile(RSCC_SCRIPT):
    raise FileNotFoundError(f"RSCC SCRIPT NOT FOUND:{RSCC_SCRIPT}")

print("FOUND PDB IDS:", PDB_LIST)

# ==================================================
# MAIN LOOP
# ==================================================

all_results = []

for pdb in PDB_LIST:
    folder = pdb.upper()

    if not os.path.isdir(folder):
        print(f"[SKIP] FOLDER {folder} NOT FOUND")
        continue

    resolution = RESOLUTION_MAP[pdb]
    print(f"\n[RUN] PROCESSING {folder} | RESOLUTION = {resolution:.1f} Å")

    # --------------------------------------
    # FIND DEPOSITED (SINGLE-CONFORMER) PDB
    # --------------------------------------
    deposited = None
    pdb_files = [f for f in os.listdir(folder) if f.lower().endswith(".pdb")]

    for f in pdb_files:
        lf = f.lower()
        if (
            "multiconformer" not in lf and
            "qfit" not in lf and
            "refined" not in lf and
            "real_space" not in lf
        ):
            deposited = f
            break

    if deposited is None:
        print(f"[SKIP] DEPOSITED PDB NOT FOUND IN {folder}")
        continue

    print(f"[INFO] USING DEPOSITED PDB: {deposited}")

    # --------------------------------------
    # QFIT REFINED MODEL
    # --------------------------------------
    qfit = "multiconformer_ligand_bound_with_protein_box_cleaned_real_space_refined_000.pdb"

    if not os.path.isfile(os.path.join(folder, qfit)):
        print(f"[SKIP] QFIT REFINED MODEL NOT FOUND IN {folder}")
        continue

    # --------------------------------------
    # CRYO-EM MAP SELECTION
    # --------------------------------------
    em_map = None
    map_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".map", ".mrc", ".ccp4"))
    ]

    if not map_files:
        print(f"[SKIP] NO CRYO-EM MAP FOUND IN {folder}")
        continue

    # PRIORITY 1 — FIXED / SCALED / REFINED MAP
    for f in map_files:
        lf = f.lower()
        if "fixed" in lf or "scaled" in lf or "refine" in lf:
            em_map = f
            break

    # PRIORITY 2 — ANY MAP
    if em_map is None:
        em_map = map_files[0]

    print(f"[INFO] USING MAP: {em_map}")

    # --------------------------------------
    # RUN RSCC (CRYO-EM, VOXEL-BASED)
    # --------------------------------------
    output_csv = f"{pdb}_rscc.csv"

    cmd = [
        "python",          # 🔑 qfit/phenix ENV
        RSCC_SCRIPT,
        deposited,
        em_map,
        "--resolution", str(resolution),
        "--comp_pdb", qfit,
        "--pdb", pdb,
        "--directory", "./"
    ]

    print("EXECUTING:", " ".join(cmd))

    try:
        subprocess.run(
            cmd,
            cwd=folder,   # == cd <PDB>
            check=True
        )
    except subprocess.CalledProcessError:
        print(f"[ERROR] RSCC CALCULATION CRASHED FOR {pdb}")
        continue

    # --------------------------------------
    # COLLECT RESULTS
    # --------------------------------------
    csv_path = os.path.join(folder, output_csv)

    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df["PDB"] = pdb
            all_results.append(df)
            print(f"[OK] RSCC RESULT COLLECTED FOR {pdb}")
        except Exception as e:
            print(f"[WARN] FAILED TO READ RSCC CSV FOR {pdb}: {e}")
    else:
        print(f"[WARN] NO RSCC OUTPUT FOR {pdb}")

# ==================================================
# MERGE ALL RSCC RESULTS
# ==================================================

if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("RSCC_all_results.csv", index=False)
    print("\n🎉 ALL CRYO-EM RSCC RESULTS MERGED → RSCC_all_results.csv")
else:
    print("\n⚠ NO RSCC FILES GENERATED")

