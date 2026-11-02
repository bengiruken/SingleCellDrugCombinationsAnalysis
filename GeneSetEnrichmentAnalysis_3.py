# Enrich the most active Leiden community in each cell against KEGG pathways.

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import pandas as pd

# Input and output paths
BASE_DIR = Path("BreastTumorDataInput_UMIcounts")
ACTIVITY_FILE = BASE_DIR / "CellSpecificSubnetworks_LeidenCommunity_ActivityScores.tsv"

OUT_BASE = BASE_DIR / "CELL_SPECIFIC_KEGG_ENRICHMENT"

# Enrichment settings
KEGG_LIBRARY = "KEGG_2021_Human"     # KEGG gene set library
ORGANISM = "Human"
SIGNALING_SUBSTRING = "signaling"    # Keep pathway names containing this text, ignoring case.
MIN_GENES_FOR_ENRICH = 10            # Minimum gene list size for enrichment
TOP_TERMS_PER_CELL = 20              # Maximum enriched terms retained per cell
SLEEP_SECONDS = 0.0                  # Delay between requests, in seconds
SKIP_IF_PATIENT_OUTPUT_EXISTS = False

# Use the same candidate genes as the preceding analysis.
CO_TARGETING_CANDIDATES = [
    "AKT1", "PIK3R1", "PIK3CA", "PIK3R2", "PIK3R3", "PIK3CB", "KRAS", "RAF1",
    "MAPK1", "MAP2K1", "GSK3B", "BRAF", "IGF1R", "FGFR3", "FGFR2", "FASLG",
    "NTRK1", "SOCS2", "SOCS1", "JAK2", "STAT3", "PTPN11", "PTPN6", "FYN",
    "EP300", "ESR1", "HSP90AB1", "HSP90AA1", "NF1", "HSPA1A", "HSPA8",
    "CDKN2A", "PKM", "JAK3", "JAK1", "FN1", "YWHAZ", "YWHAG", "YWHAQ",
    "PTK2", "ARAF", "ERBB2", "ERBB3", "TP53", "MDM2", "MYC", "CTNNB1",
    "AXIN1", "GRB2", "EGFR", "PLCG1", "SRC", "CALM1"
]
CO_TARGETING_SET: Set[str] = {g.upper() for g in CO_TARGETING_CANDIDATES}


# Timestamped progress messages
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# Read gene names and scores from the activity table.
def parse_gene_score_string(s: str) -> Dict[str, float]:
    """Parse 'GENE:score, GENE2:score2, ...' into dict."""
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if not s or s.lower() == "not available":
        return {}
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        gene, val = part.split(":", 1)
        gene = gene.strip().upper()
        try:
            out[gene] = float(val.strip())
        except ValueError:
            continue
    return out


def gene_list_from_genes_and_scores(s: str) -> List[str]:
    return sorted(parse_gene_score_string(s).keys())


def parse_cotarget_names(s: str) -> Set[str]:
    """Parse 'GENE1, GENE2, ...' into uppercase set; handles 'Not available'."""
    if not isinstance(s, str):
        return set()
    s = s.strip()
    if not s or s.lower() == "not available":
        return set()
    return {g.strip().upper() for g in s.split(",") if g.strip()}


def compute_num_cotargeting_candidates_in_cell(cell_df: pd.DataFrame) -> int:
    """
    Count unique co-targeting candidate genes expressed in the cell.
    Preferred: union of Step 5 'CoTargetingGeneNamesInCommunity' across communities.
    Fallback: union all expressed genes from GenesFromCellAndScores and intersect with CO_TARGETING_SET.
    """
    if "CoTargetingGeneNamesInCommunity" in cell_df.columns:
        genes = set()
        for x in cell_df["CoTargetingGeneNamesInCommunity"].tolist():
            genes |= parse_cotarget_names(x)
        return int(len(genes))

    genes_all = set()
    for x in cell_df.get("GenesFromCellAndScores", pd.Series([], dtype=str)).tolist():
        genes_all |= set(gene_list_from_genes_and_scores(x))
    return int(len(genes_all & CO_TARGETING_SET))


def pick_most_active_community_per_cell(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (DatasetID, PatientID, CellID), pick the row with max CommunityActivityScore.
    Tie-breaker: NumGenesFromCellInCommunity (if present).
    """
    required = {"DatasetID", "PatientID", "CellID", "CommunityID", "CommunityActivityScore", "GenesFromCellAndScores"}
    missing = required - set(activity_df.columns)
    if missing:
        raise ValueError(f"Activity file missing required columns: {sorted(missing)}")

    work = activity_df.copy()
    work["CommunityActivityScore"] = pd.to_numeric(work["CommunityActivityScore"], errors="coerce").fillna(0.0)

    if "NumGenesFromCellInCommunity" in work.columns:
        work["NumGenesFromCellInCommunity"] = pd.to_numeric(
            work["NumGenesFromCellInCommunity"], errors="coerce"
        ).fillna(0.0)
    else:
        work["NumGenesFromCellInCommunity"] = 0.0

    out_rows = []
    for (ds, pt, cell), g in work.groupby(["DatasetID", "PatientID", "CellID"], sort=True):
        g = g.sort_values(
            ["CommunityActivityScore", "NumGenesFromCellInCommunity"],
            ascending=[False, False],
            kind="mergesort",
        )
        top = g.iloc[0]
        out_rows.append({
            "DatasetID": str(ds),
            "PatientID": str(pt),
            "CellID": str(cell),
            "MostActiveLeidenCommunityID": str(top["CommunityID"]),
            "MostActiveLeidenCommunityScore": float(top["CommunityActivityScore"]),
            "NumCoTargetingCandidatesInCell": int(compute_num_cotargeting_candidates_in_cell(g)),
            "GenesFromCellAndScores": str(top.get("GenesFromCellAndScores", "")),
        })
    return pd.DataFrame(out_rows)


def normalize_gseapy_results(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize gseapy.enrichr results columns across versions."""
    if df is None or df.empty:
        return pd.DataFrame()

    colmap = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "term":
            colmap[c] = "Term"
        elif lc in {"p-value", "pvalue", "p value"}:
            colmap[c] = "PValue"
        elif lc in {"adjusted p-value", "adj p-value", "adjusted p value", "fdr", "q-value"}:
            colmap[c] = "AdjPValue"
        elif lc in {"odds ratio", "oddsratio"}:
            colmap[c] = "OddsRatio"
        elif lc in {"combined score", "combinedscore"}:
            colmap[c] = "CombinedScore"
        elif lc == "genes":
            colmap[c] = "Genes"
        elif lc == "overlap":
            colmap[c] = "Overlap"

    out = df.rename(columns=colmap).copy()

    for c in ["PValue", "AdjPValue", "OddsRatio", "CombinedScore"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if "Term" in out.columns:
        out["Term"] = out["Term"].astype(str)

    return out


# Install gseapy before running this script: python -m pip install gseapy


# Check the input paths and preview the activity table.

from pathlib import Path
print("ACTIVITY_FILE:", ACTIVITY_FILE, type(ACTIVITY_FILE))
print("OUT_BASE:", OUT_BASE, type(OUT_BASE))

ACTIVITY_FILE = Path(ACTIVITY_FILE)
OUT_BASE = Path(OUT_BASE)

print("exists?", ACTIVITY_FILE.exists())
if ACTIVITY_FILE.exists():
    print("size:", ACTIVITY_FILE.stat().st_size)

act = pd.read_csv(ACTIVITY_FILE, sep="\t")
print("act shape:", act.shape)
print("act columns:", act.columns.tolist()[:50])
print(act.head(2))


import gseapy


# Select the most active communities and run KEGG enrichment.
if not ACTIVITY_FILE.exists():
    raise FileNotFoundError(
        f"Missing Step 5 output: {ACTIVITY_FILE}\n"
        "Run your pipeline through Step 5 first (it creates the ActivityScores TSV)."
    )

OUT_BASE.mkdir(parents=True, exist_ok=True)

log(f"Step 7: reading activity file: {ACTIVITY_FILE}")
act = pd.read_csv(ACTIVITY_FILE, sep="\t")
if act.empty:
    raise ValueError("Activity file is empty; nothing to enrich.")

# Summarize the most active community and candidate gene count for each cell.
top_cells = pick_most_active_community_per_cell(act)

# Save the combined cell summary.
global_summary_path = OUT_BASE / "ALL_PatientCell_MostActiveLeidenCommunity.txt"
top_cells[[
    "PatientID",
    "CellID",
    "MostActiveLeidenCommunityID",
    "MostActiveLeidenCommunityScore",
    "NumCoTargetingCandidatesInCell",
]].to_csv(global_summary_path, sep="\t", index=False)
log(f"Step 7: wrote global per-cell summary: {global_summary_path}")

# Run KEGG enrichment on the most active community in each cell.
try:
    import gseapy as gp
except ImportError as e:
    raise ImportError("Step 7 requires gseapy. Install with: pip install gseapy") from e

all_global_rows = []
all_global_sig_rows = []

signaling_sub = (SIGNALING_SUBSTRING or "signaling").lower()

for (ds, pt), sub_df in top_cells.groupby(["DatasetID", "PatientID"], sort=True):
    ds = str(ds)
    pt = str(pt)

    patient_out = OUT_BASE / ds / pt
    patient_out.mkdir(parents=True, exist_ok=True)

    # Save the cell summary for this patient.
    patient_summary_path = patient_out / "PatientCell_MostActiveLeidenCommunity.txt"
    sub_df[[
        "PatientID",
        "CellID",
        "MostActiveLeidenCommunityID",
        "MostActiveLeidenCommunityScore",
        "NumCoTargetingCandidatesInCell",
    ]].to_csv(patient_summary_path, sep="\t", index=False)
    log(f"Step 7: wrote patient summary: {patient_summary_path}")

    patient_all_path = patient_out / "KEGG_Enrichment_AllTopTerms_ByCell.tsv"
    patient_sig_path = patient_out / "KEGG_Enrichment_SignalingOnly_ByCell.tsv"
    patient_stats_path = patient_out / "KEGG_Signaling_Pathway_Stats.tsv"

    if SKIP_IF_PATIENT_OUTPUT_EXISTS and patient_all_path.exists() and patient_sig_path.exists():
        log(f"Step 7: skipping enrichment for dataset={ds} patient={pt} (outputs exist).")
        continue

    patient_rows = []

    for _, r in sub_df.iterrows():
        cell = str(r["CellID"])
        comm_id = str(r["MostActiveLeidenCommunityID"])
        comm_score = float(r["MostActiveLeidenCommunityScore"])
        n_cot = int(r["NumCoTargetingCandidatesInCell"])

        genes = gene_list_from_genes_and_scores(r.get("GenesFromCellAndScores", ""))
        if len(genes) < MIN_GENES_FOR_ENRICH:
            continue

        log(f"Step 7: KEGG enrichr dataset={ds} patient={pt} cell={cell} genes={len(genes)}")

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=KEGG_LIBRARY,
                organism=ORGANISM,
                outdir=None,
                cutoff=1.0,   # Rank and select the top terms below.
            )
        except Exception as e:
            log(f"WARN: gseapy.enrichr failed for dataset={ds} patient={pt} cell={cell}: {e}")
            continue

        df_res = normalize_gseapy_results(getattr(enr, "results", None))
        if df_res.empty or "Term" not in df_res.columns:
            continue

        # Rank by adjusted p-value when available, then by p-value.
        sort_cols = [c for c in ["AdjPValue", "PValue"] if c in df_res.columns]
        if sort_cols:
            df_res = df_res.sort_values(sort_cols, ascending=True, kind="mergesort")
        df_res = df_res.head(TOP_TERMS_PER_CELL).copy()

        for _, rr in df_res.iterrows():
            patient_rows.append({
                "DatasetID": ds,
                "PatientID": pt,
                "CellID": cell,
                "MostActiveLeidenCommunityID": comm_id,
                "MostActiveLeidenCommunityScore": comm_score,
                "NumCoTargetingCandidatesInCell": n_cot,
                "GeneListSize": len(genes),
                "KEGG_Library": KEGG_LIBRARY,
                "Term": rr.get("Term", ""),
                "PValue": rr.get("PValue", np.nan),
                "AdjPValue": rr.get("AdjPValue", np.nan),
                "OddsRatio": rr.get("OddsRatio", np.nan),
                "CombinedScore": rr.get("CombinedScore", np.nan),
                "Overlap": rr.get("Overlap", ""),
                "Genes": rr.get("Genes", ""),
            })

        if SLEEP_SECONDS and SLEEP_SECONDS > 0:
            import time
            time.sleep(float(SLEEP_SECONDS))

    # Save enrichment tables for this patient.
    patient_all = pd.DataFrame(patient_rows)
    if patient_all.empty:
        log(f"Step 7: no enrichment results for dataset={ds} patient={pt}")
        continue

    patient_all.to_csv(patient_all_path, sep="\t", index=False)

    patient_sig = patient_all[patient_all["Term"].astype(str).str.lower().str.contains(signaling_sub, na=False)].copy()
    patient_sig.to_csv(patient_sig_path, sep="\t", index=False)

    # Summarize signaling pathways across cells.
    if not patient_sig.empty:
        stats = (
            patient_sig.groupby("Term", as_index=False)
            .agg(
                NumCells=("CellID", "nunique"),
                MinAdjPValue=("AdjPValue", "min"),
                MedianAdjPValue=("AdjPValue", "median"),
                MedianCombinedScore=("CombinedScore", "median"),
            )
            .sort_values(["NumCells", "MinAdjPValue"], ascending=[False, True], kind="mergesort")
        )
        stats.to_csv(patient_stats_path, sep="\t", index=False)

    log(f"Step 7: wrote patient KEGG all terms: {patient_all_path}")
    log(f"Step 7: wrote patient KEGG signaling-only: {patient_sig_path}")
    if patient_stats_path.exists():
        log(f"Step 7: wrote patient signaling stats: {patient_stats_path}")

    all_global_rows.extend(patient_rows)
    all_global_sig_rows.extend(patient_sig.to_dict("records"))

# Save combined results across the processed patients.
if all_global_rows:
    pd.DataFrame(all_global_rows).to_csv(OUT_BASE / "ALL_KEGG_Enrichment_AllTopTerms_ByCell.tsv", sep="\t", index=False)
if all_global_sig_rows:
    pd.DataFrame(all_global_sig_rows).to_csv(OUT_BASE / "ALL_KEGG_Enrichment_SignalingOnly_ByCell.tsv", sep="\t", index=False)

log(f"Step 7 complete. Outputs under: {OUT_BASE.resolve()}")
