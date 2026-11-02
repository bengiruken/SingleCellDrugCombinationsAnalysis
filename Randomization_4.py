# Run matched-permutation controls for the KEGG enrichment results.
# Match genes by HIPPIE network degree and observed detectability.
# Count cells with pathway enrichment p-values at or below P_THRESH.

from __future__ import annotations

from IPython.display import display

from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Iterable, Optional

import numpy as np
import pandas as pd

from scipy.stats import hypergeom

# Input and output paths
BASE_DIR = Path("BreastTumorDataInput_UMIcounts")
ACTIVITY_FILE = BASE_DIR / "CellSpecificSubnetworks_LeidenCommunity_ActivityScores.tsv"

# HIPPIE interaction file used for the cell subnetworks
HIPPIE_FILE = Path("hippie_parsed.txt")

# Use the same KEGG library as the enrichment analysis.
KEGG_LIBRARY = "KEGG_2021_Human"
ORGANISM = "Human"

# Permutation test settings
MIN_GENES_FOR_CELL = 10         # Minimum genes per cell
P_THRESH = 0.05                 # Enrichment p-value threshold for counting cells
N_PERM = 300                    # Permutations per patient and pathway
SEED = 1

# Gene matching settings
N_DEG_BINS = 5                  # Number of network-degree bins
N_DET_BINS = 5                  # Number of detectability bins

# Candidate pathway settings
TOP_TERMS_TO_TEST = 25          # Select pathways by their observed cell frequency.
SIGNALING_ONLY = True           # Restrict candidates to pathway names containing the signaling text.
SIGNALING_SUBSTRING = "signaling"

# Runtime and output settings
MAX_CELLS_PER_PATIENT: Optional[int] = None   # Use all cells when None.
OUT_DIR = BASE_DIR / "CELL_SPECIFIC_KEGG_ENRICHMENT" / "RANDOMIZATION_CONTROLS"


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
        ).head(1)
        r = g.iloc[0].to_dict()
        out_rows.append({
            "DatasetID": str(ds),
            "PatientID": str(pt),
            "CellID": str(cell),
            "MostActiveLeidenCommunityID": str(r.get("CommunityID", "")),
            "MostActiveLeidenCommunityScore": float(r.get("CommunityActivityScore", 0.0)),
            "GenesFromCellAndScores": r.get("GenesFromCellAndScores", ""),
        })
    return pd.DataFrame(out_rows)


# Count gene connections in the HIPPIE interaction file.
def load_ppi_degree_from_hippie(filepath: Path) -> Dict[str, int]:
    """Read gene pairs from a whitespace-separated HIPPIE file and count connections."""
    if not filepath.exists():
        raise FileNotFoundError(f"HiPPie file not found: {filepath.resolve()}")

    deg = Counter()
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a = parts[0].upper()
            b = parts[1].upper()
            if a == b:
                continue
            deg[a] += 1
            deg[b] += 1
    return dict(deg)


# Fetch the KEGG gene sets once for the permutation tests.
def load_kegg_library_as_sets(library_name: str, organism: str) -> Dict[str, Set[str]]:
    """Fetch the KEGG library once and convert gene names to uppercase sets.
    
    A local GMT file can be used instead; see the commented example below."""
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError("Please install gseapy first: pip install gseapy") from e

    lib = gp.get_library(name=library_name, organism=organism)  # Pathway names mapped to gene lists
    out: Dict[str, Set[str]] = {}
    for term, genes in lib.items():
        out[str(term)] = {str(g).upper() for g in genes if isinstance(g, str) and g.strip()}
    return out

# Alternative: load the gene sets from a local GMT file.
# from gseapy.parser import read_gmt
# gmt = read_gmt("KEGG_2021_Human.gmt")  # returns dict term -> list
# kegg_sets = {t: {g.upper() for g in gs} for t, gs in gmt.items()}


# Adjust p-values with the Benjamini-Hochberg procedure.
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = p[idx] * n / rank
        if val < prev:
            prev = val
        q[idx] = min(prev, 1.0)
    return q


# Group genes by network degree and detectability.
def build_bins(
    universe: List[str],
    deg: Dict[str, int],
    det: Dict[str, float],
    n_deg_bins: int,
    n_det_bins: int
) -> Tuple[Dict[Tuple[int, int], np.ndarray], Dict[str, Tuple[int, int]]]:
    """
    Returns:
      bins[(det_bin, deg_bin)] = np.array([genes...])
      gene_to_bin[gene] = (det_bin, deg_bin)
    Uses rank-based qcut to avoid "duplicate bin edges" issues.
    """
    u = [g for g in universe if (g in deg) and (g in det)]
    if len(u) < 100:
        raise ValueError("Universe too small after filtering for degree+detectability. Check inputs.")

    s_deg = pd.Series({g: float(deg.get(g, 0)) for g in u})
    s_det = pd.Series({g: float(det.get(g, 0.0)) for g in u})

    # Rank values before binning to handle ties.
    deg_bin = pd.qcut(s_deg.rank(method="first"), q=n_deg_bins, labels=False, duplicates="drop")
    det_bin = pd.qcut(s_det.rank(method="first"), q=n_det_bins, labels=False, duplicates="drop")

    gene_to_bin: Dict[str, Tuple[int, int]] = {}
    bins: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    for g in u:
        b = (int(det_bin[g]), int(deg_bin[g]))
        gene_to_bin[g] = b
        bins[b].append(g)

    bins_np = {b: np.array(gs, dtype=object) for b, gs in bins.items()}
    return bins_np, gene_to_bin


# Sample random gene sets with the same bin counts as each cell.
def sample_gene_set_from_bin_counts(
    bin_counts: Counter,
    bins: Dict[Tuple[int, int], np.ndarray],
    universe_arr: np.ndarray,
    rng: np.random.Generator,
    target_size: int
) -> Set[str]:
    """Sample a gene set with the requested bin counts.
    
    Draw without replacement within each bin when possible. If a bin is too
    small, fill the remaining positions from the gene universe. Return unique genes."""
    sampled: List[str] = []

    for b, n in bin_counts.items():
        pool = bins.get(b, None)
        if pool is None or len(pool) == 0 or n <= 0:
            continue

        if n <= len(pool):
            picks = rng.choice(pool, size=n, replace=False)
            sampled.extend(picks.tolist())
        else:
            # Use the whole bin and draw the remaining genes from the full universe.
            sampled.extend(pool.tolist())
            rem = n - len(pool)
            fill = rng.choice(universe_arr, size=rem, replace=False if rem <= len(universe_arr) else True)
            sampled.extend(fill.tolist())

    out = set(sampled)

    # Draw extra genes if removing duplicates reduced the set size.
    if len(out) < target_size:
        need = target_size - len(out)
        fill_pool = universe_arr
        fill = rng.choice(fill_pool, size=need, replace=False if need <= len(fill_pool) else True)
        out |= set(fill.tolist())

    # Trim the set if it exceeds the target size.
    if len(out) > target_size:
        out = set(rng.choice(np.array(list(out), dtype=object), size=target_size, replace=False).tolist())

    return out


def ora_pvalue(M: int, K: int, n: int, x: int) -> float:
    """One-sided enrichment p-value using hypergeometric survival function."""
    if x <= 0:
        return 1.0
    return float(hypergeom.sf(x - 1, M, K, n))


# Test pathway prominence within one patient group.
def permutation_test_prominence(
    cell_gene_sets: List[Set[str]],
    cell_bin_counts: List[Counter],
    term_genes: Set[str],
    universe_arr: np.ndarray,
    bins: Dict[Tuple[int, int], np.ndarray],
    n_perm: int,
    p_thresh: float,
    seed: int
) -> Dict[str, float]:
    """
    Statistic: number of cells in group with ORA p <= p_thresh for this pathway.
    Null: matched permutations of each cell's gene set.
    """
    rng = np.random.default_rng(seed)
    M = int(len(universe_arr))
    term_genes = set(term_genes)  # Use a set for gene intersections.
    K = int(len(term_genes))

    sizes = np.array([len(s) for s in cell_gene_sets], dtype=int)

    # Count cells meeting the enrichment threshold in the observed data.
    obs_sig = 0
    for gs, n in zip(cell_gene_sets, sizes):
        x = len(gs & term_genes)
        p = ora_pvalue(M, K, int(n), int(x))
        if p <= p_thresh:
            obs_sig += 1

    # Repeat the count with matched random gene sets.
    null_sig = np.zeros(n_perm, dtype=int)
    for b in range(n_perm):
        sig = 0
        for bc, n in zip(cell_bin_counts, sizes):
            rand = sample_gene_set_from_bin_counts(bc, bins, universe_arr, rng, target_size=int(n))
            x = len(rand & term_genes)
            p = ora_pvalue(M, K, int(n), int(x))
            if p <= p_thresh:
                sig += 1
        null_sig[b] = sig

    p_emp = (1.0 + float(np.sum(null_sig >= obs_sig))) / (1.0 + float(n_perm))
    mu = float(null_sig.mean())
    sd = float(null_sig.std(ddof=1)) if n_perm > 1 else float("nan")
    z = (float(obs_sig) - mu) / sd if (sd and np.isfinite(sd) and sd > 0) else float("nan")

    return {
        "NumCells": float(len(cell_gene_sets)),
        "Obs_NumSigCells": float(obs_sig),
        "NullMean_NumSigCells": mu,
        "NullSD_NumSigCells": sd,
        "Z_NumSigCells": z,
        "EmpiricalP": p_emp,
        "Obs_FracSigCells": float(obs_sig) / float(len(cell_gene_sets)) if cell_gene_sets else float("nan"),
        "NullMean_FracSigCells": mu / float(len(cell_gene_sets)) if cell_gene_sets else float("nan"),
    }


# Load the inputs and run the permutation tests.
OUT_DIR.mkdir(parents=True, exist_ok=True)

log(f"Reading activity file: {ACTIVITY_FILE}")
act = pd.read_csv(ACTIVITY_FILE, sep="\t")
top_cells = pick_most_active_community_per_cell(act)

# Build a gene set for each cell and filter by its size.
top_cells["GeneSet"] = top_cells["GenesFromCellAndScores"].apply(lambda s: set(gene_list_from_genes_and_scores(s)))
top_cells["GeneSetSize"] = top_cells["GeneSet"].apply(len)
top_cells = top_cells[top_cells["GeneSetSize"] >= MIN_GENES_FOR_CELL].copy()
log(f"Cells kept (GeneSetSize >= {MIN_GENES_FOR_CELL}): {top_cells.shape[0]}")

# Load the KEGG pathway gene sets.
log(f"Loading KEGG library once: {KEGG_LIBRARY} ({ORGANISM})")
kegg_sets = load_kegg_library_as_sets(KEGG_LIBRARY, ORGANISM)

# Select candidate pathways from the saved enrichment results when available.
obs_file = BASE_DIR / "CELL_SPECIFIC_KEGG_ENRICHMENT" / "ALL_KEGG_Enrichment_AllTopTerms_ByCell.tsv"
candidate_terms: List[str]

if obs_file.exists():
    obs = pd.read_csv(obs_file, sep="\t")
    if SIGNALING_ONLY:
        obs = obs[obs["Term"].astype(str).str.lower().str.contains(SIGNALING_SUBSTRING.lower(), na=False)]
    term_counts = obs.groupby("Term")["CellID"].nunique().sort_values(ascending=False)
    candidate_terms = term_counts.head(TOP_TERMS_TO_TEST).index.astype(str).tolist()
    log(f"Candidate terms from observed output: {len(candidate_terms)}")
else:
    # Use the first KEGG pathways if no observed results are available.
    candidate_terms = list(kegg_sets.keys())[:TOP_TERMS_TO_TEST]
    log(f"No observed Step-7 file found; using first {len(candidate_terms)} KEGG terms")

# Keep candidate pathways that exist in the KEGG library.
candidate_terms = [t for t in candidate_terms if t in kegg_sets]
if not candidate_terms:
    raise ValueError("No candidate terms to test. Check SIGNALING_ONLY / TOP_TERMS_TO_TEST.")

# Use genes shared by the cell gene sets, candidate KEGG pathways, and HIPPIE.
log("Loading HiPPie degree...")
ppi_deg = load_ppi_degree_from_hippie(HIPPIE_FILE)

genes_in_cells = set().union(*top_cells["GeneSet"].tolist())
genes_in_kegg = set().union(*[kegg_sets[t] for t in candidate_terms])
universe = sorted(list(genes_in_cells & genes_in_kegg & set(ppi_deg.keys())))
universe_set = set(universe)

log(f"Universe size (cells ∩ KEGG ∩ HiPPie): {len(universe)}")
if len(universe) < 500:
    log("WARN: Universe is quite small; consider relaxing intersections (but keep it biologically meaningful).")

# Restrict each cell gene set to the shared gene universe.
top_cells["GeneSet"] = top_cells["GeneSet"].apply(lambda s: s & universe_set)
top_cells["GeneSetSize"] = top_cells["GeneSet"].apply(len)
top_cells = top_cells[top_cells["GeneSetSize"] >= MIN_GENES_FOR_CELL].copy()
log(f"Cells after universe restriction: {top_cells.shape[0]}")

# Count how often each gene appears in a cell community.
log("Computing detectability proxy from top_cells gene sets...")
det = Counter()
for gs in top_cells["GeneSet"]:
    for g in gs:
        det[g] += 1
det = {g: float(det.get(g, 0)) for g in universe}

# Look up network degree for genes in the universe.
deg = {g: int(ppi_deg.get(g, 0)) for g in universe}

# Build bins for matched sampling.
log(f"Building bins: degree={N_DEG_BINS}, detectability={N_DET_BINS}")
bins, gene_to_bin = build_bins(universe, deg, det, n_deg_bins=N_DEG_BINS, n_det_bins=N_DET_BINS)
universe_arr = np.array(list(gene_to_bin.keys()), dtype=object)  # Genes assigned to matching bins

# Count genes in each matching bin for every cell.
def cell_to_bin_counts(gs: Set[str]) -> Counter:
    c = Counter()
    for g in gs:
        b = gene_to_bin.get(g, None)
        if b is not None:
            c[b] += 1
    return c

top_cells["BinCounts"] = top_cells["GeneSet"].apply(cell_to_bin_counts)

# Run the tests separately for each dataset and patient.
results_rows = []

group_cols = ["DatasetID", "PatientID"]
for (ds, pt), gdf in top_cells.groupby(group_cols, sort=True):
    gdf = gdf.copy()
    if MAX_CELLS_PER_PATIENT is not None and gdf.shape[0] > MAX_CELLS_PER_PATIENT:
        gdf = gdf.sample(n=MAX_CELLS_PER_PATIENT, random_state=SEED)
        log(f"Subsampled cells for ds={ds}, pt={pt} to n={gdf.shape[0]}")

    cell_gene_sets = gdf["GeneSet"].tolist()
    cell_bin_counts = gdf["BinCounts"].tolist()

    log(f"Permutation tests for ds={ds}, pt={pt}, cells={len(cell_gene_sets)}, terms={len(candidate_terms)}")

    for term in candidate_terms:
        term_genes = (kegg_sets[term] & set(universe_arr.tolist()))
        if len(term_genes) < 5:
            continue  # Skip pathways with fewer than five genes in the universe.

        stats = permutation_test_prominence(
            cell_gene_sets=cell_gene_sets,
            cell_bin_counts=cell_bin_counts,
            term_genes=term_genes,
            universe_arr=universe_arr,
            bins=bins,
            n_perm=N_PERM,
            p_thresh=P_THRESH,
            seed=SEED + (hash(f"{ds}|{pt}|{term}") % 10_000_000),
        )

        results_rows.append({
            "DatasetID": str(ds),
            "PatientID": str(pt),
            "Term": str(term),
            "TermSize": int(len(term_genes)),
            "UniverseSize": int(len(universe_arr)),
            **stats,
        })

res = pd.DataFrame(results_rows)
if res.empty:
    raise ValueError("No results produced. Check candidate_terms and universe intersections.")

# Adjust empirical p-values across pathways within each patient.
res["EmpiricalQ_withinPatient"] = np.nan
for (ds, pt), idx in res.groupby(["DatasetID", "PatientID"]).groups.items():
    res.loc[idx, "EmpiricalQ_withinPatient"] = bh_fdr(res.loc[idx, "EmpiricalP"].values)

# Also adjust across all patient and pathway tests.
res["EmpiricalQ_global"] = bh_fdr(res["EmpiricalP"].values)

# Save the sorted permutation test results.
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"KEGG_Prominence_MatchedPermutations_deg{N_DEG_BINS}_det{N_DET_BINS}_perm{N_PERM}.tsv"
res.sort_values(["EmpiricalP", "EmpiricalQ_global", "Obs_FracSigCells"], ascending=[True, True, False]).to_csv(out_path, sep="\t", index=False)
log(f"Wrote randomization control results: {out_path.resolve()}")

# Show the twenty pathways with the smallest empirical p-values.
display(res.sort_values(["EmpiricalP", "Obs_FracSigCells"], ascending=[True, False]).head(20))
