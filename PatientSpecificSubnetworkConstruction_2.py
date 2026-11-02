# Load the HIPPIE interaction network and set PageRank parameters.
import os
from pathlib import Path
import networkx as nx
from typing import List, Set


# Base output directory
BASE_OUTPUT_FOLDER = Path(" BreastTumorDataInput_UMIcounts")
BASE_OUTPUT_FOLDER.mkdir(exist_ok=True)

# Directory for cell subnetworks
SUBNETWORK_FOLDER = BASE_OUTPUT_FOLDER / "CELL_SPECIFIC_SUBNETWORKS"
SUBNETWORK_FOLDER.mkdir(exist_ok=True)

# HIPPIE interaction file
HIPPIE_FILE = "hippie_parsed.txt"  # HIPPIE interaction file

# Co-targeting candidate genes
CO_TARGETING_CANDIDATES = [
    "AKT1","PIK3R1","PIK3CA","PIK3R2","PIK3R3","PIK3CB","KRAS","RAF1",
    "MAPK1","MAP2K1","GSK3B","BRAF","IGF1R","FGFR3","FGFR2","FASLG",
    "NTRK1","SOCS2","SOCS1","JAK2","STAT3","PTPN11","PTPN6","FYN",
    "EP300","ESR1","HSP90AB1","HSP90AA1","NF1","HSPA1A","HSPA8",
    "CDKN2A","PKM","JAK3","JAK1","FN1","YWHAZ","YWHAG","YWHAQ",
    "PTK2","ARAF","ERBB2","ERBB3","TP53","MDM2","MYC","CTNNB1",
    "AXIN1","GRB2","EGFR","PLCG1","SRC","CALM1"
]

# Use uppercase gene names for matching.
CO_TARGETING_SET: Set[str] = {gene.upper() for gene in CO_TARGETING_CANDIDATES}

# PageRank and subnetwork settings
PAGERANK_ALPHA: float = 0.85      # PageRank damping factor
TOP_N_PAGERANK: int = 10_000_000  # Maximum subnetwork nodes per cell

def load_ppi_graph_from_hippie(filepath: str) -> nx.Graph:
    """
    Load a protein-protein interaction (PPI) network from a HIPPIE file.
    
    Each line in the file should contain two gene names separated by whitespace.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"PPI file not found at: {filepath}")
    
    G = nx.Graph()
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            gene_a, gene_b = parts[0].upper(), parts[1].upper()
            G.add_edge(gene_a, gene_b, weight=1.0)
    
    return G

# Load the interaction network.
print("\n--- Loading PPI graph from HIPPIE ---")
ppi_graph = load_ppi_graph_from_hippie(HIPPIE_FILE)
print(f"PPI graph loaded: {ppi_graph.number_of_nodes()} nodes, {ppi_graph.number_of_edges()} edges")


# Inspect the input directory.
import os

def list_directory_contents(root_path):
    """
    Recursively list all folders and files in a directory.
    
    Args:
        root_path: Path to the root directory to explore
    """
    print(f"Directory structure for: {root_path}\n")
    print("="*80)
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Indent nested folders by directory depth.
        level = dirpath.replace(root_path, '').count(os.sep)
        indent = "  " * level
        
        # Show the current folder.
        folder_name = os.path.basename(dirpath) if level > 0 else root_path
        print(f"{indent}{folder_name}/")
        
        # Show the files in this folder.
        sub_indent = "  " * (level + 1)
        for filename in filenames:
            print(f"{sub_indent}{filename}")

# List the directory contents.
root_folder = "BreastTumorDataInput_UMIcounts"
list_directory_contents(root_folder)

# Count the folders and files.
total_folders = sum([len(dirnames) for _, dirnames, _ in os.walk(root_folder)])
total_files = sum([len(filenames) for _, _, filenames in os.walk(root_folder)])

print("\n" + "="*80)
print(f"\nSummary:")
print(f"Total folders: {total_folders}")
print(f"Total files: {total_files}")


# Load preprocessed expression matrices and map gene names to columns.
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
import numpy as np

print("\n--- Step 3C: Preparing expression matrices and gene mappings ---")

# Directory containing the preprocessed inputs
FINAL_INPUT_FOLDER = Path("BreastTumorDataInput_UMIcounts/H5AD_Malignant_TreatmentNaive")
if not FINAL_INPUT_FOLDER.exists():
    raise FileNotFoundError(f"Folder not found: {FINAL_INPUT_FOLDER}")

# Keep each dataset and its gene mappings in dictionaries.
adata_dict = {}  # Dataset name to AnnData object
counts_dict = {}  # Dataset name to CSR matrix
gene_to_index_dict = {}  # Dataset name to gene-column mapping
gene_names_upper_dict = {}  # Dataset name to uppercase gene names

for h5ad_file in FINAL_INPUT_FOLDER.glob("*.h5ad"):
    print(f"\nLoading {h5ad_file.name} ...")
    
    adata = sc.read(h5ad_file)
    adata_dict[h5ad_file.stem] = adata  
    
    # Select the expression matrix.
    if "counts" in adata.layers:
        counts = adata.layers["counts"]
        print("  Using counts from adata.layers['counts'].")
    else:
        counts = adata.X
        print("  WARNING: 'counts' layer missing → using adata.X.")

    # Convert counts to CSR format.
    counts_csr = counts.tocsr() if issparse(counts) else csr_matrix(counts)
    counts_dict[h5ad_file.stem] = counts_csr
    
    print(f"  counts_csr shape (cells x genes): {counts_csr.shape}")
    print("  Example values (first 10 genes from a random cell):",
          counts_csr[np.random.randint(0, counts_csr.shape[0]), :10].toarray())

    # Use uppercase gene names.
    gene_names_upper = np.array([g.upper() for g in adata.var_names.values])
    gene_names_upper_dict[h5ad_file.stem] = gene_names_upper
    
    # Map each gene name to its column index.
    gene_to_index = {g: i for i, g in enumerate(gene_names_upper)}
    gene_to_index_dict[h5ad_file.stem] = gene_to_index
    
    print(f"  Total unique genes: {len(gene_names_upper)}")
    print(f"  First 5 genes: {gene_names_upper[:5]}")

print("\nDone: All expression matrices, uppercase genes, and gene→index mappings are ready.")


print("adata.raw exists:", adata.raw is not None)

if adata.raw is not None:
    print("adata.raw.X shape:", adata.raw.X.shape)


print("Max value:", adata.X.max())
print("Unique nonzero values (sample):", np.unique(adata.X.data[:20]))


print("Available layers:", list(adata.layers.keys()))


print("Max value in X:", adata.X.max())


# Inspect expression layers, matrix contents, and gene names.
print(adata.layers.keys())

# Inspect the expression matrix.
print(type(adata.X))
print(adata.X.shape)
print(adata.X[:5, :5])  # First five cells and genes

# Check whether the matrix is sparse.
from scipy.sparse import issparse
print("Is adata.X sparse?", issparse(adata.X))

# Inspect the first ten gene names.
print(adata.var_names[:10])


# Calculate personalized PageRank for individual cells.
import copy
from pathlib import Path

def compute_cell_personalized_pagerank_top20(
    cell_index: int,
    ppi_graph: nx.Graph,
    counts_csr: csr_matrix,
    gene_names_upper: np.ndarray,
    co_targeting_set: set,
    base_personalization: dict,
    alpha: float = PAGERANK_ALPHA,
    top_fraction: float = 0.2
) -> dict:
    """
    Compute personalized PageRank for one cell using top 20% expressed genes + co-targeting.
    """
    personalization = copy.deepcopy(base_personalization)
    row = counts_csr[cell_index]
    indices, values = row.indices, row.data

    if len(values) > 0:
        top_n = max(1, int(len(values) * top_fraction))
        top_indices_sorted = indices[np.argsort(values)[-top_n:]]

        for idx in top_indices_sorted:
            gene_name = gene_names_upper[idx]
            if gene_name in ppi_graph:
                personalization[gene_name] = float(values[np.where(indices == idx)[0][0]])

    # Include candidate genes and multiply their personalization weights by five.
    for gene_name in co_targeting_set:
        if gene_name in ppi_graph:
            personalization[gene_name] = personalization.get(gene_name, 0.0) * 5.0 if personalization.get(gene_name, 0.0) > 0 else 5.0

    pagerank_scores = nx.pagerank(ppi_graph, alpha=alpha, personalization=personalization, weight="weight")
    return pagerank_scores


def sanitize_for_filename(s: str) -> str:
    """Replace problematic characters for safe filenames."""
    for ch in ["/", "\\", ":", " ", "|"]:
        s = s.replace(ch, "_")
    return s


print(adata.obs.columns.tolist())


cell_index = 0  # First cell
patient_id = str(adata.obs.iloc[cell_index]["patient"])
print(patient_id)


cell_id = adata.obs_names[0]
patient_id = adata.obs.loc[cell_id, "patient"]
print(patient_id)


def get_patient_id(adata, cell_idx):
    """
    Robustly extract patient/sample ID for a given cell index.
    """
    if "patient" in adata.obs.columns:
        return str(adata.obs.iloc[cell_idx]["patient"])
    elif "orig.ident" in adata.obs.columns:
        return str(adata.obs.iloc[cell_idx]["orig.ident"])
    elif "sample" in adata.obs.columns:
        return str(adata.obs.iloc[cell_idx]["sample"])
    else:
        return "ALL_CELLS"


def save_cell_subnetwork(
    cell_index,
    cell_id,
    patient_id,
    pagerank_scores,
    ppi_graph,
    counts_csr,
    gene_names_upper,
    co_targeting_set,
    adata,
    subnetwork_folder,
):
    """
    Save node and edge lists for one cell-specific subnetwork.
    """

    # Create the patient and cell output directories.
    patient_folder = subnetwork_folder / str(patient_id)
    patient_folder.mkdir(exist_ok=True)

    cell_folder = patient_folder / sanitize_for_filename(cell_id)
    cell_folder.mkdir(exist_ok=True)

    nodes_path = cell_folder / "nodes.txt"
    edges_path = cell_folder / "edges.txt"

    # Save node scores and expression values.
    rows = []
    row = counts_csr[cell_index]
    expr_dict = {
        gene_names_upper[i]: v
        for i, v in zip(row.indices, row.data)
    }

    for gene, pr in pagerank_scores.items():
        if gene not in ppi_graph:
            continue

        rows.append({
            "GeneSymbol": gene,
            "PageRank": pr,
            "RawCount": expr_dict.get(gene, 0.0),
            "IsCoTargeting": int(gene in co_targeting_set),
        })

    nodes_df = pd.DataFrame(rows)
    nodes_df.sort_values("PageRank", ascending=False, inplace=True)
    nodes_df.to_csv(nodes_path, sep="\t", index=False)

    # Save edges between retained genes.
    kept_genes = set(nodes_df["GeneSymbol"])
    edge_rows = []

    for u, v, d in ppi_graph.edges(kept_genes, data=True):
        if u in kept_genes and v in kept_genes:
            edge_rows.append({
                "GeneA": u,
                "GeneB": v,
                "Weight": d.get("weight", 1.0),
            })

    edges_df = pd.DataFrame(edge_rows)
    edges_df.to_csv(edges_path, sep="\t", index=False)

    print(f"        Saved subnetwork → {cell_folder}")


import pandas as pd


# Build and save cell subnetworks, grouped by dataset and patient.
print("\n--- Step 3F: Constructing cell-specific subnetworks and saving to disk ---")

# Directory for the saved cell subnetworks
subnetwork_base_folder = Path("BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_SUBNETWORKS")
subnetwork_base_folder.mkdir(exist_ok=True)

# Start every network node with a personalization weight of one.
base_personalization = {node: 1.0 for node in ppi_graph.nodes}

MAX_CELLS_PER_PATIENT = None  # Use all cells when None.

for dataset_name in counts_dict.keys():
    print(f"\nProcessing dataset: {dataset_name}")

    counts_csr = counts_dict[dataset_name]
    gene_names_upper = gene_names_upper_dict[dataset_name]
    adata = adata_dict[dataset_name]

    dataset_folder = subnetwork_base_folder / dataset_name
    dataset_folder.mkdir(exist_ok=True)

    # Find patient identifiers in the available metadata.
    if "patient" in adata.obs.columns:
        patient_ids = adata.obs["patient"].astype(str).unique()
    elif "orig.ident" in adata.obs.columns:
        patient_ids = adata.obs["orig.ident"].astype(str).unique()
    elif "sample" in adata.obs.columns:
        patient_ids = adata.obs["sample"].astype(str).unique()
    else:
        patient_ids = ["ALL_CELLS"]
        adata.obs["patient"] = patient_ids[0]

    print(f"  Patients found: {list(patient_ids)}")
    print(f"  Output folder: {dataset_folder.resolve()}")

    for patient in patient_ids:
        print(f"\n  Processing patient: {patient}")

        # Select the cells for this patient.
        if "patient" in adata.obs.columns:
            cell_mask = (adata.obs["patient"].astype(str) == str(patient))
        elif "orig.ident" in adata.obs.columns:
            cell_mask = (adata.obs["orig.ident"].astype(str) == str(patient))
        elif "sample" in adata.obs.columns:
            cell_mask = (adata.obs["sample"].astype(str) == str(patient))
        else:
            cell_mask = np.ones(adata.n_obs, dtype=bool)  # Include all cells.

        cell_indices = np.where(cell_mask.values)[0]

        if MAX_CELLS_PER_PATIENT is not None:
            cell_indices = cell_indices[:MAX_CELLS_PER_PATIENT]

        print(f"    Number of cells to process for {patient}: {len(cell_indices)}")

        for i, cell_idx in enumerate(cell_indices, start=1):
            cell_id = adata.obs_names[cell_idx]
            patient_id = get_patient_id(adata, cell_idx)

            print(f"      Cell {i}/{len(cell_indices)}: {cell_id} (Patient: {patient_id})")

            # Calculate PageRank using the top 20% of expressed genes and candidate genes.
            pr_scores = compute_cell_personalized_pagerank_top20(
                cell_index=cell_idx,
                ppi_graph=ppi_graph,
                counts_csr=counts_csr,
                gene_names_upper=gene_names_upper,
                co_targeting_set=CO_TARGETING_SET,
                base_personalization=base_personalization,
                alpha=PAGERANK_ALPHA,
                top_fraction=0.2
            )

            # Save network nodes and edges under the patient and cell identifiers.
            save_cell_subnetwork(
                cell_index=cell_idx,
                cell_id=cell_id,
                patient_id=patient_id,
                pagerank_scores=pr_scores,
                ppi_graph=ppi_graph,
                counts_csr=counts_csr,
                gene_names_upper=gene_names_upper,
                co_targeting_set=CO_TARGETING_SET,
                adata=adata,
                subnetwork_folder=dataset_folder
            )

            # Release scores before processing the next cell.
            del pr_scores

print("\nStep 3 complete: cell-specific subnetworks saved to:")
print(f"   {subnetwork_base_folder.resolve()}")


# Set up Leiden community detection.
import os
import pandas as pd
from pathlib import Path

# Required packages: igraph and leidenalg.
# Install with: pip install igraph leidenalg

try:
    import igraph as ig
    import leidenalg as la
except ImportError as e:
    print(
        "ERROR: This step requires 'python-igraph' and 'leidenalg'.\n"
        "Install with:\n"
        "  pip install igraph leidenalg\n"
    )
    raise e

# Directory containing the cell subnetworks
SUBNETWORK_BASE_FOLDER = Path("BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_SUBNETWORKS")

# Directory for the Leiden outputs
LEIDEN_BASE_FOLDER = Path("BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_LEIDEN_COMMUNITY")
LEIDEN_BASE_FOLDER.mkdir(exist_ok=True)

# Leiden resolution
LEIDEN_RESOLUTION = 1.0

print(f"\nLeiden community detection will read subnetworks from: {SUBNETWORK_BASE_FOLDER.resolve()}")
print(f"Leiden communities will be saved to: {LEIDEN_BASE_FOLDER.resolve()}")


# Inspect the saved subnetwork directories.
import os

def list_directory_contents(root_path):
    """
    Recursively list all folders and files in a directory.
    
    Args:
        root_path: Path to the root directory to explore
    """
    print(f"Directory structure for: {root_path}\n")
    print("="*80)
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Indent nested folders by directory depth.
        level = dirpath.replace(root_path, '').count(os.sep)
        indent = "  " * level
        
        # Show the current folder.
        folder_name = os.path.basename(dirpath) if level > 0 else root_path
        print(f"{indent}{folder_name}/")
        
        # Show the files in this folder.
        sub_indent = "  " * (level + 1)
        for filename in filenames:
            print(f"{sub_indent}{filename}")

# List the directory contents.
root_folder = "BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_SUBNETWORKS" 
list_directory_contents(root_folder)

# Count the folders and files.
total_folders = sum([len(dirnames) for _, dirnames, _ in os.walk(root_folder)])
total_files = sum([len(filenames) for _, _, filenames in os.walk(root_folder)])

print("\n" + "="*80)
print(f"\nSummary:")
print(f"Total folders: {total_folders}")
print(f"Total files: {total_files}")


# Run Leiden community detection for one cell subnetwork.
def run_leiden_for_cell_folder(
    cell_folder: Path,
    dataset_name: str,
    patient_name: str,
    leiden_base_folder: Path,
    resolution: float = LEIDEN_RESOLUTION,
):
    """
    Run Leiden community detection for a single cell-specific subnetwork stored as:
        cell_folder/nodes.txt
        cell_folder/edges.txt
    and save results under:
        leiden_base_folder/dataset_name/patient_name/<cell_folder.name>/
    """
    nodes_file = cell_folder / "nodes.txt"
    edges_file = cell_folder / "edges.txt"
    base_name = cell_folder.name

    if not nodes_file.exists() or not edges_file.exists():
        print(f"  [WARN] nodes.txt/edges.txt missing in {cell_folder}, skipping.")
        return

    # Read the node and edge tables.
    nodes_df = pd.read_csv(nodes_file, sep="\t")
    edges_df = pd.read_csv(edges_file, sep="\t")

    if nodes_df.empty:
        print(f"  [WARN] Empty nodes.txt for {cell_folder}, skipping.")
        return

    # Use uppercase gene names.
    if "GeneSymbol" not in nodes_df.columns:
        raise ValueError(f"'GeneSymbol' column not found in {nodes_file}")

    nodes_df["GeneSymbol"] = nodes_df["GeneSymbol"].astype(str).str.upper()
    gene_list = nodes_df["GeneSymbol"].tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Convert edges to igraph vertex indices.
    edge_tuples, edge_weights = [], []
    if not edges_df.empty:
        # Use a weight of one when Weight is missing.
        for _, row in edges_df.iterrows():
            u = str(row.get("GeneA", "")).upper()
            v = str(row.get("GeneB", "")).upper()
            if u in gene_to_idx and v in gene_to_idx and u != "" and v != "":
                edge_tuples.append((gene_to_idx[u], gene_to_idx[v]))
                w = row.get("Weight", 1.0)
                try:
                    edge_weights.append(float(w))
                except Exception:
                    edge_weights.append(1.0)

    # Build the igraph network.
    g = ig.Graph()
    g.add_vertices(len(gene_list))
    g.vs["name"] = gene_list
    if edge_tuples:
        g.add_edges(edge_tuples)
        g.es["weight"] = edge_weights

    # Run Leiden when edges are present; otherwise assign a single community.
    if g.vcount() > 1 and g.ecount() > 0:
        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
        )
        gene_to_comm = {}
        for cid, members in enumerate(partition):
            for vid in members:
                gene_to_comm[gene_list[vid]] = cid
    else:
        gene_to_comm = {gname: 0 for gname in gene_list}

    # Save results under dataset/patient/cell.
    out_folder = leiden_base_folder / dataset_name / patient_name / base_name
    out_folder.mkdir(parents=True, exist_ok=True)

    out_df = nodes_df.copy()
    out_df["LeidenCommunity"] = out_df["GeneSymbol"].map(gene_to_comm)
    out_df.to_csv(out_folder / "leiden_communities.txt", sep="\t", index=False)

    # Summarize gene counts and expression by community.
    if "RawCount" in out_df.columns:
        comm_summary = (
            out_df.groupby("LeidenCommunity")
            .agg(NumGenes=("GeneSymbol", "nunique"), SumRawCount=("RawCount", "sum"))
            .reset_index()
            .sort_values("LeidenCommunity")
        )
    else:
        comm_summary = (
            out_df.groupby("LeidenCommunity")
            .agg(NumGenes=("GeneSymbol", "nunique"))
            .reset_index()
            .sort_values("LeidenCommunity")
        )

    comm_summary.to_csv(out_folder / "leiden_community_summary.txt", sep="\t", index=False)

    print(f"  Saved Leiden for {dataset_name}/{patient_name}/{base_name} → {out_folder}")


# Run Leiden across all saved cell subnetworks.
print("\n--- Step 4C: Running Leiden for all cell-specific subnetworks ---")

if not SUBNETWORK_BASE_FOLDER.exists():
    raise FileNotFoundError(f"Subnetwork folder '{SUBNETWORK_BASE_FOLDER}' not found.")

dataset_folders = sorted([f for f in SUBNETWORK_BASE_FOLDER.iterdir() if f.is_dir()])
print(f"  Found {len(dataset_folders)} datasets: {[f.name for f in dataset_folders]}")

for dataset_folder in dataset_folders:
    dataset_name = dataset_folder.name
    print(f"\nProcessing dataset: {dataset_name} ({dataset_folder.resolve()})")

    patient_folders = sorted([f for f in dataset_folder.iterdir() if f.is_dir()])
    print(f"  Found {len(patient_folders)} patients: {[f.name for f in patient_folders]}")

    for patient_folder in patient_folders:
        patient_name = patient_folder.name
        print(f"    Checking patient folder: {patient_folder.resolve()}")

        # Find cell directories containing both nodes.txt and edges.txt.
        cell_folders = sorted([
            f for f in patient_folder.iterdir()
            if f.is_dir() and (f / "nodes.txt").exists() and (f / "edges.txt").exists()
        ])

        print(f"    Patient {patient_name} → {len(cell_folders)} cell subnetworks found")

        for i, cell_folder in enumerate(cell_folders, start=1):
            print(f"      [{i}/{len(cell_folders)}] Leiden → {cell_folder.name}")

            run_leiden_for_cell_folder(
                cell_folder=cell_folder,
                dataset_name=dataset_name,
                patient_name=patient_name,
                leiden_base_folder=LEIDEN_BASE_FOLDER,
                resolution=LEIDEN_RESOLUTION,
            )

print("\nStep 4 complete: Leiden communities saved under:")
print(f"   {LEIDEN_BASE_FOLDER.resolve()}")


# Set input paths and candidate genes for community activity scores.
import os
import json
import pandas as pd
from pathlib import Path

# Base output directory
BASE_OUTPUT_FOLDER = Path("BreastTumorDataInput_UMIcounts")  

# Directory containing the Leiden outputs
LEIDEN_BASE_FOLDER = BASE_OUTPUT_FOLDER / "CELL_SPECIFIC_LEIDEN_COMMUNITY"

# Combined activity score file
OUTPUT_ACTIVITY_FILE = BASE_OUTPUT_FOLDER / "CellSpecificSubnetworks_LeidenCommunity_ActivityScores.tsv"

# Co-targeting candidate genes
co_targeting_candidates = [ 
    "AKT1","PIK3R1","PIK3CA","PIK3R2","PIK3R3","PIK3CB","KRAS","RAF1",
    "MAPK1","MAP2K1","GSK3B","BRAF","IGF1R","FGFR3","FGFR2","FASLG",
    "NTRK1","SOCS2","SOCS1","JAK2","STAT3","PTPN11","PTPN6","FYN",
    "EP300","ESR1","HSP90AB1","HSP90AA1","NF1","HSPA1A","HSPA8",
    "CDKN2A","PKM","JAK3","JAK1","FN1","YWHAZ","YWHAG","YWHAQ",
    "PTK2","ARAF","ERBB2","ERBB3","TP53","MDM2","MYC","CTNNB1",
    "AXIN1","GRB2","EGFR","PLCG1","SRC","CALM1"
]
CO_TARGETING_SET = {g.upper() for g in co_targeting_candidates}

print(f"Leiden communities will be read from: {LEIDEN_BASE_FOLDER.resolve()}")
print(f"Activity scores will be saved to: {OUTPUT_ACTIVITY_FILE.resolve()}")
print(f"Total co-targeting candidate genes: {len(CO_TARGETING_SET)}")


# Read dataset, patient, and cell identifiers from a membership file path.
def parse_dataset_patient_cell_from_membership_path(membership_path: Path, leiden_base: Path):
    """
    Given a path like:
      <leiden_base>/<dataset>/<patient>/<cell>/leiden_membership.tsv

    returns:
      dataset_id, patient_id, cell_id

    If structure is unexpected, returns UNKNOWN_* fallbacks.
    """
    try:
        rel = membership_path.relative_to(leiden_base)
        parts = rel.parts  # Dataset, patient, cell, filename
        dataset_id = parts[0] if len(parts) >= 1 else "UNKNOWN_DATASET"
        patient_id = parts[1] if len(parts) >= 2 else "UNKNOWN_PATIENT"
        cell_id    = parts[2] if len(parts) >= 3 else membership_path.parent.name
        return dataset_id, patient_id, cell_id
    except Exception:
        # Use fallback identifiers when the relative path cannot be resolved.
        return "UNKNOWN_DATASET", "UNKNOWN_PATIENT", membership_path.parent.name


# Read Leiden outputs and calculate community activity scores for one cell.
from pathlib import Path
import pandas as pd

def parse_dataset_patient_cell_from_cell_dir(cell_dir: Path, leiden_base_folder: Path):
    """
    Expects cell_dir like:
      <leiden_base>/<dataset>/<patient>/<cell>/

    Returns (dataset_id, patient_id, cell_id) with safe fallbacks.
    """
    try:
        rel = cell_dir.relative_to(leiden_base_folder)
        parts = rel.parts
        dataset_id = parts[0] if len(parts) >= 1 else "UNKNOWN_DATASET"
        patient_id = parts[1] if len(parts) >= 2 else "UNKNOWN_PATIENT"
        cell_id    = parts[2] if len(parts) >= 3 else cell_dir.name
        return dataset_id, patient_id, cell_id
    except Exception:
        return "UNKNOWN_DATASET", "UNKNOWN_PATIENT", cell_dir.name


def _read_leiden_table_for_cell(cell_dir: Path) -> pd.DataFrame | None:
    """
    Prefer leiden_communities.* if present (often includes RawCount).
    Otherwise fallback to leiden_membership.tsv (gene + community only).
    """
    # Look for the community table first.
    candidates = [
        cell_dir / "leiden_communities.txt",
        cell_dir / "leiden_communities.tsv",
        cell_dir / "leiden_communities.csv",
    ]
    communities_file = next((p for p in candidates if p.exists()), None)

    if communities_file is not None:
        # Try tab-separated input, then detect the delimiter if reading fails.
        try:
            df = pd.read_csv(communities_file, sep="\t")
        except Exception:
            df = pd.read_csv(communities_file, sep=None, engine="python")
        df["_source_file"] = communities_file.name
        return df

    # Use the membership table if no community table exists.
    membership_file = cell_dir / "leiden_membership.tsv"
    if membership_file.exists():
        try:
            df = pd.read_csv(membership_file, sep="\t")
        except Exception:
            df = pd.read_csv(membership_file, sep=None, engine="python")
        df["_source_file"] = membership_file.name
        return df

    return None


def compute_activity_scores_for_cell(
    cell_dir: Path,
    leiden_base_folder: Path,
    co_targeting_set: set
):
    """
    Compute community activity scores for a single cell folder:
      <leiden_base>/<dataset>/<patient>/<cell>/

    Returns
    -------
    rows : list of dict
        One dict per community with the required output columns.
    """
    cell_dir = Path(cell_dir)

    df_raw = _read_leiden_table_for_cell(cell_dir)
    if df_raw is None or df_raw.empty:
        print(f"  [WARN] No Leiden outputs found in {cell_dir}, skipping.")
        return []

    # Strip whitespace from column names.
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower() for c in df.columns}

    # Find gene, community, count, and candidate columns.
    # Community tables use GeneSymbol and LeidenCommunity.
    # Membership tables may use gene and community.
    gene_col = None
    comm_col = None
    rawcount_col = None
    iscotarget_col = None

    for c in df.columns:
        cl = c.lower()
        if cl in {"genesymbol", "gene", "gene_symbol"}:
            gene_col = c
        if cl in {"leidencommunity", "community", "cluster"}:
            comm_col = c
        if cl in {"rawcount", "raw_count", "count", "umis", "umi"}:
            rawcount_col = c
        if cl in {"iscotargetingcandidate", "is_cotargeting_candidate"}:
            iscotarget_col = c

    # If headers are unrecognized, use the first two columns for gene and community.
    if gene_col is None or comm_col is None:
        if df.shape[1] >= 2:
            gene_col = df.columns[0]
            comm_col = df.columns[1]
        else:
            print(f"  [WARN] Malformed Leiden file in {cell_dir}, skipping.")
            return []

    # Create a table with consistent gene and community columns.
    work = pd.DataFrame({
        "GeneSymbol": df[gene_col].astype(str).str.strip().str.upper(),
        "LeidenCommunity": df[comm_col],
    })

    # Keep numeric community labels and convert them to integers.
    work["LeidenCommunity"] = pd.to_numeric(work["LeidenCommunity"], errors="coerce")
    work = work.dropna(subset=["LeidenCommunity"])
    work["LeidenCommunity"] = work["LeidenCommunity"].astype(int)

    # Use the count column when available; otherwise assign a count of one per gene.
    if rawcount_col is not None:
        work["RawCount"] = pd.to_numeric(df[rawcount_col], errors="coerce").fillna(0.0)
    else:
        work["RawCount"] = 1.0  # Fallback for membership-only input

    # Use the candidate flag when available, or match genes against the candidate set.
    if iscotarget_col is not None:
        work["IsCoTargetingCandidate"] = (
            df[iscotarget_col].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        )
    else:
        work["IsCoTargetingCandidate"] = work["GeneSymbol"].isin({g.upper() for g in co_targeting_set})

    # Read identifiers from the directory structure.
    dataset_id, patient_id, cell_id = parse_dataset_patient_cell_from_cell_dir(cell_dir, leiden_base_folder)
    base_name = f"{patient_id}__{cell_id}"  # Patient and cell identifier

    # Number communities within each cell from one.
    unique_comms = sorted(work["LeidenCommunity"].unique())
    comm_number_map = {comm_label: i + 1 for i, comm_label in enumerate(unique_comms)}

    rows = []
    for comm_label in unique_comms:
        df_comm = work[work["LeidenCommunity"] == comm_label].copy()

        # Keep genes with positive counts, including the membership-only fallback counts.
        df_expr = df_comm[df_comm["RawCount"] > 0].copy()

        community_activity = float(df_expr["RawCount"].sum())
        num_genes_from_cell = int(df_expr.shape[0])

        if num_genes_from_cell > 0:
            # Store each gene with its count; fallback counts are written as one.
            genes_from_cell_and_scores = ", ".join(
                f"{g}:{int(rc)}" if float(rc).is_integer() else f"{g}:{rc}"
                for g, rc in zip(df_expr["GeneSymbol"], df_expr["RawCount"])
            )
        else:
            genes_from_cell_and_scores = "Not available"

        # Summarize candidate genes in this community.
        df_expr_co = df_expr[df_expr["IsCoTargetingCandidate"]]
        num_co_genes = int(df_expr_co.shape[0])
        co_genes_str = ", ".join(df_expr_co["GeneSymbol"].tolist()) if num_co_genes > 0 else "Not available"

        community_number_in_cell = comm_number_map[comm_label]

        # Include the dataset identifier to keep community IDs unique.
        community_id = f"{dataset_id}__{base_name}__C{comm_label}"

        rows.append({
            "DatasetID": dataset_id,  
            "PatientID": patient_id,
            "CellID": cell_id,
            "CommunityNumberInCell": community_number_in_cell,
            "CommunityID": community_id,
            "CommunityActivityScore": community_activity,
            "NumGenesFromCellInCommunity": num_genes_from_cell,
            "GenesFromCellAndScores": genes_from_cell_and_scores,
            "NumCoTargetingGenesInCommunity": num_co_genes,
            "CoTargetingGeneNamesInCommunity": co_genes_str,
            "LeidenSourceFile": str(df_raw.get("_source_file", "unknown")),
        })

    return rows


# Calculate and save activity scores from the nested cell directories.
all_rows = []
for cell_dir in sorted(LEIDEN_BASE_FOLDER.glob("*/*/*")):  # Dataset/patient/cell
    if cell_dir.is_dir():
        all_rows.extend(compute_activity_scores_for_cell(cell_dir, LEIDEN_BASE_FOLDER, CO_TARGETING_SET))

out_df = pd.DataFrame(all_rows)
out_df.to_csv(OUTPUT_ACTIVITY_FILE, sep="\t", index=False)


# Collect activity scores across datasets and save the combined table.
print("\n--- Step 5D: Computing activity scores for all cell-specific Leiden communities ---")

if not LEIDEN_BASE_FOLDER.exists():
    raise FileNotFoundError(
        f"Leiden folder '{LEIDEN_BASE_FOLDER}' not found. Make sure Step 4C has been executed."
    )

def has_leiden_outputs(cell_dir: Path) -> bool:
    """
    Returns True if the directory contains Step 4C Leiden outputs for a cell.
    """
    if not cell_dir.is_dir():
        return False

    if (cell_dir / "leiden_membership.tsv").exists():
        return True

    for fname in ["leiden_communities.txt", "leiden_communities.tsv", "leiden_communities.csv"]:
        if (cell_dir / fname).exists():
            return True

    return False


all_rows = []

# Find the dataset directories.
dataset_folders = sorted([f for f in LEIDEN_BASE_FOLDER.iterdir() if f.is_dir()])
print(f"  Found {len(dataset_folders)} datasets: {[f.name for f in dataset_folders]}")

for dataset_folder in dataset_folders:
    print(f"\nProcessing dataset: {dataset_folder.name}")

    # Check for direct or nested cell directories.
    # Direct layout: <dataset>/<patient__cell>/
    direct_cell_dirs = sorted([d for d in dataset_folder.iterdir() if d.is_dir() and has_leiden_outputs(d)])

    # Nested layout: <dataset>/<patient>/<cell>/
    nested_cell_dirs = []
    if not direct_cell_dirs:
        patient_folders = sorted([p for p in dataset_folder.iterdir() if p.is_dir()])
        for patient_folder in patient_folders:
            for cell_dir in sorted([c for c in patient_folder.iterdir() if c.is_dir()]):
                if has_leiden_outputs(cell_dir):
                    nested_cell_dirs.append(cell_dir)

    cell_dirs = direct_cell_dirs if direct_cell_dirs else nested_cell_dirs
    print(f"  Found {len(cell_dirs)} cell folders with Leiden outputs in dataset '{dataset_folder.name}'")

    for i, cell_dir in enumerate(cell_dirs, start=1):
        # Read identifiers for the progress message.
        dataset_id, patient_id, cell_id = parse_dataset_patient_cell_from_cell_dir(cell_dir, LEIDEN_BASE_FOLDER)
        print(f"    [{i}/{len(cell_dirs)}] Processing Dataset={dataset_id}, Patient={patient_id}, Cell={cell_id}")

        rows = compute_activity_scores_for_cell(
            cell_dir=cell_dir,
            leiden_base_folder=LEIDEN_BASE_FOLDER,
            co_targeting_set=CO_TARGETING_SET
        )
        all_rows.extend(rows)

if not all_rows:
    print("No activity scores were computed (no valid Leiden communities found).")
else:
    activity_df = pd.DataFrame(all_rows)

    # Create the output directory if needed.
    OUTPUT_ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)

    activity_df.to_csv(OUTPUT_ACTIVITY_FILE, sep="\t", index=False)

    print(f"\nStep 5 complete.")
    print(f"   Activity scores saved to: {OUTPUT_ACTIVITY_FILE.resolve()}")
    print(f"   Rows written: {activity_df.shape[0]}")


# Set up comparisons between community gene sets.
import pandas as pd
import numpy as np
from pathlib import Path

# Directory containing the Leiden communities
LEIDEN_BASE_FOLDER = Path("BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_LEIDEN_COMMUNITY")

# Minimum community size for similarity comparisons
MIN_GENES_PER_COMMUNITY = 10

# Jaccard threshold and output paths
JACCARD_THRESHOLD = 0.5
OUT_DIR = Path("BreastTumorDataInput_UMIcounts")

SIMILARITY_OUTPUT_FILE = OUT_DIR / "CellSpecificSubnetworks_LeidenCommunity_SimilarityScores.txt"
SIMILARITY_THRESHOLD_OUTPUT_FILE = OUT_DIR / (
    f"CellSpecificSubnetworks_LeidenCommunity_SimilarityScores_JaccardGE{JACCARD_THRESHOLD}.txt"
)

# Leiden filename suffix and required columns
LEIDEN_FILE_SUFFIX = "__leiden_communities.txt"
REQUIRED_LEIDEN_COLUMNS = {"GeneSymbol", "LeidenCommunity"}

print(f"Leiden communities folder: {LEIDEN_BASE_FOLDER.resolve()}")
print(f"Similarity scores will be saved to: {SIMILARITY_OUTPUT_FILE.resolve()}")
print(
    f"Filtered (Jaccard ≥ {JACCARD_THRESHOLD}) similarity scores will be saved to: "
    f"{SIMILARITY_THRESHOLD_OUTPUT_FILE.resolve()}"
)

# Check that the Leiden directory exists before scanning datasets.
if not LEIDEN_BASE_FOLDER.exists():
    raise FileNotFoundError(
        f"Leiden folder '{LEIDEN_BASE_FOLDER}' not found. Make sure Step 4 has been executed."
    )

DATASET_FOLDERS = sorted([p for p in LEIDEN_BASE_FOLDER.iterdir() if p.is_dir()])
print(f"Found {len(DATASET_FOLDERS)} datasets under LEIDEN_BASE_FOLDER.")


# Extract patient and cell identifiers from a combined folder name.
def parse_patient_cell_from_base_name(base_name: str):
    """
    Parse PatientID and CellID from a base_name.

    Expected most common pattern:
        <patient>__<cell>

    Robust behavior:
        If there are more than 2 chunks split by '__' (e.g., dataset__patient__cell),
        we take the LAST TWO chunks as (patient, cell).

    Parameters
    ----------
    base_name : str
        Folder/base identifier for a cell-specific Leiden run.

    Returns
    -------
    patient_id : str
    cell_id : str
    """
    parts = str(base_name).split("__")

    if len(parts) >= 2:
        patient_id = parts[-2]
        cell_id = parts[-1]
    else:
        patient_id, cell_id = "UNKNOWN_PATIENT", str(base_name)

    return patient_id, cell_id


from pathlib import Path

root_dir = Path("BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_LEIDEN_COMMUNITY")

for path in root_dir.rglob("*"):
    if path.is_dir():
        print("Folder:", path)
    else:
        print("File:", path)


# Load community gene sets from the dataset, patient, and cell directories.
import pandas as pd
import numpy as np
from pathlib import Path

LEIDEN_BASE_FOLDER = Path("BreastTumorDataInput_UMIcounts/CELL_SPECIFIC_LEIDEN_COMMUNITY")
MIN_GENES_PER_COMMUNITY = 10  # Minimum genes per retained community

print("\n--- Step 6C: Loading communities from Leiden output ---")
print(f"Leiden base folder: {LEIDEN_BASE_FOLDER.resolve()}")

if not LEIDEN_BASE_FOLDER.exists():
    raise FileNotFoundError(
        f"Leiden folder '{LEIDEN_BASE_FOLDER}' not found. Make sure Step 4 has been executed."
    )

def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in df from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _normalize_comm_label(comm_label) -> str:
    """Normalize community labels so 0 and 0.0 become '0', etc."""
    try:
        if isinstance(comm_label, (int, np.integer)):
            return str(int(comm_label))
        if isinstance(comm_label, (float, np.floating)):
            if np.isfinite(comm_label) and float(comm_label).is_integer():
                return str(int(comm_label))
            return str(float(comm_label))
    except Exception:
        pass
    s = str(comm_label)
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s

communities = []  # Community records

# Find datasets directly under the Leiden output directory.
dataset_folders = sorted([d for d in LEIDEN_BASE_FOLDER.iterdir() if d.is_dir()])
print(f"  Found {len(dataset_folders)} datasets: {[d.name for d in dataset_folders]}")

missing_files = 0
malformed_files = 0
total_files_seen = 0

for dataset_folder in dataset_folders:
    dataset_id = dataset_folder.name
    print(f"\nProcessing dataset: {dataset_id}")

    # Find patient directories within the dataset.
    patient_folders = sorted([p for p in dataset_folder.iterdir() if p.is_dir()])
    print(f"  Found {len(patient_folders)} patient folders in dataset '{dataset_id}'")

    for patient_folder in patient_folders:
        patient_id = patient_folder.name

        # Find cell directories within the patient.
        cell_folders = sorted([c for c in patient_folder.iterdir() if c.is_dir()])
        if len(cell_folders) == 0:
            continue

        for cell_folder in cell_folders:
            cell_id = cell_folder.name

            # Accept either TXT or TSV community tables.
            communities_file_txt = cell_folder / "leiden_communities.txt"
            communities_file_tsv = cell_folder / "leiden_communities.tsv"

            if communities_file_txt.exists():
                communities_file = communities_file_txt
            elif communities_file_tsv.exists():
                communities_file = communities_file_tsv
            else:
                missing_files += 1
                # Detailed warnings are disabled during directory scans.
                # print(f"  [WARN] Missing leiden_communities file in: {cell_folder}")
                continue

            total_files_seen += 1

            # Read the community table.
            try:
                df = pd.read_csv(communities_file, sep="\t")
            except Exception as e:
                malformed_files += 1
                print(f"  [WARN] Could not read {communities_file}: {e}")
                continue

            if df.empty:
                malformed_files += 1
                continue

            # Find community and gene columns using the accepted column names.
            comm_col = _pick_column(df, ["LeidenCommunity", "leiden_community", "Community", "community", "cluster"])
            gene_col = _pick_column(df, ["GeneSymbol", "gene_symbol", "Gene", "gene", "Symbol", "symbol"])

            if comm_col is None or gene_col is None:
                malformed_files += 1
                # print(f"  [WARN] Malformed file (missing columns) for: {communities_file}")
                continue

            # Use uppercase gene names.
            df[gene_col] = df[gene_col].astype(str).str.upper()

            # Combine patient and cell identifiers for the base name.
            base_name = f"{patient_id}__{cell_id}"

            # Group genes by community label.
            for comm_label, df_comm in df.groupby(comm_col, dropna=True):
                gene_set = set(df_comm[gene_col].dropna())
                if len(gene_set) < MIN_GENES_PER_COMMUNITY:
                    continue

                comm_label_str = _normalize_comm_label(comm_label)

                # Use the same community identifier format as the activity scores.
                community_id = f"{dataset_id}__{patient_id}__{cell_id}__C{comm_label_str}"

                communities.append({
                    "dataset_id": dataset_id,
                    "patient_id": patient_id,
                    "cell_id": cell_id,
                    "base_name": base_name,
                    "community_label": comm_label_str,
                    "community_id": community_id,
                    "genes": gene_set,
                    "num_genes": len(gene_set),
                    "leiden_source_file": str(communities_file.resolve()),
                })

print(f"\n  Loaded {len(communities)} communities with ≥ {MIN_GENES_PER_COMMUNITY} genes.")
print(f"  Files seen/read: {total_files_seen}")
print(f"  Missing leiden_communities files: {missing_files}")
print(f"  Malformed/unreadable files: {malformed_files}")


# Compare community gene sets across cells within each patient.
import pandas as pd
from collections import defaultdict
from pathlib import Path

# Path for the full similarity table
SIMILARITY_OUTPUT_FILE = Path(
    "BreastTumorDataInput_UMIcounts/CellSpecificSubnetworks_LeidenCommunity_SimilarityScores.txt"
)

# Threshold and path for the filtered similarity table
JACCARD_THRESHOLD = 0.5
SIMILARITY_THRESHOLD_OUTPUT_FILE = Path(
    f"BreastTumorDataInput_UMIcounts/CellSpecificSubnetworks_LeidenCommunity_SimilarityScores_JaccardGE{JACCARD_THRESHOLD}.txt"
)

print("\n--- Step 6D: Computing Jaccard similarities across communities ---")

if not communities:
    print("  No communities available for similarity analysis (after filtering).")
    similarity_df = pd.DataFrame()

else:
    # Group communities by dataset and patient.
    communities_by_dataset_patient = defaultdict(list)
    for comm in communities:
        key = (comm["dataset_id"], comm["patient_id"])
        communities_by_dataset_patient[key].append(comm)

    similarity_rows = []

    for (dataset_id, patient_id), comm_list in communities_by_dataset_patient.items():
        n = len(comm_list)
        print(f"  Dataset={dataset_id} | Patient={patient_id}: {n} communities to compare.")

        # Compare community pairs within each patient.
        for i in range(n):
            c1 = comm_list[i]
            genes1 = c1["genes"]
            size1 = len(genes1)

            for j in range(i + 1, n):
                c2 = comm_list[j]

                # Compare communities from different cells.
                if c1["cell_id"] == c2["cell_id"]:
                    continue

                genes2 = c2["genes"]
                size2 = len(genes2)

                intersection = genes1.intersection(genes2)
                inter_size = len(intersection)
                if inter_size == 0:
                    continue  # Skip pairs with no shared genes.

                union_size = size1 + size2 - inter_size
                jaccard = inter_size / union_size if union_size > 0 else 0.0

                similarity_rows.append({
                    "DatasetID": dataset_id,
                    "PatientID": patient_id,

                    "CellID_1": c1["cell_id"],
                    "CommunityID_1": c1["community_id"],
                    "NumGenes_1": size1,
                    "LeidenSourceFile_1": c1.get("leiden_source_file", "Not available"),

                    "CellID_2": c2["cell_id"],
                    "CommunityID_2": c2["community_id"],
                    "NumGenes_2": size2,
                    "LeidenSourceFile_2": c2.get("leiden_source_file", "Not available"),

                    "IntersectionSize": inter_size,
                    "UnionSize": union_size,
                    "JaccardIndex": jaccard,
                    "GenesInIntersection": ", ".join(sorted(intersection)),
                })

    if similarity_rows:
        similarity_df = pd.DataFrame(similarity_rows)
    else:
        similarity_df = pd.DataFrame()
        print("  No non-zero Jaccard similarities found across communities.")

# Save all nonzero similarities.
if not similarity_df.empty:
    SIMILARITY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(SIMILARITY_OUTPUT_FILE, sep="\t", index=False)
    print(f"\nStep 6D complete: similarity scores saved to {SIMILARITY_OUTPUT_FILE.resolve()}")

    # Save pairs that meet the Jaccard threshold.
    similarity_df_thr = similarity_df[similarity_df["JaccardIndex"] >= JACCARD_THRESHOLD].copy()
    if not similarity_df_thr.empty:
        similarity_df_thr.to_csv(SIMILARITY_THRESHOLD_OUTPUT_FILE, sep="\t", index=False)
        print(f"Filtered (Jaccard ≥ {JACCARD_THRESHOLD}) saved to {SIMILARITY_THRESHOLD_OUTPUT_FILE.resolve()}")
    else:
        print(f"No similarities passed Jaccard ≥ {JACCARD_THRESHOLD} (filtered file not created).")

else:
    print("\nStep 6D complete: no similarities computed.")


# Save full and threshold-filtered community similarities.
from pathlib import Path

# Jaccard threshold for the filtered output
JACCARD_THRESHOLD = 0.6

# Similarity output paths
SIMILARITY_OUTPUT_FILE = Path(
    "BreastTumorDataInput_UMIcounts/CellSpecificSubnetworks_LeidenCommunity_SimilarityScores.txt"
)
SIMILARITY_THRESHOLD_OUTPUT_FILE = Path(
    f"BreastTumorDataInput_UMIcounts/CellSpecificSubnetworks_LeidenCommunity_SimilarityScores_JaccardGE{JACCARD_THRESHOLD}.txt"
)

# Use the similarity table calculated above.
if "similarity_df" in globals() and similarity_df is not None and not similarity_df.empty:

    # Save all nonzero Jaccard similarities.
    SIMILARITY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(SIMILARITY_OUTPUT_FILE, sep="\t", index=False)

    print(f"\nStep 6E complete (full similarity file).")
    print(f"   All non-zero similarities saved to: {SIMILARITY_OUTPUT_FILE.resolve()}")

    # Save similarities at or above the threshold.
    if "JaccardIndex" not in similarity_df.columns:
        print("  [WARN] similarity_df has no 'JaccardIndex' column. Thresholding skipped.")
    else:
        similarity_df_thr = similarity_df[similarity_df["JaccardIndex"] >= JACCARD_THRESHOLD].copy()

        if not similarity_df_thr.empty:
            similarity_df_thr["JaccardThreshold"] = JACCARD_THRESHOLD
            similarity_df_thr.to_csv(SIMILARITY_THRESHOLD_OUTPUT_FILE, sep="\t", index=False)

            print(f"\nThresholded similarities saved (Jaccard ≥ {JACCARD_THRESHOLD}):")
            print(f"   {SIMILARITY_THRESHOLD_OUTPUT_FILE.resolve()}")
        else:
            print(f"\nNo community pairs with JaccardIndex ≥ {JACCARD_THRESHOLD}. (No thresholded file written)")

else:
    print("\nStep 6E finished, but no similarity scores were written (similarity_df missing or empty).")


# Calculate community coactivity using weighted Jaccard scores.
import pandas as pd
import numpy as np
from pathlib import Path

# Base directory
ROOT = Path("BreastTumorDataInput_UMIcounts")

# Input files
ACTIVITY_FILE = ROOT / "CellSpecificSubnetworks_LeidenCommunity_ActivityScores.tsv"

JACCARD_THRESHOLD = 0.6
SIMILARITY_THRESHOLD_FILE = ROOT / f"CellSpecificSubnetworks_LeidenCommunity_SimilarityScores_JaccardGE{JACCARD_THRESHOLD}.txt"

# Output file
COACTIVITY_OUTPUT_FILE = ROOT / "CellSpecificSubnetworks_LeidenCommunity_CoactivityScores.tsv"


def parse_gene_score_string(s: str) -> dict:
    """
    Parse 'GENE1:10, GENE2:5' → { 'GENE1':10.0, 'GENE2':5.0 }
    Handles floats as well (e.g. 'LMNA:2.3001370').
    """
    if not isinstance(s, str):
        return {}
    s = s.strip()
    if (not s) or (s.lower() == "not available"):
        return {}

    result = {}
    for part in s.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        gene, val = part.split(":", 1)
        gene = gene.strip().upper()
        try:
            score = float(val.strip())
        except ValueError:
            continue
        result[gene] = score
    return result


def _safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


print("\n--- Step 6F: Loading community activity scores ---")
if not ACTIVITY_FILE.exists():
    raise FileNotFoundError(
        f"Activity score file not found:\n  {ACTIVITY_FILE}\n"
        f"Make sure Step 5 has been executed and saved to the current pipeline folder."
    )

activity_df = pd.read_csv(ACTIVITY_FILE, sep="\t")
if activity_df.empty:
    raise ValueError("Activity score file is empty – cannot compute coactivity.")

required_activity_cols = {"CommunityID", "GenesFromCellAndScores"}
missing_a = required_activity_cols - set(activity_df.columns)
if missing_a:
    raise ValueError(
        f"Activity file is missing required columns: {sorted(missing_a)}\n"
        f"Found columns: {list(activity_df.columns)}"
    )

# Map each community ID to its gene activity scores.
community_to_gene_scores = {}
dup_count = 0
for _, row in activity_df.iterrows():
    cid = str(row["CommunityID"])
    genes_scores_str = row.get("GenesFromCellAndScores", "Not available")
    gene_scores = parse_gene_score_string(genes_scores_str)

    if cid in community_to_gene_scores:
        dup_count += 1
    community_to_gene_scores[cid] = gene_scores

print(f"  Loaded activity for {len(community_to_gene_scores)} communities.")
if dup_count > 0:
    print(f"  [WARN] Found {dup_count} duplicate CommunityID rows in activity file (last one wins).")


print("\n--- Step 6F: Computing community coactivity scores (Weighted Jaccard) ---")
if not SIMILARITY_THRESHOLD_FILE.exists():
    raise FileNotFoundError(
        f"Thresholded similarity file not found:\n  {SIMILARITY_THRESHOLD_FILE}\n"
        f"Make sure Step 6E was executed with JACCARD_THRESHOLD={JACCARD_THRESHOLD}."
    )

sim_thr_df = pd.read_csv(SIMILARITY_THRESHOLD_FILE, sep="\t")
if sim_thr_df.empty:
    print("Thresholded similarity file is empty – no pairs to compute coactivity for.")
    coactivity_df = pd.DataFrame()
else:
    required_sim_cols = {"CommunityID_1", "CommunityID_2", "JaccardIndex"}
    missing_s = required_sim_cols - set(sim_thr_df.columns)
    if missing_s:
        raise ValueError(
            f"Similarity threshold file missing required columns: {sorted(missing_s)}\n"
            f"Found columns: {list(sim_thr_df.columns)}"
        )

    coactivity_rows = []

    # Check whether dataset and source-file columns are available.
    has_dataset = "DatasetID" in sim_thr_df.columns
    has_src1 = "LeidenSourceFile_1" in sim_thr_df.columns
    has_src2 = "LeidenSourceFile_2" in sim_thr_df.columns

    for _, row in sim_thr_df.iterrows():
        dataset_id = str(row.get("DatasetID", "UNKNOWN_DATASET")) if has_dataset else "UNKNOWN_DATASET"
        patient_id = str(row.get("PatientID", "UNKNOWN_PATIENT"))

        cell1 = str(row.get("CellID_1", "NA"))
        cell2 = str(row.get("CellID_2", "NA"))
        comm1 = str(row.get("CommunityID_1", "NA"))
        comm2 = str(row.get("CommunityID_2", "NA"))

        jaccard = _safe_float(row.get("JaccardIndex", np.nan), default=np.nan)
        inter_size = _safe_int(row.get("IntersectionSize", 0), default=0)
        union_size = _safe_int(row.get("UnionSize", 0), default=0)
        genes_in_intersection = str(row.get("GenesInIntersection", ""))

        leiden_src1 = str(row.get("LeidenSourceFile_1", "NA")) if has_src1 else "NA"
        leiden_src2 = str(row.get("LeidenSourceFile_2", "NA")) if has_src2 else "NA"

        # Look up gene activity scores for both communities.
        gs1 = community_to_gene_scores.get(comm1, {})
        gs2 = community_to_gene_scores.get(comm2, {})

        if (not gs1) and (not gs2):
            coactivity_score = np.nan
            weighted_intersection = np.nan
            weighted_union = np.nan
            num_genes_union = 0
            num_genes_nonzero_both = 0
            genes_nonzero_both = "Not available"
        else:
            genes_union = set(gs1.keys()) | set(gs2.keys())

            weighted_intersection = 0.0
            weighted_union = 0.0
            genes_nonzero_both_list = []

            for g in genes_union:
                s1 = gs1.get(g, 0.0)
                s2 = gs2.get(g, 0.0)
                weighted_intersection += min(s1, s2)
                weighted_union += max(s1, s2)
                if s1 > 0 and s2 > 0:
                    genes_nonzero_both_list.append(g)

            num_genes_union = len(genes_union)
            num_genes_nonzero_both = len(genes_nonzero_both_list)
            genes_nonzero_both = (
                ", ".join(sorted(genes_nonzero_both_list)) if genes_nonzero_both_list else "Not available"
            )

            coactivity_score = (weighted_intersection / weighted_union) if weighted_union > 0 else np.nan

        out_row = {
            "DatasetID": dataset_id,
            "PatientID": patient_id,

            "CellID_1": cell1,
            "CommunityID_1": comm1,
            "CellID_2": cell2,
            "CommunityID_2": comm2,

            "IntersectionSize": inter_size,
            "UnionSize": union_size,
            "JaccardIndex": jaccard,

            "NumGenesInUnionForActivity": num_genes_union,
            "WeightedIntersectionActivity": weighted_intersection,
            "WeightedUnionActivity": weighted_union,
            "CoactivityScore_WeightedJaccard": coactivity_score,

            "NumGenesNonzeroInBoth": num_genes_nonzero_both,
            "GenesNonzeroInBoth": genes_nonzero_both,

            "GenesInIntersection": genes_in_intersection,
            "JaccardThresholdUsed": JACCARD_THRESHOLD,
        }

        # Retain source filenames when available.
        if has_src1:
            out_row["LeidenSourceFile_1"] = leiden_src1
        if has_src2:
            out_row["LeidenSourceFile_2"] = leiden_src2

        coactivity_rows.append(out_row)

    coactivity_df = pd.DataFrame(coactivity_rows)


if coactivity_df is not None and not coactivity_df.empty:
    COACTIVITY_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    coactivity_df.to_csv(COACTIVITY_OUTPUT_FILE, sep="\t", index=False)
    print("\nStep 6F complete. Community coactivity scores saved to:")
    print(f"   {COACTIVITY_OUTPUT_FILE.resolve()}")
else:
    print("\nStep 6F finished, but no coactivity scores were computed (empty coactivity_df).")
