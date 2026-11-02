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

# List the input directory.
root_folder = "BreastTumorDataInput_UMIcounts"
list_directory_contents(root_folder)

# Count the folders and files.
total_folders = sum([len(dirnames) for _, dirnames, _ in os.walk(root_folder)])
total_files = sum([len(filenames) for _, _, filenames in os.walk(root_folder)])

print("\n" + "="*80)
print(f"\nSummary:")
print(f"Total folders: {total_folders}")
print(f"Total files: {total_files}")


import os

import pandas as pd
import anndata as ad
from scipy.io import mmread


# Set the input data and metadata directories.
ROOT = "BreastTumorDataInput_UMIcounts"

DATA_DIR = os.path.join(ROOT, "Data_Wu2021_Breast")
META_DIR = os.path.join(ROOT, "Meta-data_Wu2021_Breast")


# Build AnnData objects from raw UMI counts and cell metadata.
import os
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path


# Settings


# Input directory
BASE_DIR = "BreastTumorDataInput_UMIcounts"

# Directory for the saved AnnData files
ALLCELLS_DIR = os.path.join(BASE_DIR, "H5AD_AllCells")
os.makedirs(ALLCELLS_DIR, exist_ok=True)

# Input filenames within each Data_* directory
MTX_NAME        = "Exp_data_UMIcounts.mtx"
GENES_NAME      = "Genes.txt"
CELLS_CSV_NAME  = "Cells.csv"

# Possible cell identifier columns
POSSIBLE_ID_COLS = [
    "cell_name",
    "cell_id",
    "CellID",
    "cell",
    "Cell",
    "barcode",
    "Barcode"
]


# Print progress messages.

def dbg(*args):
    print(*args)


# Process each dataset.


for dataset in os.listdir(BASE_DIR):
    dataset_dir = os.path.join(BASE_DIR, dataset)

    # Process directories whose names start with Data_.
    if not os.path.isdir(dataset_dir) or not dataset.startswith("Data_"):
        continue

    dbg("\n" + "=" * 80)
    dbg(f"Processing dataset: {dataset}")
    dbg("=" * 80)


    # Locate the input files.

    mtx_path   = os.path.join(dataset_dir, MTX_NAME)
    genes_path = os.path.join(dataset_dir, GENES_NAME)
    cells_path = os.path.join(dataset_dir, CELLS_CSV_NAME)

    # Check that each input file exists.
    for fp in [mtx_path, genes_path, cells_path]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing required file: {fp}")

    dbg("Found all required input files")


    # Read the UMI count matrix.

    # The input matrix has genes in rows and cells in columns.
    adata = sc.read_mtx(mtx_path)
    dbg(f"Loaded matrix: {adata.shape} (genes × cells)")

    # Transpose to put cells in rows and genes in columns.
    adata = adata.T
    dbg(f"After transpose: {adata.n_obs} cells × {adata.n_vars} genes")


    # Read and assign gene names.

    genes = pd.read_csv(genes_path, sep="\t", header=None)

    if genes.shape[0] != adata.n_vars:
        raise ValueError(
            f"Gene count mismatch: "
            f"{genes.shape[0]} genes in file vs {adata.n_vars} in matrix"
        )

    adata.var_names = genes.iloc[:, 0].astype(str).values
    adata.var_names_make_unique()

    dbg(f"Assigned {adata.n_vars} unique gene names")


    # Read cell metadata and align it with the matrix.

    meta = pd.read_csv(cells_path, dtype=str)
    dbg(f"Loaded cell metadata: {meta.shape[0]} cells")

    # Find the cell identifier column.
    id_col = next(
        (c for c in POSSIBLE_ID_COLS if c in meta.columns),
        meta.columns[0]
    )
    dbg(f"Using '{id_col}' as cell ID column")

    # Strip whitespace from cell identifiers.
    meta.index = meta[id_col].astype(str).str.strip()
    adata.obs_names = adata.obs_names.astype(str).str.strip()

    # Match cells by identifier when possible.
    common_cells = adata.obs_names.intersection(meta.index)

    if len(common_cells) > 0:
        dbg(f"ID-based alignment ({len(common_cells)} cells)")

        adata = adata[common_cells, :].copy()
        adata.obs = meta.loc[common_cells].copy()
        adata.obs_names = meta.loc[common_cells].index.values

    else:
        dbg("No matching cell IDs — using positional alignment")

        if adata.n_obs != meta.shape[0]:
            raise RuntimeError(
                "Cannot align cells:\n"
                f"  Matrix cells   = {adata.n_obs}\n"
                f"  Metadata cells = {meta.shape[0]}"
            )

        adata.obs = meta.copy()
        adata.obs_names = meta.index.values

    dbg(f"Final shape: {adata.n_obs} cells × {adata.n_vars} genes")


    # Keep a copy of the raw counts.

    adata.raw = adata.copy()
    dbg("Stored raw counts in adata.raw")


    # Check dimensions and identifier uniqueness.

    assert adata.n_obs > 0, "Zero cells detected"
    assert adata.n_vars > 0, "Zero genes detected"
    assert adata.obs_names.is_unique, "Duplicate cell IDs"
    assert adata.var_names.is_unique, "Duplicate gene names"


    # Save the AnnData object.

    out_path = os.path.join(
        ALLCELLS_DIR,
        f"{dataset}_allcells_raw.h5ad"
    )

    adata.write(out_path)
    dbg(f"Saved AnnData to: {out_path}")


# Plot cell type counts for each dataset.
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import os
import pandas as pd


# Settings

ALLCELLS_DIR = "BreastTumorDataInput_UMIcounts/H5AD_AllCells"
FIGURES_DIR = os.path.join(ALLCELLS_DIR, "FIGURES")
os.makedirs(FIGURES_DIR, exist_ok=True)

CELLTYPE_COL = "cell_type"  # Cell type column in adata.obs

# Plot style
sns.set_theme(style="whitegrid", palette="pastel", font="Arial", font_scale=1.0)


# Process each saved dataset.

for h5ad_file in os.listdir(ALLCELLS_DIR):
    if not h5ad_file.endswith("_allcells_raw.h5ad"):
        continue

    dataset_name = h5ad_file.replace("_allcells_raw.h5ad", "")
    adata_path = os.path.join(ALLCELLS_DIR, h5ad_file)

    print(f"Plotting cell type counts for: {dataset_name}")

    # Load the AnnData object.
    adata = sc.read_h5ad(adata_path)

    if CELLTYPE_COL not in adata.obs.columns:
        print(f"Column '{CELLTYPE_COL}' not found, skipping.")
        continue

    # Count cells by cell type.
    counts = adata.obs.groupby(CELLTYPE_COL).size().reset_index(name="Count")

    # Show the most common cell types first.
    counts = counts.sort_values("Count", ascending=False)

    # Plot counts in descending order.
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(
        data=counts,
        x=CELLTYPE_COL,
        y="Count",
        order=counts[CELLTYPE_COL],  # Keep the sorted order.
        palette=sns.color_palette("pastel", n_colors=len(counts))
    )

    # Rotate labels so they fit.
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("")
    plt.ylabel("Number of cells", fontsize=12)
    plt.title(f"{dataset_name} — Cell type counts", fontsize=14, fontweight="bold")

    plt.tight_layout()

    # Save the plot as PNG and SVG.
    png_path = os.path.join(FIGURES_DIR, f"{dataset_name}_celltype_counts.png")
    svg_path = os.path.join(FIGURES_DIR, f"{dataset_name}_celltype_counts.svg")

    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

    print(f"Saved plots to: {png_path}, {svg_path}")


# Save the cell type counts for the current dataset.
counts = adata.obs.groupby(CELLTYPE_COL).size().reset_index(name="Count")

# Show the most common cell types first.
counts = counts.sort_values("Count", ascending=False)

# Print the exact counts.
print(f"\nExact cell type counts for {dataset_name}:\n")
print(counts)

# Save counts to CSV.
csv_path = os.path.join(FIGURES_DIR, f"{dataset_name}_celltype_counts.csv")
counts.to_csv(csv_path, index=False)
print(f"Saved exact counts to: {csv_path}")


# Plot cell type composition across samples.
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Settings

ALLCELLS_DIR = "BreastTumorDataInput_UMIcounts/H5AD_AllCells"
FIGURES_DIR = os.path.join(ALLCELLS_DIR, "FIGURES")
os.makedirs(FIGURES_DIR, exist_ok=True)

CELLTYPE_COL = "cell_type"
SAMPLE_COL   = "sample"

sns.set_theme(style="whitegrid", palette="pastel", font="Arial", font_scale=1.0)


# Process each saved dataset.

for h5ad_file in os.listdir(ALLCELLS_DIR):
    if not h5ad_file.endswith("_allcells_raw.h5ad"):
        continue

    dataset_name = h5ad_file.replace("_allcells_raw.h5ad", "")
    adata_path = os.path.join(ALLCELLS_DIR, h5ad_file)

    print(f"Creating stacked bar plot for dataset: {dataset_name}")

    adata = sc.read_h5ad(adata_path)

    # Check for the required metadata columns.
    if CELLTYPE_COL not in adata.obs.columns:
        print(f"Column '{CELLTYPE_COL}' not found, skipping.")
        continue
    if SAMPLE_COL not in adata.obs.columns:
        print(f"Column '{SAMPLE_COL}' not found, skipping.")
        continue

    # Count each cell type within each sample.
    counts_df = pd.crosstab(adata.obs[SAMPLE_COL], adata.obs[CELLTYPE_COL])

    # Sort samples alphabetically.
    counts_df_sorted = counts_df.sort_index()
    cell_types = counts_df_sorted.columns

    # Choose a color for each cell type.
    colors = sns.color_palette("pastel", n_colors=len(cell_types))

    # Plot the counts.
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = pd.Series([0]*len(counts_df_sorted), index=counts_df_sorted.index)

    for idx, ct in enumerate(cell_types):
        ax.bar(
            counts_df_sorted.index,
            counts_df_sorted[ct],
            bottom=bottom,
            label=ct,
            color=colors[idx]
        )
        bottom += counts_df_sorted[ct]

    ax.set_ylabel("Number of cells", fontsize=12)
    ax.set_xlabel("Sample", fontsize=12)
    ax.set_title(f"{dataset_name} — Cell type composition per sample", fontsize=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Cell type", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    # Save the output.
    png_path = os.path.join(FIGURES_DIR, f"{dataset_name}_stacked_celltypes_per_sample.png")
    svg_path = os.path.join(FIGURES_DIR, f"{dataset_name}_stacked_celltypes_per_sample.svg")

    plt.savefig(png_path, dpi=600)
    plt.savefig(svg_path)
    plt.close()

    print(f"Saved stacked bar plot to: {png_path}, {svg_path}")


# Compare the saved AnnData object with the original input files.
import os
import pandas as pd
import scanpy as sc
from scipy.io import mmread

# Input directory
BASE_DIR = "BreastTumorDataInput_UMIcounts"
H5AD_DIR = os.path.join(BASE_DIR, "H5AD_AllCells")
DATASET = "Data_Wu2021_Breast"

# Load the AnnData object.
adata_path = os.path.join(H5AD_DIR, f"{DATASET}_allcells_raw.h5ad")
adata = sc.read_h5ad(adata_path)
print(f"Loaded AnnData: {adata.shape} (cells x genes)")

# Read the original gene names.
genes_path = os.path.join(BASE_DIR, DATASET, "Genes.txt")
genes = pd.read_csv(genes_path, sep="\t", header=None)
print(f"Original gene count: {genes.shape[0]}, AnnData gene count: {adata.n_vars}")

# Read the original cell metadata.
cells_path = os.path.join(BASE_DIR, DATASET, "Cells.csv")
meta = pd.read_csv(cells_path, dtype=str)
print(f"Original cell count: {meta.shape[0]}, AnnData cell count: {adata.n_obs}")

# Compare the first ten gene names.
print("AnnData var_names[:10]:", adata.var_names[:10])
print("Original gene names[:10]:", genes.iloc[:,0].head(10).tolist())

# Compare the first few metadata rows.
print("AnnData obs (first 5 rows):")
print(adata.obs.head())
print("Original metadata (first 5 rows):")
print(meta.head())


adata


print("Cells (n_obs):", adata.n_obs)
print("Genes (n_vars):", adata.n_vars)
print("X type:", type(adata.X))


adata[:5, :5].X.toarray()


adata.var_names[:10]


adata.var_names.is_unique


adata.obs_names[:10]


# Summarize UMI counts per cell.
adata.obs["n_counts"] = adata.X.sum(axis=1).A1

adata.obs["n_counts"].describe()


# Check the fraction of zero counts across cells and genes.
cell_sparsity = 1 - (adata.X != 0).sum(axis=1) / adata.n_vars
print("Average cell sparsity:", np.mean(cell_sparsity))

# Calculate the fraction of zeros per gene.
gene_sparsity = 1 - (adata.X != 0).sum(axis=0) / adata.n_obs
print("Average gene sparsity:", np.mean(gene_sparsity))


gene = "PIK3CA"

adata[:, gene].X[:10].toarray().ravel()


# Check overall matrix sparsity.
import numpy as np

nonzero = adata.X.count_nonzero()
total = adata.n_obs * adata.n_vars

print(f"Sparsity: {1 - nonzero / total:.4%}")


assert adata.n_obs > 0
assert adata.n_vars > 0
assert adata.obs_names.is_unique
assert adata.var_names.is_unique


# Calculate QC metrics and apply the initial cell and gene filters.
import os
import scanpy as sc
import numpy as np
import pandas as pd


# Input and output directories


BASE_DIR = "BreastTumorDataInput_UMIcounts"

ALLCELLS_DIR = os.path.join(BASE_DIR, "H5AD_AllCells")
QC_DIR       = os.path.join(BASE_DIR, "H5AD_AllCells_QC")
os.makedirs(QC_DIR, exist_ok=True)


# Print progress messages.

def dbg(*args):
    print(*args)


# Process each unfiltered AnnData file.


for fname in os.listdir(ALLCELLS_DIR):
    if not fname.endswith(".h5ad"):
        continue

    dataset_name = fname.replace(".h5ad", "")
    in_path = os.path.join(ALLCELLS_DIR, fname)

    dbg("\n" + "=" * 80)
    dbg(f"STEP 2 — QC for dataset: {dataset_name}")
    dbg("=" * 80)


    # Load the AnnData object.

    adata = sc.read_h5ad(in_path)
    dbg(f"Loaded AnnData: {adata.n_obs} cells × {adata.n_vars} genes")


    # Calculate QC metrics.


    # Identify human mitochondrial genes by the MT- prefix.
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    dbg(f"Mitochondrial genes detected: {adata.var['mt'].sum()}")

    # Calculate QC metrics.
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        inplace=True
    )

    dbg("QC metrics added to adata.obs:")
    dbg([c for c in adata.obs.columns if
         "n_genes" in c or "total_counts" in c or "pct_counts_mt" in c])


    # Apply the initial QC filters.


    # Thresholds for the initial QC filters
    MIN_GENES_PER_CELL = 200
    MIN_CELLS_PER_GENE = 5
    MAX_MITO_FRAC      = 0.25   # 25%

    dbg(
        f"QC thresholds:\n"
        f"  min_genes_per_cell = {MIN_GENES_PER_CELL}\n"
        f"  min_cells_per_gene = {MIN_CELLS_PER_GENE}\n"
        f"  max_mito_fraction  = {MAX_MITO_FRAC * 100:.0f}%"
    )

    # Filter cells by the number of detected genes.
    pre_cells = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=MIN_GENES_PER_CELL)
    dbg(f"Cells removed by min_genes filter: {pre_cells - adata.n_obs}")

    # Filter cells by the mitochondrial count fraction.
    pre_cells = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < (MAX_MITO_FRAC * 100), :].copy()
    dbg(f"Cells removed by mito filter: {pre_cells - adata.n_obs}")

    # Keep genes detected in at least the minimum number of cells.
    pre_genes = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=MIN_CELLS_PER_GENE)
    dbg(f"Genes removed by min_cells filter: {pre_genes - adata.n_vars}")

    dbg(f"After QC: {adata.n_obs} cells × {adata.n_vars} genes")


    # Store raw counts if they are not already available.

    if adata.raw is None:
        dbg("adata.raw is None → storing raw counts")
        adata.raw = adata.copy()


    # Save the QC-filtered data.

    out_path = os.path.join(
        QC_DIR,
        f"{dataset_name}_QC.h5ad"
    )

    adata.write(out_path)
    dbg(f"Saved QC AnnData → {out_path}")


adata.obs.columns


adata.obs[["n_genes_by_counts", "total_counts", "pct_counts_mt"]].describe()


sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"])


# Run Scrublet and save both annotated cells and the singlet subset.
import os
import scanpy as sc
import numpy as np
import scipy.sparse as sp
import scrublet as scr


# Input and output directories


BASE_DIR = "BreastTumorDataInput_UMIcounts"

QC_DIR       = os.path.join(BASE_DIR, "H5AD_AllCells_QC")
DOUBLET_DIR  = os.path.join(BASE_DIR, "H5AD_AllCells_Doublets")
os.makedirs(DOUBLET_DIR, exist_ok=True)

# Expected doublet rate for Scrublet
EXPECTED_DOUBLET_RATE = 0.06


# Print progress messages.

def dbg(*args):
    print(*args)


# Process each QC-filtered dataset.


for fname in os.listdir(QC_DIR):
    if not fname.endswith(".h5ad"):
        continue

    dataset_name = fname.replace(".h5ad", "")
    in_path = os.path.join(QC_DIR, fname)

    dbg("\n" + "=" * 80)
    dbg(f"STEP 3 — Doublet detection for: {dataset_name}")
    dbg("=" * 80)


    # Load the QC-filtered data.

    adata = sc.read_h5ad(in_path)
    dbg(f"Loaded QC AnnData: {adata.n_obs} cells × {adata.n_vars} genes")


    # Select the count matrix for Scrublet.

    # Use raw UMI counts when available.
    if adata.raw is not None:
        counts = adata.raw.X
        dbg("Using adata.raw.X for Scrublet input")
    else:
        counts = adata.X
        dbg("adata.raw is None → using adata.X")

    # Convert sparse counts to CSR format.
    if sp.issparse(counts):
        counts = counts.tocsr()
        dbg("Converted counts to CSR format")


    # Run Scrublet.

    scrub = scr.Scrublet(
        counts,
        expected_doublet_rate=EXPECTED_DOUBLET_RATE
    )

    doublet_scores, predicted_doublets = scrub.scrub_doublets()

    dbg(f"Predicted doublets: {predicted_doublets.sum()} / {adata.n_obs}")


    # Add doublet scores and calls to the cell metadata.

    adata.obs["doublet_score"] = doublet_scores
    adata.obs["doublet_call"]  = predicted_doublets.astype(bool)

    dbg("Doublet score summary:")
    dbg(pd.Series(doublet_scores).describe())


    # Save all cells with doublet annotations.

    annotated_out = os.path.join(
        DOUBLET_DIR,
        f"{dataset_name}_doubletAnnotated.h5ad"
    )
    adata.write(annotated_out)
    dbg(f"Saved doublet-annotated AnnData → {annotated_out}")


    # Remove predicted doublets and save the singlets.

    singlet_mask = ~adata.obs["doublet_call"]
    ad_singlets = adata[singlet_mask].copy()

    dbg(f"Singlets retained: {ad_singlets.n_obs} cells")

    singlet_out = os.path.join(
        DOUBLET_DIR,
        f"{dataset_name}_doubletRemoved.h5ad"
    )
    ad_singlets.write(singlet_out)
    dbg(f"Saved singlet-only AnnData → {singlet_out}")


adata.obs["doublet_call"].value_counts()


adata.obs["doublet_score"].describe()


import matplotlib.pyplot as plt

plt.hist(adata.obs["doublet_score"], bins=50, color="skyblue", edgecolor="black")
plt.xlabel("Doublet score")
plt.ylabel("Number of cells")
plt.title("Distribution of Scrublet doublet scores")
plt.show()


print(adata.obs.columns.tolist())


# Normalize library sizes and apply the log1p transformation.
import os
import scanpy as sc


# Input and output directories


BASE_DIR = "BreastTumorDataInput_UMIcounts"

DOUBLET_DIR = os.path.join(BASE_DIR, "H5AD_AllCells_Doublets")
NORM_DIR    = os.path.join(BASE_DIR, "H5AD_AllCells_Normalized")
os.makedirs(NORM_DIR, exist_ok=True)


# Print progress messages.

def dbg(*args):
    print(*args)


# Process each singlet-only dataset.


for fname in os.listdir(DOUBLET_DIR):
    if not fname.endswith("_doubletRemoved.h5ad"):
        continue

    dataset_name = fname.replace("_doubletRemoved.h5ad", "")
    in_path = os.path.join(DOUBLET_DIR, fname)

    dbg("\n" + "=" * 80)
    dbg(f"STEP 4 — Normalization for: {dataset_name}")
    dbg("=" * 80)


    # Load the singlet subset.

    adata = sc.read_h5ad(in_path)
    dbg(f"Loaded singlets: {adata.n_obs} cells × {adata.n_vars} genes")


    # Store raw counts if they are not already available.

    if adata.raw is None:
        dbg("adata.raw is None → storing raw counts")
        adata.raw = adata.copy()


    # Normalize total counts per cell.

    sc.pp.normalize_total(
        adata,
        target_sum=1e4
    )
    dbg("Applied library-size normalization (target_sum=1e4)")


    # Apply the log1p transformation.

    sc.pp.log1p(adata)
    dbg("Applied log1p transformation")


    # Save the normalized data.

    out_path = os.path.join(
        NORM_DIR,
        f"{dataset_name}_normalized.h5ad"
    )

    adata.write(out_path)
    dbg(f"Saved normalized AnnData → {out_path}")


# Inspect total normalized expression per cell.
import numpy as np
import matplotlib.pyplot as plt

# total normalized expression per cell
cell_sums = np.array(adata.X.sum(axis=1)).flatten()

plt.hist(cell_sums, bins=50)
plt.xlabel("Total normalized expression per cell (log1p)")
plt.ylabel("Number of cells")
plt.title("Distribution of normalized library size")
plt.show()


vals = adata.X.data if hasattr(adata.X, "data") else adata.X.flatten()

plt.hist(vals, bins=100)
plt.xlabel("Normalized log1p expression")
plt.ylabel("Frequency")
plt.title("Distribution of normalized gene expression values")
plt.show()


adata.X[:5, :5]


adata.raw.X[:5, :5]


adata.obs["total_counts"].describe()


# Apply patient-specific QC thresholds to the normalized data.
import os
import scanpy as sc
import pandas as pd
import numpy as np


# Input and output directories

BASE_DIR = "BreastTumorDataInput_UMIcounts"
NORM_DIR = os.path.join(BASE_DIR, "H5AD_AllCells_Normalized")
FINAL_DIR = os.path.join(BASE_DIR, "H5AD_AllCells_AdaptiveQC")
THR_DIR = os.path.join(FINAL_DIR, "Thresholds")

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(THR_DIR, exist_ok=True)

POSSIBLE_PATIENT_COLS = [
    "patient", "patient_id", "sample", 
    "sample_id", "tumor", "donor", "subject"
]

MIN_CELLS_PER_PATIENT = 300


# Print progress messages.

def dbg(*args):
    print(*args)


# Process each normalized dataset.

for fname in os.listdir(NORM_DIR):
    if not fname.endswith("_normalized.h5ad"):
        continue

    dataset_name = fname.replace("_normalized.h5ad", "")
    in_path = os.path.join(NORM_DIR, fname)

    dbg("\n" + "="*80)
    dbg(f"STEP 5 — Adaptive QC for: {dataset_name}")
    dbg("="*80)


    # Load the normalized data.

    adata = sc.read_h5ad(in_path)
    dbg(f"Loaded normalized AnnData: {adata.n_obs} cells × {adata.n_vars} genes")


    # Find the patient or sample column.

    patient_col = next((c for c in POSSIBLE_PATIENT_COLS if c in adata.obs.columns), None)

    if patient_col is None:
        dbg("No patient/sample column found → skipping adaptive QC")
        out_path = os.path.join(FINAL_DIR, f"{dataset_name}_FinalInput.h5ad")
        adata.write(out_path)
        dbg(f"Saved FinalInput (no adaptive QC) → {out_path}")
        continue

    dbg(f"Using '{patient_col}' as grouping variable")


    # Keep patients with enough cells.

    counts_per_patient = adata.obs[patient_col].value_counts()
    keep_patients = counts_per_patient[counts_per_patient >= MIN_CELLS_PER_PATIENT].index
    dbg(f"Patients before filtering: {counts_per_patient.shape[0]}")
    dbg(f"Patients with ≥{MIN_CELLS_PER_PATIENT} cells: {len(keep_patients)}")

    adata = adata[adata.obs[patient_col].isin(keep_patients)].copy()
    dbg(f"Cells retained after patient filter: {adata.n_obs}")


    # Calculate a gene-count threshold for each patient.

    thresholds = {}
    final_adata_list = []

    for pid in adata.obs[patient_col].unique():
        mask = adata.obs[patient_col] == pid
        n_genes = adata.obs.loc[mask, "n_genes_by_counts"]

        # Skip patients whose gene counts are all missing.
        if n_genes.isna().all():
            dbg(f"All n_genes_by_counts missing for patient {pid} → skipping")
            continue

        median = n_genes.median()
        iqr = n_genes.quantile(0.75) - n_genes.quantile(0.25)
        thr = max(1, median - 1.5 * iqr)
        thresholds[pid] = thr

        keep_idx = n_genes[n_genes >= thr].index
        dbg(f"Patient {pid}: median={median:.1f}, IQR={iqr:.1f}, threshold={thr:.1f}, cells kept={len(keep_idx)}")

        if len(keep_idx) > 0:
            final_adata_list.append(adata[keep_idx])

    # Combine the retained cells from each patient.
    if len(final_adata_list) == 0:
        dbg("No cells passed adaptive QC → saving empty AnnData")
        adata_final = adata[[]].copy()
    else:
        adata_final = final_adata_list[0].concatenate(*final_adata_list[1:], batch_key=None, index_unique=None)

    dbg(f"Cells retained after adaptive QC: {adata_final.n_obs}")


    # Save the patient thresholds and filtered data.

    thr_df = pd.DataFrame.from_dict(thresholds, orient="index", columns=["adaptive_min_genes"])
    thr_df.index.name = patient_col
    thr_out = os.path.join(THR_DIR, f"{dataset_name}_adaptive_thresholds.csv")
    thr_df.to_csv(thr_out)
    dbg(f"Saved adaptive thresholds → {thr_out}")

    out_path = os.path.join(FINAL_DIR, f"{dataset_name}_FinalInput.h5ad")
    adata_final.write(out_path)
    dbg(f"Saved FinalInput AnnData → {out_path}")


# Select malignant cells using the cell type annotations.
import scanpy as sc
import os
import pandas as pd

BASE_DIR = "BreastTumorDataInput_UMIcounts"
ALLCELLS_DIR = os.path.join(BASE_DIR, "H5AD_AllCells_AdaptiveQC")
MALIGNANT_DIR = os.path.join(BASE_DIR, "H5AD_Malignant")
os.makedirs(MALIGNANT_DIR, exist_ok=True)

dataset_name = "Data_Wu2021_Breast_allcells_raw_QC"
in_path = os.path.join(ALLCELLS_DIR, f"{dataset_name}_FinalInput.h5ad")
out_path = os.path.join(MALIGNANT_DIR, f"{dataset_name}_malignant.h5ad")


# Open the dataset in read-only backed mode.

adata = sc.read_h5ad(in_path, backed='r')  # Keep the expression matrix on disk.
print(f"AnnData loaded in backed mode: {adata.n_obs} cells × {adata.n_vars} genes")


# Find cells annotated as tumor or malignant.

cell_type_col = "cell_type"

if cell_type_col not in adata.obs.columns:
    raise ValueError(f"Column '{cell_type_col}' not found in adata.obs!")

# Match cell type labels without regard to case.
malignant_mask = adata.obs[cell_type_col].str.lower().str.contains("tumor|malignant", na=False)


# Load only the malignant subset into memory.

adata_mal = adata[malignant_mask, :].to_memory()  # Load the selected cells into memory.
print(f"Malignant cells retained: {adata_mal.n_obs} / {adata.n_obs}")


# Save the malignant subset.

adata_mal.write(out_path)
print(f"Saved malignant AnnData → {out_path}")


# Keep treatment-naive malignant cells.
import os
import scanpy as sc
import pandas as pd
import unicodedata

# Input and output directories
MALIGNANT_DIR = "BreastTumorDataInput_UMIcounts/H5AD_Malignant"
NAIVE_DIR = "BreastTumorDataInput_UMIcounts/H5AD_Malignant_TreatmentNaive"
os.makedirs(NAIVE_DIR, exist_ok=True)

dataset_name = "Data_Wu2021_Breast_allcells_raw_QC"
in_path = os.path.join(MALIGNANT_DIR, f"{dataset_name}_malignant.h5ad")
out_path = os.path.join(NAIVE_DIR, f"{dataset_name}_malignant_naive.h5ad")

# Load the malignant subset.
adata_mal = sc.read_h5ad(in_path)
print(f"Loaded malignant AnnData: {adata_mal.n_obs} cells × {adata_mal.n_vars} genes")

# Remove accents, whitespace, and case differences from treatment labels.
def normalize_text(x):
    if pd.isna(x):
        return ""
    return unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode("ascii").strip().lower()

treatment_clean = adata_mal.obs["treatment"].apply(normalize_text)

# Keep cells labeled naive.
naive_mask = treatment_clean == "naive"
adata_naive = adata_mal[naive_mask, :].copy()
print(f"Cells retained (malignant + treatment-naive): {adata_naive.n_obs}")

# Save the output.
if adata_naive.n_obs > 0:
    adata_naive.write(out_path)
    print(f"Saved treatment-naive malignant AnnData → {out_path}")
else:
    print("No treatment-naive malignant cells found in this dataset")


# Inspect the current AnnData object and its raw counts.
print(adata)
print("adata.raw:", adata.raw)
print("adata.X type:", type(adata.X))


# Export cell metadata and detected gene counts.
import pandas as pd
import scanpy as sc

# Load the saved treatment-naive malignant subset.
adata_naive = sc.read_h5ad(
    "BreastTumorDataInput_UMIcounts/H5AD_Malignant_TreatmentNaive/"
        "Data_Wu2021_Breast_allcells_raw_QC_malignant_naive.h5ad"
        )

        # Copy the cell metadata.
df_cells = adata_naive.obs.copy()

        # Calculate detected gene counts if the column is missing.
if "n_genes_by_counts" not in df_cells.columns:
    sc.pp.calculate_qc_metrics(adata_naive, inplace=True)
    df_cells["n_genes_by_counts"] = adata_naive.obs["n_genes_by_counts"]

    # Choose the columns to export.
    columns_to_save = [
        "cell_name", "patient", "cell_type", "cell_subtype", 
            "treatment", "n_genes_by_counts"
            ]

            # Keep only columns present in the metadata.
    columns_to_save = [c for c in columns_to_save if c in df_cells.columns]

    df_out = df_cells[columns_to_save].copy()

    # Save as a tab-separated file.
    output_file = "BreastTumorDataInput_UMIcounts/H5AD_Malignant_TreatmentNaive/final_malignant_naive_cells.tsv"
    df_out.to_csv(output_file, sep="\t", index=False)

    print(f"Saved {df_out.shape[0]} cells × {df_out.shape[1]} columns → {output_file}")


# Inspect the saved treatment-naive malignant subset.
import scanpy as sc

# Load the saved dataset.
adata_naive = sc.read_h5ad(
    "BreastTumorDataInput_UMIcounts/H5AD_Malignant_TreatmentNaive/Data_Wu2021_Breast_allcells_raw_QC_malignant_naive.h5ad"
)

# Show dimensions and available metadata.
print(f"Cells × Genes: {adata_naive.n_obs} × {adata_naive.n_vars}")
print("Observation columns (metadata):", adata_naive.obs.columns.tolist())
print("Variable columns (gene info):", adata_naive.var.columns.tolist())
print("Unstructured info in .uns:", adata_naive.uns.keys())
print("Raw counts stored:", "Yes" if adata_naive.raw is not None else "No")


# Inspect the first five metadata rows.
print(adata_naive.obs.head())

# Count cells by cell type, treatment, and patient.
print("Cell types:\n", adata_naive.obs['cell_type'].value_counts())
print("Treatment column:\n", adata_naive.obs['treatment'].value_counts())
print("Patients:\n", adata_naive.obs['patient'].value_counts())


# Inspect the first five genes.
print(adata_naive.var.head())

# Count mitochondrial genes.
if 'mt' in adata_naive.var.columns:
    print(f"Mito genes: {adata_naive.var['mt'].sum()} / {adata_naive.n_vars}")


print(f"X type: {type(adata_naive.X)}")
print(f"Mean counts per cell: {adata_naive.X.sum(axis=1).mean()}")
print(f"Mean counts per gene: {adata_naive.X.sum(axis=0).mean()}")


# Count cells per patient.
print(adata_naive.obs['patient'].value_counts())

# Count cells by cell type.
print(adata_naive.obs['cell_type'].value_counts())

# Summarize total counts per cell.
print(adata_naive.obs['total_counts'].describe())

# Summarize detected gene counts per cell.
print(adata_naive.obs['n_genes_by_counts'].describe())


# Inspect the saved adaptive QC output.
import scanpy as sc

# Load the adaptive QC output.
adata_qc = sc.read_h5ad("BreastTumorDataInput_UMIcounts/H5AD_AllCells_AdaptiveQC/Data_Wu2021_Breast_allcells_raw_QC_FinalInput.h5ad")

print("=== Adaptive QC (All Cells) ===")
print(f"Cells × Genes: {adata_qc.n_obs} × {adata_qc.n_vars}")
print("Columns in obs (metadata):", adata_qc.obs.columns.tolist())
print("First 5 cells:")
print(adata_qc.obs.head())


# Inspect the saved malignant subset.
adata_mal = sc.read_h5ad("BreastTumorDataInput_UMIcounts/H5AD_Malignant/Data_Wu2021_Breast_allcells_raw_QC_malignant.h5ad")

print("=== Malignant Cells ===")
print(f"Cells × Genes: {adata_mal.n_obs} × {adata_mal.n_vars}")
print("Cell type counts:")
print(adata_mal.obs['cell_type'].value_counts())
