# Breast Cancer Single-Cell Network Analysis

Python scripts for preprocessing breast cancer single-cell RNA-seq data, constructing cell-specific interaction networks, identifying Leiden communities, testing KEGG pathway enrichment, and running matched permutation controls.

## Workflow

Install the Python packages imported by the scripts and check input paths before running in this order:

1. `PreprocessingSingleCellBreastUMIcounts_1.py`: preprocessing and quality control.
2. `PatientSpecificSubnetworkConstruction_2.py`: network construction, community detection, and community comparisons.
3. `GeneSetEnrichmentAnalysis_3.py`: KEGG enrichment of the most active community in each cell.
4. `Randomization_4.py`: matched permutation controls for pathway enrichment.

Inputs include `BreastTumorDataInput_UMIcounts/` and the parsed interaction file `hippie_parsed.txt`.

## Data availability

Single-cell RNA-seq data, gene annotations, cell-type labels, and sample metadata from the Wu et al. breast cancer cohort were accessed through [3CA](https://www.weizmann.ac.il/sites/3CA/). Network and pathway analyses used the [HIPPIE](https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/) interactome and `KEGG_2021_Human` gene sets from [Enrichr](https://maayanlab.cloud/Enrichr/).
