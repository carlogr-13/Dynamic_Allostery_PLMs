# Topological characterization of dynamic allostery in protein kinase A using ESM-2
*Carlos González Ruiz 2026*

---

This repository contains the data, Jupyter Notebooks, and Python scripts used for the Bachelor's Thesis "Topological 
characterization of dynamic allostery in protein kinase A using ESM-2". It provides a computational pipeline to study 
dynamical allostery and perform attention-based analyses using the protein language model **ESM-2** on de catalytic 
subunit of **Protein Kinase A (PKA)**, comparing the wild-type (WT) and the I150A mutant. It also includes tools to 
construct and analyze Maximum Spanning Tree (MST) from evolutionary covariance, identifying residues with high betweenness
centrality values to detect and map the allosteric wiring and functional bottlenecks in kinase regulation.

---

## Repository layout

```bash
data/               
  processed/      # Clean PDB and filtered attention tensors
  raw/            # Original FASTA and 1ATP.pdb
deprecated/       # Archived and discarded scripts
notebooks/        # External analysis
reference_code/   # amoyag and jdlg-42 scripts
results/
  figures/        # Plots (heatmaps, MST...)
  networks/       # Binary adjacency matrices (MST)
  tables/         # Tabular data 
scripts/          # Core Python modules (pipeline)
  visualization/  # PyMOL scripts 
```
---

## Installation and dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/Dynamic_Allostery_PLMs.git
cd Dynamic_Allostery_PLMs

# Create conda environment (required for ESM-2 and graph analysis)
conda create -n plms_allostery python=3.12.13
conda activate plms_allostery

# Install dependencies
pip install -r requirements.txt
```
---

## Usage

```bash
python scripts/PKA_analysis.py
```

That contains:

```python
if __name__ == "__main__":
    analyzer = AllostericNetworkAnalyzer()

    # 1. PKA Catalytic Subunit (PDB: 1ATP) - Biological Reference Data
    target_allosteric_site = [133, 134, 204, 280, 327, 328, 329, 330]

    pka_sequence = "VKEFIVSGKVRFI...RVSINEKCGKEFTE"

    # 2. Mutational Probing Setup
    mutations = {
        "I150A": [["I", 150, "A"]]
    }

    # 3. Pipeline Ignition
    analyzer.execute_pipeline(
        project_name="PKA_Allostery",
        pdb_id="1ATP", 
        chain="E", 
        canonical_sequence=pka_sequence,
        offset=14, #To align fasta with pdb coordinates
        target_residues=target_allosteric_site,
        mutational_dict=mutations,
        seed=7355608
    )
```
---

## References

1. Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites. *bioRxiv*. DOI: https://doi.org/10.1101/2024.09.28.615583 
2. Trenfield & Lin (2025). Sparse networks of conformational fluctuations communicate signals within proteins. *bioRxiv*. DOI: https://doi.org/10.1101/2025.05.28.656549 
3. Allosteric Analyzer: https://github.com/amoyag/PLMs_Dynamic_Allostery & https://github.com/jdlg-42/GPCRAllostericAnalysis
4. ESM-2: https://github.com/facebookresearch/esm

---

## Author 

Carlos González Ruiz

Universidad de Málaga