"""
Execution Script: EGFR Dynamic Allostery Assessment
===================================================
Orchestrates the structural parsing of the Epidermal Growth Factor Receptor (EGFR)
and maps the dynamic energy flow targeting the canonical allosteric pocket
(residues K745 and D855).
"""

from supervised_allosteric_analyzer import SupervisedAllostericAnalyzer


def main():
    # Experimental Matrix Configuration
    protein_name = "EGFR"
    pdb_code = "5D41"  # Or your chosen basal scaffold for EGFR
    chain = "A"

    # Allosteric Targets based on experimental literature (e.g., EAI045 binding pocket)
    allosteric_targets = [745, 855]

    print(f"Initializing statistical framework for {protein_name}...")

    # Initialize the engine
    analyzer = SupervisedAllostericAnalyzer(model_name="esm2_t33_650M_UR50D", threshold=0.3)

    # Execute the supervised pathway extraction
    analyzer.execute_pipeline(
        name=protein_name,
        pdb_id=pdb_code,
        chain=chain,
        allosteric_pdb_sites=allosteric_targets
    )


if __name__ == "__main__":
    main()