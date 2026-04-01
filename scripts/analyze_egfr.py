"""
EGFR Kinase Allosteric Network Analysis
=======================================

This executable script orchestrates the ab initio extraction of the
thermodynamic topology for the human Epidermal Growth Factor Receptor
(EGFR) kinase domain (PDB: 3POZ).

It utilizes the unified KinaseMistAnalyzer framework to process the
Wild-Type state alongside its primary oncogenic variant (L858R) and
the epistatic drug-resistant double mutant (L858R + T790M).

Execution:
    $ python analyze_egfr.py

Output:
    - Purified PDB scaffolds and FASTA sequences.
    - Absolute and differential ESM-2 covariance tensors (.npy).
    - Maximum Information Spanning Trees (.graphml).
    - Topological bottleneck metrics (.csv).
    - PyMOL 3D rendering scripts (.py).
"""

import logging
from typing import Dict, List, Tuple
from mist_analyzer import KinaseMistAnalyzer


def main() -> None:
    """
    Main execution function. Defines the strictly typed experimental matrix
    for the EGFR kinase and sequentially triggers the analytical phases.
    """
    # 1. Strict Typed Experimental Matrix Definition
    # Variables are explicitly annotated to satisfy static type checkers (e.g., mypy)
    # and prevent Union type ambiguity during method invocation.
    kinase_name: str = "EGFR"
    pdb_id: str = "3POZ"
    chain_id: str = "A"

    # Structural mapping of the mutational variants.
    # The T790M mutation acts as a gatekeeper altering the binding pocket,
    # often evaluated in tandem with L858R to study thermodynamic epistasis.
    mutations: Dict[str, List[Tuple[str, int, str]]] = {
        "WT": [],  # Basal thermodynamic state (Reference)
        "L858R": [("L", 858, "R")],  # Primary activating oncogenic mutation
        "L858R_T790M_Epistatic": [("L", 858, "R"), ("T", 790, "M")]  # Acquired resistance double mutation
    }

    # 2. Framework Instantiation
    # Initializes the underlying analytical class, allocating the ESM-2
    # Transformer architecture to hardware-accelerated memory (GPU) if available.
    logging.info("Instantiating the KinaseMistAnalyzer framework...")
    analyzer = KinaseMistAnalyzer(model_name="esm2_t33_650M_UR50D")

    # 3. Pipeline Execution
    # Dispatches the strictly typed configuration to the master orchestrator,
    # ensuring the sequential calculation of 1D attention, 3D constraints, and graph centrality.
    logging.info("Deploying unified analytical pipeline for EGFR Kinase...")
    analyzer.execute_full_pipeline(
        kinase_name=kinase_name,
        pdb_id=pdb_id,
        chain_id=chain_id,
        mutations=mutations
    )


if __name__ == "__main__":
    # Ensures the pipeline is only triggered when executed as the primary script,
    # preventing unintended executions during parallel module imports.
    main()