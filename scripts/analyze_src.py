"""
SRC Kinase Allosteric Network Analysis
======================================

This executable script orchestrates the ab initio extraction of the
thermodynamic topology for the human SRC kinase (PDB: 2SRC).

It utilizes the unified KinaseMistAnalyzer framework to process the
Wild-Type state alongside its active (E310A) and inhibitory (T338G) variants.

Execution:
    $ python analyze_src.py

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
    for the SRC kinase and sequentially triggers the analytical phases.
    """
    # 1. Strict Typed Experimental Matrix Definition
    # Variables are explicitly annotated to satisfy static type checkers (e.g., mypy)
    # and prevent Union type ambiguity during method invocation.
    kinase_name: str = "SRC"
    pdb_id: str = "2SRC"
    chain_id: str = "A"

    mutations: Dict[str, List[Tuple[str, int, str]]] = {
        "WT": [],  # Basal thermodynamic state (Reference)
        "E310A_Active": [("E", 310, "A")],  # Constitutively active mutation
        "T338G_Inhibitory": [("T", 338, "G")]  # Gatekeeper mutation (inhibitor sensitization)
    }

    # 2. Framework Instantiation
    # Initializes the underlying analytical class, allocating the ESM-2
    # Transformer architecture to hardware-accelerated memory (GPU) if available.
    logging.info("Instantiating the KinaseMistAnalyzer framework...")
    analyzer = KinaseMistAnalyzer(model_name="esm2_t33_650M_UR50D")

    # 3. Pipeline Execution
    # Dispatches the strictly typed configuration to the master orchestrator,
    # ensuring the sequential calculation of 1D attention, 3D constraints, and graph centrality.
    logging.info("Deploying unified analytical pipeline for SRC Kinase...")
    analyzer.execute_full_pipeline(
        kinase_name=kinase_name,
        pdb_id=pdb_id,
        chain_id=chain_id,
        mutations=mutations
    )

if __name__ == "__main__":
    # Ensures the pipeline is only triggered when executed as the primary script,
    # preventing unintended executions during parallel imports.
    main()