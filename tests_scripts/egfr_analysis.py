"""
EGFR Kinase Epistatic Analysis Execution Script
===============================================
This script serves as the dedicated entry point for the thermodynamic and
topological analysis of the Epidermal Growth Factor Receptor (EGFR) kinase domain.
It imports the master framework architecture and strictly defines the biological
variables, the crystallographic scaffold, and the mutational microstate matrix
(including clinical resistance epistasis).

Author: TFG Bioinformatics Pipeline
"""

import logging
import sys

# Import the orchestrator class from the main library
try:
    from analysis_scripts.allosteric_network_analyzer import AllostericNetworkAnalyzer
except ImportError:
    logging.error("Failed to locate the module 'allosteric_network_analyzer.py'. "
                  "Ensure the master script is in the same directory or within the PYTHONPATH.")
    sys.exit(1)

# Scientific logging configuration for the execution layer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main() -> None:
    """
    Main orchestration function for the evaluation of the EGFR system.
    It injects the rigid empirical parameters into the execution pipeline.

    :return: None
    """
    logger = logging.getLogger("EGFR_Execution")

    # 1. Biological and Structural Parameter Definition
    project_name = "EGFR"
    pdb_id = "2GS2"  # RCSB PDB accession code for the EGFR kinase domain scaffold
    chain = "A"

    # The absolute sequence offset linking the 0-indexed FASTA array to the biological
    # numbering of the PDB structure. The EGFR kinase domain structural numbering
    # typically begins around residue 701.
    offset = 671

    # 2. Canonical Sequence Definition
    # One-dimensional continuous sequence extracted strictly to align topologically
    # with the crystal structure, preventing the ESM-2 inference engine from crashing
    # against missing or unresolved spatial coordinates.
    canonical_sequence = "GEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLI"

    # 3. Kinematic Perturbation Matrix Definition
    # - L858R_Active: Primary oncogenic mutation located in the activation loop.
    # - L858R_T790M_Epistatic: Incorporation of the gatekeeper mutation that confers
    #   acquired clinical resistance to first-generation Tyrosine Kinase Inhibitors.
    mutational_dict = {
        "L858R": ["L834R"]
    }

    # 4. Execution of the Unified Biophysical Pipeline
    logger.info(f"Instantiating the analytical environment for {project_name} (PDB: {pdb_id}).")

    # Instantiate the analyzer object (parentheses are required)
    analyzer = AllostericNetworkAnalyzer()

    try:
        logger.info("Initiating epistatic inference and MST topology extraction.")

        # Injection of the empirical variables into the black-box computational engine
        analyzer.execute_pipeline(
            project_name=project_name,
            pdb_id=pdb_id,
            chain=chain,
            canonical_sequence=canonical_sequence,
            mutational_dict=mutational_dict,
            offset=offset,
            target_residues=[742, 753, 811, 832, 872]
        )

        logger.info(f"Execution for system {project_name} completed.")

    except Exception as e:
        logger.error(f"Critical failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()