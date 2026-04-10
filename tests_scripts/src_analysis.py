"""
SRC Kinase Epistatic Analysis Execution Script
==============================================
This script serves as the dedicated entry point for the thermodynamic and
topological analysis of the c-Src kinase. It imports the master framework
architecture and strictly defines the biological variables, the crystallographic
scaffold, and the mutational microstate matrix.

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
    Main orchestration function for the evaluation of the c-Src system.
    It injects the rigid empirical parameters into the execution pipeline.

    :return: None
    """
    logger = logging.getLogger("SRC_Execution")

    # 1. Biological and Structural Parameter Definition
    project_name = "SRC"
    pdb_id = "2SRC"  # RCSB PDB accession code for the c-Src kinase domain scaffold
    chain = "A"

    # The absolute sequence offset linking the 0-indexed FASTA array to the biological
    # numbering of the PDB structure.
    offset = 83

    # 2. Canonical Sequence Definition
    # One-dimensional continuous sequence extracted strictly to align topologically
    # with the crystal structure, preventing the ESM-2 inference engine from crashing
    # against missing or unresolved spatial coordinates.
    canonical_sequence = "TTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL"

    # 3. Kinematic Perturbation Matrix Definition
    # - E310A_Active: Cleavage of the critical K295-E310 salt bridge, triggering activation.
    # - T338G_Inhibitory: Introduction of a steric void at the gatekeeper residue position.
    mutational_dict = {
        "E310A_Active": ["E310A"],
        "T338G_Inhibitory": ["T338G"]
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
            offset=offset
        )

        logger.info(f"Execution for system {project_name} completed with absolute parametric rigor.")

    except Exception as e:
        logger.error(f"Critical failure during Jacobian tensor evaluation or topology extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()