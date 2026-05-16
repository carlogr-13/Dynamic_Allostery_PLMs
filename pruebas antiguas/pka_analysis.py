import os
import logging
import sys

# Import the orchestrator class and the full module from the main library
import allosteric_network_analyzer
from allosteric_network_analyzer import AllostericNetworkAnalyzer

# Scientific logging configuration for the execution layer
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def main() -> None:
    logger = logging.getLogger("PKA_Execution")

    # 1. Biological and Structural Parameter Definition
    project_name = "PKA"
    pdb_id = "1ATP"  # RCSB PDB accession code for the PKA catalytic domain (mouse)
    chain = "E"

    # The absolute sequence offset linking the 0-indexed FASTA array to the biological
    # numbering of the PDB structure.
    offset = 14

    # 2. Canonical Sequence Definition
    # One-dimensional continuous sequence extracted strictly to align topologically
    # with the crystal structure, preventing the ESM-2 inference engine from crashing
    # against missing or unresolved spatial coordinates.
    canonical_sequence = "VKEFLAKAKEDFLKKWETPSQNTAQLDQFDRIKTLGTGSFGRVMLVKHKESGNHYAMKILDKQKVVKLKQIEHTLNEKRILQAVNFPFLVKLEFSFKDNSNLYMVMEYVAGGEMFSHLRRIGRFSEPHARFYAAQIVLTFEYLHSLDLIYRDLKPENLLIDQQGYIQVTDFGFAKRVKGRTWTLCGTPEYLAPEIILSKGYNKAVDWWALGVLIYEMAAGYPPFFADQPIQIYEKIVSGKVRFPSHFSSDLKDLLRNLLQVDLTKRFGNLKNGVNDIKNHKWFATTDWIAIYQRKVEAPFIPKFKGPGDTSNFDDYEEEEIRVSINEKCGKEFTE"

    # 3. Kinematic Perturbation Matrix Definition
    mutational_dict = {
        "I150A": ["I150A"]
    }

    # 4. Directory Configuration
    project_root = os.path.dirname(os.path.abspath(str(allosteric_network_analyzer.__file__)))
    master_dir = os.path.join(project_root, f"Data_{project_name}")
    os.makedirs(master_dir, exist_ok=True)

    # 5. Execution of the Unified Biophysical Pipeline
    logger.info(f"Instantiating the analytical environment for {project_name} (PDB: {pdb_id}).")
    logger.info(f"All serialized outputs will be routed to: {master_dir}")

    # Instantiate the analyzer object (parentheses are required)
    analyzer = AllostericNetworkAnalyzer()

    try:
        logger.info("Initiating epistatic inference and MST topology extraction.")

        # Injection of the empirical variables into the black-box computational engine
        # Stage 1 to 8 will be executed sequentially
        analyzer.execute_pipeline(
            project_name=project_name,
            pdb_id=pdb_id,
            chain=chain,
            canonical_sequence=canonical_sequence,
            mutational_dict=mutational_dict,
            offset=offset,
            target_residues=[57, 70, 95, 106, 128, 164, 172, 173, 174, 185, 220, 222, 227, 231],
            seed=42,  # Or None
            base_dir=master_dir
        )

        logger.info(f"Execution for system {project_name} completed.")

    except Exception as e:
        logger.error(f"Critical failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()