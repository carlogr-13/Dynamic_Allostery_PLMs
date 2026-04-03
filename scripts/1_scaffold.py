"""
Module 1: Spatial Scaffold Curation
===================================
This module is exclusively responsible for the retrieval, spatial purification,
and storage of the 3D crystallographic scaffolds required for the topological
mapping of dynamic allostery.

By strictly isolating the physical coordinate extraction from the genetic
sequence curation (Module 2), it ensures that downstream thermodynamic models
are not biased by crystallographic artifacts, such as unresolved flexible loops
(missing electron density).

Author: TFG Bioinformatics Pipeline
"""

import os
import logging
import warnings
from Bio.PDB import PDBList, PDBParser, PDBIO, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Configure strict scientific logging for process traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Suppress discontinuous chain warnings common in raw PDB files
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class ChainAndProteinSelect(Select):
    """
    Filtering heuristic to isolate the biological unit of interest.
    It overrides the Select base class to reject solvent, ligands,
    and non-target chains during the PDBIO writing process.
    """

    def __init__(self, target_chain: str):
        """
        Initializes the spatial filter.

        :param target_chain: str, The strict chain identifier to retain (e.g., 'A').
        """
        self.target_chain = target_chain

    def accept_chain(self, chain: Chain) -> int:
        """
        Evaluates chain compliance against the target identifier.

        :param chain: Bio.PDB.Chain.Chain, The chain object under evaluation.
        :return: int, 1 to retain the chain, 0 to discard it.
        """
        return 1 if chain.id == self.target_chain else 0

    def accept_residue(self, residue: Residue) -> int:
        """
        Evaluates residue compliance, discarding heteroatoms (HETATM).

        :param residue: Bio.PDB.Residue.Residue, The residue object under evaluation.
        :return: int, 1 if it is a standard amino acid, 0 otherwise.
        """
        # residue.id[0] is ' ' for standard amino acids, and 'H_' or 'W' for heteroatoms/water
        return 1 if residue.id[0] == ' ' else 0


class SpatialScaffoldCurator:
    """
    Orchestrates the downloading and spatial cleaning of kinase crystal structures.
    """

    def __init__(self):
        """
        Initializes the curation environment, resolving absolute paths for the
        project root and defining the specific target structures.
        """
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)

        # Centralized directory for the Differential Spectral Pipeline
        self.data_dir: str = os.path.join(self.project_root, "data_dynamic_allostery")

        self.dirs: dict = {
            "raw": os.path.join(self.data_dir, "raw_pdb"),
            "processed": os.path.join(self.data_dir, "processed_pdb")
        }

        self._setup_directories()

        # Structural targets maintaining the highly resolved PDBs specified in the experimental design
        self.targets: dict = {
            "SRC": {"pdb_id": "2SRC", "chain": "A"},
            "EGFR": {"pdb_id": "3POZ", "chain": "A"}
        }

    def _setup_directories(self) -> None:
        """
        Generates the directory tree from scratch to host the spatial coordinates.

        :return: None
        """
        os.makedirs(self.data_dir, exist_ok=True)
        for name, path in self.dirs.items():
            os.makedirs(path, exist_ok=True)
        logging.info(f"Directory structure initialized at: {self.data_dir}")

    def retrieve_pdb(self, pdb_id: str) -> str:
        """
        Retrieves the specified crystallographic structure from the PDB via FTP.

        :param pdb_id: str, The 4-character alphanumeric PDB ID.
        :return: str, The absolute local path to the downloaded PDB file.
        :raises FileNotFoundError: If the PDB download fails critically.
        """
        pdb_id = pdb_id.lower()
        pdbl = PDBList(verbose=False)
        expected_pdb_path = os.path.join(self.dirs["raw"], f"{pdb_id}.pdb")

        if os.path.exists(expected_pdb_path):
            logging.info(f"Structure {pdb_id.upper()} already exists in local repository. Skipping download.")
            return expected_pdb_path

        logging.info(f"Initiating FTP download for {pdb_id.upper()}...")
        file_path = pdbl.retrieve_pdb_file(
            pdb_code=pdb_id,
            file_format="pdb",
            pdir=self.dirs["raw"],
            overwrite=True
        )

        if file_path and file_path.endswith('.ent'):
            os.rename(file_path, expected_pdb_path)
            logging.info(f"Successfully downloaded and standardized nomenclature for {pdb_id.upper()}.")
            return expected_pdb_path

        error_msg = f"Critical failure downloading PDB structure: {pdb_id.upper()}"
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)

    def purify_scaffold(self, kinase_name: str, config: dict) -> None:
        """
        Parses the raw PDB file, applies the ChainAndProteinSelect filter to remove
        solvents and ligands, and serializes the purified protein scaffold.

        :param kinase_name: str, The biological identifier of the target (e.g., 'SRC').
        :param config: dict, Target configuration containing the 'pdb_id' and 'chain'.
        :return: None
        """
        pdb_id = config["pdb_id"].upper()
        target_chain = config["chain"]

        raw_path = os.path.join(self.dirs["raw"], f"{pdb_id.lower()}.pdb")
        clean_path = os.path.join(self.dirs["processed"], f"{kinase_name}_scaffold_clean.pdb")

        parser = PDBParser()
        try:
            structure = parser.get_structure(pdb_id, raw_path)
        except Exception as e:
            logging.error(f"Failed to parse structural geometry for {pdb_id}: {str(e)}")
            return

        io = PDBIO()
        io.set_structure(structure)

        # Apply strict filtering mechanism
        selector = ChainAndProteinSelect(target_chain=target_chain)
        io.save(clean_path, selector)

        logging.info(f"Purified 3D spatial scaffold generated and saved for {kinase_name}.")

    def execute_curation(self) -> None:
        """
        Master orchestration method. Iterates through the structural targets matrix,
        ensures the presence of raw data, and dispatches the purification algorithms.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING SPATIAL SCAFFOLD CURATION PIPELINE")
        logging.info("===================================================")

        for kinase, config in self.targets.items():
            logging.info(f"Processing structural geometry for: {kinase}")
            try:
                self.retrieve_pdb(config["pdb_id"])
                self.purify_scaffold(kinase, config)
            except FileNotFoundError:
                logging.warning(f"Skipping spatial curation for {kinase} due to missing raw data.")

        logging.info("SPATIAL SCAFFOLD PIPELINE SUCCESSFULLY COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    curator = SpatialScaffoldCurator()
    curator.execute_curation()