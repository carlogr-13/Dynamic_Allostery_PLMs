"""
Module 6: Topological Integration
=================================
This module bridges the 1D thermodynamic variance data (Z-Scores) with the
3D spatial coordinates of the kinase domains. It iterates over the pristine
crystallographic scaffolds and overwrites the B-Factor column of each atom
with the computed allosteric rewiring signal.

This generates spatially mapped structures ready for gradient rendering.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging
import pandas as pd
from Bio.PDB import PDBParser, PDBIO

# Scientific logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class TopologicalIntegrator:
    """
    Handles the parsing of structural coordinates and the precise mathematical
    injection of statistical thermodynamic data into the crystallographic B-factor attributes.
    """

    def __init__(self):
        """
        Initializes the topological integration environment, configuring
        directory paths and absolute sequence alignment offsets.
        """
        # Directory Management
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)

        self.pdb_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "processed_pdb")
        self.csv_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "rewiring_data")
        self.out_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "mapped_pdb")

        os.makedirs(self.out_dir, exist_ok=True)

        # Sequence Offsets for 1D to 3D Alignment
        # SRC is set to 83, corresponding to a PDB starting at biological residue 84.
        # EGFR must be verified against the exact absolute starting position of the 3POZ chain.
        self.sequence_offsets: dict = {
            "SRC": 83,
            "EGFR": 700
        }

    def _map_zscores_to_bfactor(self, kinase: str, pdb_path: str, csv_path: str, out_path: str) -> None:
        """
        Parses the 3D structure, aligns the indices, and injects the Z-Scores
        into the atomic B-factor fields.

        :param kinase: str, The kinase identifier to fetch the appropriate offset.
        :param pdb_path: str, Absolute path to the clean 3D scaffold.
        :param csv_path: str, Absolute path to the CSV containing the differential Z-scores.
        :param out_path: str, Absolute path for saving the mapped PDB file.
        :return: None
        """
        # 1. Load and align thermodynamic statistical data
        df = pd.read_csv(csv_path)
        offset = self.sequence_offsets.get(kinase, 0)

        # Algebraic mapping: Absolute_Biological_Position = Relative_Index + 1 + Offset
        z_score_map = {
            int(row["Relative_Index"]) + 1 + offset: float(row["Z_Score"])
            for _, row in df.iterrows()
        }

        # 2. Parse pristine 3D Scaffold
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(kinase, pdb_path)

        # 3. Spatial Injection (B-Factor Overwrite)
        mapped_residues = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Extract absolute crystallographic sequence number
                    res_id = residue.get_id()[1]

                    # Assign Z-score if present; default to 0.0 to prevent rendering artifacts
                    z_value = z_score_map.get(res_id, 0.0)

                    if res_id in z_score_map:
                        mapped_residues += 1

                    # Inject data into all atoms of the given residue
                    for atom in residue:
                        atom.set_bfactor(z_value)

        logging.info(f"   -> Successfully injected thermodynamic data into {mapped_residues} spatial residues.")

        # 4. Serialize the mapped structural manifold
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_path)

    def execute_integration(self) -> None:
        """
        Master orchestration routine. Identifies all available differential rewiring
        datasets, matches them with their corresponding pristine spatial scaffolds,
        and executes the topological data injection.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING TOPOLOGICAL INTEGRATION (1D -> 3D)")
        logging.info("===================================================")

        csv_files = glob.glob(os.path.join(self.csv_dir, "Rewiring_*.csv"))
        if not csv_files:
            logging.error(f"No differential rewiring datasets found in {self.csv_dir}.")
            return

        for csv_path in csv_files:
            filename = os.path.basename(csv_path)

            # File structure parsing: Rewiring_EGFR_L858R_Active.csv
            parts = filename.replace("Rewiring_", "").replace(".csv", "").split("_", 1)
            if len(parts) < 2:
                continue

            kinase_system = parts[0]
            state_name = parts[1]

            logging.info(f"Mapping thermodynamic microstate: {kinase_system} [{state_name}]")

            # Retrieve pristine structural scaffold
            pdb_path = os.path.join(self.pdb_dir, f"{kinase_system}_scaffold_clean.pdb")
            if not os.path.exists(pdb_path):
                logging.error(f"   -> [CRITICAL] Clean scaffold not found for {kinase_system} at {pdb_path}. Skipping.")
                continue

            out_pdb_path = os.path.join(self.out_dir, f"{kinase_system}_{state_name}_Mapped.pdb")

            # Execute injection
            self._map_zscores_to_bfactor(kinase_system, pdb_path, csv_path, out_pdb_path)
            logging.info(f"   -> Mapped topological scaffold saved to: {os.path.basename(out_pdb_path)}")

        logging.info("TOPOLOGICAL INTEGRATION COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    integrator = TopologicalIntegrator()
    integrator.execute_integration()