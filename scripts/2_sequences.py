"""
Module 2: Genetic Microstate Definition
=======================================
This module handles the generation of discrete 1D sequence representations
(genetic microstates) for both Wild-Type (WT) and pathologically perturbed
kinase domains. It strictly decouples sequence data from 3D crystallographic
coordinates to prevent positional embedding artifacts in downstream Language Models.

Author: TFG Bioinformatics Pipeline
"""

import os
import logging
from typing import Dict, List, Tuple

# Scientific logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class GeneticMicrostateGenerator:
    """
    A class dedicated to in silico mutagenesis and FASTA sequence curation.
    It processes canonical UniProt sequences and generates precise mutational
    variants for differential thermodynamic analysis.
    """

    def __init__(self):
        """
        Initializes the sequence generation environment, dynamically resolving
        output directories and defining the biological matrices.
        """
        # Directory Management
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)
        self.fasta_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "fasta_sequences")
        os.makedirs(self.fasta_dir, exist_ok=True)

        # Canonical Kinase Domain Sequences (UniProt standard)
        # IMPORTANT: Replace these placeholder sequences with the EXACT kinase domain
        # sequences you wish to analyze (e.g., from UniProt).
        self.canonical_sequences: Dict[str, str] = {
            "SRC": "TTFVALYDYESRTETDLSFKKGERLQIVNNTEGDWWLAHSLSTGQTGYIPSNYVAPSDSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKGLNVKHYKIRKLDSGGFYITSRTQFNSLQQLVAYYSKHADGLCHRLTTVCPTSKPQTQGLAKDAWEIPRESLRLEVKLGQGCFGEVWMGTWNGTTRVAIKTLKPGTMSPEAFLQEAQVMKKLRHEKLVQLYAVVSEEPIYIVTEYMSKGSLLDFLKGETGKYLRLPQLVDMAAQIASGMAYVERMNYVHRDLRAANILVGENLVCKVADFGLARLIEDNEYTARQGAKFPIKWTAPEAALYGRFTIKSDVWSFGILLTELTTKGRVPYPGMVNREVLDQVERGYRMPCPPECPESLHDLMCQCWRKEPEERPTFEYLQAFLEDYFTSTEPQYQPGENL",
            "EGFR": "QALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYL"
        }

        # Epistatic Mutational Matrix
        # Format: "State_Name": [("Original_AA", Canonical_Position, "Mutant_AA")]
        self.mutational_matrix: Dict[str, Dict[str, List[Tuple[str, int, str]]]] = {
            "SRC": {
                "WT": [],
                "E310A_Active": [("E", 310, "A")],
                "T338G_Inhibitory": [("T", 338, "G")]
            },
            "EGFR": {
                "WT": [],
                "L858R_Active": [("L", 858, "R")],
                "L858R_T790M_Epistatic": [("L", 858, "R"), ("T", 790, "M")]
            }
        }

        # Offset dictionary to map UniProt absolute numbering to our local string index.
        # If your sequence string starts at amino acid 250 of the full protein,
        # the offset should be 249. If you paste the FULL protein sequence, offset is 0.
        self.sequence_offsets: Dict[str, int] = {
            "SRC": 83,  # Set to the starting residue number - 1
            "EGFR": 700  # Set to the starting residue number - 1
        }

    def _apply_mutations(self, kinase_name: str, base_sequence: str, mutations: List[Tuple[str, int, str]]) -> str:
        """
        Applies a set of point mutations to a given canonical sequence string.

        :param kinase_name: str, The identifier of the kinase (e.g., 'EGFR').
        :param base_sequence: str, The continuous string of amino acids.
        :param mutations: List of Tuples containing (Original_AA, Absolute_Position, Mutant_AA).

        :return: str, The newly mutated continuous sequence.
        :raises ValueError: If the expected Original_AA does not match the residue found at Absolute_Position.
        """
        # Convert string to list for mutable indexing
        seq_list = list(base_sequence)
        offset = self.sequence_offsets.get(kinase_name, 0)

        for aa_orig, abs_pos, aa_mut in mutations:
            # Map absolute biological position to 0-indexed Python list
            list_index = abs_pos - 1 - offset

            # Boundary checks
            if list_index < 0 or list_index >= len(seq_list):
                error_msg = f"Mutation position {abs_pos} is out of bounds for {kinase_name} sequence."
                logging.error(error_msg)
                raise IndexError(error_msg)

            # Biophysical safety check: Ensure we are mutating the correct wild-type residue
            found_aa = seq_list[list_index]
            if found_aa != aa_orig:
                error_msg = f"Biophysical Mismatch in {kinase_name} at pos {abs_pos}: Expected '{aa_orig}', found '{found_aa}'."
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Apply in silico substitution
            seq_list[list_index] = aa_mut
            logging.debug(f"Applied mutation {aa_orig}{abs_pos}{aa_mut} to {kinase_name}.")

        return "".join(seq_list)

    def _write_fasta(self, kinase_name: str, state_name: str, sequence: str) -> None:
        """
        Formats and exports the sequence string into a strict 80-character
        line-wrapped FASTA file.

        :param kinase_name: str, The identifier of the kinase.
        :param state_name: str, The thermodynamic state identifier (e.g., 'WT', 'L858R').
        :param sequence: str, The continuous amino acid string.

        :return: None
        """
        file_name = f"{kinase_name}_{state_name}.fasta"
        file_path = os.path.join(self.fasta_dir, file_name)

        try:
            with open(file_path, "w") as fasta_file:
                # Write header
                fasta_file.write(f">{kinase_name}_{state_name}\n")

                # Apply 80-character strict wrapping standard
                for i in range(0, len(sequence), 80):
                    fasta_file.write(f"{sequence[i:i + 80]}\n")

            logging.info(f"Successfully generated FASTA microstate: {file_name} (Length: {len(sequence)} aa).")
        except IOError as e:
            logging.error(f"Failed to write FASTA file {file_name}: {str(e)}")

    def generate_microstates(self) -> None:
        """
        Master orchestration method. Iterates through the mutational matrix,
        applies perturbations to the canonical sequences, and delegates the
        export of the resulting FASTA microstates.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING GENETIC MICROSTATE CURATION PIPELINE")
        logging.info("===================================================")

        for kinase, states in self.mutational_matrix.items():
            logging.info(f"Processing Target System: {kinase}")
            canonical_seq = self.canonical_sequences.get(kinase, "")

            if not canonical_seq or "PLACEHOLDER" in canonical_seq:
                logging.warning(f"Canonical sequence for {kinase} is missing or is a placeholder. Skipping.")
                continue

            for state_name, mutations in states.items():
                try:
                    # Generate the mutated string
                    mutated_seq = self._apply_mutations(kinase, canonical_seq, mutations)

                    # Export to file
                    self._write_fasta(kinase, state_name, mutated_seq)
                except (ValueError, IndexError) as e:
                    logging.error(f"Skipping generation of {kinase}_{state_name} due to preceding errors.")

        logging.info("GENETIC MICROSTATE PIPELINE COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    curator = GeneticMicrostateGenerator()
    curator.generate_microstates()