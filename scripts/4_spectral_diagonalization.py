"""
Module 4: Spectral Diagonalization and Orthogonal Alignment
===========================================================
This module isolates the thermodynamic sectors of the kinase domain by performing
an eigendecomposition of the asymmetrical Jacobian tensors generated in Module 3.

To resolve the bio-mathematical problem of Eigenvalue Crossing during mutagenesis,
it implements a Dot-Product Alignment heuristic. This ensures that the dynamic
conformational entropy vector extracted from the mutant states is strictly
homologous to the reference dynamic vector of the Wild-Type state.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging
import numpy as np
from typing import Dict, Tuple

# Scientific logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class SpectralDiagonalizationEngine:
    """
    Handles the algebraic decomposition of epistatic covariance matrices,
    isolates the orthogonal dynamic eigenvectors, and aligns them across
    different thermodynamic microstates to enable differential analysis.
    """

    def __init__(self, target_eigenvector_index: int = 1):
        """
        Initializes the spectral engine and resolves I/O topological paths.

        :param target_eigenvector_index: int, The 0-based index of the eigenvector
                                         to isolate. Default is 1 (EV2), assuming EV1
                                         (index 0) absorbs the rigid structural core.
        """
        # Directory Management
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)
        self.tensor_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "results")
        self.spectral_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "spectral_data")

        os.makedirs(self.spectral_dir, exist_ok=True)

        # Target dimension for dynamic allostery (Orthogonal to the primary structural vector)
        self.target_ev_idx: int = target_eigenvector_index

    def _symmetrize_and_decompose(self, jacobian: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Symmetrizes the raw Jacobian matrix, neutralizes the auto-covariance diagonal,
        and performs a rigorous orthogonal eigendecomposition.

        :param jacobian: np.ndarray, The raw NxN asymmetric epistatic tensor.
        :return: Tuple[np.ndarray, np.ndarray], A tuple containing the sorted eigenvalues
                 (1D array) and the corresponding sorted eigenvectors (NxN matrix).
        """
        # 1. Algebraic Symmetrization: J_sym = (J + J^T) / 2
        sym_tensor = (jacobian + jacobian.T) / 2.0

        # 2. Neutralize self-dependence (diagonal = 0)
        np.fill_diagonal(sym_tensor, 0.0)

        # 3. Orthogonal Eigendecomposition (valid for real symmetric matrices)
        eigenvalues, eigenvectors = np.linalg.eigh(sym_tensor)

        # 4. Sort eigen-pairs by absolute magnitude of variance (descending)
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        return eigenvalues, eigenvectors

    def _align_eigenvector(self, ref_vector: np.ndarray, mut_eigenvectors: np.ndarray) -> np.ndarray:
        """
        Solves the Eigenvalue Crossing problem. Computes the dot product (cosine overlap)
        between the Reference WT vector and all eigenvectors of the mutant state to
        identify and extract the strictly homologous functional sector.

        :param ref_vector: np.ndarray, The isolated dynamic eigenvector of the WT state (1D array).
        :param mut_eigenvectors: np.ndarray, The full matrix of eigenvectors from the mutant state (NxN).
        :return: np.ndarray, The single mutant eigenvector homologous to the reference.
        """
        # Compute absolute dot product overlap across all dimensions
        overlaps = np.abs(np.dot(ref_vector, mut_eigenvectors))

        # Identify the dimension with maximal homology
        homologous_idx = np.argmax(overlaps)
        aligned_vector = mut_eigenvectors[:, homologous_idx]

        # Sign-invariance resolution: Ensure parallel vector orientation for downstream subtraction
        if np.dot(ref_vector, aligned_vector) < 0:
            aligned_vector = -aligned_vector

        logging.info(
            f"   -> Homology resolved: Mutant EV{homologous_idx + 1} maps to Reference EV{self.target_ev_idx + 1} (Overlap: {overlaps[homologous_idx]:.4f})")

        return aligned_vector

    def execute_diagonalization(self) -> None:
        """
        Master orchestration routine. Groups tensors by kinase system, processes the
        Wild-Type state to establish the reference coordinate system, and subsequently
        aligns and extracts the dynamic vectors for all associated mutant states.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING SPECTRAL DIAGONALIZATION & ALIGNMENT")
        logging.info("===================================================")

        tensor_files = glob.glob(os.path.join(self.tensor_dir, "Jacobian_*.npy"))
        if not tensor_files:
            logging.error(f"No Jacobian tensors found in {self.tensor_dir}.")
            return

        # Group microstates by Target Kinase (e.g., 'EGFR', 'SRC')
        kinase_systems: Dict[str, Dict[str, str]] = {}
        for file_path in tensor_files:
            filename = os.path.basename(file_path).replace(".npy", "")
            # Expected format: Jacobian_KINASE_STATE
            parts = filename.split("_", 2)
            if len(parts) >= 3:
                kinase = parts[1]
                state = parts[2]
            else:
                continue

            if kinase not in kinase_systems:
                kinase_systems[kinase] = {}
            kinase_systems[kinase][state] = file_path

        # Process each Kinase System independently
        for kinase, states in kinase_systems.items():
            logging.info(f"Processing Spectral Topology for System: {kinase}")

            # Identify WT baseline
            wt_state_key = "WT"
            if wt_state_key not in states:
                logging.error(f"   -> [CRITICAL] Wild-Type (WT) baseline missing for {kinase}. Skipping system.")
                continue

            # Load and decompose WT Tensor
            wt_tensor = np.load(states[wt_state_key])
            _, wt_eigenvectors = self._symmetrize_and_decompose(wt_tensor)

            # Isolate the Reference Dynamic Eigenvector
            ref_dynamic_vector = wt_eigenvectors[:, self.target_ev_idx]
            logging.info(f"   -> Reference Dynamic Vector established from {kinase}_WT (EV{self.target_ev_idx + 1})")

            # Dictionary to store aligned dynamic vectors for all states of this kinase
            aligned_vectors: Dict[str, np.ndarray] = {
                wt_state_key: ref_dynamic_vector
            }

            # Process all Mutant microstates
            for state_name, file_path in states.items():
                if state_name == wt_state_key:
                    continue

                logging.info(f"   -> Aligning microstate: {state_name}")
                mut_tensor = np.load(file_path)
                _, mut_eigenvectors = self._symmetrize_and_decompose(mut_tensor)

                # Execute Orthogonal Alignment
                homologous_vector = self._align_eigenvector(ref_dynamic_vector, mut_eigenvectors)
                aligned_vectors[state_name] = homologous_vector

            # Serialize the aligned dynamic manifold into a compressed dictionary
            out_file = os.path.join(self.spectral_dir, f"Dynamic_Manifold_{kinase}.npz")
            np.savez_compressed(out_file, **aligned_vectors)
            logging.info(f"Successfully serialized orthogonal manifold to: Dynamic_Manifold_{kinase}.npz")

        logging.info("SPECTRAL DIAGONALIZATION COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    engine = SpectralDiagonalizationEngine(target_eigenvector_index=1)
    engine.execute_diagonalization()