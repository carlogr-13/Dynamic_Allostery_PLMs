"""
Module 5: Differential Rewiring Analysis
========================================
This module quantifies the allosteric rewiring induced by pathogenic mutations.
It extracts the aligned thermodynamic vectors from the spectral manifold,
computes the absolute differential magnitude (Delta V), and normalizes the
signal using Z-scores to identify statistically significant nodes of
conformational entropy gain or loss.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
from typing import Dict

# Scientific logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DifferentialRewiringAnalyzer:
    """
    Handles the statistical computation of the allosteric delta vector.
    Extracts the pure signal of thermodynamic rewiring from the pre-aligned
    spectral components.
    """

    def __init__(self):
        """
        Initializes the analytical environment and establishes the input/output
        directory topologies.
        """
        # Directory Management
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)
        self.spectral_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "spectral_data")
        self.out_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "rewiring_data")

        os.makedirs(self.out_dir, exist_ok=True)

    def _compute_delta_and_zscore(self, v_wt: np.ndarray, v_mut: np.ndarray) -> pd.DataFrame:
        """
        Performs the core biophysical calculation: extracts the absolute difference
        in variance magnitude and normalizes it statistically.

        Math:
        Delta V = |V_mut| - |V_wt|
        Z = (Delta V - mean(Delta V)) / std(Delta V)

        :param v_wt: np.ndarray, The aligned dynamic eigenvector of the WT state.
        :param v_mut: np.ndarray, The aligned dynamic eigenvector of the mutant state.
        :return: pd.DataFrame, A tabular dataset containing relative indices,
                 raw Delta V, and normalized Z-scores.
        """
        # 1. Absolute Magnitude Subtraction (Thermodynamic differential)
        delta_v = np.abs(v_mut) - np.abs(v_wt)

        # 2. Parametric Standardization (Z-Score)
        mean_delta = np.mean(delta_v)
        std_delta = np.std(delta_v)

        # Epsilon prevents division by zero in mathematically flat systems
        z_scores = (delta_v - mean_delta) / (std_delta + 1e-10)

        # 3. Data Structuring
        # The relative index (0 to N-1) maps directly to the continuous sequence string
        df = pd.DataFrame({
            "Relative_Index": np.arange(len(delta_v)),
            "Delta_V": delta_v,
            "Z_Score": z_scores
        })

        return df

    def execute_differential_analysis(self) -> None:
        """
        Master orchestration routine. Iterates through all compressed spectral
        manifolds, isolates the baseline (WT), and computes the differential
        statistics for every mutant microstate.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING DIFFERENTIAL REWIRING ANALYSIS")
        logging.info("===================================================")

        manifold_files = glob.glob(os.path.join(self.spectral_dir, "Dynamic_Manifold_*.npz"))
        if not manifold_files:
            logging.error(f"No spectral manifolds found in {self.spectral_dir}.")
            return

        for file_path in manifold_files:
            # Extract system name (e.g., 'EGFR' or 'SRC')
            filename = os.path.basename(file_path)
            kinase_system = filename.replace("Dynamic_Manifold_", "").replace(".npz", "")
            logging.info(f"Processing Manifold for System: {kinase_system}")

            # Load the aligned topological space
            try:
                manifold_data = np.load(file_path)
            except Exception as e:
                logging.error(f"   -> Failed to load manifold {filename}: {str(e)}")
                continue

            available_states = manifold_data.files
            wt_key = "WT"

            if wt_key not in available_states:
                logging.error(f"   -> [CRITICAL] Wild-Type (WT) baseline missing in {filename}. Skipping.")
                continue

            v_wt = manifold_data[wt_key]

            # Compute rewiring for each mutant state
            for state_name in available_states:
                if state_name == wt_key:
                    continue

                logging.info(f"   -> Quantifying allosteric rewiring for microstate: {state_name}")
                v_mut = manifold_data[state_name]

                # Execute biophysical statistics
                rewiring_df = self._compute_delta_and_zscore(v_wt, v_mut)

                # Sort by absolute significance to easily identify primary hubs at a glance
                rewiring_df["Abs_Z_Score"] = rewiring_df["Z_Score"].abs()
                rewiring_df = rewiring_df.sort_values(by="Abs_Z_Score", ascending=False).drop(columns=["Abs_Z_Score"])

                # Serialize to CSV
                out_csv_path = os.path.join(self.out_dir, f"Rewiring_{kinase_system}_{state_name}.csv")
                rewiring_df.to_csv(out_csv_path, index=False)
                logging.info(f"      Data serialized to: Rewiring_{kinase_system}_{state_name}.csv")

        logging.info("DIFFERENTIAL REWIRING ANALYSIS COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    analyzer = DifferentialRewiringAnalyzer()
    analyzer.execute_differential_analysis()