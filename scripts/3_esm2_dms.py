"""
Module 3: Epistatic Inference Engine (ESM-2)
============================================
This module performs in silico Deep Mutational Scanning (DMS) using the ESM-2
Protein Language Model. It iterates through curated genetic microstates (FASTA),
systematically applies Alanine substitutions, and evaluates the global
thermodynamic perturbation using the Jensen-Shannon Divergence (JSD).

The output is an NxN asymmetrical Jacobian tensor representing the epistatic
covariance of the kinase domain, which forms the mathematical basis for the
downstream orthogonal spectral decomposition.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging
import torch
import numpy as np
from tqdm import tqdm
from typing import Tuple

# Attempt to import ESM; will fail gracefully if not installed
try:
    import esm
except ImportError:
    raise ImportError("The 'esm' library is required. Install via: pip install fair-esm")

# Configure rigorous scientific logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class EpistaticInferenceEngine:
    """
    Orchestrates the loading of the ESM-2 transformer model, execution of the
    forward passes, and the algebraic computation of the Jacobian tensors.
    """

    def __init__(self, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Initializes the neural network engine, configures hardware acceleration,
        and resolves the absolute paths for I/O operations.

        :param model_name: str, The specific ESM-2 architecture to load.
                           Default is the 650M parameter model for optimal resolution.
        """
        # Directory Management
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)
        self.fasta_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "fasta_sequences")
        self.results_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "results")

        os.makedirs(self.results_dir, exist_ok=True)

        # Hardware Acceleration Topology
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Hardware acceleration allocated to: {self.device.type.upper()}")

        # Neural Network Instantiation
        logging.info(f"Instantiating Protein Language Model: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)

        # Lock model parameters (Inference mode) to prevent memory leaks
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        logging.info("Model weights loaded and frozen for inference.")

    def _read_fasta(self, file_path: str) -> Tuple[str, str]:
        """
        Parses a strict FASTA file to extract the identifier and the continuous sequence.

        :param file_path: str, Absolute path to the FASTA file.
        :return: Tuple[str, str], The sequence identifier and the amino acid string.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()

        seq_id = lines[0].strip().replace(">", "")
        sequence = "".join([line.strip() for line in lines[1:]])
        return seq_id, sequence

    def _get_probability_distribution(self, sequence: str) -> torch.Tensor:
        """
        Executes a forward pass through the ESM-2 network and extracts the
        Softmax probability distribution from the terminal attention layer.

        :param sequence: str, The amino acid string to evaluate.
        :return: torch.Tensor, An [N x Vocab] probability matrix mapped to the sequence length.
        """
        # Data preparation mapping
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        # Gradient-free execution context
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.model.num_layers])

        # Logits extraction: batch_size=1. Slice [1:-1] drops <cls> and <eos> structural tokens.
        logits = results["logits"].squeeze(0)[1:-1, :]

        # Thermodynamically equivalent normalization
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return probabilities

    def _jensen_shannon_divergence(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pointwise Jensen-Shannon Divergence between two probability distributions.

        Math: D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M) where M = 0.5 * (P + Q)

        :param p: torch.Tensor, Basal probability distribution.
        :param q: torch.Tensor, Perturbed probability distribution.
        :return: torch.Tensor, 1D array of divergence scalar values of length N.
        """
        m = 0.5 * (p + q)

        # Epsilon scalar to prevent log(0) computational singularities
        epsilon = 1e-10

        log_p = torch.log(p + epsilon)
        log_q = torch.log(q + epsilon)
        log_m = torch.log(m + epsilon)

        # Kullback-Leibler algebraic formulations
        kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
        kl_qm = torch.sum(q * (log_q - log_m), dim=-1)

        jsd = 0.5 * kl_pm + 0.5 * kl_qm
        return jsd

    def compute_jacobian(self, seq_id: str, sequence: str) -> np.ndarray:
        """
        Executes the in silico Deep Mutational Scanning algorithm. Generates the
        N x N epistatic covariance matrix by perturbing each residue sequentially.

        :param seq_id: str, The identifier of the current microstate.
        :param sequence: str, The amino acid sequence to perturb.
        :return: np.ndarray, The dense Jacobian tensor in float32 format.
        """
        n_res = len(sequence)
        jacobian = np.zeros((n_res, n_res), dtype=np.float32)

        logging.info(f"Computing Basal State Vector for {seq_id} (N={n_res})")
        p_wt = self._get_probability_distribution(sequence)

        # Iterative mutational scanning with Progress Bar
        for j in tqdm(range(n_res), desc=f"Scanning {seq_id}", unit="residue"):
            seq_list = list(sequence)
            native_aa = seq_list[j]

            # Sub-routine: Force side-chain perturbation
            mut_aa = 'A' if native_aa != 'A' else 'G'
            seq_list[j] = mut_aa
            mut_sequence = "".join(seq_list)

            # Retrieve perturbed thermodynamic state
            p_mut = self._get_probability_distribution(mut_sequence)

            # Calculate informational variance
            jsd_vector = self._jensen_shannon_divergence(p_wt, p_mut)

            # Populate the Jacobian column vector (effect of mutating j on all i)
            jacobian[:, j] = jsd_vector.cpu().numpy()

        return jacobian

    def execute_inference(self) -> None:
        """
        Master orchestration routine. Reads all FASTA files, computes their
        respective Jacobian tensors, and serializes the data.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING EPISTATIC INFERENCE ENGINE")
        logging.info("===================================================")

        fasta_files = glob.glob(os.path.join(self.fasta_dir, "*.fasta"))

        if not fasta_files:
            logging.error(f"No FASTA microstates found in {self.fasta_dir}.")
            return

        for file_path in fasta_files:
            seq_id, sequence = self._read_fasta(file_path)
            out_tensor_path = os.path.join(self.results_dir, f"Jacobian_{seq_id}.npy")

            if os.path.exists(out_tensor_path):
                logging.info(f"Tensor for {seq_id} already exists. Skipping inference.")
                continue

            # Compute tensor
            jacobian_tensor = self.compute_jacobian(seq_id, sequence)

            # Serialize mathematically dense output
            np.save(out_tensor_path, jacobian_tensor)
            logging.info(f"Successfully serialized tensor: Jacobian_{seq_id}.npy")

        logging.info("EPISTATIC INFERENCE COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    engine = EpistaticInferenceEngine()
    engine.execute_inference()