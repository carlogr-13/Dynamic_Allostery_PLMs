import os
import torch
import numpy as np
import logging
import esm
from tqdm import tqdm

# Academic traceability configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class AttentionTensorExtractor:
    def __init__(self):
        """
        Initializes the tensor inference environment. Defines the ESM-2 architecture
        parameters and establishes the directory topology for data ingestion
        and tensor serialization.
        """
        # Explicit string casting to resolve strict IDE type-hinting warnings
        script_dir = str(os.path.dirname(os.path.abspath(str(__file__))))
        project_root = str(os.path.dirname(script_dir))
        self.data_dir = str(os.path.join(project_root, "data"))

        self.dirs = {
            "fasta": str(os.path.join(self.data_dir, "fasta_sequences")),
            "tensors": str(os.path.join(self.data_dir, "attention_tensors"))
        }
        os.makedirs(self.dirs["tensors"], exist_ok=True)

        # Experimental network parameters
        self.model_name = "esm2_t33_650M_UR50D"
        self.start_layer = 16  # Lower bound to isolate global tertiary contacts

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Hardware acceleration detected: {self.device.type.upper()}")

        self.model, self.alphabet, self.batch_converter = self._load_model()

    def _load_model(self):
        """
        Loads the pre-trained weights of the Meta FAIR protein language model into memory.

        :return: tuple (torch.nn.Module, esm.data.Alphabet, esm.data.BatchConverter)
        """
        logging.info(f"Loading {self.model_name} architecture on {self.device}...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        model = model.eval().to(self.device)
        batch_converter = alphabet.get_batch_converter()
        logging.info("Model instantiated and locked in evaluation mode.")
        return model, alphabet, batch_converter

    @staticmethod
    def _read_fasta(filepath: str) -> str:
        """
        Extracts the pure linear amino acid sequence from a FASTA file.

        :param filepath: str, Absolute path to the target FASTA file
        :return: str, Continuous amino acid sequence
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])
        return sequence

    def extract_global_attention(self, sequence: str, seq_name: str) -> np.ndarray:
        """
        Executes the forward pass through the Transformer architecture, extracts the
        multidimensional attention tensor, spatially crops the artificial tokens,
        and computes the statistical mean across deep layers.

        :param sequence: str, Linear amino acid sequence
        :param seq_name: str, Identifier for the biological sequence
        :return: numpy.ndarray, N x N asymmetric matrix representing global covariance
        """
        data = [(seq_name, sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[], need_head_weights=True)

        # Original tensor dimensions: (Batch, Layers, Heads, SeqLen+2, SeqLen+2)
        attentions = results["attentions"].squeeze(0)

        # 1. Spatial cropping: Remove <cls> (index 0) and <eos> (index -1) tokens
        attentions_cropped = attentions[:, :, 1:-1, 1:-1]

        # 2. Deep layer isolation: Extract layers responsible for tertiary contacts
        deep_attentions = attentions_cropped[self.start_layer:, :, :, :]

        # 3. Statistical collapse: Average across selected layers and all heads
        global_attention_matrix = torch.mean(deep_attentions, dim=(0, 1)).cpu().numpy()

        return global_attention_matrix

    def execute_differential_pipeline(self, target_systems: dict):
        """
        Iterates over the biological systems, computes the basal (WT) tensor,
        contrasts it with the mutated tensors, and serializes the differential matrices.

        :param target_systems: dict, Mapping of systems to their corresponding variants
        :return: None
        """
        logging.info("\n--- INITIATING PHASE 2: TENSOR INFERENCE AND DIFFERENTIAL CALCULUS ---")

        for system, variants in target_systems.items():
            logging.info(f"\nProcessing thermodynamic topology for: {system}")

            # 1. Obtain the basal matrix (Thermodynamic Reference State)
            wt_name = f"{system}_WT"
            wt_fasta = os.path.join(self.dirs["fasta"], f"{wt_name}.fasta")

            if not os.path.exists(wt_fasta):
                logging.error(f"Basal file missing: {wt_fasta}. Skipping system.")
                continue

            wt_seq = self._read_fasta(wt_fasta)
            logging.info(f"Extracting basal tensor ({wt_name})...")
            wt_tensor = self.extract_global_attention(wt_seq, wt_name)

            # Serialize the WT tensor for experimental rigor
            np.save(os.path.join(self.dirs["tensors"], f"tensor_{wt_name}.npy"), wt_tensor)

            # 2. Iterate over mutational variants using tqdm
            mutant_variants = [v for v in variants if v != "WT"]

            for mut_suffix in tqdm(mutant_variants, desc=f"Computing Delta Tensors for {system}", unit="mutant"):
                mut_name = f"{system}_{mut_suffix}"
                mut_fasta = os.path.join(self.dirs["fasta"], f"{mut_name}.fasta")

                if not os.path.exists(mut_fasta):
                    logging.warning(f"File not found: {mut_fasta}. Skipping.")
                    continue

                mut_seq = self._read_fasta(mut_fasta)
                mut_tensor = self.extract_global_attention(mut_seq, mut_name)

                # 3. DIFFERENTIAL CALCULUS: Isolate the signal redistribution
                delta_tensor = mut_tensor - wt_tensor

                # 4. Serialize data to disk
                out_path = os.path.join(self.dirs["tensors"], f"delta_tensor_{mut_name}.npy")
                np.save(out_path, delta_tensor)

        logging.info("\n--- PHASE 2 SUCCESSFULLY COMPLETED ---")


if __name__ == "__main__":
    # Experimental target configuration
    experimental_targets = {
        "SRC": ["WT", "E310A_Active", "T338G_Inhibitory"],
        "EGFR": ["WT", "L858R", "L858R_T790M_Epistatic"]
    }

    extractor = AttentionTensorExtractor()
    extractor.execute_differential_pipeline(experimental_targets)