import torch
import esm
import logging
import numpy as np
from tqdm import tqdm
import os
import pandas as pd

# Configure strict logging for execution audit
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s', datefmt='%H:%M:%S')

def initialize_plm_environment(model_name: str = "esm2_t33_650M_UR50D"):
    """
    Loads the ESM-2 Transformer architecture, freezes its weights for deterministic
    inference and dynamically allocates it to the optimal hardware device.
    :param model_name: str, ID of pre-trained ESM-2 Transformer architecture.
    :return: Tuple:
            - model: torch.nn.Module (frozen PyTorch model)
            - alphabet: esm.data.Alphabet (vocabulary dictionary)
            - batch_converter: callable (function to tokenize string sequence)
            - device: torch.device (hardware device (CUDA/CPU))
    """

    logging.info(f"Loading ESM-2 Transformer architecture: {model_name}")

    # 1. Load pre-trained weights and vocabulary alphabet from Meta AI
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    except Exception as e:
        logging.error(f"Failed to load ESM-2 Transformer architecture: {model_name}")
        raise

    # 2. Extract the batch converter to tokenize string sequences into integer tensors
    batch_converter = alphabet.get_batch_converter()

    # 3. Freeze network weights to prevent gradient computation (memory optimization)
    model.eval()

    # 4. Dynamic hardware allocation (GPU over CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    logging.info(f"Inference engine successfully configured and transferred to device: {device.type.upper()}")
    return model, alphabet, batch_converter, device

def extract_raw_attention_tensor(model, batch_converter, device, sequence_id: str, sequence: str) -> np.ndarray:
    """
    Tokenizes the input sequence, performs a forward pass through the PLM and
    extracts the raw multi-head self-attention tensor.
    :param model: torch.nn.Module (initialized and frozen ESM-2 model)
    :param batch_converter: callable (tokenizer associated with alphabet)
    :param device: torch.device (hardware device (CUDA/CPU))
    :param sequence_id: str (ID of sequence)
    :param sequence: str (AA sequence)
    :return: np.ndarray (4D NumPy array of shape: Layers, Heads, Seq_Len +2, Seq_Len + 2)
    """
    logging.info(f"Preparing sequence '{sequence_id}' (Length: {len(sequence)} AA) for tokenization.")

    # 1. Formatting data for ESM-2 Batch Converter
    data = [(sequence_id, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # 2. Transfer tokens to the computation device
    batch_tokens = batch_tokens.to(device)
    seq_len_with_tokens = batch_tokens.shape[1]

    logging.info(f"Tokens generated: {seq_len_with_tokens} (with <cls> and <eos>)")

    # 3. Forward pass without gradient tracking
    with torch.no_grad():
        # return_contacts=False limits unnecessary computation overhead
        # need_head_weights=True forces model export covariance matrices
        results = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False, need_head_weights=True)

        # Extract the attention tensor
        # Original shape: [Batch, Layers, Heads, Seq_len, Seq_len] => discard [0] (sequence:time)
        raw_attentions = results["attentions"].squeeze(0).cpu().numpy()

    logging.info("Forward pass finished: Attention tensor extracted")
    return raw_attentions

def compute_monte_carlo_pvalue(attention_matrix: np.ndarray, target_idx: int, permutations: int = 1000) -> float:
    """

    :param attention_matrix:
    :param target_idx:
    :param permutations:
    :return:
    """
    # 1. Extract target signal vector (excluding self-attention)
    target_vector = np.copy(attention_matrix[target_idx])
    target_vector[target_idx] = np.nan
    observed_mean = np.nanmean(target_vector)

    # 2. Population pool
    off_diagonal_mask = ~np.eye(attention_matrix.shape[0], dtype=bool)
    population_pool = attention_matrix[off_diagonal_mask]

    # 3. Vectorized stochastic simulation
    random_means = np.zeros(permutations)
    for i in range(permutations):
        # Sampling without replacement to simulate random vectors of identical length
        randon_sample = np.random.choice(population_pool, size=len(target_vector)-1, replace=False)
        random_means[i] = np.mean(randon_sample)

    # 4. Calculation of the conditional probability
    p_value = np.sum(random_means >= observed_mean) / permutations
    return float(p_value)

def compute_impact_score(attention_matrix: np.ndarray, target_idx: int) -> float:
    """

    :param attention_matrix:
    :param target_idx:
    :return:
    """
    # 1. Extract target signal vector (excluding self-attention)
    target_vector = np.copy(attention_matrix[target_idx, :])
    target_vector[target_idx] = np.nan
    observed_mean = np.nanmean(target_vector)

    # 2. Population pool
    off_diagonal_mask = ~np.eye(attention_matrix.shape[0], dtype=bool)
    population_pool = attention_matrix[off_diagonal_mask]

    # 3. Compute population stadistics
    pop_mean = np.mean(population_pool)
    pop_std = np.std(population_pool)

    # Prevent ZeroDivisionError
    if pop_std == 0:
        return 0.0

    # 4. Calculate Z-score
    impact_score = (observed_mean - pop_mean) / pop_std
    return float(impact_score)

def distill_allosteric_signal(raw_attention: np.ndarray, target_idx: int,
                              p_value_threshold: float = 0.01,
                              impact_score_threshold: float = 1.0) -> list:
    """

    :param raw_attention:
    :param target_idx:
    :param p_value_threshold:
    :param impact_score_threshold:
    :return:
    """
    num_layers, num_heads, _, _ = raw_attention.shape
    validate_heads = []

    # Calculate total iterations
    total_matrices = num_layers * num_heads
    logging.info(f"Filtering over {total_matrices} latent matrices")

    # TQDM progress bar
    with tqdm(total=total_matrices, desc="Distilling latent space", unit="head", dynamic_ncols=True) as pbar:
        for layer in range(num_layers):
            for head in range(num_heads):
                matrix = raw_attention[layer, head, :, :]

                # Simultaneous statistical evaluation
                p_val = compute_monte_carlo_pvalue(matrix, target_idx)
                impact = compute_impact_score(matrix, target_idx)

                # Strict logical sieve
                if p_val < p_value_threshold and impact > impact_score_threshold:
                    validate_heads.append({
                        "Layer": layer,
                        "Head": head,
                        "P-value": p_val,
                        "Impact_score": impact
                    })
                # Update progress bar
                pbar.update(1)
    return validate_heads

def aggregate_and_export_signal (raw_attention: np.ndarray, validated_heads: list,
                                 sequence: str, sequence_id: str, esm_target_idx: int,
                                 output_dir: str = "data/attention_results"):

    os.makedirs(output_dir, exist_ok=True)

    if not validated_heads:
        logging.warning(f"No validated heads to aggregate for {sequence_id}")
        return

    seq_len_with_tokens = raw_attentions.shape[2]
    accumulated_matrix = np.zeros((seq_len_with_tokens, seq_len_with_tokens))

    # 1. Tensor aggregation: statistically significant heads
    for head_data in validated_heads:
        layer = head_data["Layer"]
        head = head_data["Head"]
        accumulated_matrix += raw_attention[layer, head, :, :]
    # Average the accumulated signal
    accumulated_matrix /= len(validated_heads)

    # 2. Vector extraction: isolate interaction array from the mutation coordinate [1:-1]
    attention_vector = accumulated_matrix[esm_target_idx, 1:-1]

    # 3. Export
    df_heads = pd.DataFrame(accumulated_matrix)
    heads_path = os.path.join(output_dir, f"{sequence_id}_head_statistics.csv")
    df_heads.to_csv(heads_path, index=False)

    # Export the 1D thermodynamic signal
    df_signal = pd.DataFrame({
        "Residue_index": np.arange(len(sequence)),
        "AA": list(sequence),
        "Filtered_attention_score": attention_vector
    })
    signal_path = os.path.join(output_dir, f"{sequence_id}_distilled_signal.csv")
    df_signal.to_csv(signal_path, index=False)

    logging.info(f"Data exportes to: {output_dir}")


if __name__ == "__main__":
    try:
        logging.info(f"--- ABL1 WT Signal Extraction ---")

        model, alphabet, batch_converter, device = initialize_plm_environment()

        # ABL1 Kinase Domain
        abl1_id = "ABL1_WT"
        abl1_seq = "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTVYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRT"

        mutation_coordinate = 314
        esm_target_idx = mutation_coordinate + 1

        # 1. Forward pass
        raw_tensor = extract_raw_attention_tensor(model, batch_converter, device, abl1_id, abl1_seq)

        # 2. Monte Carlo
        surviving_heads = distill_allosteric_signal(raw_tensor, esm_target_idx, p_value_threshold=0.01, impact_score_threshold=1)

        survival_rate = (len(surviving_heads) / 660) * 100
        logging.info(f"Validated heads: {len(surviving_heads)}/660 ({survival_rate:.2f}%)")

        # 3. Aggregation and export
        aggregate_and_export_signal(raw_tensor, surviving_heads, abl1_seq, abl1_id, esm_target_idx)

    except Exception as e:
        import traceback
        logging.error(f"Execution failed. Technical traceback:\n{traceback.format_exc()}")