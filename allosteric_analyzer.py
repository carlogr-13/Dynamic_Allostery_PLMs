"""
Allosteric Sensitivity Analysis in ESM2 Attention Heads
=====================================================

This module implements the methodology from Dong et al. (2024) for analyzing
allosteric sensitivity in protein language model attention heads, using ESM2.

Background
----------
Attention heads in Protein Language Models (PLMs) are specialized components that learn to focus on different aspects of relationships between amino acids in a protein sequence. Each attention head acts like a spotlight that can highlight specific connections or patterns between different positions in the sequence. For example, one head might focus on nearby amino acids that form local structural elements, while another might detect long-range interactions between distant parts of the protein. In technical terms, each attention head computes a weighted sum of all positions in the sequence for each position, where the weights (attention scores) indicate how much each position should influence the current position. These attention heads are organized in layers, with each layer containing multiple heads that work in parallel to capture different types of relationships. The combined output of these attention heads helps the model understand both local and global patterns in protein sequences, which is crucial for tasks like predicting protein structure, function, or in this case, allosteric sites.

Allostery is a mechanism where binding at one site affects protein function at
another site. Dong et al. 2024 shows that protein language models capture
allosteric relationships in their attention heads. This code identifies which
attention heads are most sensitive to allosteric sites.

Classes
-------
AllosticHeadAnalyzer:
    Main class for analyzing allosteric sensitivity in ESM2 attention heads.

Example
-------
>>> from allosteric_analyzer import AllosticHeadAnalyzer
>>>
>>> # Initialize analyzer
>>> analyzer = AllosticHeadAnalyzer(threshold=0.3)
>>>
>>> # Example protein sequence (Adenosine A2A receptor)
>>> sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNSNLQNVTNYFVVSLAAADIAVGVLAIPFAITISTGFCAACHGCLFIACFVLVLTQSSIFSLLAIAIDRYIAIRIPLRYNGLVTGTRAKGIIAICWVLSFAIGLTPMLGWNNCGQPKEGKQHSQGCGEGQVACLFEDVVPMNYMVYFNFFACVLVPLLLMLGVYLRIFLAARRQLKQMESQPLPGERARSTLQKEVHAAKSLAIIVGLFALCWLPLHIINCFTFFCPDCSHAPLWLMYLAIVLSHTNSVVNPFIYAYRIREFRQTFRKIIRSHVLRQQEPFKA"
>>> allosteric_sites = [85, 89, 246, 253]
>>>
>>> # Analyze protein
>>> results = analyzer.analyze_protein(sequence, allosteric_sites)
>>>
>>> # Get scores for each head
>>> impact_scores = results["impacts"].squeeze().tolist()
>>>
>>> # Print results
>>> for head_idx, score in enumerate(impact_scores):
...     print(f"Head {head_idx}: {score:.3f}")
>>>
>>> # Identify most sensitive heads
>>> mean_score = sum(impact_scores) / len(impact_scores)
>>> sensitive_heads = [i for i, score in enumerate(impact_scores)
...                   if score > mean_score]
>>> print(f"Most sensitive heads: {sensitive_heads}")



Dependencies
-----------
- torch
- fair-esm
- numpy
- pandas
- tqdm

Authors:
--------
Aurelio A. Moya-García
Date: February 10, 2025

References
----------
Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites.
bioRxiv. DOI: pending
"""

import torch
from esm import pretrained
import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List
from scipy.stats import ttest_1samp


class AllosticHeadAnalyzer:
    def __init__(self, threshold: float = 0.3, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Initialize the analyzer with ESM2 model

        Args:
            threshold: Minimum score for attention values to be considered significant
            model_name: Name of the ESM2 model to use
        """
        # Set up device for M1 Mac (MPS) or fallback to CPU
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        # print(f"Using device: {self.device}")

        # Initialize ESM2 model
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.threshold = threshold

        # Get model dimensions - these will be initialized after first run
        self.num_layers = self.model.num_layers
        self.num_heads = self.model.attention_heads

    def get_attention_maps(self, sequence: str) -> torch.Tensor:
        """
        Get attention maps for a protein sequence

        Args:
            sequence: Amino acid sequence

        Returns:
            Tensor of shape [num_layers, num_heads, seq_len, seq_len]
        """
        # Tokenize the sequence
        batch_converter = self.alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter([("protein", sequence)])
        batch_tokens = batch_tokens.to(self.device)

        # Get logits output with attention maps
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.num_layers], return_contacts=True)

        attentions = results["attentions"][..., 1:-1, 1:-1] # En el script antiguo no se eliminaban los tokens <cls> y <eos>
        return attentions

    # This function is replaced with the new version below
    def old_compute_allosteric_impact(
            self,
            attention_maps: List[torch.Tensor],
            allosteric_sites: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores for each attention head

        Args:
            attention_maps: List of attention tensors from each layer
            allosteric_sites: List of allosteric site indices (1-based indexing)

        Returns:
            Tuple of (allosteric_impacts, snr_values)
        """
        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]

        impacts = []
        snrs = []

        # Compute impact for each layer and head
        for layer_attention in attention_maps:
            layer_impacts = []
            layer_snrs = []

            # Squeeze out batch dimension if present
            if layer_attention.dim() == 4:
                layer_attention = layer_attention.squeeze(0)

            for head in range(layer_attention.size(1)):
                attention = layer_attention[:, head]
                # print(f"Attention shape: {attention.shape}")  # Debug print

                # Calculate w_allo (eq. 1 from paper)
                w_allo = 0
                mask = attention > self.threshold
                for site in allo_sites:
                    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()

                    # Calculate total activity w (eq. 2 from paper)
                w_total = torch.sum(attention > self.threshold).item()

                # Calculate impact score p (eq. after eq. 2 in paper)
                # Normalize impact by sequence length factor
                seq_len = attention_maps[0].size(-1)
                base_factor = seq_len / len(allosteric_sites)  # Ratio of sequence length to number of sites
                length_factor = np.log2(base_factor)  # Logarithmic scaling

                impact = (w_allo / w_total) * length_factor if w_total > 0 else 0
                layer_impacts.append(impact)
                layer_snrs.append(impact)

            impacts.append(layer_impacts)
            snrs.append(layer_snrs)

        impacts = torch.tensor(impacts)
        snrs = torch.tensor(snrs)

        return impacts, snrs

    def seq_normal_compute_allosteric_impact(
            self,
            attention_maps: List[torch.Tensor],
            allosteric_sites: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores for each attention head. Sequence length normalization.

        Args:
            attention_maps: List of attention tensors from each layer
            allosteric_sites: List of allosteric site indices (1-based indexing)

        Returns:
            Tuple of (allosteric_impacts, snr_values)
        """
        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]

        impacts = []
        snrs = []

        # Compute impact for each layer and head
        for layer_attention in attention_maps:
            layer_impacts = []
            layer_snrs = []

            # Squeeze out batch dimension if present
            if layer_attention.dim() == 4:
                layer_attention = layer_attention.squeeze(0)

            for head in range(layer_attention.size(1)):
                attention = layer_attention[:, head]

                # Calculate w_allo (eq. 1 from paper)
                w_allo = 0
                mask = attention > self.threshold
                for site in allo_sites:
                    w_allo += torch.sum(attention[:, site][mask[:, site]]).item()

                    # Calculate total activity w (eq. 2 from paper)
                w_total = torch.sum(attention > self.threshold).item()

                # Calculate impact score p (eq. after eq. 2 in paper)
                # Normalize impact by sequence length factor
                seq_len = attention_maps[0].size(-1)
                base_factor = seq_len / len(allosteric_sites)
                length_factor = np.log2(base_factor)

                impact = (w_allo / w_total) * length_factor if w_total > 0 else 0
                layer_impacts.append(impact)
                layer_snrs.append(impact)

            impacts.append(layer_impacts)
            snrs.append(layer_snrs)

        return torch.tensor(impacts), torch.tensor(snrs)

    def compute_allosteric_impact(self, attention_maps: torch.Tensor, allosteric_sites: List[int],
                                  n_random_trials: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores with proper random baseline,
        excluding allosteric sites from random selection.

        Args:
            attention_maps: Tensor of shape [1, num_layers, num_heads, seq_len, seq_len]
            allosteric_sites: List of 1-based indices
            n_random_trials: Number of random samples for baseline

        Returns:
            Tuple of (impact scores, snr values), both of shape [num_layers, num_heads]
        """
        # Remove batch dimension
        attention_maps = attention_maps[0]  # shape: [num_layers, num_heads, seq_len, seq_len]

        allo_sites = [site - 1 for site in allosteric_sites]  # Convert to 0-based indexing
        n_allo_sites = len(allo_sites)
        num_layers, num_heads, seq_len, _ = attention_maps.shape

        impacts = []
        snrs = []

        for layer_idx in range(num_layers):
            layer_attention = attention_maps[layer_idx]  # [num_heads, seq_len, seq_len]
            layer_impacts = []
            layer_snrs = []

            for head_idx in range(num_heads):
                attention = layer_attention[head_idx]  # [seq_len, seq_len]
                mask = attention > self.threshold

                # Compute attention to allosteric sites
                w_allo = sum(
                    torch.sum(attention[:, site][mask[:, site]]).item()
                    for site in allo_sites
                )

                # Create array of non-allosteric positions
                non_allo_positions = np.array([i for i in range(seq_len) if i not in allo_sites])

                # Random baseline
                random_w_values = []
                for _ in range(n_random_trials):
                    random_sites = np.random.choice(non_allo_positions, size=n_allo_sites, replace=False)
                    random_w = sum(
                        torch.sum(attention[:, site][mask[:, site]]).item()
                        for site in random_sites
                    )
                    random_w_values.append(random_w)

                expected_random = np.mean(random_w_values)
                random_std = np.std(random_w_values)

                impact = w_allo / expected_random if expected_random > 0 else 0
                snr = (w_allo - expected_random) / (random_std + 1e-10)

                layer_impacts.append(impact)
                layer_snrs.append(snr)

            impacts.append(layer_impacts)
            snrs.append(layer_snrs)

        return torch.tensor(impacts), torch.tensor(snrs)

    def compute_allosteric_impact_with_significance(
            self,
            attention_maps: torch.Tensor,
            allosteric_sites: List[int],
            n_random_trials: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute allosteric impact scores and perform statistical significance testing
        using a one-sample t-test.

        Returns:
            impacts: [num_layers, num_heads]
            snrs: [num_layers, num_heads]
            p_values: [num_layers, num_heads] from t-test
        """
        attention_maps = attention_maps[0]  # Remove batch
        allo_sites = [site - 1 for site in allosteric_sites]
        n_allo_sites = len(allo_sites)
        num_layers, num_heads, seq_len, _ = attention_maps.shape

        impacts, snrs, pvals = [], [], []

        for layer_idx in range(num_layers):
            layer_impacts, layer_snrs, layer_pvals = [], [], []
            for head_idx in range(num_heads):
                attention = attention_maps[layer_idx, head_idx]  # [seq_len, seq_len]
                mask = attention > self.threshold

                # Atención a sitios alostéricos
                w_allo = sum(
                    torch.sum(attention[:, site][mask[:, site]]).item()
                    for site in allo_sites
                )

                # Baseline aleatorio
                non_allo_positions = np.array([i for i in range(seq_len) if i not in allo_sites])
                random_w_values = []
                for _ in range(n_random_trials):
                    random_sites = np.random.choice(non_allo_positions, size=n_allo_sites, replace=False)
                    w_random = sum(
                        torch.sum(attention[:, site][mask[:, site]]).item()
                        for site in random_sites
                    )
                    random_w_values.append(w_random)

                expected_random = np.mean(random_w_values)
                std_random = np.std(random_w_values)

                impact = w_allo / expected_random if expected_random > 0 else 0
                snr = (w_allo - expected_random) / (std_random + 1e-10)

                # t-test: Is w_allo significantly > mean(random)?
                t_stat, p_value = ttest_1samp(random_w_values, w_allo, alternative='less')

                layer_impacts.append(impact)
                layer_snrs.append(snr)
                layer_pvals.append(p_value)

            impacts.append(layer_impacts)
            snrs.append(layer_snrs)
            pvals.append(layer_pvals)

        return (
            torch.tensor(impacts),  # shape: [num_layers, num_heads]
            torch.tensor(snrs),
            torch.tensor(pvals)
        )

    def analyze_protein(
            self,
            sequence: str,
            allosteric_sites: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze a single protein sequence

        Args:
            sequence: Amino acid sequence
            allosteric_sites: List of allosteric site indices (1-based)

        Returns:
            Dictionary with impact scores and SNR values
        """
        attention_maps = self.get_attention_maps(sequence)
        impacts, snrs, p_values = self.compute_allosteric_impact_with_significance(attention_maps, allosteric_sites)

        return {
            "impacts": impacts,
            "snrs": snrs,
            "p_values": p_values
        }

    def analyze_dataset(
            self,
            sequences: List[str],
            allosteric_sites: List[List[int]]
    ) -> pd.DataFrame:
        """
        Analyze a dataset of proteins

        Args:
            sequences: List of protein sequences
            allosteric_sites: List of lists containing allosteric sites for each sequence

        Returns:
            DataFrame with impact scores for each layer-head combination
        """
        all_impacts = []

        for seq, sites in tqdm(zip(sequences, allosteric_sites), total=len(sequences)):
            results = self.analyze_protein(seq, sites)
            all_impacts.append(results["impacts"])

        # Stack impacts from all proteins
        impacts_tensor = torch.stack(all_impacts)

        # Calculate mean and std across proteins
        mean_impacts = torch.mean(impacts_tensor, dim=0)
        std_impacts = torch.std(impacts_tensor, dim=0)
        snr = mean_impacts / (std_impacts + 1e-10)

        # Create DataFrame
        rows = []
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                rows.append({
                    "layer": layer,
                    "head": head,
                    "mean_impact": mean_impacts[layer, head].item(),
                    "std_impact": std_impacts[layer, head].item(),
                    "snr": snr[layer, head].item()
                })

        return pd.DataFrame(rows)

    def compute_accumulated_average_attention(self, attention_maps: torch.Tensor, sequence: str,
                                              allosteric_sites: list, sensitive_heads: list) -> dict:
        """
        Calcula la atención promedio acumulada hacia los residuos del sitio alostérico.

        Parámetros:
        - attention_maps: Tensor [1, layers, heads, seq_len, seq_len]
        - sequence: cadena de aminoácidos
        - allosteric_sites: lista de posiciones (1-based)
        - sensitive_heads: lista de tuplas (layer, head)

        Devuelve:
        - Diccionario {i: Ā(i)} con las posiciones i (1-based) y su atención promedio acumulada.
        """
        assert attention_maps.dim() == 5, "El tensor de atención debe tener 5 dimensiones [1, layers, heads, seq_len, seq_len]"

        seq_len = len(sequence)
        accumulated_attention = {i + 1: [] for i in range(seq_len)}  # 1-based index

        for i in range(seq_len):  # i es el residuo que presta atención
            for j in allosteric_sites:  # j es el residuo alostérico que recibe atención
                values = []
                for (layer, head) in sensitive_heads:
                    if j - 1 < seq_len:  # asegurarse que j está en rango
                        value = attention_maps[0, layer, head, i, j - 1].item()
                        values.append(value)
                if values:
                    avg_ij = sum(values) / len(values)
                    accumulated_attention[i + 1].append(avg_ij)

        # Sumar atención acumulada por residuo
        accumulated_avg_attention = {}
        for i, values in accumulated_attention.items():
            accumulated_avg_attention[i] = sum(values) if values else 0.0

        return accumulated_avg_attention

    def analyze_sequence_length_effect(self, sequence: str, allosteric_sites: List[int]) -> Dict:
        """Analyze how sequence length affects attention scores"""
        seq_len = len(sequence)
        num_sites = len(allosteric_sites)

        # Basic statistics
        stats = {
            "sequence_length": seq_len,
            "num_allosteric_sites": num_sites,
            "sites_ratio": num_sites / seq_len,
            "expected_random_attention": 1.0 / seq_len
        }

        # Get attention maps
        attention_maps = self.get_attention_maps(sequence)

        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]

        # Take first layer and squeeze batch dimension if present
        layer_attention = attention_maps[0]  # Shape: [num_heads, seq_len, seq_len]
        if layer_attention.dim() == 4:
            layer_attention = layer_attention.squeeze(0)

        # Create masks for allosteric sites
        allo_mask = torch.zeros(layer_attention.size(-1), dtype=torch.bool, device=layer_attention.device)
        allo_mask[allo_sites] = True
        non_allo_mask = ~allo_mask

        # Calculate attention statistics
        # 1. First get attention scores above threshold
        threshold_mask = layer_attention > self.threshold

        # 2. Calculate mean attention for allosteric sites (averaging across heads)
        allo_attention = 0
        non_allo_attention = 0

        for head in range(layer_attention.size(0)):
            head_attention = layer_attention[head]
            head_threshold_mask = threshold_mask[head]

            # For allosteric sites
            if allo_mask.any():
                valid_attention = head_attention[head_threshold_mask & allo_mask.unsqueeze(0)]
                if len(valid_attention) > 0:
                    allo_attention += valid_attention.mean().item()

            # For non-allosteric sites
            if non_allo_mask.any():
                valid_attention = head_attention[head_threshold_mask & non_allo_mask.unsqueeze(0)]
                if len(valid_attention) > 0:
                    non_allo_attention += valid_attention.mean().item()

        # Average across heads
        num_heads = layer_attention.size(0)
        allo_attention /= num_heads
        non_allo_attention /= num_heads

        stats.update({
            "mean_attention_allosteric": allo_attention,
            "mean_attention_non_allosteric": non_allo_attention,
            "attention_ratio": allo_attention / non_allo_attention if non_allo_attention > 0 else 0
        })

        return stats

    def analyze_distance_effect(self, attention_maps, allosteric_sites, head_idx):
        """Analyze how attention varies with distance from allosteric sites"""
        # Get attention for specific head and reshape if needed
        head_attention = attention_maps[0, head_idx]  # Shape: [seq_len, seq_len]
        if head_attention.dim() > 2:
            head_attention = head_attention.squeeze()  # Remove extra dimensions

        distances = {}

        for site in allosteric_sites:
            # Get attention from all positions to this allosteric site
            site_attention = head_attention[:, site - 1]  # 0-based indexing

            # Calculate distances and attention values
            for pos in range(len(site_attention)):
                distance = abs(pos - (site - 1))
                # Get mean attention value across all positions
                attention_value = site_attention[pos].mean().cpu().item()

                if distance not in distances:
                    distances[distance] = []
                distances[distance].append(attention_value)

        # Average attention by distance
        avg_attention = {d: np.mean(v) for d, v in distances.items()}
        return avg_attention

    def calculate_allosteric_attention_scores(self, sequence: str, allosteric_sites: List[int],
                                              sensitive_heads: List[int]) -> Dict[int, float]:
        """
        Calculate attention scores for each amino acid based on how much they attend to allosteric sites
        in allostery-sensitive heads.

        Args:
            sequence: Protein sequence
            allosteric_sites: List of allosteric site positions (1-based indexing)
            sensitive_heads: List of head indices identified as allosteric-sensitive

        Returns:
            Dictionary mapping position (1-based) to attention score
        """
        attention_maps = self.get_attention_maps(sequence)
        position_scores = {}

        # Convert to 0-based indexing
        allo_sites = [site - 1 for site in allosteric_sites]

        # For each position in the sequence
        for pos in range(len(sequence)):
            total_attention = 0

            # Average attention across sensitive heads
            for head in sensitive_heads:
                head_attention = attention_maps[0, head]

                # Get attention from this position to all allosteric sites
                for site in allo_sites:
                    total_attention += head_attention[pos, site].item()

            # Average across heads and sites
            avg_attention = total_attention / (len(sensitive_heads) * len(allo_sites))
            position_scores[pos + 1] = avg_attention  # Convert back to 1-based indexing

        return position_scores

    def analyze_window_scores(self, attention_maps: torch.Tensor, allosteric_sites: List[int], window_size: int = 5) -> \
    Dict[int, float]:
        """
        Analyze attention patterns in windows around allosteric sites.
        Returns head scores based on attention to allosteric regions.
        """
        head_scores = {}

        print(f"\nDebug - Attention maps shape: {attention_maps.shape}")

        # Reshape attention maps if needed
        if attention_maps.dim() == 5:  # [1, 33, 20, 318, 318]
            attention_maps = attention_maps.squeeze(0)  # Now [33, 20, 318, 318]

        for head in range(self.num_heads):
            # Get the correct attention matrix for this head
            head_attention = attention_maps[0, head]  # Get first layer, specific head
            print(f"\nDebug - Head {head} attention shape: {head_attention.shape}")
            print(
                f"Debug - Head {head} attention range: [{head_attention.min().item():.6f}, {head_attention.max().item():.6f}]")

            total_score = 0
            valid_sites = 0

            for site in allosteric_sites:
                # Define window boundaries (convert to 0-based indexing)
                site_idx = site - 1
                start = max(0, site_idx - window_size)
                end = min(head_attention.shape[0], site_idx + window_size + 1)

                if end > start:
                    # Get window attention and apply threshold
                    window_attention = head_attention[start:end, start:end]
                    mask = window_attention > self.threshold
                    filtered_attention = window_attention[mask]

                    print(f"Debug - Site {site}:")
                    print(f"Window shape: {window_attention.shape}")
                    print(f"Window boundaries: [{start}, {end}]")
                    print(f"Values above threshold: {filtered_attention.numel()}")

                    if filtered_attention.numel() > 0:
                        window_score = filtered_attention.mean().item()
                        print(f"Window score: {window_score:.6f}")
                        total_score += window_score
                        valid_sites += 1

            # Average across valid allosteric sites
            if valid_sites > 0:
                head_scores[head] = total_score / valid_sites
            else:
                head_scores[head] = 0.0

            print(f"Debug - Head {head} final score: {head_scores[head]:.6f}")

        return head_scores

    def visualize_head_attention(self, attention_maps: torch.Tensor, allosteric_sites: List[int],
                                 orthosteric_sites: List[int],
                                 pathway_sites: List[int], head_idx: int, sequence: str, layer_idx: int = 0) -> None:
        """
        Visualize attention patterns for a specific head

        Args:
            head_idx: Index of the attention head to visualize
            layer_idx: Index of the layer to visualize (default: 0)
            attention_maps: Attention maps from the model
            allosteric_sites: List of allosteric site positions (1-based indexing)
            orthosteric_sites: List of residues that form part of the site for the orthosteric ligand
            pathway_sites: List of pathway site positions (1-based indexing) obtained through a PNC analysis
            sequence: Protein sequence
        """

        # Check if attention_maps is 5D
        if attention_maps.dim() == 5:  # [batch, num_layers, num_heads, seq_len, seq_len]
            # Select the first batch and the specified layer
            attention_maps = attention_maps[0, layer_idx]  # Now [num_heads, seq_len, seq_len]
        elif attention_maps.dim() == 4:  # [batch, num_heads, seq_len, seq_len]
            # Select the first batch
            attention_maps = attention_maps[0]  # Now [num_heads, seq_len, seq_len]
        elif attention_maps.dim() != 3:  # [num_heads, seq_len, seq_len]
            raise ValueError(f"Unexpected attention map shape: {attention_maps.shape}")

        # Validate head_idx
        num_heads = attention_maps.shape[0]
        if head_idx < 0 or head_idx >= num_heads:
            raise ValueError(f"head_idx {head_idx} is out of range (0 to {num_heads - 1})")

        # Obtain attention map for specific head
        attention = attention_maps[head_idx]  # [seq_len, seq_len]

        # Convert to a numpy array and squeeze out batch dimension if present
        attention_2d = attention.cpu().numpy().squeeze()

        # Debug print statements
        # print(f"Attention map shape: {attention_2d.shape}")
        # print(f"Head index: {head_idx}")
        # print(f"Allosteric sites: {allosteric_sites}")
        # print(f"Orthosteric sites: {orthosteric_sites}")
        # print(f"Pathway sites: {pathway_sites}")

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create a heatmap of the attention map.
        ## Use a dynamic tick spacing to avoid crowding in short sequences
        seq_len = len(sequence)
        tick_spacing = 50
        default_ticks = set(i + 1 for i in range(seq_len) if (i + 1) % tick_spacing == 0)
        highlight_ticks = default_ticks.union(allosteric_sites).union(orthosteric_sites).union(pathway_sites)

        xticks = [
            f"{i}({sequence[i - 1]})" if i in highlight_ticks and i <= seq_len else ''
            for i in range(1, seq_len + 1)
        ]

        highlight_ticks_y = default_ticks.union(orthosteric_sites).union(pathway_sites)
        yticks = [
            f"{i}({sequence[i - 1]})" if i in highlight_ticks_y and i <= seq_len else ''
            for i in range(1, seq_len + 1)
        ]

        # Calcular el percentil
        threshold_annot = np.percentile(attention_2d, 98)

        # Crear matriz de anotaciones vacía
        annot = np.full_like(attention_2d, '', dtype=object)

        # Rellenar solo donde la atención supera el percentil escogido
        for i in range(seq_len):
            for j in range(seq_len):
                # Comprobar si el residuo atendido (j) está en allosteric_sites y si el valor de atención es alto
                if attention_2d[i, j] >= threshold_annot and (
                        j + 1) in allosteric_sites:  # Usamos j + 1 porque allosteric_sites usa 1-based indexing
                    # Solo asignamos formato si el valor es un número
                    annot[i, j] = f"{attention_2d[i, j]:.4f}"
        ax = sns.heatmap(
            attention_2d,
            cmap='viridis',
            xticklabels=xticks,
            yticklabels=yticks,
            annot=annot,  # Añade los valores de atención
            fmt="",  # Formato con 2 decimales
            annot_kws={"size": 7}
        )

        # Highlight allosteric sites
        for site in allosteric_sites:
            if 1 <= site <= seq_len:  # Asegura que el sitio está dentro del rango
                plt.axhline(y=site - 0.5, color='r', alpha=0.3, linestyle='--')
                plt.axvline(x=site - 0.5, color='r', alpha=0.3, linestyle='--')
            else:
                print(f"Warning: Allosteric site {site} is out of range (1 to {seq_len})")
        # Rotar y resaltar etiquetas de ejes
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Highlight orthosteric sites
        for site in orthosteric_sites:
            if 1 <= site <= seq_len:  # Asegura que el sitio está dentro del rango
                plt.axhline(y=site - 0.5, color='lime', alpha=0.3, linestyle='--')
                plt.axvline(x=site - 0.5, color='lime', alpha=0.3, linestyle='--')
            else:
                print(f"Warning: Orthosteric site {site} is out of range (1 to {seq_len})")
        # Rotar y resaltar etiquetas de ejes
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        # Líneas para sitios del pathway alostérico en amarillo
        for site in pathway_sites:
            if 1 <= site <= seq_len:
                ax.axhline(y=site - 0.5, color='yellow', alpha=0.3, linestyle='--')
                ax.axvline(x=site - 0.5, color='yellow', alpha=0.3, linestyle='--')
            else:
                print(f"Warning: Pathway site {site} is out of range (1 to {seq_len})")

        for label in ax.get_xticklabels():
            try:
                val = int(label.get_text())
                if val in allosteric_sites:
                    label.set_color('red')
                    label.set_fontweight('bold')
                elif val in orthosteric_sites:
                    label.set_color('lime')
                    label.set_fontweight('bold')
                elif val in pathway_sites:
                    label.set_color('goldenrod')
                    label.set_fontweight('bold')

            except ValueError:
                continue

        for label in ax.get_yticklabels():
            try:
                val = int(label.get_text())
                if val in allosteric_sites:
                    label.set_color('red')
                    label.set_fontweight('bold')
                elif val in orthosteric_sites:
                    label.set_color('lime')
                    label.set_fontweight('bold')
                elif val in pathway_sites:
                    label.set_color('goldenrod')
                    label.set_fontweight('bold')
            except ValueError:
                continue

        plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Position j (attended to)')
        plt.ylabel('Position i (attending from)')
        plt.show()

    def visualize_average_head_attention(self, attention_maps: torch.Tensor, impact_scores: list, snr_values: list,
                                         p_values: list,
                                         allosteric_sites: list, orthosteric_sites: list, pathway_sites: list,
                                         sequence: str, layer_idx: int = 0, cmap="viridis") -> None:
        """
        Visualiza el mapa de atención promedio de las cabezas sensibles en una capa específica,
        basado en impact > mean, SNR > 2.0 y p_value < 0.01
        """

        import matplotlib.pyplot as plt
        import seaborn as sns

        # 1. Seleccionar cabezas sensibles con los 3 criterios
        mean_impact = np.mean(impact_scores)
        sensitive_heads = [
            i for i, (impact, snr, pval) in enumerate(zip(impact_scores, snr_values, p_values))
            if snr.item() > 2.0 and pval < 0.01
        ]

        if not sensitive_heads:
            print(f"[Layer {layer_idx}] No sensitive heads found (impact > mean, SNR > 2.0, p < 0.01)")
            return

        # 2. Preparar el tensor de atención
        if attention_maps.dim() == 5:
            attention_maps = attention_maps[0, layer_idx]  # [num_heads, seq_len, seq_len]
        elif attention_maps.dim() == 4:
            attention_maps = attention_maps[0]  # [num_heads, seq_len, seq_len]
        elif attention_maps.dim() != 3:
            raise ValueError("Expected attention_maps to have 3, 4, or 5 dimensions.")

        # 3. Extraer y promediar las cabezas sensibles
        selected_maps = attention_maps[sensitive_heads]  # [num_sensitive_heads, seq_len, seq_len]
        average_attention = selected_maps.mean(dim=0).cpu().numpy()

        # 4. Visualizar mapa de atención promedio
        seq_len = len(sequence)
        # tick_spacing = max(1, seq_len // 50)
        tick_spacing = 50
        default_ticks = set(i + 1 for i in range(seq_len) if i % tick_spacing == 0)
        highlight_ticks = default_ticks.union(allosteric_sites).union(orthosteric_sites).union(pathway_sites)
        xticks = [
            f"{i}({sequence[i - 1]})" if i in highlight_ticks else ''
            for i in range(1, seq_len + 1)
        ]

        highlight_ticks_y = default_ticks.union(orthosteric_sites).union(pathway_sites)
        yticks = [
            f"{i}({sequence[i - 1]})" if i in highlight_ticks_y else ''
            for i in range(1, seq_len + 1)
        ]

        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(average_attention, cmap=cmap, xticklabels=xticks, yticklabels=yticks)
        ax.set_title(f'Mapa de atención promedio de las cabezas alostéricamente atentas.')
        ax.set_xlabel('Residuo j (Atento)')
        ax.set_ylabel('Residuo i (Atendido)')

        # 5. Añadir líneas punteadas para sitios clave
        for site in allosteric_sites:
            # plt.axhline(x=site-0.5, color='red', linestyle='--', linewidth=0.7)
            plt.axvline(x=site - 0.5, color='red', linestyle='--', linewidth=0.7)
        for site in orthosteric_sites:
            plt.axhline(y=site - 0.5, color='lime', linestyle='--', linewidth=0.3)
            # plt.axvline(x=site-0.5, color='lime', linestyle='--', linewidth=0.7)
        for site in pathway_sites:
            plt.axhline(y=site - 0.5, color='goldenrod', linestyle='--', linewidth=0.7)
            # plt.axvline(x=site-0.5, color='goldenrod', linestyle='--', linewidth=0.7)

        # 6. Colorear etiquetas de los ejes

        for label in ax.get_xticklabels():
            idx = int(label.get_text().split('(')[0]) if '(' in label.get_text() else -1
            if idx in allosteric_sites:
                label.set_color('red')
                label.set_fontweight('bold')
            elif idx in orthosteric_sites:
                label.set_color('lime')
                label.set_fontweight('bold')
            elif idx in pathway_sites:
                label.set_color('goldenrod')
                label.set_fontweight('bold')

        for label in ax.get_yticklabels():
            idx = int(label.get_text().split('(')[0]) if '(' in label.get_text() else -1
            if idx in allosteric_sites:
                label.set_color('red')
                label.set_fontweight('bold')
            elif idx in orthosteric_sites:
                label.set_color('lime')
                label.set_fontweight('bold')
            elif idx in pathway_sites:
                label.set_color('goldenrod')
                label.set_fontweight('bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=00)

        plt.tight_layout()
        plt.show()

    def analyze_head_connections(self, head_idx: int, attention_maps: torch.Tensor,
                                 sequence: str) -> None:
        """
        Analyze top attention connections for a head

        Args:
            head_idx: Index of the attention head to analyze
            attention_maps: Attention maps from the model [batch, layers, heads, seq_len, seq_len]
            sequence: Protein sequence

        Results in meaningless self- connections output. Needs to be fixed.
        """
        # Get attention map for specific head (from first layer and first batch)
        attention = attention_maps[0, 0, head_idx]  # Shape: [seq_len, seq_len]

        print(f"\nHead {head_idx} - Top attention connections:")
        print("From\tTo\tScore\tAA_From->AA_To")
        print("-" * 45)

        # Get top attention scores
        values, indices = attention.view(-1).sort(descending=True)
        seq_len = len(sequence)
        count = 0
        i = 0

        # Print top 10 connections with scores above threshold
        while count < 10 and i < len(values):
            score = values[i].item()
            if score > self.threshold:
                # Calculate positions
                from_pos = (indices[i].item() // attention.size(1))
                to_pos = (indices[i].item() % attention.size(1))

                # Check if indices are within sequence bounds
                if 0 <= from_pos < seq_len and 0 <= to_pos < seq_len:
                    aa_from = sequence[from_pos]
                    aa_to = sequence[to_pos]
                    print(f"{from_pos + 1}\t{to_pos + 1}\t{score:.3f}\t{aa_from}->{aa_to}")
                    count += 1
            i += 1