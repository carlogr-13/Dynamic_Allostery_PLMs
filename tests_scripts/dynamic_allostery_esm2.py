"""
Dynamic Allosteric PLM Analyzer (ESM-2)
=======================================

This module implements a zero-shot computational pipeline to extract, filter,
and analyze allosteric communication networks in proteins using Protein Language
Models (ESM-2) and graph theory (MST extraction).

Dependencies
------------
    - torch
    - fair-esm
    - numpy
    - pandas
    - scipy
    - networkx
    - biopython
    - tqdm

Author
------
[Tu Nombre / Tu Institución]

References
----------
    1. Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science.
    2. Dong, T., et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites.
    3. Di Paola, L., et al. (2013). Protein contact networks: an emerging paradigm in chemistry. Chemical Reviews.
    4. Trenfield, K., & Lin, M. M. (2025). Sparse networks of conformational fluctuations communicate signals within proteins.
    5. Sethi, A., et al. (2009). Dynamical networks in tRNA:protein complexes. PNAS.
    6. Madan, et al. (2023). The "violin model": Looking at community networks for dynamic allostery. JCP.
"""

import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import cdist
from Bio.PDB import PDBList, PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.Data import IUPACData

try:
    import esm
except ImportError:
    raise ImportError("The 'esm' library is required. Install via: pip install fair-esm")

try:
    from scipy.stats import ttest_1samp
except ImportError:
    raise ImportError("The 'scipy' library is required. Install via: pip install scipy")

# =====================================================================
# GLOBAL CONFIGURATION & HELPER CLASSES
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class ChainAndProteinSelect(Select):
    """
    Filtering heuristic to isolate the biological unit of interest from a PDB.
    Rejects solvent (HOH, WAT), ligands, and non-target chains to prevent
    spatial noise during the 3D distance masking in Phase 5.
    """

    def __init__(self, target_chain: str) -> None:
        self.target_chain = target_chain

    def accept_chain(self, chain: Any) -> int:
        return 1 if chain.get_id() == self.target_chain else 0

    def accept_residue(self, residue: Any) -> int:
        return 1 if residue.id[0] == " " and residue.resname not in ["HOH", "WAT"] else 0


# =====================================================================
# MASTER ANALYZER CLASS
# =====================================================================
class AllostericNetworkAnalyzer:
    """
    Orchestrates the dynamic allosteric network workflow using a 6-phase pipeline:
    Phase 0: Scaffold preparation and FASTA generation.
    Phase 1: Zero-shot ESM-2 inference.
    Phase 2 & 3: Statistical head filtering (Dong et al., 2024) and sink purging.
    Phase 4: Symmetrization and sequential neighborhood purging (|i-j| < 5).
    Phase 5: Spatial constraint mapping, Thermodynamic Distance scaling, and MIST extraction.
    Phase 6: Topological CGO rendering (Trenfield & Lin, 2025).
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("AllostericAnalyzer")
        self.root_dir: Path = Path("")
        self.pdb_dir: Path = Path("")
        self.fasta_dir: Path = Path("")
        self.tensor_dir: Path = Path("")
        self.graph_dir: Path = Path("")
        self.cgo_dir: Path = Path("")
        self.current_offset: int = 0
        self.spatial_cutoff: float = 15.0  # Angstroms (Sethi et al., 2009)

    def _set_deterministic_seed(self, seed: int = 42) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        self.logger.info(f"Deterministic PRNG seed anchored globally at: {seed}")

    def _setup_directories(self, project_name: str, base_dir: Optional[str]) -> None:
        if base_dir is None:
            current_script_dir = Path(__file__).resolve().parent
            self.root_dir = current_script_dir.parent / f"Data_{project_name}"
        else:
            self.root_dir = Path(base_dir).resolve()

        self.pdb_dir = self.root_dir / "processed_pdb"
        self.fasta_dir = self.root_dir / "fasta_sequences"
        self.tensor_dir = self.root_dir / "results"
        self.graph_dir = self.root_dir / "graph_centrality"
        self.cgo_dir = self.root_dir / "pymol_cgo_scripts"

        for directory in [self.pdb_dir, self.fasta_dir, self.tensor_dir, self.graph_dir, self.cgo_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Directory topology configured at: {self.root_dir}")

    # =================================================================
    # PHASE 0: SCAFFOLD CURATION AND SEQUENCE GENERATION
    # =================================================================
    def _curate_scaffold(self, project_name: str, pdb_id: str, chain: str) -> str:
        self.logger.info(f"Phase 0: Curating spatial scaffold for {project_name} ({pdb_id}, Chain {chain})")
        clean_path = self.pdb_dir / f"{project_name}_scaffold_clean.pdb"

        if clean_path.exists():
            self.logger.info("Curated scaffold already exists. Skipping download.")
            return str(clean_path)

        pdbl = PDBList(pdb=str(self.pdb_dir))
        raw_pdb_path = pdbl.retrieve_pdb_file(pdb_id, pdir=str(self.pdb_dir), file_format="pdb")
        parser = PDBParser(QUIET=True)

        try:
            structure = parser.get_structure(project_name, raw_pdb_path)
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(clean_path), ChainAndProteinSelect(target_chain=chain))
        except Exception as e:
            self.logger.error(f"Crystallographic parsing failed: {e}")
            raise

        return str(clean_path)

    def _generate_microstates(self, project_name: str, canonical_seq: str,
                              mutational_dict: Dict[str, List[Any]], offset: int) -> None:
        self.logger.info(f"Phase 0: Generating sequence FASTA files for {project_name}")

        wt_path = self.fasta_dir / f"{project_name}_WT.fasta"
        with open(wt_path, "w") as f:
            f.write(f">{project_name}_WT\n{canonical_seq}\n")

        for state_name, mutation_list in mutational_dict.items():
            seq_list = list(canonical_seq)

            for mutation_data in mutation_list:
                wt_aa = str(mutation_data[0])
                pos = int(mutation_data[1])
                mut_aa = str(mutation_data[2])
                rel_idx = pos - 1 - offset

                if seq_list[rel_idx] != wt_aa:
                    self.logger.warning(
                        f"Residue mismatch at PDB pos {pos}. Expected '{wt_aa}', found '{seq_list[rel_idx]}'. "
                        f"Check the offset value ({offset})."
                    )
                seq_list[rel_idx] = mut_aa

            mut_seq = "".join(seq_list)
            mut_path = self.fasta_dir / f"{project_name}_{state_name}.fasta"
            with open(mut_path, "w") as f:
                f.write(f">{project_name}_{state_name}\n{mut_seq}\n")

    # =================================================================
    # PHASE 1, 2 & 3: INFERENCE AND FILTERING (WT-ANCHORED)
    # =================================================================
    def _extract_esm_attention(self, sequence: str) -> torch.Tensor:
        self.logger.info("   -> Initializing ESM-2 Model (esm2_t33_650M_UR50D)...")
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        data = [("protein", sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            # repr_layers=[33] optimizes RAM footprint by not storing hidden embeddings for all layers,
            # while need_head_weights=True ensures attention maps are still returned for all 33 layers.
            results = model(batch_tokens, repr_layers=[33], return_contacts=False, need_head_weights=True)
            attentions = results["attentions"].squeeze(0).cpu()

        # Isolate true biological sequence by slicing out <cls> and <eos> tokens
        biological_attentions = attentions[..., 1:-1, 1:-1]
        return biological_attentions

    def _filter_allo_allo(self, attention_tensor: torch.Tensor, target_indices: List[int], sequence: str) -> Tuple[
        List[Tuple[int, int]], torch.Tensor]:
        """
        Isolates allostery-sensitive attention heads via an empirical null model (Dong et al., 2024).
        Safeguards against infinite SNR caused by zero-variance backgrounds in highly sparse heads.
        """
        layers, heads, seq_len, _ = attention_tensor.shape
        threshold = 0.3
        n_random_trials = 1000

        head_stats = []
        non_allo_positions = np.array([i for i in range(seq_len) if i not in target_indices])
        n_allo_sites = len(target_indices)

        self.logger.info(
            f"   -> Executing Monte Carlo statistical filtering across {layers * heads} attention heads...")

        total_iterations = layers * heads
        with tqdm(total=total_iterations, desc="   -> SNR Filtering", unit="head", leave=False) as pbar:
            for l in range(layers):
                for h in range(heads):
                    matrix = attention_tensor[l, h]
                    mask = matrix > threshold

                    w_allo = sum(torch.sum(matrix[:, site][mask[:, site]]).item() for site in target_indices)

                    random_w_values = []
                    for _ in range(n_random_trials):
                        random_sites = np.random.choice(non_allo_positions, size=n_allo_sites, replace=False)
                        w_random = sum(torch.sum(matrix[:, site][mask[:, site]]).item() for site in random_sites)
                        random_w_values.append(w_random)

                    expected_random = np.mean(random_w_values)
                    std_random = np.std(random_w_values)

                    if std_random < 1e-6:
                        impact, snr, p_val = 0.0, 0.0, 1.0
                    else:
                        impact = w_allo / expected_random if expected_random > 0 else 0
                        snr = (w_allo - expected_random) / std_random
                        t_stat, p_val = ttest_1samp(random_w_values, w_allo, alternative='less')

                    head_stats.append((l, h, impact, snr, p_val))
                    pbar.update(1)

        impacts = [stat[2] for stat in head_stats]
        mean_impact = np.mean(impacts)

        valid_heads = []
        selected_matrices = []

        print("=" * 50)
        print("The allosteric residues are: ")
        for site in target_indices:
            pdb_num = site + 1 + self.current_offset
            aa_3l = IUPACData.protein_letters_1to3[sequence[site].upper()]
            print(f"{aa_3l} {pdb_num}")
        print("=" * 50)

        print("\nAllosteric sensitivity analysis per attention head:")
        print("Layer | Head | Impact Score | SNR")
        print("-" * 40)
        for layer, head, impact, snr, p_val in head_stats:
            if impact > 0.001:
                print(f"{layer:5d} | {head:4d} | {impact:11.3f} | {snr:6.2f}")

            if impact > mean_impact and snr > 2.0 and p_val < 0.01:
                valid_heads.append((layer, head))
                selected_matrices.append(attention_tensor[layer, head])

        absolute_target_sites = [i + 1 + self.current_offset for i in target_indices]
        print(f"\nMost sensitive heads to allosteric sites {absolute_target_sites}:")
        print(f"(Impact > {mean_impact:.3f}, p < 0.01 and SNR > 2.0)")
        print(f"(Layer, Head) pairs: {valid_heads}")

        if not selected_matrices:
            return [], torch.empty(0)

        return valid_heads, torch.stack(selected_matrices)

    def _purge_column_sinks(self, filtered_tensor: torch.Tensor, valid_heads: List[Tuple[int, int]],
                            seq_len: int) -> Tuple[List[Tuple[int, int]], torch.Tensor]:
        """
        Discards attention heads where a single residue absorbs >10% of total probability mass.
        Prevents softmax dimensionality collapse artifacts from posing as allosteric networks
        (Moya-García, 2026).

        Returns:
            Tuple containing the strictly curated list of (layer, head) pairs and the cleaned tensor.
        """
        sink_threshold = seq_len * 0.10
        clean_matrices = []
        clean_valid_heads = []
        purged_count = 0

        for idx, matrix in enumerate(filtered_tensor):
            col_sums = matrix.sum(dim=0)
            max_col_sum = col_sums.max().item()

            if max_col_sum < sink_threshold:
                clean_matrices.append(matrix)
                clean_valid_heads.append(valid_heads[idx])
            else:
                purged_count += 1
                self.logger.debug(f"      - Purged Head {valid_heads[idx]} (Column sum exceeds 10% threshold)")

        self.logger.info(f"   -> Purged {purged_count} column-sink artifacts.")
        if not clean_matrices:
            return [], torch.empty(0)

        return clean_valid_heads, torch.stack(clean_matrices)

    def _run_inference_and_filtering(self, project_name: str, target_residues_pdb: List[int], offset: int) -> None:
        """
        Enforces isogenic probability spaces by determining the definitive valid heads
        strictly on the WT sequence (Post-SNR and Post-Sink Purge).
        Mutants are constrained identically to these heads to quantify topological decay.
        """
        self.logger.info(f"PHASE 1-3: Executing Inference and Filtering for {project_name}")

        fasta_files = list(self.fasta_dir.glob(f"{project_name}_*.fasta"))
        if not fasta_files:
            self.logger.error("No FASTA files found. Phase 0 must be executed first.")
            return

        # 1. Evaluate WT Baseline
        wt_fasta = next((f for f in fasta_files if "WT" in f.name), None)
        if not wt_fasta:
            self.logger.error("WT FASTA file not found. A WT sequence is required as the baseline.")
            return

        self.logger.info(f"Establishing baseline architecture using: {wt_fasta.name}")
        with open(wt_fasta, "r") as f:
            wt_seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

        seq_len = len(wt_seq)
        target_idx = [p - 1 - offset for p in target_residues_pdb]
        valid_targets = [i for i in target_idx if 0 <= i < seq_len]

        if not valid_targets:
            self.logger.error("Target mapping failed. No valid indices found within sequence bounds.")
            return

        raw_wt_attention = self._extract_esm_attention(wt_seq)

        # 2. Extract significant heads (Allo-Allo)
        valid_heads, p2_tensor_wt = self._filter_allo_allo(raw_wt_attention, valid_targets, wt_seq)

        if p2_tensor_wt.shape[0] == 0:
            self.logger.error("WT sequence failed to produce valid attention heads. Pipeline halted.")
            return

        # 3. Purge Softmax Sinks to finalize WT Architecture
        final_valid_heads, clean_tensor_wt = self._purge_column_sinks(p2_tensor_wt, valid_heads, seq_len)

        if len(final_valid_heads) == 0:
            self.logger.error("All significant heads were discarded as column-sinks. Pipeline halted.")
            return

        self.logger.info(
            f"   -> Definitive baseline established: {len(final_valid_heads)} functional heads identified.")
        torch.save(clean_tensor_wt, self.tensor_dir / f"clean_attention_{project_name}_WT.pt")

        # 4. Process Mutants using STRICT WT Architecture
        mutant_fastas = [f for f in fasta_files if "WT" not in f.name]
        for mut_fasta in mutant_fastas:
            state_name = mut_fasta.stem.replace(f"{project_name}_", "")
            self.logger.info(f"Processing Mutant State: {state_name} (Using Definitive WT Architecture)")

            with open(mut_fasta, "r") as f:
                mut_seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

            raw_mut_attention = self._extract_esm_attention(mut_seq)

            # Isolate ONLY the heads definitively validated in the WT run
            # DO NOT re-apply purge filters. This ensures a 1:1 isogenic topological comparison.
            selected_mut_matrices = []
            for (l, h) in final_valid_heads:
                selected_mut_matrices.append(raw_mut_attention[l, h])

            mut_tensor = torch.stack(selected_mut_matrices)
            out_path = self.tensor_dir / f"clean_attention_{project_name}_{state_name}.pt"
            torch.save(mut_tensor, out_path)
            self.logger.info(f"   -> Saved mutant tensor strictly mapped to WT architecture: {out_path.name}")

    # =================================================================
    # PHASE 4: CONSENSUS AND SYMMETRIZATION
    # =================================================================
    def _build_symmetric_network(self, project_name: str) -> None:
        """
        Collapses valid head matrices into a consensus representation.
        Enforces algebraic graph symmetry. Purges sequential neighbors (|i-j| < 5)
        to eliminate covalent biases, strictly isolating tertiary allosteric communication
        (Di Paola et al., 2013).
        """
        self.logger.info(f"PHASE 4: Executing Symmetrization and Neighborhood Purge for {project_name}")

        tensor_files = list(self.tensor_dir.glob(f"clean_attention_{project_name}_*.pt"))
        if not tensor_files:
            self.logger.error("No clean attention tensors found. Execute Phases 1-3 first.")
            return

        for tensor_path in tensor_files:
            state_name = tensor_path.stem.replace(f"clean_attention_{project_name}_", "")
            self.logger.info(f"Processing State: {state_name}")

            # 1. Load tensor [K, L, L] and average over valid heads to create [L, L] consensus
            clean_tensor = torch.load(tensor_path, weights_only=True)
            consensus_matrix = torch.mean(clean_tensor, dim=0).cpu().numpy()

            # 2. Heuristic algebraic symmetrization to allow undirected graph topological analysis
            symmetric_matrix = (consensus_matrix + consensus_matrix.T) / 2.0
            seq_len = symmetric_matrix.shape[0]

            # 3. Vectorized sequential neighborhood purge (|i - j| < 5)
            # Creates an LxL boolean mask mapping the primary sequence distance between any two residues
            sequence_distance_matrix = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len))
            neighborhood_mask = (sequence_distance_matrix >= 5).astype(float)

            # Apply mask to eliminate covalent and local secondary structure bias (LRO isolation)
            final_network = symmetric_matrix * neighborhood_mask

            # Save the final undirected topological matrix
            out_path = self.tensor_dir / f"undirected_network_{project_name}_{state_name}.npy"
            np.save(out_path, final_network)

            # Metrics and Logging
            global_sparsity = (final_network == 0).sum() / final_network.size * 100
            max_w = np.max(final_network)
            mean_w = np.mean(final_network[final_network > 0]) if np.any(final_network > 0) else 0.0

            self.logger.info(f"   -> Graph Matrix Shape : {final_network.shape}")
            self.logger.info(f"   -> Masked Sparsity    : {global_sparsity:.2f}% (Diagonals & Neighbors)")
            self.logger.info(f"   -> Max Edge Weight    : {max_w:.6f}")
            self.logger.info(f"   -> Mean Active Edge   : {mean_w:.6f}")
            self.logger.info(f"   -> Matrix saved to    : {out_path.name}")

    # =================================================================
    # PHASE 5: SPATIAL MASK, THERMODYNAMIC DISTANCE, AND MST
    # =================================================================
    def _extract_mist_and_centrality(self, project_name: str, pdb_path: str, chain_id: str, offset: int) -> None:
        """
        Imposes a 12 Angstrom Euclidean mask based on C-alpha coordinates to isolate
        physically realizable non-covalent interactions and prevent unphysical statistical
        covariance (Jones et al., 2023).

        Converts symmetrized attention probabilities into topological distances (D = -log(P))
        (Madan et al., 2023) to ensure additive shortest-path calculations.
        Extracts the Minimum Spanning Tree (MST) as a topological proxy for the allosteric
        communication network, followed by Betweenness Centrality analysis to identify
        critical signal bottlenecks.
        """
        self.logger.info(f"PHASE 5/6: Extracting Spatial MST and Centralities for {project_name}")

        # 1. Structural Parsing and Coordinate Extraction
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("scaffold", pdb_path)

        coords_dict = {}
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        # Ensure residue is a standard amino acid (hetero flag is ' ')
                        if 'CA' in residue and residue.id[0] == ' ':
                            seq_idx = residue.id[1] - 1 - offset
                            coords_dict[seq_idx] = residue['CA'].coord

        network_files = list(self.tensor_dir.glob(f"undirected_network_{project_name}_*.npy"))

        for file_path in network_files:
            state_name = file_path.stem.replace(f"undirected_network_{project_name}_", "")
            self.logger.info(f"   -> Processing State: {state_name}")

            symmetric_matrix = np.load(file_path)
            seq_len = symmetric_matrix.shape[0]

            coords_array = np.full((seq_len, 3), np.nan)
            for idx, coord in coords_dict.items():
                if 0 <= idx < seq_len:
                    coords_array[idx] = coord

            # 2. Euclidean Distance Masking (12 Angstrom Threshold)
            dist_matrix = cdist(coords_array, coords_array, metric='euclidean')
            dist_matrix[np.isnan(dist_matrix)] = np.inf

            spatial_mask = (dist_matrix <= self.spatial_cutoff).astype(float)
            physical_matrix = symmetric_matrix * spatial_mask

            # 3. Logarithmic Transformation and Graph Construction
            G = nx.Graph()
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    prob = physical_matrix[i, j]
                    if prob > 0:
                        # Transform probability to topological distance
                        thermo_dist = -np.log(prob)
                        G.add_edge(i, j, weight=thermo_dist, probability=prob)

            # 4. MST Extraction and Betweenness Centrality Calculation
            if G.number_of_edges() > 0:
                mst = nx.minimum_spanning_tree(G, weight='weight')
                # NetworkX interprets 'weight' as distance for shortest path computations
                centrality = nx.betweenness_centrality(mst, weight='weight', normalized=True)

                fasta_path = self.fasta_dir / f"{project_name}_{state_name}.fasta"
                with open(fasta_path, 'r') as f:
                    seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

                nodes_data = [{"Residue_PDB": k + 1 + offset,
                               "Amino_Acid": seq[k],
                               "Betweenness_Centrality": v}
                              for k, v in centrality.items()]

                edges_data = [{"Source_PDB": u + 1 + offset, "Source_AA": seq[u],
                               "Target_PDB": v + 1 + offset, "Target_AA": seq[v],
                               "Distance": d['weight'],
                               "Probability": d['probability']}
                              for u, v, d in mst.edges(data=True)]

                df_nodes = pd.DataFrame(nodes_data).sort_values(by="Betweenness_Centrality", ascending=False)
                df_edges = pd.DataFrame(edges_data).sort_values(by="Probability", ascending=False)

                # 5. Bottleneck Identification (Top 5%)
                if not df_nodes.empty:
                    threshold_95 = df_nodes["Betweenness_Centrality"].quantile(0.95)
                    df_nodes["Is_Bottleneck"] = df_nodes["Betweenness_Centrality"] >= threshold_95

                    bottleneck_count = df_nodes["Is_Bottleneck"].sum()
                    self.logger.info(f"      - Percentile 95 Threshold: {threshold_95:.4f}")
                    self.logger.info(f"      - Identified {bottleneck_count} critical bottlenecks (Top 5%).")

                df_nodes.to_csv(self.graph_dir / f"Centrality_{project_name}_{state_name}.csv", index=False)
                df_edges.to_csv(self.graph_dir / f"Edges_{project_name}_{state_name}.csv", index=False)

    # =================================================================
    # PHASE 6: SPARSE NETWORK CGO COMPILATION
    # =================================================================
    @staticmethod
    def _get_color_gradient(value: float, vmin: float, vmax: float) -> Tuple[float, float, float]:
        """
        Maps quantitative metrics strictly to an RGB thermal gradient (Blue -> Yellow -> Red).
        Low centrality/probability = Blue (Cool)
        High centrality/probability = Red (Hot)
        """
        if vmax == vmin:
            return 0.0, 0.0, 1.0  # Default to blue if variance is 0

        norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))

        if norm < 0.5:
            # Interpolate Blue (0,0,1) to Yellow (1,1,0)
            r = 2.0 * norm
            g = 2.0 * norm
            b = 1.0 - 2.0 * norm
        else:
            # Interpolate Yellow (1,1,0) to Red (1,0,0)
            r = 1.0
            g = 1.0 - 2.0 * (norm - 0.5)
            b = 0.0

        return float(r), float(g), float(b)

    def _compile_cgo(self, project_name: str, pdb_path: str, chain: str) -> None:
        """
        Renders the entire topological MST extracted in Phase 5 without edge deletion.
        Applies steep polynomial scaling (quadratic for edges, cubic for nodes) to visually
        isolate the scale-free nature of allosteric bottlenecks from background communication
        noise, aligning with the sparse network paradigm.
        """
        self.logger.info("PHASE 6: Compiling Sparse Network Topologies (PyMOL CGO Render)")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("scaffold", pdb_path)
        # Extract native PDB numbering coordinates
        coords = {res.id[1]: res['CA'].get_coord() for res in structure[0][chain] if
                    'CA' in res and res.id[0] == ' '}

        centrality_files = list(self.graph_dir.glob(f"Centrality_{project_name}_*.csv"))

        for cent_file in centrality_files:
            state_name = cent_file.stem.replace(f"Centrality_{project_name}_", "")
            edges_file = self.graph_dir / f"Edges_{project_name}_{state_name}.csv"

            if not edges_file.exists():
                continue

            df_nodes: pd.DataFrame = pd.read_csv(cent_file, engine="python")
            df_edges: pd.DataFrame = pd.read_csv(edges_file, engine="python")

            if df_nodes.empty or df_edges.empty:
                continue

            # Identify the absolute maximums for structural normalization
            max_prob = df_edges["Probability"].max() if not df_edges.empty else 1.0
            max_cent = df_nodes["Betweenness_Centrality"].max() if not df_nodes.empty else 1.0

            script_lines = [
                "from pymol.cgo import *",
                "from pymol import cmd",
                "cmd.reinitialize()",
                "cmd.bg_color('white')",
                f"cmd.load(r'{Path(pdb_path).resolve()}', '{project_name}_scaffold')",
                f"cmd.show_as('cartoon', '{project_name}_scaffold')",
                f"cmd.color('gray80', '{project_name}_scaffold')",
                f"cmd.set('cartoon_transparency', 0.85, '{project_name}_scaffold')",
                "obj = []"
            ]

            # Render ALL MST edges applying a quadratic exponential decay to thickness
            # This visually isolates high-probability vectors from structural noise
            for _, row in df_edges.iterrows():
                u, v = int(row["Source_PDB"]), int(row["Target_PDB"])
                if u in coords and v in coords:
                    c1, c2 = coords[u], coords[v]
                    prob = float(row["Probability"])

                    norm_prob = prob / max_prob if max_prob > 0 else 0
                    r, g, b = self._get_color_gradient(norm_prob, 0.0, 1.0)

                    # Quadratic thickness limits visual saturation (High pass filter)
                    thickness = 0.03 + (norm_prob ** 2) * 0.45
                    script_lines.append(
                        f"obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, {c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {thickness:.3f}, {r:.2f}, {g:.2f}, {b:.2f}, {r:.2f}, {g:.2f}, {b:.2f}])"
                    )

            # Render ALL MST nodes applying a steep cubic expansion to the radius
            # This mimics the "hubs" highlighted in sparse allosteric networks
            for _, row in df_nodes.iterrows():
                node = int(row["Residue_PDB"])
                if node in coords:
                    c = coords[node]
                    cent = float(row["Betweenness_Centrality"])

                    norm_cent = cent / max_cent if max_cent > 0 else 0
                    r, g, b = self._get_color_gradient(norm_cent, 0.0, 1.0)

                    # Cubic expansion creates clear hierarchical distinction for Centrality Hubs
                    radius = 0.4 + (norm_cent ** 3) * 2.2
                    script_lines.append(f"obj.extend([COLOR, {r:.2f}, {g:.2f}, {b:.2f}])")
                    script_lines.append(f"obj.extend([SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {radius:.3f}])")

            script_lines.append(f"cmd.load_cgo(obj, '{project_name}_{state_name}_Pathway')")

            cgo_path = self.cgo_dir / f"Render_CGO_{project_name}_{state_name}.py"
            with open(cgo_path, "w") as f:
                f.write("\n".join(script_lines))

            self.logger.info(f"   -> Sparse Network CGO generated: {cgo_path.name}")

    # =================================================================
    # EXECUTION ORCHESTRATOR
    # =================================================================
    def execute_pipeline(self, project_name: str, pdb_id: str, chain: str, canonical_sequence: str, offset: int,
                         target_residues: List[int],
                         mutational_dict: Optional[Dict[str, List[Any]]] = None,
                         base_dir: Optional[str] = None,
                         seed: Optional[int] = 42) -> None:
        """
        Triggers the absolute sequential execution of the dynamic allosteric
        network analysis. Enforces statistical determinism if a seed is provided.
        """
        self.logger.info("=====================================================")
        self.logger.info(f"STARTING ALLOSTERIC NETWORK PIPELINE: {project_name}")
        self.logger.info("=====================================================")

        if seed is not None:
            self._set_deterministic_seed(seed=seed)
        else:
            self.logger.warning("No deterministic seed provided. Execution will be stochastic.")

        self.current_offset = offset
        safe_mut_dict: Dict[str, List[Any]] = mutational_dict if mutational_dict is not None else {}

        if not target_residues:
            raise ValueError(
                "A list of target_residues (allosteric site) is strictly required for Phase 2 filtering.")

        # Phase 0: Topology and Scaffold Setup
        self._setup_directories(project_name, base_dir)
        clean_pdb_path = self._curate_scaffold(project_name, pdb_id, chain)
        self._generate_microstates(project_name, canonical_sequence, safe_mut_dict, offset)

        # Phases 1-3: ESM-2 Inference, SNR Filtering (Allo-Allo) and Sink Purge
        self._run_inference_and_filtering(project_name, target_residues, offset)

        # Phase 4: Thermodynamic Graph Preparation (Symmetrization & Diagonal Purge)
        self._build_symmetric_network(project_name)

        # Phases 5 & 6: Spatial Masking, MIST Extraction, and Centrality Computation
        self._extract_mist_and_centrality(project_name, clean_pdb_path, chain, offset)

        # Phase 7: Visualization CGO compilation
        self._compile_cgo(project_name, clean_pdb_path, chain)

        self.logger.info("=====================================================")
        self.logger.info(f"WORKFLOW COMPLETED SUCCESSFULLY FOR {project_name}")
        self.logger.info("=====================================================")


# =====================================================================
# USAGE EXECUTION BLOCK
# =====================================================================
if __name__ == "__main__":
    analyzer = AllostericNetworkAnalyzer()

    # 1. PKA Catalytic Subunit (PDB: 1ATP) - Biological Reference Data
    # It focuses exclusively on the external dynamic sensors: P+1 loop (204) (Madan et al., 2023),
    # alpha-D sensors (133,134) (Madan et al., 2023), GHI tether (280) (Taylor & Kornev, 2010),
    # and C-tail motif FDDY (327-330) (Madan et al., 2023).
    target_allosteric_site = [133, 134, 204, 280, 327, 328, 329, 330]

    pka_sequence = "VKEFLAKAKEDFLKKWETPSQNTAQLDQFDRIKTLGTGSFGRVMLVKHKESGNHYAMKILDKQKVVKLKQIEHTLNEKRILQAVNFPFLVKLEFSFKDNSNLYMVMEYVAGGEMFSHLRRIGRFSEPHARFYAAQIVLTFEYLHSLDLIYRDLKPENLLIDQQGYIQVTDFGFAKRVKGRTWTLCGTPEYLAPEIILSKGYNKAVDWWALGVLIYEMAAGYPPFFADQPIQIYEKIVSGKVRFPSHFSSDLKDLLRNLLQVDLTKRFGNLKNGVNDIKNHKWFATTDWIAIYQRKVEAPFIPKFKGPGDTSNFDDYEEEEIRVSINEKCGKEFTE"

    # 2. Mutational Probing Setup
    mutations = {
        "I150A": [["I", 150, "A"]]
    }

    # 3. Pipeline Ignition
    analyzer.execute_pipeline(
        project_name="PKA_Allostery",
        pdb_id="1ATP",
        chain="E",
        canonical_sequence=pka_sequence,
        offset=14,
        target_residues=target_allosteric_site,
        mutational_dict=mutations,
        seed=7355608
    )