"""
Dynamic Allosteric PLM Analyzer (ESM-2)
=======================================

This module implements the methodology from Dong et al. (2024) for analyzing
allosteric sensitivity in protein language model attention heads, using ESM2.

Background
----------
Attention heads in Protein Language Models (PLMs) are specialized components that learn to focus on different aspects of relationships between amino acids in a protein sequence. Each attention head acts like a spotlight that can highlight specific connections or patterns between different positions in the sequence. For example, one head might focus on nearby amino acids that form local structural elements, while another might detect long-range interactions between distant parts of the protein. In technical terms, each attention head computes a weighted sum of all positions in the sequence for each position, where the weights (attention scores) indicate how much each position should influence the current position. These attention heads are organized in layers, with each layer containing multiple heads that work in parallel to capture different types of relationships. The combined output of these attention heads helps the model understand both local and global patterns in protein sequences, which is crucial for tasks like predicting protein structure, function, or in this case, allosteric sites.

Allostery is a mechanism where binding at one site affects protein function at
another site. The paper shows that protein language models (PLMs) capture
allosteric relationships in their attention heads. This code identifies which
attention heads are most sensitive to allosteric sites.

Author
------
Carlos González Ruiz / Universidad de Málaga

References
----------
1. Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites. *bioRxiv*. DOI: https://doi.org/10.1101/2024.09.28.615583
2. Trenfield & Lin (2025). Sparse networks of conformational fluctuations communicate signals within proteins. *bioRxiv*. DOI: https://doi.org/10.1101/2025.05.28.656549
3. Allosteric Analyzer: https://github.com/amoyag/PLMs_Dynamic_Allostery & https://github.com/jdlg-42/GPCRAllostericAnalysis
4. ESM-2: https://github.com/facebookresearch/esm

Date: 2026
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
    Crystallographic PDB filter.
    Rejects solvent (HOH, WAT), ligands, and non-target chains to isolate
    the pure polypeptide scaffold.
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
    Phase 0: Structural scaffold curation and FASTA microstate generation.
    Phase 1: Zero-shot ESM-2 inference.
    Phase 2: Statistical head filtering via Monte Carlo null model (Dong et al., 2024).
    Phase 3: Consensus matrix construction, symmetrization, and covalent/sequential purge.
    Phase 4: Spatial mask (<12 Å), Thermodynamic distance scaling, and MST extraction.
    Phase 5: Topological centrality calculation and PyMOL CGO rendering.
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
        self.spatial_cutoff: float = 12.0  # Euclidean interaction threshold in Angstroms

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
            self.root_dir = current_script_dir / f"Data_{project_name}"
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
        self.logger.info(f"Phase 0: Curating spatial scaffold for {project_name} (PDB: {pdb_id}, Chain: {chain})")
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
        self.logger.info(f"Phase 0: Generating FASTA microstates for {project_name}")

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
                        f"Residue mismatch at PDB pos {pos}. Expected '{wt_aa}', found '{seq_list[rel_idx]}'."
                    )
                seq_list[rel_idx] = mut_aa

            mut_seq = "".join(seq_list)
            mut_path = self.fasta_dir / f"{project_name}_{state_name}.fasta"
            with open(mut_path, "w") as f:
                f.write(f">{project_name}_{state_name}\n{mut_seq}\n")

    # =================================================================
    # PHASE 1 & 2: ESM-2 INFERENCE AND STATISTICAL FILTERING
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
            results = model(batch_tokens, repr_layers=[33], return_contacts=False, need_head_weights=True)
            attentions = results["attentions"].squeeze(0).cpu()

        # Isolate true biological sequence by slicing out <cls> and <eos> tokens
        biological_attentions = attentions[..., 1:-1, 1:-1]
        return biological_attentions

    def _filter_allo_allo(self, attention_tensor: torch.Tensor, target_indices: List[int], sequence: str) -> Tuple[
        List[Tuple[int, int]], torch.Tensor]:
        """
        Isolates allostery-sensitive attention heads via an empirical Monte Carlo null model.
        Requires heads to exhibit an impact above the global mean, SNR > 2.0, and p-value < 0.01.
        """
        layers, heads, seq_len, _ = attention_tensor.shape
        threshold = 0.3
        n_random_trials = 1000

        head_stats = []
        non_allo_positions = np.array([i for i in range(seq_len) if i not in target_indices])
        n_allo_sites = len(target_indices)

        self.logger.info(f"   -> Executing Monte Carlo sampling across {layers * heads} attention heads...")

        total_iterations = layers * heads
        with tqdm(total=total_iterations, desc="   -> SNR Filtering", unit="head", leave=False) as pbar:
            for l in range(layers):
                for h in range(heads):
                    matrix = attention_tensor[l, h]
                    mask = matrix > threshold

                    w_total = torch.sum(matrix[mask]).item()
                    if w_total == 0:
                        pbar.update(1)
                        continue

                    w_allo = sum(torch.sum(matrix[:, site][mask[:, site]]).item() for site in target_indices)
                    p_allo = w_allo / w_total

                    random_p_values = []
                    for _ in range(n_random_trials):
                        random_sites = np.random.choice(non_allo_positions, size=n_allo_sites, replace=False)
                        w_random = sum(torch.sum(matrix[:, site][mask[:, site]]).item() for site in random_sites)
                        p_random = w_random / w_total
                        random_p_values.append(p_random)

                    expected_p_random = np.mean(random_p_values)
                    std_p_random = np.std(random_p_values)

                    # Safeguard against zero-variance backgrounds
                    if std_p_random < 1e-6:
                        impact, snr, p_val = p_allo, 0.0, 1.0
                    else:
                        impact = p_allo
                        snr = (p_allo - expected_p_random) / std_p_random
                        t_stat, p_val = ttest_1samp(random_p_values, p_allo, alternative='less')

                    head_stats.append((l, h, impact, snr, p_val))
                    pbar.update(1)

        if not head_stats:
            self.logger.warning("No attention heads surpassed the initial threshold criteria.")
            return [], torch.empty(0)

        impacts = [stat[2] for stat in head_stats]
        mean_impact = np.mean(impacts)

        df_all_heads = pd.DataFrame(head_stats, columns=["Layer", "Head", "Impact", "SNR", "P_val"])
        df_all_heads.to_csv(self.tensor_dir / "Raw_Head_Stats.csv", index=False)

        valid_heads = []
        selected_matrices = []

        # Formatted logging of statistically validated heads
        self.logger.info("   -> Summary of statistically validated heads:")
        self.logger.info(f"{'Layer':>6} | {'Head':>6} | {'Impact':>10} | {'SNR':>6} | {'p-value':>10}")
        self.logger.info("-" * 47)

        for layer, head, impact, snr, p_val in head_stats:
            if impact > mean_impact and snr > 2.0 and p_val < 0.01:
                valid_heads.append((layer, head))
                selected_matrices.append(attention_tensor[layer, head])
                self.logger.info(f"{layer:6d} | {head:6d} | {impact:10.3f} | {snr:6.2f} | {p_val:.4e}")

        if not selected_matrices:
            self.logger.warning("No attention heads survived the rigorous statistical filtering.")
            return [], torch.empty(0)

        return valid_heads, torch.stack(selected_matrices)

    def _run_inference_and_filtering(self, project_name: str, target_residues_pdb: List[int], offset: int) -> None:
        """
        Enforces isogenic probability spaces by determining the definitive valid heads
        strictly on the WT sequence. Mutants are evaluated by enforcing this native
        architecture to allow robust differential topological comparison.
        """
        self.logger.info(f"PHASE 1/2: Executing Inference and Filtering for {project_name}")

        fasta_files = list(self.fasta_dir.glob(f"{project_name}_*.fasta"))
        if not fasta_files:
            self.logger.error("No FASTA files found. Phase 0 must be executed first.")
            return

        wt_fasta = next((f for f in fasta_files if "WT" in f.name), None)
        if not wt_fasta:
            self.logger.error("WT FASTA file not found. A baseline WT sequence is mandatory.")
            return

        self.logger.info(f"Establishing native baseline architecture using: {wt_fasta.name}")
        with open(wt_fasta, "r") as f:
            wt_seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

        seq_len = len(wt_seq)
        target_idx = [p - 1 - offset for p in target_residues_pdb]
        valid_targets = [i for i in target_idx if 0 <= i < seq_len]

        if not valid_targets:
            self.logger.error("Target mapping failed. Indices out of bounds.")
            return

        # 1. Evaluate WT Baseline
        raw_wt_attention = self._extract_esm_attention(wt_seq)
        valid_heads, clean_tensor_wt = self._filter_allo_allo(raw_wt_attention, valid_targets, wt_seq)

        if len(valid_heads) == 0:
            self.logger.error("WT sequence failed to produce valid attention heads. Pipeline aborted.")
            return

        self.logger.info(f"   -> Final native architecture: {len(valid_heads)} functional heads retained.")
        torch.save(clean_tensor_wt, self.tensor_dir / f"clean_attention_{project_name}_WT.pt")

        # 2. Process Mutants enforcing WT Architecture
        mutant_fastas = [f for f in fasta_files if "WT" not in f.name]
        for mut_fasta in mutant_fastas:
            state_name = mut_fasta.stem.replace(f"{project_name}_", "")
            self.logger.info(f"Processing Mutant State: {state_name} (Enforcing WT topology)")

            with open(mut_fasta, "r") as f:
                mut_seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

            raw_mut_attention = self._extract_esm_attention(mut_seq)

            selected_mut_matrices = []
            for (l, h) in valid_heads:
                selected_mut_matrices.append(raw_mut_attention[l, h])

            mut_tensor = torch.stack(selected_mut_matrices)
            out_path = self.tensor_dir / f"clean_attention_{project_name}_{state_name}.pt"
            torch.save(mut_tensor, out_path)
            self.logger.info(f"   -> Mutant tensor saved: {out_path.name}")

    # =================================================================
    # PHASE 3: CONSENSUS AND SYMMETRIZATION
    # =================================================================
    def _build_symmetric_network(self, project_name: str) -> None:
        """
        Collapses attention matrices into a symmetric consensus graph.
        Applies a sequential filter (|i-j| >= 5) to eradicate covalent and
        local secondary structure bias.
        """
        self.logger.info(f"PHASE 3: Executing Symmetrization and Covalent Purge for {project_name}")

        tensor_files = list(self.tensor_dir.glob(f"clean_attention_{project_name}_*.pt"))
        for tensor_path in tensor_files:
            state_name = tensor_path.stem.replace(f"clean_attention_{project_name}_", "")

            clean_tensor = torch.load(tensor_path, weights_only=True)
            consensus_matrix = torch.mean(clean_tensor, dim=0).cpu().numpy()

            # Algebraic symmetrization for undirected graph analysis
            symmetric_matrix = (consensus_matrix + consensus_matrix.T) / 2.0

            seq_len = symmetric_matrix.shape[0]
            sequence_distance_matrix = np.abs(np.arange(seq_len)[:, None] - np.arange(seq_len))
            neighborhood_mask = (sequence_distance_matrix >= 5).astype(float)

            final_network = symmetric_matrix * neighborhood_mask
            out_path = self.tensor_dir / f"undirected_network_{project_name}_{state_name}.npy"
            np.save(out_path, final_network)

            max_w = np.max(final_network)
            self.logger.info(f"   -> State {state_name}: Matrix saved with max edge weight = {max_w:.6f}")

    # =================================================================
    # PHASE 4: SPATIAL MASK, THERMODYNAMIC DISTANCE, AND MST
    # =================================================================
    def _extract_mist_and_centrality(self, project_name: str, pdb_path: str, chain_id: str, offset: int) -> None:
        """
        Applies a Euclidean spatial constraint (<12 Å), performs a logarithmic
        transformation to derive thermodynamic distances, and extracts the Minimum
        Spanning Tree (MST) as a proxy for the optimal communication network.
        """
        self.logger.info(f"PHASE 4: Extracting Spatial MST and Centralities for {project_name}")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("scaffold", pdb_path)

        coords_dict = {}
        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
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

            # Euclidean Distance Masking
            dist_matrix = cdist(coords_array, coords_array, metric='euclidean')
            dist_matrix[np.isnan(dist_matrix)] = np.inf
            spatial_mask = (dist_matrix <= self.spatial_cutoff).astype(float)

            physical_matrix = symmetric_matrix * spatial_mask

            G = nx.Graph()
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    prob = physical_matrix[i, j]
                    if prob > 0:
                        # Shannon/Thermodynamic distance conversion
                        thermo_dist = -np.log(prob)
                        G.add_edge(i, j, weight=thermo_dist, probability=prob)

            if G.number_of_edges() > 0:
                # Minimum Spanning Tree extraction
                mst = nx.minimum_spanning_tree(G, weight='weight')
                centrality = nx.betweenness_centrality(mst, weight='weight', normalized=True)

                fasta_path = self.fasta_dir / f"{project_name}_{state_name}.fasta"
                with open(fasta_path, 'r') as f:
                    seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

                nodes_data = [{"Residue_PDB": k + 1 + offset, "Amino_Acid": seq[k], "Betweenness_Centrality": v}
                              for k, v in centrality.items()]
                edges_data = [{"Source_PDB": u + 1 + offset, "Target_PDB": v + 1 + offset,
                               "Distance": d['weight'], "Probability": d['probability']}
                              for u, v, d in mst.edges(data=True)]

                df_nodes = pd.DataFrame(nodes_data).sort_values(by="Betweenness_Centrality", ascending=False)
                df_edges = pd.DataFrame(edges_data).sort_values(by="Probability", ascending=False)

                if not df_nodes.empty:
                    threshold_95 = df_nodes["Betweenness_Centrality"].quantile(0.95)
                    df_nodes["Is_Bottleneck"] = df_nodes["Betweenness_Centrality"] >= threshold_95
                    self.logger.info(f"      - Percentile 95 Threshold: {threshold_95:.4f}")
                    self.logger.info(f"      - Identified {df_nodes['Is_Bottleneck'].sum()} critical bottlenecks.")

                df_nodes.to_csv(self.graph_dir / f"Centrality_{project_name}_{state_name}.csv", index=False)
                df_edges.to_csv(self.graph_dir / f"Edges_{project_name}_{state_name}.csv", index=False)

    # =================================================================
    # PHASE 5: CGO COMPILATION FOR PYMOL
    # =================================================================
    @staticmethod
    def _get_color_gradient(value: float, vmin: float, vmax: float) -> Tuple[float, float, float]:
        """Maps quantitative metrics to an RGB thermal gradient (Blue -> Yellow -> Red)."""
        if vmax == vmin: return 0.0, 0.0, 1.0
        norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        if norm < 0.5:
            return 2.0 * norm, 2.0 * norm, 1.0 - 2.0 * norm
        return 1.0, 1.0 - 2.0 * (norm - 0.5), 0.0

    def _compile_cgo(self, project_name: str, pdb_path: str, chain: str) -> None:
        """Renders the MST applying polynomial scaling for sparse network visualization."""
        self.logger.info("PHASE 5: Compiling Sparse Networks (PyMOL CGO Render)")

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("scaffold", pdb_path)
        coords = {res.id[1]: res['CA'].get_coord() for res in structure[0][chain] if 'CA' in res and res.id[0] == ' '}

        centrality_files = list(self.graph_dir.glob(f"Centrality_{project_name}_*.csv"))

        for cent_file in centrality_files:
            state_name = cent_file.stem.replace(f"Centrality_{project_name}_", "")
            edges_file = self.graph_dir / f"Edges_{project_name}_{state_name}.csv"

            if not edges_file.exists(): continue
            df_nodes = pd.read_csv(cent_file)
            df_edges = pd.read_csv(edges_file)
            if df_nodes.empty or df_edges.empty: continue

            max_prob = df_edges["Probability"].max()
            max_cent = df_nodes["Betweenness_Centrality"].max()

            script_lines = [
                "from pymol.cgo import *",
                "from pymol import cmd",
                "cmd.reinitialize()",
                "cmd.bg_color('white')",
                f"cmd.load(r'{Path(pdb_path).resolve()}', '{project_name}_scaffold')",
                f"cmd.show_as('cartoon', '{project_name}_scaffold')",
                f"cmd.color('gray80', '{project_name}_scaffold')",
                f"cmd.set('cartoon_transparency', 0.55, '{project_name}_scaffold')",
                "obj = []"
            ]

            # Edges: Quadratic exponential decay to thickness
            for _, row in df_edges.iterrows():
                u, v = int(row["Source_PDB"]), int(row["Target_PDB"])
                if u in coords and v in coords:
                    c1, c2 = coords[u], coords[v]
                    norm_prob = float(row["Probability"]) / max_prob if max_prob > 0 else 0
                    r, g, b = self._get_color_gradient(norm_prob, 0.0, 1.0)
                    thickness = 0.03 + (norm_prob ** 2) * 0.45
                    script_lines.append(
                        f"obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, {c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {thickness:.3f}, {r:.2f}, {g:.2f}, {b:.2f}, {r:.2f}, {g:.2f}, {b:.2f}])")

            # Nodes: Cubic expansion to radius
            for _, row in df_nodes.iterrows():
                node = int(row["Residue_PDB"])
                if node in coords:
                    c = coords[node]
                    norm_cent = float(row["Betweenness_Centrality"]) / max_cent if max_cent > 0 else 0
                    r, g, b = self._get_color_gradient(norm_cent, 0.0, 1.0)
                    radius = 0.4 + (norm_cent ** 3) * 2.2
                    script_lines.append(f"obj.extend([COLOR, {r:.2f}, {g:.2f}, {b:.2f}])")
                    script_lines.append(f"obj.extend([SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {radius:.3f}])")

            script_lines.append(f"cmd.load_cgo(obj, '{project_name}_{state_name}_Pathway')")

            cgo_path = self.cgo_dir / f"Render_CGO_{project_name}_{state_name}.py"
            with open(cgo_path, "w") as f:
                f.write("\n".join(script_lines))

            self.logger.info(f"   -> CGO Script generated: {cgo_path.name}")

    # =================================================================
    # PIPELINE ORCHESTRATOR
    # =================================================================
    def execute_pipeline(self, project_name: str, pdb_id: str, chain: str, canonical_sequence: str, offset: int,
                         target_residues: List[int], mutational_dict: Optional[Dict[str, List[Any]]] = None,
                         base_dir: Optional[str] = None, seed: Optional[int] = 42) -> None:
        """
        Triggers the absolute sequential execution of the network analysis.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING ALLOSTERIC PIPELINE: {project_name}")
        self.logger.info("=" * 60)

        if seed is not None:
            self._set_deterministic_seed(seed=seed)

        self.current_offset = offset
        safe_mut_dict = mutational_dict or {}

        if not target_residues:
            raise ValueError("A list of target residues (allosteric site) is strictly required.")

        self._setup_directories(project_name, base_dir)
        clean_pdb_path = self._curate_scaffold(project_name, pdb_id, chain)
        self._generate_microstates(project_name, canonical_sequence, safe_mut_dict, offset)
        self._run_inference_and_filtering(project_name, target_residues, offset)
        self._build_symmetric_network(project_name)
        self._extract_mist_and_centrality(project_name, clean_pdb_path, chain, offset)
        self._compile_cgo(project_name, clean_pdb_path, chain)

        self.logger.info("=" * 60)
        self.logger.info(f"WORKFLOW COMPLETED SUCCESSFULLY FOR {project_name}")
        self.logger.info("=" * 60)


# =====================================================================
# COMMAND LINE INTERFACE (CLI)
# =====================================================================
#def main():
#    parser = argparse.ArgumentParser(
#        description="Dynamic Allosteric Network Analyzer based on PLMs (ESM-2).",
#        formatter_class=argparse.ArgumentDefaultsHelpFormatter
#    )
#    parser.add_argument("--project", type=str, default="PKA_Allostery", help="Name of the analysis project.")
#    parser.add_argument("--pdb", type=str, default="1ATP", help="PDB ID to download as structural scaffold.")
#    parser.add_argument("--chain", type=str, default="E", help="Target chain from the PDB structure.")
#    parser.add_argument("--offset", type=int, default=14, help="Sequence offset relative to PDB numbering.")
#    parser.add_argument("--seed", type=int, default=7355608, help="Deterministic seed for Monte Carlo sampling.")

#    args = parser.parse_args()

    # Biological defaults for PKA if executed without custom arguments
#    pka_sequence = "VKEFLAKAKEDFLKKWETPSQNTAQLDQFDRIKTLGTGSFGRVMLVKHKESGNHYAMKILDKQKVVKLKQIEHTLNEKRILQAVNFPFLVKLEFSFKDNSNLYMVMEYVAGGEMFSHLRRIGRFSEPHARFYAAQIVLTFEYLHSLDLIYRDLKPENLLIDQQGYIQVTDFGFAKRVKGRTWTLCGTPEYLAPEIILSKGYNKAVDWWALGVLIYEMAAGYPPFFADQPIQIYEKIVSGKVRFPSHFSSDLKDLLRNLLQVDLTKRFGNLKNGVNDIKNHKWFATTDWIAIYQRKVEAPFIPKFKGPGDTSNFDDYEEEEIRVSINEKCGKEFTE"
#    target_site = [133, 134, 204, 280, 327, 328, 329, 330]
#    mutations = {"I150A": [["I", 150, "A"]]}

#    analyzer = AllostericNetworkAnalyzer()
#    analyzer.execute_pipeline(
#        project_name=args.project,
#        pdb_id=args.pdb,
#        chain=args.chain,
#        canonical_sequence=pka_sequence,
#        offset=args.offset,
#        target_residues=target_site,
#        mutational_dict=mutations,
#        seed=args.seed
#    )


#if __name__ == "__main__":
#    main()