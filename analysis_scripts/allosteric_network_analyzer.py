"""
Dynamic Allosteric PLM Analyzer
======================================================
A fully integrated bioinformatic framework for the mapping of dynamic allostery
using Protein Language Models (ESM-2) and Maximum Spanning Tree (MST) Graph Theory.

1. Spatial Scaffold Curation (PDB purification)
2. Genetic Microstate Generation (canonical FASTA sequence and in silico mutagenesis)
3. Epistatic Inference Engine (ESM-2 Mutational Sensitivity Tensors via JSD)
4. MST Topological Extraction (Graph Centrality with absolute PDB indexing and AA mapping)
5. High-Res CGO Topology Compilation (PyMOL rendering)
6. Quantitative Analytics (Long-range epistasis and target sensitivity profiling)
7. Attention Map Sensitivity (Extraction of PLM internal representations)

 Example
 -------
from allosteric_network_analyzer import AllostericNetworkAnalyzer

if __name__ == "__main__":
    analyzer = AllostericNetworkAnalyzer()

    # Define your biological parameters
    analyzer.execute_pipeline(
        project_name="Target_Name",         # e.g., "PKA"
        pdb_id="XXXX",                      # e.g., "1ATP"
        chain="X",                          # e.g., "E"
        canonical_sequence="SEQ...",        # Insert continuous 1D amino acid string
        offset=0,                           # Shift to map 0-indexed array to PDB numbering
        mutational_dict={
            "Mutant_1": ["I150A"],          # In silico perturbations (Optional)
        },
        target_residues=[50, 75, 231],    # Nodes for directed epistatic & attention sensitivity
        seed=42 (or None),
        base_dir="."                        # Directory for data serialization
    )

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
 Carlos González Ruiz

 References
 ----------
 Dong et al. (2024). Allo-Allo: Data-efficient prediction of allosteric sites.
 bioRxiv. DOI: https://doi.org/10.1101/2024.09.28.615583

 GPCRAllostericAnalysis. https://github.com/jdlg-42/GPCRAllostericAnalysis

"""

import os
import glob
import logging
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from Bio.PDB import PDBList, PDBParser, PDBIO, Select
from Bio.PDB.PDBExceptions import PDBConstructionWarning

try:
    import esm
except ImportError:
    raise ImportError("The 'esm' library is strictly required. Install via: pip install fair-esm")

try:
    from scipy.stats import ttest_1samp
except ImportError:
    raise ImportError("The 'scipy' library is required for attention stats. Install via: pip install scipy")

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
    Rejects solvent (HOH, WAT), ligands, and non-target chains.
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
    Unified class orchestrating the complete dynamic allosteric network workflow.
    It manages dynamic directory topology, spatial scaffold curation, mutational
    sensitivity inference via ESM-2, MST graph theory topology, 3D PyMOL compilations,
    and differential attention deconvolution.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger("AllostericAnalyzer")
        self.root_dir: str = ""
        self.pdb_dir: str = ""
        self.fasta_dir: str = ""
        self.tensor_dir: str = ""
        self.graph_dir: str = ""
        self.cgo_dir: str = ""
        self.analytics_dir: str = ""
        self.plots_dir: str = ""
        self.current_offset: int = 0

    def _set_deterministic_seed(self, seed: int = 42) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.logger.info(f"Deterministic PRNG seed anchored globally at: {seed}")

    def _setup_directories(self, project_name: str, base_dir: Optional[str]) -> None:
        """
        Dynamically constructs the data topology in the project root directory.
        """
        if base_dir is None:
            current_script_dir = str(os.path.dirname(os.path.abspath(__file__)))
            safe_base_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
            self.root_dir = os.path.join(safe_base_dir, f"Data_{project_name}")
        else:
            self.root_dir = os.path.abspath(str(base_dir))

        self.pdb_dir = os.path.join(self.root_dir, "processed_pdb")
        self.fasta_dir = os.path.join(self.root_dir, "fasta_sequences")
        self.tensor_dir = os.path.join(self.root_dir, "results")
        self.graph_dir = os.path.join(self.root_dir, "graph_centrality")
        self.cgo_dir = os.path.join(self.root_dir, "pymol_cgo_scripts")
        self.analytics_dir = os.path.join(self.root_dir, "quantitative_metrics")

        for directory in [self.pdb_dir, self.fasta_dir, self.tensor_dir, self.graph_dir,
                          self.cgo_dir, self.analytics_dir]:
            os.makedirs(directory, exist_ok=True)

        self.logger.info(f"Directory topology dynamically configured at: {self.root_dir}")

    def execute_pipeline(self, project_name: str, pdb_id: str, chain: str, canonical_sequence: str, offset: int,
                         mutational_dict: Optional[Dict[str, List[str]]] = None,
                         target_residues: Optional[List[int]] = None,
                         base_dir: Optional[str] = None,
                         seed: Optional[int] = 42) -> None:
        """
        Triggers the absolute execution of the 7-stage analysis logic.
        """
        self.logger.info("=====================================")
        self.logger.info(f"STARTING  ANALYSIS: {project_name}")
        self.logger.info("=====================================")

        if seed is not None:
            self._set_deterministic_seed(seed=seed)
        else:
            self.logger.warning("No deterministic seed provided. Monte Carlo execution will be fully stochastic.")

        self.current_offset = offset
        safe_mut_dict: Dict[str, List[str]] = mutational_dict if mutational_dict is not None else {}
        safe_target_res: List[int] = target_residues if target_residues is not None else []

        self._setup_directories(project_name, base_dir)

        clean_pdb_path = self._curate_scaffold(project_name, pdb_id, chain)
        self._generate_microstates(project_name, canonical_sequence, safe_mut_dict, offset)
        self._run_inference(project_name)
        self._extract_mst(project_name, offset)
        self._compile_cgo(project_name, clean_pdb_path, chain)
        self._run_quantitative_analytics(project_name, clean_pdb_path, chain, offset, safe_target_res)
        self._run_attention_analytics(project_name, offset, safe_target_res)

        self.logger.info("===================================================")
        self.logger.info(f"WORKFLOW COMPLETED SUCCESSFULLY FOR {project_name}")
        self.logger.info("===================================================")

    # -----------------------------------------------------------------
    # STAGE 1 & 2: SCAFFOLD CURATION AND SEQUENCE GENERATION
    # -----------------------------------------------------------------
    def _curate_scaffold(self, project_name: str, pdb_id: str, chain: str) -> str:
        self.logger.info(f"STAGE 1: Executing Spatial Curation for {project_name} ({pdb_id}_{chain})")
        clean_path = os.path.join(self.pdb_dir, f"{project_name}_scaffold_clean.pdb")

        if os.path.exists(clean_path):
            self.logger.info("   -> Curated scaffold already exists. Bypassing spatial processing.")
            return clean_path

        pdbl = PDBList(pdb=self.pdb_dir)
        raw_pdb_path = pdbl.retrieve_pdb_file(pdb_id, pdir=self.pdb_dir, file_format="pdb")
        parser = PDBParser(QUIET=True)

        try:
            # Parse the raw crystallographic coordinates into a hierarchical structure
            structure = parser.get_structure(project_name, raw_pdb_path)
            io = PDBIO()
            io.set_structure(structure)

            # Isolate the target peptidic chain and discard heteroatoms/solvent
            io.save(clean_path, ChainAndProteinSelect(target_chain=chain))
        except Exception as e:
            self.logger.error(f"Crystallographic parsing failed: {e}")
            raise

        return clean_path

    def _generate_microstates(self, project_name: str, canonical_seq: str, mutational_dict: Dict[str, List[str]],
                              offset: int) -> None:
        self.logger.info(f"STAGE 2: Generating Genetic Microstates for {project_name}")

        wt_path = os.path.join(self.fasta_dir, f"{project_name}_WT.fasta")
        with open(wt_path, "w") as f:
            f.write(f">{project_name}_WT\n{canonical_seq}\n")

        for state_name, mutations in mutational_dict.items():
            seq_list = list(canonical_seq)

            for mut in mutations:
                wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
                # Transform biological PDB numbering to 0-indexed Python array
                rel_idx = pos - 1 - offset
                seq_list[rel_idx] = mut_aa

            mut_seq = "".join(seq_list)

            with open(os.path.join(self.fasta_dir, f"{project_name}_{state_name}.fasta"), "w") as f:
                f.write(f">{project_name}_{state_name}\n{mut_seq}\n")

    # -----------------------------------------------------------------
    # STAGE 3: EPISTATIC INFERENCE ENGINE
    # -----------------------------------------------------------------
    def _run_inference(self, project_name: str) -> None:
        self.logger.info("STAGE 3: Initiating Epistatic Inference Engine (ESM-2)")
        fasta_files = glob.glob(os.path.join(self.fasta_dir, f"{project_name}_*.fasta"))

        if not fasta_files:
            return

        all_exist = all(
            os.path.exists(
                os.path.join(self.tensor_dir, f"EpistaticTensor_{os.path.basename(fp).replace('.fasta', '')}.npy"))
            for fp in fasta_files)
        if all_exist:
            self.logger.info("   -> All epistatic tensors already exist. Skipping ESM-2 model initialization.")
            return

        # Multi-platform Hardware Allocation (NVIDIA CUDA -> Apple MPS -> CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        self.logger.info(f"   -> Loading PLM onto hardware: {device}...")

        # Initialize the ESM-2 Transformer (650M parameter variant, 33 layer)
        model_instance, alphabet_instance = esm.pretrained.esm2_t33_650M_UR50D()
        model_instance.eval()
        model_instance.to(device)
        batch_converter_func = alphabet_instance.get_batch_converter()
        num_layers_int: int = 33

        for file_path in fasta_files:
            filename = os.path.basename(file_path).replace(".fasta", "")
            out_tensor = os.path.join(self.tensor_dir, f"EpistaticTensor_{filename}.npy")
            if os.path.exists(out_tensor):
                continue

            self.logger.info(f"   -> Computing Deep Mutational Scan for: {filename}...")
            with open(file_path, 'r') as f:
                seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

            seq_len = len(seq)
            # Initialize the directional mutational sensitivity matrix (N x N)
            epistatic_tensor = np.zeros((seq_len, seq_len))
            data = [("WT", seq)]
            _, _, batch_tokens = batch_converter_func(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model_instance(batch_tokens, repr_layers=[num_layers_int], return_contacts=True)

                # Extract logits and slice array [1:seq_len+1] to discard <cls> and <eos> tokens
                logits_wt = results["logits"][0, 1:seq_len + 1, :].cpu().numpy()

                # Apply softmax function to normalize logits into a bounded probability distribution
                probs_wt = np.exp(logits_wt) / np.sum(np.exp(logits_wt), axis=-1, keepdims=True)

                # Extract and serialize the internal attention maps for Stage 7
                attention_maps = results["attentions"]
                torch.save(attention_maps, os.path.join(self.tensor_dir, f"Attention_{filename}.pt"))
                self.logger.info(f"      -> Attention maps extracted and serialized: Attention_{filename}.pt")

            # In Silico Mutagenesis (Alanine Scanning)
            for i in tqdm(range(seq_len), desc=f"Scanning {filename}", unit="residue"):
                mut_aa = 'A' if seq[i] != 'A' else 'G'
                mut_seq = seq[:i] + mut_aa + seq[i + 1:]
                _, _, mut_tokens = batch_converter_func([("MUT", mut_seq)])
                mut_tokens = mut_tokens.to(device)

                with torch.no_grad():
                    res_mut = model_instance(mut_tokens, repr_layers=[], return_contacts=False)
                    logits_mut = res_mut["logits"][0, 1:seq_len + 1, :].cpu().numpy()
                    probs_mut = np.exp(logits_mut) / np.sum(np.exp(logits_mut), axis=-1, keepdims=True)

                # Compute Jensen-Shannon Divergence (JSD) to quantify directional epistatic perturbation
                # The 1e-10 constant prevents log(0) computational underflow errors.
                m = 0.5 * (probs_wt + probs_mut)
                kl_wt_m = np.sum(probs_wt * np.log(probs_wt / (m + 1e-10)), axis=-1)  # (eq. 2 for WT)
                kl_mut_m = np.sum(probs_mut * np.log(probs_mut / (m + 1e-10)), axis=-1)  # (eq. 2 for mut)
                epistatic_tensor[i, :] = 0.5 * (kl_wt_m + kl_mut_m)  # (eq. 1)

            np.save(out_tensor, epistatic_tensor)

    # -----------------------------------------------------------------
    # STAGE 4, 5 & 6: MST, CGO, AND QUANTITATIVE ANALYTICS
    # -----------------------------------------------------------------
    def _extract_mst(self, project_name: str, offset: int) -> None:
        self.logger.info("STAGE 4: Extracting Maximum Spanning Tree Topologies (Absolute Indexing)")
        for file_path in glob.glob(os.path.join(self.tensor_dir, f"EpistaticTensor_{project_name}_*.npy")):
            state = os.path.basename(file_path).replace(f"EpistaticTensor_{project_name}_", "").replace(".npy", "")

            fasta_path = os.path.join(self.fasta_dir, f"{project_name}_{state}.fasta")
            if not os.path.exists(fasta_path):
                self.logger.error(f"   -> Cannot extract AA names: Sequence file {fasta_path} missing.")
                continue

            with open(fasta_path, 'r') as f:
                seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

            epistatic_tensor = np.load(file_path)

            # Symmetrize the tensor to construct an undirected graph. This assumes the
            # biophysical principle of allosteric reciprocity (the energetic coupling
            # pathway from node i to j is identical to j to i in equilibrium). (Eq. 3)
            sym_epistatic_tensor = (epistatic_tensor + epistatic_tensor.T) / 2.0

            # Eliminate self-loops to prevent artificial inflation of node centrality.
            np.fill_diagonal(sym_epistatic_tensor, 0.0)

            graph = nx.Graph()
            graph.add_nodes_from(range(epistatic_tensor.shape[0]))
            rows, cols = np.where(sym_epistatic_tensor> 0)
            for i, j in zip(rows, cols):
                if i < j:
                    graph.add_edge(i, j, weight=sym_epistatic_tensor[i, j])

            if graph.number_of_edges() > 0:
                mst = nx.maximum_spanning_tree(graph, weight='weight')
                centrality = nx.betweenness_centrality(mst, normalized=True) # (eq. 4)

                nodes_data = [{"Residue_PDB": k + 1 + offset,
                               "Amino_Acid": seq[k],
                               "Betweenness_Centrality": v}
                              for k, v in centrality.items()]

                edges_data = [{"Source_PDB": u + 1 + offset, "Source_AA": seq[u],
                               "Target_PDB": v + 1 + offset, "Target_AA": seq[v],
                               "Weight": w}
                              for u, v, w in mst.edges(data='weight')]

                pd.DataFrame(nodes_data).sort_values(by="Betweenness_Centrality", ascending=False).to_csv(
                    os.path.join(self.graph_dir, f"Centrality_{project_name}_{state}.csv"), index=False)
                pd.DataFrame(edges_data).sort_values(by="Weight", ascending=False).to_csv(
                    os.path.join(self.graph_dir, f"Edges_{project_name}_{state}.csv"), index=False)

    @staticmethod
    def _get_color_gradient(value: float, vmin: float, vmax: float) -> Tuple[float, float, float]:
        if vmax == vmin:
            return 1.0, 1.0, 0.0
        norm = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        if norm < 0.5:
            return 1.0 - 0.2 * norm * 2, 1.0 - 0.8 * norm * 2, 0.6 * norm * 2
        else:
            return 0.8 - 0.5 * (norm - 0.5) * 2, 0.2 - 0.2 * (norm - 0.5) * 2, 0.6 - 0.1 * (norm - 0.5) * 2

    def _compile_cgo(self, project_name: str, pdb_path: str, chain: str) -> None:
        self.logger.info("STAGE 5: Compiling Multi-Dimensional CGO Graphics")
        abs_pdb_path = os.path.abspath(pdb_path)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("scaffold", abs_pdb_path)

        # Apply a coarse-grained approximation using only Carbon Alpha (CA) coordinates.
        # This mitigates spatial noise from highly flexible side-chain rotamers.
        coords = {res.id[1]: res['CA'].get_coord() for res in structure[0][chain] if 'CA' in res and res.id[0] == ' '}

        from pathlib import Path

        for cent_path_str in glob.glob(os.path.join(self.graph_dir, f"Centrality_{project_name}_*.csv")):
            state = os.path.basename(cent_path_str).replace(f"Centrality_{project_name}_", "").replace(".csv", "")
            edges_path_str = os.path.join(self.graph_dir, f"Edges_{project_name}_{state}.csv")

            if not os.path.exists(edges_path_str):
                continue

            cent_file = Path(cent_path_str)
            edges_file = Path(edges_path_str)

            raw_nodes = pd.read_csv(cent_file, engine="python")
            raw_edges = pd.read_csv(edges_file, engine="python")

            if not isinstance(raw_nodes, pd.DataFrame) or not isinstance(raw_edges, pd.DataFrame):
                continue

            df_nodes: pd.DataFrame = raw_nodes
            df_edges: pd.DataFrame = raw_edges

            max_weight = df_edges["Weight"].max() if not df_edges.empty else 1.0
            max_cent = df_nodes["Betweenness_Centrality"].max() if not df_nodes.empty else 1.0

            script_lines = [
                "from pymol.cgo import *", "from pymol import cmd", "cmd.reinitialize()", "cmd.bg_color('white')",
                f"cmd.load(r'{abs_pdb_path}', '{project_name}_scaffold')",
                f"cmd.show_as('cartoon', '{project_name}_scaffold')",
                f"cmd.color('gray80', '{project_name}_scaffold')",
                f"cmd.set('cartoon_transparency', 0.65, '{project_name}_scaffold')", "obj = []"
            ]

            for _, row in df_edges.iterrows():
                abs_u, abs_v = int(row["Source_PDB"]), int(row["Target_PDB"])
                if abs_u in coords and abs_v in coords:
                    c1, c2 = coords[abs_u], coords[abs_v]
                    w = float(row["Weight"])
                    r, g, b = self._get_color_gradient(w, 0.0, max_weight)
                    script_lines.append(
                        f"obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, {c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {0.05 + (w / max_weight) * 0.3:.3f}, {r:.2f}, {g:.2f}, {b:.2f}, {r:.2f}, {g:.2f}, {b:.2f}])")

            for _, row in df_nodes.iterrows():
                abs_idx = int(row["Residue_PDB"])
                if abs_idx in coords:
                    c = coords[abs_idx]
                    val = float(row["Betweenness_Centrality"])
                    r, g, b = self._get_color_gradient(val, 0.0, max_cent)
                    script_lines.append(f"obj.extend([COLOR, {r:.2f}, {g:.2f}, {b:.2f}])")
                    script_lines.append(
                        f"obj.extend([SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {0.4 + (val / max_cent) * 2.8:.3f}])")

            script_lines.append(f"cmd.load_cgo(obj, '{project_name}_{state}_Topology')")
            with open(os.path.join(self.cgo_dir, f"Render_CGO_{project_name}_{state}.py"), "w") as f:
                f.write("\n".join(script_lines))

    def _run_quantitative_analytics(self, project_name: str, pdb_path: str, chain: str, offset: int,
                                    target_residues: List[int]) -> None:
        self.logger.info("STAGE 6: Extracting Quantitative Allosteric Metrics")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("scaffold", os.path.abspath(pdb_path))

        coords = {res.id[1] - 1 - offset: res['CA'].get_coord() for res in structure[0][chain] if
                  'CA' in res and res.id[0] == ' '}

        for file_path in glob.glob(os.path.join(self.tensor_dir, f"EpistaticTensor_{project_name}_*.npy")):
            state = os.path.basename(file_path).replace(f"EpistaticTensor_{project_name}_", "").replace(".npy", "")

            fasta_path = os.path.join(self.fasta_dir, f"{project_name}_{state}.fasta")
            with open(fasta_path, 'r') as f:
                seq = "".join([line.strip() for line in f.readlines() if not line.startswith(">")])

            epistatic_tensor = np.load(file_path)

            for target in target_residues:
                target_rel_idx = target - 1 - offset
                if 0 <= target_rel_idx < epistatic_tensor.shape[0]:
                    df_target = pd.DataFrame({"Residue_PDB": np.arange(epistatic_tensor.shape[1]) + 1 + offset,
                                              "Amino_Acid": list(seq),
                                              "Epistatic_Coupling_JSD": epistatic_tensor[target_rel_idx, :]})
                    df_target = df_target[df_target["Residue_PDB"] != target].sort_values(
                        by="Epistatic_Coupling_JSD", ascending=False)
                    df_target.to_csv(os.path.join(self.analytics_dir, f"DirectedSensitivity_{target}_{state}.csv"),
                                     index=False)

            long_range_data = []
            sym_epsitatic_tensor = (epistatic_tensor + epistatic_tensor.T) / 2.0
            rows, cols = np.triu_indices(sym_epsitatic_tensor.shape[0], k=1)
            for u, v in zip(rows, cols):
                abs_u = u + 1 + offset
                abs_v = v + 1 + offset
                if u in coords and v in coords:
                    dist = np.linalg.norm(coords[u] - coords[v])
                    # Strict > 15 Angstrom filter. This guarantees that only true
                    # dynamic allosteric communication is recorded, discarding primary and
                    # secondary coordination spheres (e.g., hydrogen bonds, sterics).
                    if dist >= 15.0:
                        long_range_data.append(
                            {"Source_PDB": abs_u, "Source_AA": seq[u],
                             "Target_PDB": abs_v, "Target_AA": seq[v],
                             "Distance_Angstroms": dist, "Epistatic_Coupling_JSD": sym_epsitatic_tensor[u, v]})

            if long_range_data:
                pd.DataFrame(long_range_data).sort_values(by="Epistatic_Coupling_JSD", ascending=False).to_csv(
                    os.path.join(self.analytics_dir, f"LongRange_Allostery_{state}.csv"), index=False)

    # -----------------------------------------------------------------
    # STAGE 7: ATTENTION MAPS SENSITIVITY
    # -----------------------------------------------------------------
    @staticmethod
    def _compute_attention_impact(
            attention_maps: torch.Tensor,
            allo_sites_relative: List[int],
            n_random_trials: int = 1000,
            threshold: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates absolute allosteric sensitivity of PLM attention heads.
        Algorithm logic derived from Dong et al. (2024), fully optimized via
        continuous vectorized tensor operations and sampling without replacement.
        """
        target_device = attention_maps.device
        attention_maps = attention_maps[0]
        n_allo_sites = len(allo_sites_relative)
        num_layers, num_heads, seq_len, _ = attention_maps.shape

        impacts = torch.zeros((num_layers, num_heads))
        snrs = torch.zeros((num_layers, num_heads))
        pvals = torch.ones((num_layers, num_heads))

        # Definition of the background space (non-allosteric residues)
        non_allo_list = [i for i in range(seq_len) if i not in allo_sites_relative]

        # Hardware synchronization: Instantiate tensors on the correct target device
        non_allo_tensor = torch.tensor(non_allo_list, dtype=torch.long, device=target_device)
        allo_sites_tensor = torch.tensor(allo_sites_relative, dtype=torch.long, device=target_device)

        total_heads = num_layers * num_heads

        with tqdm(total=total_heads, desc="Analyzing Absolute Attention (Vectorized)", unit="head") as pbar:
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    attention = attention_maps[layer_idx, head_idx]

                    # Apply a conservative probability threshold (>30%) to filter out
                    # diffuse background attention and isolate highly confident steric couplings.
                    mask = attention > threshold

                    # Vectorization Level 1: Directional summation of masked attention
                    col_sums = torch.sum(attention * mask, dim=0)

                    # Direct computation of the attention coupled to the allosteric sites (eq. 5)
                    w_allo = torch.sum(col_sums[allo_sites_tensor]).item()

                    # Extraction of the background noise summation pool
                    background_sums = col_sums[non_allo_tensor]

                    # Vectorization Level 2: Monte Carlo simulation WITHOUT replacement.
                    # Guarantees absolute statistical rigor by preventing duplicate indices.
                    rand_weights = torch.rand(n_random_trials, len(non_allo_list), device=target_device)
                    rand_idx = torch.argsort(rand_weights, dim=1)[:, :n_allo_sites]

                    # Transversal summation to generate the complete null distribution
                    random_w_tensor = torch.sum(background_sums[rand_idx], dim=1)

                    expected_random = torch.mean(random_w_tensor).item()
                    std_random = torch.std(random_w_tensor).item()

                    # Inference of impact metrics (eq. 6,7)
                    impact = w_allo / expected_random if expected_random > 0 else 0.0
                    snr = (w_allo - expected_random) / (std_random + 1e-10)

                    # Hypothesis testing (one-tailed t-test)
                    t_stat, p_value = ttest_1samp(
                        random_w_tensor.cpu().numpy(),
                        w_allo,
                        alternative='less'
                    )

                    if np.isnan(p_value):
                        p_value = 1.0

                    # Explicit casting to Python native float
                    impacts[layer_idx, head_idx] = float(impact)
                    snrs[layer_idx, head_idx] = float(snr)
                    pvals[layer_idx, head_idx] = float(p_value)

                    pbar.update(1)

        return impacts, snrs, pvals

    @staticmethod
    def _compute_differential_attention_impact(
            attention_maps_wt: torch.Tensor,
            attention_maps_mut: torch.Tensor,
            allo_sites_relative: List[int],
            n_random_trials: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes differential allosteric sensitivity by isolating the dynamic perturbation
        from the basal folding noise using continuous vectorized tensor operations.
        (Thresholding is explicitly omitted to preserve the thermodynamic noise distribution).
        """
        target_device = attention_maps_wt.device

        # Algebraic subtraction and absolute value extraction of the attention tensor (eq. 8)
        delta_attention = torch.abs(attention_maps_mut[0] - attention_maps_wt[0])
        n_allo_sites = len(allo_sites_relative)
        num_layers, num_heads, seq_len, _ = delta_attention.shape

        impacts = torch.zeros((num_layers, num_heads))
        snrs = torch.zeros((num_layers, num_heads))
        pvals = torch.ones((num_layers, num_heads))

        # Definition of the background space (non-allosteric residues)
        non_allo_list = [i for i in range(seq_len) if i not in allo_sites_relative]

        # Hardware synchronization: Instantiate tensors on the correct target device
        non_allo_tensor = torch.tensor(non_allo_list, dtype=torch.long, device=target_device)
        allo_sites_tensor = torch.tensor(allo_sites_relative, dtype=torch.long, device=target_device)

        total_heads = num_layers * num_heads

        with tqdm(total=total_heads, desc="Analyzing Differential Attention (Continuous)", unit="head") as pbar:
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    attention_diff = delta_attention[layer_idx, head_idx]

                    # Vectorization Level 1: Continuous directional summation.
                    # Bypassing the boolean mask guarantees the preservation of natural variance (sigma > 0).
                    col_sums = torch.sum(attention_diff, dim=0)

                    # Direct computation of the attention coupled to the allosteric network
                    w_allo = torch.sum(col_sums[allo_sites_tensor]).item()

                    # Extraction of the background noise summation pool (non-target residues)
                    background_sums = col_sums[non_allo_tensor]

                    # Vectorization Level 2: Monte Carlo simulation within the continuous tensor space.
                    # Stochastic sampling WITHOUT replacement to guarantee absolute statistical rigor.
                    rand_weights = torch.rand(n_random_trials, len(non_allo_list), device=target_device)
                    rand_idx = torch.argsort(rand_weights, dim=1)[:, :n_allo_sites]

                    random_w_tensor = torch.sum(background_sums[rand_idx], dim=1)

                    expected_random = torch.mean(random_w_tensor).item()
                    std_random = torch.std(random_w_tensor).item()

                    # Inference of impact metrics.
                    # The 1e-10 regularizer now only acts on truly static heads (where delta is perfectly 0.0)
                    impact = w_allo / expected_random if expected_random > 0 else 0.0
                    snr = (w_allo - expected_random) / (std_random + 1e-10)

                    # Hypothesis testing (one-tailed t-test)
                    t_stat, p_value = ttest_1samp(
                        random_w_tensor.cpu().numpy(),
                        w_allo,
                        alternative='less'
                    )

                    if np.isnan(p_value):
                        p_value = 1.0

                    # Explicit casting to Python native float
                    impacts[layer_idx, head_idx] = float(impact)
                    snrs[layer_idx, head_idx] = float(snr)
                    pvals[layer_idx, head_idx] = float(p_value)

                    pbar.update(1)

        return impacts, snrs, pvals

    def _run_attention_analytics(self, project_name: str, offset: int, target_residues: List[int]) -> None:
        self.logger.info("STAGE 7: Executing Static and Differential Attention Analytics")
        if not target_residues:
            self.logger.warning("   -> No target residues defined. Skipping Stage 7.")
            return

        att_file_pattern = os.path.join(self.tensor_dir, f"Attention_{project_name}_*.pt")
        attention_files = glob.glob(att_file_pattern)

        # 1. Location and loading of the basal structural tensor (Wild Type)
        wt_file_path = os.path.join(self.tensor_dir, f"Attention_{project_name}_WT.pt")
        has_wt_baseline = os.path.exists(wt_file_path)
        attention_maps_wt = None

        if has_wt_baseline:
            attention_maps_wt = torch.load(wt_file_path, weights_only=True)
            self.logger.info("   -> Basal structural tensor (WT) loaded for differential normalization.")
        else:
            self.logger.warning("   -> WT state not found. Only absolute statistics will be computed.")

        for att_file in attention_files:
            state = os.path.basename(att_file).replace(f"Attention_{project_name}_", "").replace(".pt", "")
            attention_maps = torch.load(att_file, weights_only=True)
            seq_len = attention_maps.shape[-1]
            allo_sites_relative = [site - 1 - offset for site in target_residues]
            valid_sites = [s for s in allo_sites_relative if 0 <= s < seq_len]

            if not valid_sites:
                self.logger.warning(f"   -> Target residues out of bounds for state {state}. Skipping.")
                continue

            # 2. Absolute Static Analysis (Applies probability thresholding to filter steric packing noise)
            self.logger.info(f"   -> Computing absolute attention sensitivity for {state}...")
            imp, snrs, pvals = self._compute_attention_impact(
                attention_maps=attention_maps,
                allo_sites_relative=valid_sites,
                n_random_trials=1000
            )

            rows_abs = []
            num_layers, num_heads = imp.shape
            for layer in range(num_layers):
                for head in range(num_heads):
                    rows_abs.append({
                        "layer": layer,
                        "head": head,
                        "mean_impact": imp[layer, head].item(),
                        "snr": snrs[layer, head].item(),
                        "p_value": pvals[layer, head].item()
                    })

            df_abs = pd.DataFrame(rows_abs).sort_values(by="snr", ascending=False)
            df_abs.to_csv(os.path.join(self.analytics_dir, f"Attention_Sensitivity_{project_name}_{state}.csv"),
                          index=False)

            # 3. Differential Dynamic Analysis (Continuous tensor evaluation bypassing threshold constraints)
            if state != "WT" and has_wt_baseline:
                self.logger.info(f"   -> Computing continuous differential attention perturbation for {state} vs WT...")
                imp_diff, snrs_diff, pvals_diff = self._compute_differential_attention_impact(
                    attention_maps_wt=attention_maps_wt,
                    attention_maps_mut=attention_maps,
                    allo_sites_relative=valid_sites,
                    n_random_trials=1000
                )

                rows_diff = []
                for layer in range(num_layers):
                    for head in range(num_heads):
                        rows_diff.append({
                            "layer": layer,
                            "head": head,
                            "delta_mean_impact": imp_diff[layer, head].item(),
                            "delta_snr": snrs_diff[layer, head].item(),
                            "p_value": pvals_diff[layer, head].item()
                        })

                df_diff = pd.DataFrame(rows_diff).sort_values(by="delta_snr", ascending=False)
                df_diff.to_csv(os.path.join(self.analytics_dir, f"Differential_Attention_{project_name}_{state}.csv"),
                               index=False)
                self.logger.info(f"      -> Differential metrics stabilized and saved for {state}.")