import os
import logging
import warnings
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
from typing import Dict, Tuple, List, Any

# Direct imports to resolve strict IDE type-hinting references
from Bio import PDB
from Bio.PDB import PDBList, PDBParser, PDBIO, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1

import esm
from tqdm import tqdm

# Academic traceability configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class _ChainAndProteinSelect(Select):
    """
    Internal boolean filter for crystallographic purification.
    Ensures that only the target polypeptide chain is extracted, removing
    water molecules, ligands, and heterogenous atoms.
    """
    def __init__(self, target_chain: str):
        self.target_chain = target_chain

    def accept_chain(self, chain: Chain) -> int:
        """Evaluates if the current chain matches the target configuration."""
        return 1 if chain.id == self.target_chain else 0

    def accept_residue(self, residue: Residue) -> int:
        """Filters out heteroatoms, keeping only standard amino acids."""
        return 1 if residue.id[0] == ' ' and PDB.is_aa(residue, standard=True) else 0


class KinaseMistAnalyzer:
    """
    Comprehensive computational framework for the ab initio analysis of allosteric
    networks using Protein Language Models (ESM-2) and Spatial Topology (MIST).
    """

    def __init__(self, project_root: str = None, model_name: str = "esm2_t33_650M_UR50D"):
        """
        Initializes the analytical environment, sets up the directory topology,
        and loads the pre-trained Transformer model into hardware-accelerated memory.

        :param project_root: str, Optional absolute path to the project root directory.
                             If None, it resolves dynamically based on the script location.
        :param model_name: str, Exact identifier of the ESM-2 architecture to instantiate.
        """
        # 1. Directory Topology Configuration
        if project_root:
            self.root = project_root
        else:
            script_dir = str(os.path.dirname(os.path.abspath(str(__file__))))
            self.root = str(os.path.dirname(script_dir))

        self.data_dir = os.path.join(self.root,"data")
        self.dirs = {
            "raw": os.path.join(self.data_dir, "raw_pdb"),
            "processed": os.path.join(self.data_dir, "processed_pdb"),
            "fasta": os.path.join(self.data_dir, "fasta_sequences"),
            "tensors": os.path.join(self.data_dir, "attention_tensors"),
            "graphs": os.path.join(self.data_dir, "topological_graphs"),
            "results": os.path.join(self.data_dir, "results_centrality"),
            "pymol": os.path.join(self.data_dir, "pymol_cgo_scripts")
        }
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

        # 2. Biophysical and Topological Constraints
        self.distance_cutoff = 8.0  # Angstroms (upper limit for physical energy transfer)
        self.statistical_percentile = 95.0  # Threshold to isolate significant covariance
        self.start_layer = 16  # Lower bound to isolate global tertiary contacts in ESM-2

        # 3. Transformer Model Initialization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"[INIT] Hardware acceleration detected: {self.device.type.upper()}")
        logging.info(f"[INIT] Loading PLM architecture {model_name}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        # State variables for coordinate tracking
        self.coords = {}
        self.residue_ids = []

    # ==========================================
    # PHASE 1: STRUCTURAL & SEQUENTIAL CURATION
    # ==========================================
    def process_structure(self, kinase_name: str, pdb_id: str, chain_id: str, mutations: Dict[str, List[Tuple[str, int, str]]]) -> Dict[str, str]:
        """
        Retrieves the crystallographic data, purifies the spatial scaffold, extracts
        the Cartesian coordinates, and generates the wild-type and mutated FASTA sequences.

        :param kinase_name: str, Common biological identifier (e.g., 'SRC', 'EGFR').
        :param pdb_id: str, 4-character standard alphanumeric PDB identification code.
        :param chain_id: str, Target polypeptide chain identifier (e.g., 'A').
        :param mutations: dict, Dictionary mapping the mutation names to a list of tuple variations.
        :return: dict, Mapping of sequence identifiers to their pure linear amino acid strings.
        """
        logging.info(f"\n--- PHASE 1: STRUCTURAL CURATION FOR {kinase_name.upper()} ---")

        pdb_id = pdb_id.lower()
        raw_path = os.path.join(self.dirs["raw"], f"{pdb_id}.pdb")

        # 1.1 Download scaffold if not locally available
        if not os.path.exists(raw_path):
            pdbl = PDBList(verbose=False)
            file_path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=self.dirs["raw"], overwrite=True)
            if file_path and file_path.endswith('.ent'):
                os.rename(file_path, raw_path)

        # 1.2 Purify the 3D scaffold to eliminate solvent and heteroatoms
        clean_path = os.path.join(self.dirs["processed"], f"{kinase_name}_scaffold_clean.pdb")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, raw_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(clean_path, _ChainAndProteinSelect(chain_id))

        # 1.3 Map Alpha Carbon (CA) spatial coordinates and extract the basal sequence
        clean_structure = parser.get_structure(kinase_name, clean_path)
        chain_obj = clean_structure[0][chain_id]

        self.coords = {res.id[1]: res['CA'].get_coord() for res in chain_obj if 'CA' in res and PDB.is_aa(res)}
        self.residue_ids = list(self.coords.keys())
        res_map = {res.id[1]: seq1(res.resname) for res in chain_obj if 'CA' in res and PDB.is_aa(res)}

        # 1.4 Perform in silico mutagenesis to generate differential sequences
        sequences = {}
        for mut_name, mut_list in mutations.items():
            mutated_seq_map = res_map.copy()
            for (aa_orig, pos, aa_mut) in mut_list:
                if pos in mutated_seq_map and mutated_seq_map[pos] == aa_orig:
                    mutated_seq_map[pos] = aa_mut

            final_seq = "".join([mutated_seq_map[idx] for idx in self.residue_ids])
            seq_id = f"{kinase_name}_{mut_name}"
            sequences[seq_id] = final_seq

            # Serialize the FASTA files enforcing an 80-character line width
            with open(os.path.join(self.dirs["fasta"], f"{seq_id}.fasta"), "w") as f:
                f.write(f">{seq_id}\n")
                for i in range(0, len(final_seq), 80):
                    f.write(f"{final_seq[i:i + 80]}\n")

        return sequences

    # ==========================================
    # PHASE 2: TENSORIAL INFERENCE (ESM-2)
    # ==========================================
    def _get_global_attention(self, sequence: str, seq_name: str) -> np.ndarray:
        """
        Executes the forward pass through the Transformer, isolates the attention weights,
        crops the artificial tokens, and computes the statistical mean across deep layers.

        :param sequence: str, Pure amino acid sequence to be processed.
        :param seq_name: str, Identifier of the biological sequence for batch logging.
        :return: numpy.ndarray, N x N asymmetric matrix representing global evolutionary covariance.
        """
        _, _, batch_tokens = self.batch_converter([(seq_name, sequence)])
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[], need_head_weights=True)

        attentions = results["attentions"].squeeze(0)
        # Spatial cropping: Remove <cls> (index 0) and <eos> (index -1) tokens
        attentions_cropped = attentions[:, :, 1:-1, 1:-1]
        # Isolate deep layers responsible for global tertiary structure
        deep_attentions = attentions_cropped[self.start_layer:, :, :, :]
        # Collapse dimensions to generate the N x N covariance matrix
        return torch.mean(deep_attentions, dim=(0, 1)).cpu().numpy()

    def compute_thermodynamic_tensors(self, kinase_name: str, sequences: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Iterates over the generated sequences, computes the absolute baseline tensor (WT),
        and applies differential calculus to isolate the mutational perturbations.

        :param kinase_name: str, Common biological identifier.
        :param sequences: dict, Output from process_structure containing sequence strings.
        :return: dict, Mapping of sequence identifiers to their corresponding matrices and states.
        """
        logging.info(f"\n--- PHASE 2: TENSORIAL INFERENCE ({kinase_name}) ---")
        tensors = {}

        wt_name = f"{kinase_name}_WT"
        if wt_name not in sequences:
            raise ValueError("The experimental configuration must include a 'WT' variant.")

        # 2.1 Calculate Absolute Basal Tensor
        wt_tensor = self._get_global_attention(sequences[wt_name], wt_name)
        tensors[wt_name] = {"matrix": wt_tensor, "is_differential": False}
        np.save(os.path.join(self.dirs["tensors"], f"tensor_{wt_name}.npy"), wt_tensor)

        # 2.2 Calculate Differential Mutant Tensors (Delta A = A_Mut - A_WT)
        mutants = [k for k in sequences.keys() if k != wt_name]
        for mut_name in tqdm(mutants, desc="Computing Differential Tensors"):
            mut_tensor = self._get_global_attention(sequences[mut_name], mut_name)
            delta_tensor = mut_tensor - wt_tensor
            tensors[mut_name] = {"matrix": delta_tensor, "is_differential": True}
            np.save(os.path.join(self.dirs["tensors"], f"delta_tensor_{mut_name}.npy"), delta_tensor)

        return tensors

    # ==========================================
    # PHASES 3 & 4: MIST TOPOLOGY & CENTRALITY
    # ==========================================
    def _build_mist_and_centrality(self, tensor: np.ndarray) -> Tuple[nx.Graph, pd.DataFrame]:
        """
        Enforces 3D spatial constraints, extracts the Maximum Information Spanning Tree (MIST),
        and quantifies the topological importance using normalized Betweenness Centrality.

        :param tensor: numpy.ndarray, N x N Absolute or Differential attention matrix.
        :return: tuple, (NetworkX graph object, Pandas DataFrame with centrality metrics).
        """
        num_res = len(self.residue_ids)
        coord_list = list(self.coords.values())

        # 1. Compute Euclidean Distance Matrix
        dist_matrix = np.zeros((num_res, num_res), dtype=np.float32)
        for i in range(num_res):
            for j in range(num_res):
                dist_matrix[i, j] = np.linalg.norm(coord_list[i] - coord_list[j])

        # Boolean mask enforcing the physical energy transfer limit
        spatial_mask = dist_matrix <= self.distance_cutoff

        # 2. Symmetrization and Statistical Thresholding
        sym_tensor = np.abs(tensor) + np.abs(tensor.T)
        threshold = np.percentile(sym_tensor, self.statistical_percentile)
        sym_tensor[sym_tensor < threshold] = 0.0

        # Apply physical constraints (Hadamard product)
        constrained_tensor = sym_tensor * spatial_mask

        # 3. Graph Instantiation and MIST Extraction (Kruskal's Algorithm)
        base_graph = nx.Graph()
        for i in range(num_res):
            base_graph.add_node(i, pdb_id=int(self.residue_ids[i]))

        for i in range(num_res):
            for j in range(i + 1, num_res):
                w = float(constrained_tensor[i, j])
                if w > 0:
                    # Invert weight sign to deceive algorithm into maximizing flow
                    base_graph.add_edge(i, j, weight=-w, flow=w)

        mist = nx.minimum_spanning_tree(base_graph, weight='weight')

        # Revert weights back to positive physical values
        for _, _, data in mist.edges(data=True):
            data['weight'] = data['flow']
            del data['flow']

        # 4. Topological Bottleneck Calculus (Normalized Betweenness Centrality)
        raw_cent = nx.betweenness_centrality(mist, weight='weight', normalized=True)
        df_data = [{"PDB_Residue_ID": int(mist.nodes[idx]['pdb_id']),
                    "Centrality": float(score)} for idx, score in raw_cent.items()]

        df = pd.DataFrame(df_data).sort_values(by='Centrality', ascending=False)

        return mist, df

    def extract_topologies(self, kinase_name: str, tensors: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Orchestrates the conversion of matrices into mathematical graphs for all variants,
        serializing the structures to GraphML and tabular formats.

        :param kinase_name: str, Common biological identifier.
        :param tensors: dict, Output from compute_thermodynamic_tensors.
        :return: dict, Mapping of sequence identifiers to their corresponding graphs and metrics.
        """
        logging.info(f"\n--- PHASE 3/4: MIST TOPOLOGY AND CENTRALITY ({kinase_name}) ---")
        topologies = {}

        for name, data in tensors.items():
            mist_graph, centrality_df = self._build_mist_and_centrality(data["matrix"])
            topologies[name] = {"graph": mist_graph, "df": centrality_df, "is_wt": not data["is_differential"]}

            # Serialize the topological and numerical data
            nx.write_graphml(mist_graph, os.path.join(self.dirs["graphs"], f"MIST_{name}.graphml"))
            centrality_df.to_csv(os.path.join(self.dirs["results"], f"Hubs_{name}.csv"), index=False)

        return topologies

    # ==========================================
    # PHASE 5: 3D GRAPHICAL RENDERING (PYMOL CGO)
    # ==========================================
    @staticmethod
    def _get_rgb(value: float, cmap_name: str) -> Tuple[float, float, float]:
        """
        Translates a normalized scalar into an RGB color tuple using scientific palettes.

        :param value: float, Normalized metric (e.g., centrality or covariance) within [0, 1].
        :param cmap_name: str, Target Matplotlib colormap identifier.
        :return: tuple, Float values for Red, Green, and Blue.
        """
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        r, g, b, _ = cmap(value)
        return r, g, b

    def generate_cgo_scripts(self, kinase_name: str, topologies: Dict[str, Dict[str, Any]]) -> None:
        """
        Compiles the mathematical graphs into native PyMOL Python scripts using
        Compiled Graphics Objects (CGO), mapping thermodynamics to geometric primitives.

        :param kinase_name: str, Common biological identifier.
        :param topologies: dict, Output from extract_topologies.
        :return: None
        """
        logging.info(f"\n--- PHASE 5: CGO 3D RENDERING COMPILATION ({kinase_name}) ---")

        for name, topo in topologies.items():
            graph = topo["graph"]
            df = topo["df"]
            is_wt = topo["is_wt"]

            # Prevent division by zero if all centralities are mathematically negligible
            max_cent = df['Centrality'].max() if not df.empty and df['Centrality'].max() > 0 else 1.0
            weights = [float(d.get('weight', 0)) for _, _, d in graph.edges(data=True)]
            max_w = max(weights) if weights else 1.0

            # Chromatic duality: Cold colors for Basal Architecture, Warm colors for Allosteric Perturbation
            edge_cmap = 'Blues' if is_wt else 'Oranges'
            node_cmap = 'Purples' if is_wt else 'Reds'

            out_script = os.path.join(self.dirs["pymol"], f"Render_{name}.py")
            with open(out_script, 'w') as f:
                f.write('from pymol.cgo import *\nfrom pymol import cmd\n\n')
                f.write(f'def build_network_{name}():\n    cgo_obj = []\n\n')

                # Render Edges (Covariance Cylinders)
                for u, v, data in graph.edges(data=True):
                    w = float(data.get('weight', 0))
                    if w > 0:
                        pdb_u, pdb_v = int(graph.nodes[u]['pdb_id']), int(graph.nodes[v]['pdb_id'])
                        if pdb_u in self.coords and pdb_v in self.coords:
                            norm_w = min(w / max_w, 1.0)
                            r, g, b = self._get_rgb(norm_w, edge_cmap)
                            radius = 0.10 + (norm_w * 0.40)  # Dynamic thickness mapping
                            c1, c2 = self.coords[pdb_u], self.coords[pdb_v]

                            f.write(f'    cgo_obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, ')
                            f.write(f'{c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {radius:.3f}, ')
                            f.write(f'{r:.3f}, {g:.3f}, {b:.3f}, {r:.3f}, {g:.3f}, {b:.3f}])\n')

                # Render Nodes (Centrality Spheres)
                for _, row in df.iterrows():
                    pdb_id = int(row['PDB_Residue_ID'])
                    score = float(row['Centrality'])
                    if score > 0.005 and pdb_id in self.coords:
                        norm_score = min(score / max_cent, 1.0)
                        r, g, b = self._get_rgb(norm_score, node_cmap)
                        radius = 0.8 + (norm_score * 1.5)  # Dynamic volume mapping
                        c = self.coords[pdb_id]

                        f.write(f'    cgo_obj.extend([COLOR, {r:.3f}, {g:.3f}, {b:.3f}, ')
                        f.write(f'SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {radius:.3f}])\n')

                # Aesthetics and execution commands
                f.write(f'\n    cmd.load_cgo(cgo_obj, "Net_{name}")\n')
                f.write('    cmd.bg_color("white")\n    cmd.show("cartoon")\n')
                f.write('    cmd.color("gray80", "polymer")\n    cmd.set("cartoon_transparency", 0.7)\n')
                f.write('    cmd.zoom()\n')
                f.write(f'build_network_{name}()\n')

            logging.info(f"3D rendering script successfully compiled: {out_script}")

    # ==========================================
    # MASTER ORCHESTRATOR
    # ==========================================
    def execute_full_pipeline(self, kinase_name: str, pdb_id: str, chain_id: str, mutations: Dict[str, List[Tuple[str, int, str]]]) -> None:
        """
        Main execution function. Sequences all the analytical phases required
        to transform raw biological inputs into mathematical topologies.

        :param kinase_name: str, Common biological identifier.
        :param pdb_id: str, PDB identification code.
        :param chain_id: str, Target polypeptide chain identifier.
        :param mutations: dict, Configuration dictionary of variants.
        :return: None
        """
        logging.info(f"\n=======================================================")
        logging.info(f"INITIATING UNIFIED ANALYTICAL PIPELINE: {kinase_name.upper()}")
        logging.info(f"=======================================================")

        sequences = self.process_structure(kinase_name, pdb_id, chain_id, mutations)
        tensors = self.compute_thermodynamic_tensors(kinase_name, sequences)
        topologies = self.extract_topologies(kinase_name, tensors)
        self.generate_cgo_scripts(kinase_name, topologies)

        logging.info(f"\n[SUCCESS] Pipeline execution for {kinase_name} completed seamlessly.")