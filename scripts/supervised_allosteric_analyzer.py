"""
Supervised Allosteric Analyzer
==============================
This module encapsulates the integration of Protein Language Models (ESM-2)
with statistical attention filtering and spatial-free graph topology.
It isolates dynamic allosteric networks by selecting attention heads
significantly directed towards a priori known allosteric sites.
"""

import os
import logging
import warnings
import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
from scipy.stats import ttest_1samp
from typing import Dict, Tuple, List, Any

from Bio import PDB
from Bio.PDB import PDBList, PDBParser, PDBIO, Select
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.SeqUtils import seq1
import esm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class _ChainAndProteinSelect(Select):
    def __init__(self, target_chain: str):
        self.target_chain = target_chain

    def accept_chain(self, chain: Chain) -> int:
        return 1 if chain.id == self.target_chain else 0

    def accept_residue(self, residue: Residue) -> int:
        return 1 if residue.id[0] == ' ' and PDB.is_aa(residue, standard=True) else 0


class SupervisedAllostericAnalyzer:
    def __init__(self, project_root: str = None, model_name: str = "esm2_t33_650M_UR50D", threshold: float = 0.3):
        if project_root:
            self.root = project_root
        else:
            self.root = str(os.path.dirname(os.path.abspath(str(__file__))))

        self.data_dir = os.path.join(self.root, "data")
        self.dirs = {
            "raw": os.path.join(self.data_dir, "raw_pdb"),
            "processed": os.path.join(self.data_dir, "processed_pdb"),
            "results": os.path.join(self.data_dir, "results"),
            "pymol": os.path.join(self.data_dir, "pymol_scripts")
        }
        for path in self.dirs.values():
            os.makedirs(path, exist_ok=True)

        self.threshold = threshold
        self.sequence_separation = 4  # Strict filter against primary/secondary backbone structure

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"[INIT] Hardware acceleration: {self.device.type.upper()}")
        logging.info(f"[INIT] Loading ESM-2 architecture: {model_name}")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

        self.num_layers = self.model.num_layers
        self.num_heads = self.model.attention_heads

        self.coords = {}
        self.idx_to_pdb = {}
        self.pdb_to_idx = {}
        self.sequence = ""

    def process_structure(self, kinase_name: str, pdb_id: str, chain_id: str) -> None:
        """Parses the crystallographic scaffold and builds index-to-PDB bidirectional mappings."""
        logging.info(f"--- STRUCTURAL CURATION ({kinase_name.upper()}) ---")
        pdb_id = pdb_id.lower()
        raw_path = os.path.join(self.dirs["raw"], f"{pdb_id}.pdb")

        if not os.path.exists(raw_path):
            pdbl = PDBList(verbose=False)
            file_path = pdbl.retrieve_pdb_file(pdb_id, file_format="pdb", pdir=self.dirs["raw"], overwrite=True)
            if file_path and file_path.endswith('.ent'):
                os.rename(file_path, raw_path)

        clean_path = os.path.join(self.dirs["processed"], f"{kinase_name}_scaffold.pdb")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, raw_path)
        io = PDBIO()
        io.set_structure(structure)
        io.save(clean_path, _ChainAndProteinSelect(chain_id))

        clean_structure = parser.get_structure(kinase_name, clean_path)
        chain_obj = clean_structure[0][chain_id]

        seq_list = []
        for i, res in enumerate([r for r in chain_obj if 'CA' in r and PDB.is_aa(r)]):
            pdb_res_num = int(res.id[1])
            self.coords[pdb_res_num] = res['CA'].get_coord()
            self.idx_to_pdb[i] = pdb_res_num
            self.pdb_to_idx[pdb_res_num] = i
            seq_list.append(seq1(res.resname))

        self.sequence = "".join(seq_list)
        logging.info(f"Structure mapped. Sequence length: {len(self.sequence)} residues.")

    def _get_attention_maps(self) -> torch.Tensor:
        """Extracts dense attention tensors from the language model."""
        _, _, batch_tokens = self.batch_converter([("protein", self.sequence)])
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=list(range(1, self.num_layers + 1)), need_head_weights=True)

        # [num_layers, num_heads, seq_len, seq_len] (cropping <cls> and <eos> tokens)
        attentions = results["attentions"].squeeze(0)[:, :, 1:-1, 1:-1]
        return attentions

    def compute_allosteric_impact(self, attention_maps: torch.Tensor, target_indices: List[int],
                                  n_random: int = 1000) -> List[Tuple[int, int]]:
        """
        Calculates the allosteric impact of each head towards the target sites.
        Filters heads based on random baseline, SNR, and a t-test.
        """
        num_layers, num_heads, seq_len, _ = attention_maps.shape
        n_targets = len(target_indices)

        impacts, snrs, pvals = [], [], []
        sensitive_heads = []

        # Iterate over all layers and heads to perform the statistical testing
        for l in range(num_layers):
            for h in range(num_heads):
                attention = attention_maps[l, h]
                mask = attention > self.threshold

                # Calculate specific attention directed to the designated target sites
                w_allo = sum(torch.sum(attention[:, site][mask[:, site]]).item() for site in target_indices)

                # Null model based on random structural distributions
                non_target_positions = np.array([i for i in range(seq_len) if i not in target_indices])
                random_w_values = []
                for _ in range(n_random):
                    random_sites = np.random.choice(non_target_positions, size=n_targets, replace=False)
                    w_random = sum(torch.sum(attention[:, site][mask[:, site]]).item() for site in random_sites)
                    random_w_values.append(w_random)

                expected_random = np.mean(random_w_values)
                std_random = np.std(random_w_values)

                impact = w_allo / expected_random if expected_random > 0 else 0
                snr = (w_allo - expected_random) / (std_random + 1e-10)

                # Statistical significance test
                _, p_value = ttest_1samp(random_w_values, w_allo, alternative='less')

                impacts.append(impact)
                if p_value < 0.01 and snr > 2.0:
                    sensitive_heads.append((l, h, impact, snr))

        mean_impact = np.mean(impacts)

        # Final strict filtering
        final_heads = [(l, h) for (l, h, imp, snr) in sensitive_heads if imp > mean_impact]
        logging.info(f"Identified {len(final_heads)} statistically significant allosteric heads.")
        return final_heads

    def _build_eigenvector_topology(self, consensus_tensor: np.ndarray) -> Tuple[nx.Graph, Dict[int, float]]:
        """
        Builds the long-range mathematical graph without Euclidean constraints,
        and computes Eigenvector Centrality for dynamic community detection.
        """
        num_res = len(self.sequence)

        # Sequence masking: enforce tertiary/quaternary tracking over backbone dominance
        idx_matrix = np.arange(num_res)
        seq_dist = np.abs(idx_matrix[:, None] - idx_matrix)
        seq_mask = np.where(seq_dist > self.sequence_separation, 1.0, 0.0)

        sym_tensor = np.abs(consensus_tensor) + np.abs(consensus_tensor.T)
        constrained_tensor = sym_tensor * seq_mask

        base_graph = nx.Graph()
        for i in range(num_res):
            base_graph.add_node(i, pdb_id=self.idx_to_pdb[i])

        for i in range(num_res):
            for j in range(i + 1, num_res):
                w = float(constrained_tensor[i, j])
                if w > 0:
                    # Invert weight for Minimum Spanning Tree (maximizes true covariance)
                    base_graph.add_edge(i, j, weight=-w, flow=w)

        mist = nx.minimum_spanning_tree(base_graph, weight='weight')

        for _, _, data in mist.edges(data=True):
            data['weight'] = data['flow']
            del data['flow']

        # Resolving dynamic resonances using eigenvector centrality
        raw_cent = nx.eigenvector_centrality_numpy(mist, weight='weight')
        pdb_cent = {self.idx_to_pdb[i]: score for i, score in raw_cent.items()}

        return mist, pdb_cent

    def _generate_pymol_script(self, name: str, graph: nx.Graph, centrality: Dict[int, float]) -> None:
        """Translates the mathematical tensor graph into physical coordinates for visualization."""
        cmap = matplotlib.colormaps.get_cmap('plasma')
        out_script = os.path.join(self.dirs["pymol"], f"Render_{name}.py")

        max_score = max(centrality.values()) if centrality else 1.0
        weights = [float(d.get('weight', 0)) for _, _, d in graph.edges(data=True)]
        max_w = max(weights) if weights else 1.0

        with open(out_script, 'w') as f:
            f.write('from pymol.cgo import *\nfrom pymol import cmd\n\n')
            f.write(f'def build_network_{name}():\n    cgo_obj = []\n\n')

            # Render long-range interactions
            for u, v, data in graph.edges(data=True):
                w = float(data.get('weight', 0))
                if w > 0:
                    pdb_u, pdb_v = self.idx_to_pdb[u], self.idx_to_pdb[v]
                    if pdb_u in self.coords and pdb_v in self.coords:
                        norm_w = min(w / max_w, 1.0)
                        radius = 0.05 + (norm_w * 0.3)
                        c1, c2 = self.coords[pdb_u], self.coords[pdb_v]
                        f.write(f'    cgo_obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, ')
                        f.write(f'{c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {radius:.3f}, ')
                        f.write('0.7, 0.7, 0.9, 0.7, 0.7, 0.9])\n')

            # Render Eigenvector Hubs
            for pdb_id, score in centrality.items():
                if pdb_id in self.coords:
                    norm_score = min(score / max_score, 1.0)
                    r, g, b, _ = cmap(norm_score)
                    radius = 0.3 + (norm_score * 2.0)
                    c = self.coords[pdb_id]
                    f.write(f'    cgo_obj.extend([COLOR, {r:.3f}, {g:.3f}, {b:.3f}, ')
                    f.write(f'SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {radius:.3f}])\n')

            f.write(f'\n    cmd.load_cgo(cgo_obj, "DynNet_{name}")\n')
            f.write('    cmd.bg_color("white")\n    cmd.show("cartoon")\n')
            f.write('    cmd.color("gray80", "polymer")\n    cmd.set("cartoon_transparency", 0.7)\n')
            f.write(f'build_network_{name}()\n')

    def execute_pipeline(self, name: str, pdb_id: str, chain: str, allosteric_pdb_sites: List[int]) -> None:
        """Main orchestrator for the supervised dynamic pathway extraction."""
        self.process_structure(name, pdb_id, chain)

        # 1. Translate external PDB coordinates to internal Tensor indices
        target_indices = [self.pdb_to_idx[pdb] for pdb in allosteric_pdb_sites if pdb in self.pdb_to_idx]
        if not target_indices:
            raise ValueError("Provided allosteric sites do not match any coordinates in the target chain.")

        # 2. Extract attention and compute statistical filtering
        att_maps = self._get_attention_maps()
        sensitive_heads = self.compute_allosteric_impact(att_maps, target_indices)

        if not sensitive_heads:
            logging.warning("No sensitive heads found mapping to the target site. Network computation aborted.")
            return

        # 3. Formulate the consensus dynamic tensor
        tensor_stack = torch.stack([att_maps[l, h] for (l, h) in sensitive_heads])
        consensus_tensor = torch.mean(tensor_stack, dim=0).cpu().numpy()

        # 4. Extract MIST and calculate Eigenvector Centrality
        mist, centrality = self._build_eigenvector_topology(consensus_tensor)

        # 5. Serialization and output
        df = pd.DataFrame([{"PDB_ID": k, "Eigenvector_Centrality": v} for k, v in centrality.items()])
        df = df.sort_values(by="Eigenvector_Centrality", ascending=False)
        df.to_csv(os.path.join(self.dirs["results"], f"{name}_Hubs.csv"), index=False)

        self._generate_pymol_script(name, mist, centrality)
        logging.info(f"Analysis complete. Results and CGO scripts saved for {name}.")