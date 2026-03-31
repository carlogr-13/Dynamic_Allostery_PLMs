import os
import logging
import numpy as np
import networkx as nx
from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# Academic traceability configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class NetworkTopologyBuilder:
    def __init__(self):
        """
        Initializes the biophysical network builder. Establishes the topological
        parameters and the directory infrastructure required for matrix processing
        and graph generation.
        """
        script_dir = str(os.path.dirname(os.path.abspath(str(__file__))))
        project_root = str(os.path.dirname(script_dir))
        self.data_dir = str(os.path.join(project_root, "data"))

        self.dirs = {
            "processed_pdb": str(os.path.join(self.data_dir, "processed_pdb")),
            "tensors": str(os.path.join(self.data_dir, "attention_tensors")),
            "graphs": str(os.path.join(self.data_dir, "topological_graphs"))
        }
        os.makedirs(self.dirs["graphs"], exist_ok=True)

        # Rigorous biophysical constraints
        self.distance_cutoff = 8.0  # Angstroms (upper limit for physical energy transfer)
        self.statistical_percentile = 95.0  # Retain only the top 5% of differential covariance

    @staticmethod
    def _compute_euclidean_distance_matrix(pdb_filepath: str, target_chain: str) -> np.ndarray:
        """
        Calculates the pairwise Euclidean distance matrix for all Alpha Carbons (CA)
        within the specified chain of the crystallographic scaffold.

        :param pdb_filepath: str, Absolute path to the purified structural scaffold (.pdb)
        :param target_chain: str, Identifier of the biological chain to parse
        :return: numpy.ndarray, N x N symmetric matrix containing spatial distances in Angstroms
        """
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('scaffold', pdb_filepath)
        chain = structure[0][target_chain]

        # Extract CA coordinates strictly matching the FASTA extraction logic
        ca_atoms = [residue['CA'] for residue in chain if 'CA' in residue and PDB.is_aa(residue)]
        num_residues = len(ca_atoms)

        distance_matrix = np.zeros((num_residues, num_residues), dtype=np.float32)
        for i in range(num_residues):
            for j in range(num_residues):
                # BioPython vectors compute the Euclidean distance upon subtraction
                distance_matrix[i, j] = ca_atoms[i] - ca_atoms[j]

        return distance_matrix

    def _symmetrize_and_threshold(self, tensor: np.ndarray) -> np.ndarray:
        """
        Symmetrizes the differential attention tensor and applies a strict statistical
        percentile filter to isolate the most significant covariance deviations.

        :param tensor: numpy.ndarray, Raw asymmetric differential tensor (Delta A)
        :return: numpy.ndarray, Symmetrized and statistically pruned matrix
        """
        # 1. Symmetrization based on absolute magnitudes of thermodynamic variation
        sym_tensor = np.abs(tensor) + np.abs(tensor.T)

        # 2. Calculate the scalar threshold to cut off the noise
        threshold = np.percentile(sym_tensor, self.statistical_percentile)

        # 3. Prune connections below the statistical significance threshold
        sym_tensor[sym_tensor < threshold] = 0.0
        return sym_tensor

    def construct_mist(self, differential_tensor: np.ndarray, distance_matrix: np.ndarray,
                       pdb_filepath: str, target_chain: str) -> nx.Graph:
        """
        Applies spatial constraints and graph theory to extract the Maximum Information
        Spanning Tree (MIST) from the thermodynamic network. Embeds real biological
        residue IDs as node attributes.

        :param differential_tensor: numpy.ndarray, Symmetrized Delta A matrix
        :param distance_matrix: numpy.ndarray, Euclidean spatial distance matrix
        :param pdb_filepath: str, Path to PDB to extract original residue IDs
        :param target_chain: str, Target chain to extract original residue IDs
        :return: networkx.Graph, The isolated maximum spanning tree representing allosteric routes
        """
        # Retrieve authentic PDB residue numbers for future structural mapping
        parser = PDB.PDBParser(QUIET=True)
        chain = parser.get_structure('scaffold', pdb_filepath)[0][target_chain]
        residue_ids = [res.id[1] for res in chain if 'CA' in res and PDB.is_aa(res)]

        # Apply strict spatial boolean mask (<= 8.0 Angstroms)
        spatial_mask = distance_matrix <= self.distance_cutoff
        physically_constrained_tensor = differential_tensor * spatial_mask

        # Instantiate the dense mathematical network
        num_nodes = physically_constrained_tensor.shape[0]
        graph = nx.Graph()

        # Populate nodes with their PDB crystal IDs as attributes
        for i in range(num_nodes):
            graph.add_node(i, pdb_id=int(residue_ids[i]))

        # Populate edges with their differential covariance as the weight
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Upper triangle only to avoid duplication
                weight = float(physically_constrained_tensor[i, j])
                if weight > 0:
                    # Negate weight because Kruskal's algorithm minimizes cost;
                    # we need to MAXIMIZE the covariance flow (Maximum Spanning Tree).
                    graph.add_edge(i, j, weight=-weight, covariance_flow=weight)

        # Extract the fundamental routes using Minimum Spanning Tree over negative weights
        mist_graph = nx.minimum_spanning_tree(graph, weight='weight')

        # Revert weights to positive mathematical space for downstream centrality calculus
        for u, v, data in mist_graph.edges(data=True):
            data['weight'] = data['covariance_flow']
            del data['covariance_flow']

        return mist_graph

    def execute_topology_pipeline(self, target_systems: dict):
        """
        Iterates over the experimental matrix, sequentially applying spatial
        and statistical constraints to generate and serialize the network graphs.

        :param target_systems: dict, Dictionary mapping biological systems to mutations
        :return: None
        """
        logging.info("\n--- INITIATING PHASE 3: BIOPHYSICAL FILTERING AND MIST GENERATION ---")

        for system, config in target_systems.items():
            scaffold_path = os.path.join(self.dirs["processed_pdb"], f"{system}_scaffold_clean.pdb")

            if not os.path.exists(scaffold_path):
                logging.error(f"Structural scaffold missing for {system}. Skipping.")
                continue

            logging.info(f"Computing 3D Euclidean space for {system}...")
            dist_matrix = self._compute_euclidean_distance_matrix(scaffold_path, config["chain"])

            mutant_variants = [v for v in config["mutations"].keys() if v != "WT"]

            for mut_suffix in mutant_variants:
                mut_name = f"{system}_{mut_suffix}"
                tensor_path = os.path.join(self.dirs["tensors"], f"delta_tensor_{mut_name}.npy")

                if not os.path.exists(tensor_path):
                    logging.warning(f"Differential tensor missing for {mut_name}. Skipping.")
                    continue

                logging.info(f"Extracting Maximum Information Spanning Tree for {mut_name}...")
                raw_tensor = np.load(tensor_path)

                sym_tensor = self._symmetrize_and_threshold(raw_tensor)
                mist_graph = self.construct_mist(sym_tensor, dist_matrix, scaffold_path, config["chain"])

                # Serialize the graph topology using standard GraphML format
                graph_out_path = os.path.join(self.dirs["graphs"], f"MIST_{mut_name}.graphml")
                nx.write_graphml(mist_graph, graph_out_path)

        logging.info("\n--- PHASE 3 SUCCESSFULLY COMPLETED ---")


if __name__ == "__main__":
    # Experimental configuration inherited from Phase 1
    experimental_targets = {
        "SRC": {
            "chain": "A",
            "mutations": {"WT": [], "E310A_Active": [], "T338G_Inhibitory": []}
        },
        "EGFR": {
            "chain": "A",
            "mutations": {"WT": [], "L858R": [], "L858R_T790M_Epistatic": []}
        }
    }

    builder = NetworkTopologyBuilder()
    builder.execute_topology_pipeline(experimental_targets)