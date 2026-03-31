import os
import logging
import networkx as nx
import pandas as pd
from typing import Dict

# Academic traceability configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class CentralityAnalyzer:
    def __init__(self):
        """
        Initializes the topological analyzer. Configures directory paths for
        ingesting differential MIST graphs and exporting the final tabular metrics.
        """
        script_dir = str(os.path.dirname(os.path.abspath(str(__file__))))
        project_root = str(os.path.dirname(script_dir))
        self.data_dir = str(os.path.join(project_root, "data"))

        self.dirs = {
            "graphs": str(os.path.join(self.data_dir, "topological_graphs")),
            "results": str(os.path.join(self.data_dir, "results_centrality"))
        }
        os.makedirs(self.dirs["results"], exist_ok=True)

    @staticmethod
    def _calculate_betweenness(graph_path: str) -> Dict[int, float]:
        """
        Loads a graphml file and computes the normalized Betweenness Centrality
        for all nodes, mapping the mathematical node index to its real PDB identifier.

        :param graph_path: str, Absolute path to the MIST .graphml file
        :return: dict, Mapping of PDB residue ID (int) to its Centrality score (float)
        """
        graph = nx.read_graphml(graph_path)

        # Calculate standard Betweenness Centrality using differential edge weights
        raw_centrality = nx.betweenness_centrality(graph, weight='weight', normalized=True)

        # Map topological nodes back to strict biological PDB IDs stored in Phase 3
        pdb_centrality = {}
        for node_idx, score in raw_centrality.items():
            pdb_id = int(graph.nodes[node_idx]['pdb_id'])
            pdb_centrality[pdb_id] = float(score)

        return pdb_centrality

    def execute_centrality_pipeline(self):
        """
        Scans the topological graphs directory, computes the differential centrality
        for each MIST graph automatically, and exports the prioritized hubs to CSV.

        :return: None
        """
        logging.info("\n--- INITIATING PHASE 4: TOPOLOGICAL BOTTLENECK DETECTION ---")

        # Auto-detect all graph files to prevent dictionary key mismatches
        graph_files = [f for f in os.listdir(self.dirs["graphs"]) if f.endswith(".graphml")]

        if not graph_files:
            logging.error("No .graphml files found in the topological_graphs directory.")
            return

        for graph_file in graph_files:
            # Extract the clean name (e.g., MIST_SRC_E310A_Active.graphml -> SRC_E310A_Active)
            mut_name = graph_file.replace("MIST_", "").replace(".graphml", "")
            graph_path = os.path.join(self.dirs["graphs"], graph_file)

            logging.info(f"Computing invariant topology for {mut_name}...")

            # 1. Compute centrality directly on the differential graph
            differential_centrality = self._calculate_betweenness(graph_path)

            # 2. Structure data into a list of dictionaries
            results_data = [{"PDB_Residue_ID": res_id, "Differential_Centrality": score}
                            for res_id, score in differential_centrality.items()]

            # 3. Sort descending by the magnitude of the topological score and serialize
            df = pd.DataFrame(results_data)
            df = df.sort_values(by='Differential_Centrality', ascending=False)

            out_csv = os.path.join(self.dirs["results"], f"Top_Hubs_{mut_name}.csv")
            df.to_csv(out_csv, index=False)

            logging.info(f"Topological bottlenecks extracted and serialized: {out_csv}")

        logging.info("\n--- PHASE 4 SUCCESSFULLY COMPLETED ---")


if __name__ == "__main__":
    analyzer = CentralityAnalyzer()
    analyzer.execute_centrality_pipeline()