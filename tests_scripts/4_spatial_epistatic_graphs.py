"""
Module 4: Epistatic Maximum Spanning Tree Extraction
====================================================
This module computes the Maximum Information Spanning Tree (MST) directly
from the evolutionary thermodynamic tensors (ESM-2 Jacobians). By omitting
Euclidean distance restrictions, it preserves long-range allosteric couplings
propagated through the hydrophobic core. The MST topology inherently eliminates
cyclic redundancy, allowing the exact computation of Betweenness Centrality
across the dominant thermodynamic routing architecture.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import networkx as nx

# Configuración del registro científico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class EpistaticMSTAnalyzer:
    """
    Constructs the strict Maximum Spanning Tree topology from the complete
    thermodynamic coupling matrix and evaluates absolute node centrality.
    """

    def __init__(self):
        """
        Inicializa el entorno de análisis topológico.
        Se han suprimido los parámetros de distancia espacial.
        """
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)

        self.tensor_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "results")
        self.graph_data_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "graph_centrality")

        os.makedirs(self.graph_data_dir, exist_ok=True)

    def _build_mst_and_analyze(self, jacobian: np.ndarray) -> tuple:
        """
        Instantiates the fully connected weighted graph, extracts the MST,
        and computes the topological Betweenness Centrality.

        :param jacobian: np.ndarray, Tensor asimétrico de acoplamiento epistático.
        :return: tuple (Dict[int, float], List[tuple]), Diccionario de centralidades
                 y lista de aristas que componen el MST.
        """
        G = nx.Graph()
        n_nodes = jacobian.shape[0]
        G.add_nodes_from(range(n_nodes))

        # 1. Simetrización del Tensor Jacobiano
        # Termodinámicamente, la capacidad de flujo de información en el estado
        # estacionario se modela de forma bidireccional para la extracción del árbol.
        sym_jacobian = (jacobian + jacobian.T) / 2.0
        np.fill_diagonal(sym_jacobian, 0.0)

        # 2. Instanciación del Grafo Completo
        rows, cols = np.where(sym_jacobian > 0)
        for i, j in zip(rows, cols):
            if i < j:  # Grafo no dirigido, se procesa la matriz triangular superior
                G.add_edge(i, j, weight=sym_jacobian[i, j])

        if G.number_of_edges() == 0:
            return {}, []

        # 3. Extracción del Árbol de Expansión de Máxima Información (MST)
        mst = nx.maximum_spanning_tree(G, weight='weight')

        # 4. Evaluación Topológica (Centralidad de Intermediación)
        centrality = nx.betweenness_centrality(mst, normalized=True)

        # 5. Extracción de la Lista de Aristas para Renderizado Espacial
        mst_edges = [(u, v, mst[u][v]['weight']) for u, v in mst.edges()]

        return centrality, mst_edges

    def execute_analysis(self) -> None:
        """
        Rutina de orquestación principal. Ejecuta la transformación sobre todos
        los microestados cinemáticos del sistema.
        """
        logging.info("===================================================")
        logging.info("INITIATING EPISTATIC MST ANALYSIS (NO SPATIAL BIAS)")
        logging.info("===================================================")

        tensor_files = glob.glob(os.path.join(self.tensor_dir, "Jacobian_*.npy"))

        for file_path in tensor_files:
            filename = os.path.basename(file_path).replace(".npy", "")
            parts = filename.split("_", 2)
            if len(parts) < 2:
                continue

            kinase = parts[1]
            state = parts[2] if len(parts) > 2 else "WT"

            logging.info(f"Processing unbiased topology for: {kinase} [{state}]")

            # Carga del tensor evolutivo
            jacobian = np.load(file_path)

            # Ejecución del algoritmo topológico
            centrality_dict, mst_edges = self._build_mst_and_analyze(jacobian)

            if not centrality_dict:
                logging.warning(f"   -> No valid graph generated for {filename}.")
                continue

            # Serialización de Nodos (Centralidad Absoluta)
            df_nodes = pd.DataFrame(list(centrality_dict.items()), columns=["Relative_Index", "Betweenness_Centrality"])
            df_nodes = df_nodes.sort_values(by="Betweenness_Centrality", ascending=False)
            df_nodes.to_csv(os.path.join(self.graph_data_dir, f"Centrality_{kinase}_{state}.csv"), index=False)

            # Serialización de Aristas (Estructura del MST)
            df_edges = pd.DataFrame(mst_edges, columns=["Source", "Target", "Weight"])
            df_edges = df_edges.sort_values(by="Weight", ascending=False)
            df_edges.to_csv(os.path.join(self.graph_data_dir, f"Edges_{kinase}_{state}.csv"), index=False)

            logging.info(f"   -> Topologies serialized (Nodes and MST Edges).")

        logging.info("MST ANALYSIS COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    analyzer = EpistaticMSTAnalyzer()
    analyzer.execute_analysis()