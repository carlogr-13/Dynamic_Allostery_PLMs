"""
Module 6.1: High-Resolution CGO Topology Compiler (Plasma Colormap)
===================================================================
This module synthesizes the Maximum Spanning Tree (MST) architectures applying
a strict multidimensional visual encoding homologous to high-impact PSN literature
(e.g., Trenfield et al.). Edges and nodes are dynamically scaled in volume and
colored using a perceptually uniform colormap (Yellow -> Magenta -> Dark Purple),
eliminating traditional red/blue artifacts and highlighting primary allosteric conduits.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging
import pandas as pd
from Bio.PDB import PDBParser

# Scientific logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class HighResTopologyCompiler:
    """
    Compiles advanced graph-theoretical 3D representations for PyMOL.
    Implements dynamic uniform RGB gradients (Plasma) for both vertices and edges.
    """

    def __init__(self):
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)

        self.pdb_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "processed_pdb")
        self.graph_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "graph_centrality")
        self.out_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "pymol_cgo_scripts")

        os.makedirs(self.out_dir, exist_ok=True)

        self.sequence_offsets: dict = {
            "SRC": 83,
            "EGFR": 700
        }

    def _get_ca_coordinates(self, pdb_path: str, kinase: str) -> dict:
        """Extracts the Euclidean coordinates of all Alpha Carbons."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(kinase, pdb_path)
        coords = {}
        for model in structure:
            for chain in model:
                for residue in chain:
                    if 'CA' in residue:
                        res_id = residue.get_id()[1]
                        coords[res_id] = residue['CA'].get_coord()
        return coords

    def _get_color_gradient(self, value: float, vmin: float, vmax: float) -> tuple:
        """
        Calculates a strictly normalized RGB tuple using a perceptually uniform
        scientific colormap (Yellow -> Magenta -> Dark Purple, typical in PSN).
        """
        # Proteccion contra division por cero en tensores nulos
        if vmax == vmin:
            return (1.0, 1.0, 0.0)  # Default to Yellow

        norm = (value - vmin) / (vmax - vmin)
        norm = max(0.0, min(1.0, norm))

        # Interpolación de la paleta científica "Plasma"
        if norm < 0.5:
            # Low to Mid: Yellow (1,1,0) to Magenta (0.8, 0.2, 0.6)
            n = norm * 2.0
            r = 1.0 - 0.2 * n
            g = 1.0 - 0.8 * n
            b = 0.0 + 0.6 * n
        else:
            # Mid to High: Magenta (0.8, 0.2, 0.6) to Dark Purple (0.3, 0.0, 0.5)
            n = (norm - 0.5) * 2.0
            r = 0.8 - 0.5 * n
            g = 0.2 - 0.2 * n
            b = 0.6 - 0.1 * n

        return (r, g, b)

    def _compile_pymol_script(self, kinase: str, state: str, pdb_path: str, df_nodes: pd.DataFrame,
                              df_edges: pd.DataFrame) -> str:
        """Constructs the multidimensional PyMOL CGO command sequence."""
        coords = self._get_ca_coordinates(pdb_path, kinase)
        offset = self.sequence_offsets.get(kinase, 0)

        script_lines = [
            "from pymol.cgo import *",
            "from pymol import cmd",
            f"cmd.reinitialize()",
            f"cmd.bg_color('white')",
            f"cmd.load(r'{pdb_path}', '{kinase}_scaffold')",
            f"cmd.show_as('cartoon', '{kinase}_scaffold')",
            f"cmd.color('gray80', '{kinase}_scaffold')",
            # Opacidad al 0.65 para anclar el grafo a la estructura sin ocultarlo
            f"cmd.set('cartoon_transparency', 0.65, '{kinase}_scaffold')",
            "obj = []"
        ]

        # Dynamic array normalization boundaries
        max_weight = df_edges["Weight"].max()
        min_weight = df_edges["Weight"].min()
        max_cent = df_nodes["Betweenness_Centrality"].max()
        min_cent = 0.0

        # 1. Generate Edges (Cylinders)
        for _, row in df_edges.iterrows():
            abs_u = int(row["Source"]) + 1 + offset
            abs_v = int(row["Target"]) + 1 + offset

            if abs_u in coords and abs_v in coords:
                c1 = coords[abs_u]
                c2 = coords[abs_v]
                weight = float(row["Weight"])

                # Dinamismo de volumen: desde 0.05 (fino) hasta 0.35 (arteria principal)
                cyl_rad = 0.05 + ((weight / max_weight) * 0.30) if max_weight > 0 else 0.05
                r, g, b = self._get_color_gradient(weight, min_weight, max_weight)

                script_lines.append(f"obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, "
                                    f"{c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {cyl_rad:.3f}, "
                                    f"{r:.2f}, {g:.2f}, {b:.2f}, {r:.2f}, {g:.2f}, {b:.2f}])")

                # 2. Generate Nodes (Spheres)
        for _, row in df_nodes.iterrows():
            abs_idx = int(row["Relative_Index"]) + 1 + offset
            val = float(row["Betweenness_Centrality"])

            if abs_idx in coords:
                c = coords[abs_idx]

                # Mapeo idéntico de color para crear continuidad visual Nodo-Arista
                r, g, b = self._get_color_gradient(val, min_cent, max_cent)
                # Nodos mínimos para las "hojas" de la red y masivos para los hubs
                radius = 0.4 + ((val / max_cent) * 2.8) if max_cent > 0 else 0.4

                script_lines.append(f"obj.extend([COLOR, {r:.2f}, {g:.2f}, {b:.2f}])")
                script_lines.append(f"obj.extend([SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {radius:.3f}])")

        script_lines.append(f"cmd.load_cgo(obj, '{kinase}_{state}_Topology')")
        script_lines.append(f"cmd.center('{kinase}_{state}_Topology')")

        return "\n".join(script_lines)

    def execute_compilation(self) -> None:
        """Orchestration routine for high-resolution rendering."""
        logging.info("===================================================")
        logging.info("INITIATING HIGH-RES CGO COMPILATION (PLASMA COLORMAP)")
        logging.info("===================================================")

        centrality_files = glob.glob(os.path.join(self.graph_dir, "Centrality_*.csv"))

        for cent_path in centrality_files:
            filename = os.path.basename(cent_path)
            parts = filename.replace("Centrality_", "").replace(".csv", "").split("_", 1)
            if len(parts) < 2:
                continue

            kinase, state = parts[0], parts[1]
            pdb_path = os.path.join(self.pdb_dir, f"{kinase}_scaffold_clean.pdb")
            edges_path = os.path.join(self.graph_dir, f"Edges_{kinase}_{state}.csv")

            if not os.path.exists(pdb_path) or not os.path.exists(edges_path):
                continue

            df_nodes = pd.read_csv(cent_path)
            df_edges = pd.read_csv(edges_path)

            pml_script = self._compile_pymol_script(kinase, state, pdb_path, df_nodes, df_edges)

            out_path = os.path.join(self.out_dir, f"Render_CGO_HighRes_{kinase}_{state}.py")
            with open(out_path, "w") as f:
                f.write(pml_script)
            logging.info(f"   -> Advanced Topology script compiled: {os.path.basename(out_path)}")

        logging.info("COMPILATION COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    compiler = HighResTopologyCompiler()
    compiler.execute_compilation()