import os
import logging
import pandas as pd
import networkx as nx
import matplotlib
from Bio import PDB
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# Academic traceability configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=PDBConstructionWarning)


class PyMolCGOGenerator:
    def __init__(self):
        """
        Initializes the PyMOL CGO generation pipeline. Configures directory
        paths for ingesting structural coordinates, topological networks, and
        centrality data, outputting executable PyMOL Python scripts.
        """
        script_dir = str(os.path.dirname(os.path.abspath(str(__file__))))
        project_root = str(os.path.dirname(script_dir))
        self.data_dir = str(os.path.join(project_root, "data"))

        self.dirs = {
            "processed_pdb": str(os.path.join(self.data_dir, "processed_pdb")),
            "graphs": str(os.path.join(self.data_dir, "topological_graphs")),
            "results": str(os.path.join(self.data_dir, "results_centrality")),
            "pymol_scripts": str(os.path.join(self.data_dir, "pymol_cgo_scripts"))
        }
        os.makedirs(self.dirs["pymol_scripts"], exist_ok=True)

    @staticmethod
    def _get_rgb(value: float, colormap_name: str = 'Reds') -> tuple:
        """
        Translates a normalized mathematical value [0, 1] into a float RGB tuple.
        """
        cmap = matplotlib.colormaps.get_cmap(colormap_name)
        r, g, b, a = cmap(value)
        return r, g, b

    def execute_cgo_pipeline(self, target_systems: dict):
        """
        Extracts coordinates from the PDB and generates a Python script containing
        Compiled Graphics Objects (CGO) to render the MIST network natively in PyMOL.
        """
        logging.info("\n--- INITIATING PHASE 5: PYMOL CGO NETWORK GENERATION ---")
        parser = PDB.PDBParser(QUIET=True)

        for system, config in target_systems.items():
            chain_id = config["chain"]
            scaffold_pdb = os.path.join(self.dirs["processed_pdb"], f"{system}_scaffold_clean.pdb")

            if not os.path.exists(scaffold_pdb):
                continue

            # Extract absolute XYZ coordinates
            structure = parser.get_structure(system, scaffold_pdb)
            chain = structure[0][chain_id]

            coords = {}
            for res in chain:
                if 'CA' in res and PDB.is_aa(res):
                    coords[res.id[1]] = res['CA'].get_coord()

            mutant_variants = [v for v in config["mutations"].keys() if v != "WT"]

            for mut_suffix in mutant_variants:
                mut_name = f"{system}_{mut_suffix}"
                graph_path = os.path.join(self.dirs["graphs"], f"MIST_{mut_name}.graphml")
                csv_path = os.path.join(self.dirs["results"], f"Top_Hubs_{mut_name}.csv")

                if not (os.path.exists(graph_path) and os.path.exists(csv_path)):
                    continue

                logging.info(f"Compiling PyMOL CGO script for {mut_name}...")

                graph = nx.read_graphml(graph_path)
                df_centrality = pd.read_csv(csv_path)

                max_cent = df_centrality['Differential_Centrality'].max()
                weights = [float(data.get('weight', 0)) for _, _, data in graph.edges(data=True)]
                max_weight = max(weights) if weights else 1.0

                out_script = os.path.join(self.dirs["pymol_scripts"], f"Render_MIST_{mut_name}.py")

                with open(out_script, 'w') as f:
                    f.write('from pymol.cgo import *\n')
                    f.write('from pymol import cmd\n\n')
                    f.write(f'def build_network_{mut_name}():\n')
                    f.write('    cgo_obj = []\n\n')

                    # 1. Render Edges (Covariance Cylinders)
                    f.write('    # --- EVOLUTIONARY COVARIANCE (EDGES) ---\n')
                    for u, v, data in graph.edges(data=True):
                        w = float(data.get('weight', 0))
                        if w > 0:
                            pdb_u = int(graph.nodes[u]['pdb_id'])
                            pdb_v = int(graph.nodes[v]['pdb_id'])

                            if pdb_u in coords and pdb_v in coords:
                                norm_w = min(w / max_weight, 1.0)
                                r, g, b = self._get_rgb(norm_w, 'Oranges')
                                radius = 0.10 + (norm_w * 0.40)  # Gradiente de grosor (0.1 a 0.5 A)

                                c1 = coords[pdb_u]
                                c2 = coords[pdb_v]

                                # PyMOL CYLINDER syntax: CYLINDER, x1, y1, z1, x2, y2, z2, radius, r1, g1, b1, r2, g2, b2
                                f.write(f'    cgo_obj.extend([CYLINDER, {c1[0]:.3f}, {c1[1]:.3f}, {c1[2]:.3f}, ')
                                f.write(f'{c2[0]:.3f}, {c2[1]:.3f}, {c2[2]:.3f}, {radius:.3f}, ')
                                f.write(f'{r:.3f}, {g:.3f}, {b:.3f}, {r:.3f}, {g:.3f}, {b:.3f}])\n')

                    # 2. Render Nodes (Centrality Spheres)
                    f.write('\n    # --- ALLOSTERIC HUBS (NODES) ---\n')
                    for _, row in df_centrality.iterrows():
                        pdb_id = int(row['PDB_Residue_ID'])
                        score = float(row['Differential_Centrality'])

                        if score > 0.005 and pdb_id in coords:
                            norm_score = min(score / max_cent, 1.0)
                            r, g, b = self._get_rgb(norm_score, 'Reds')
                            radius = 0.8 + (norm_score * 1.5)  # Gradiente de tamaño (0.8 a 2.3 A)

                            c = coords[pdb_id]

                            # PyMOL SPHERE syntax: COLOR, r, g, b, SPHERE, x, y, z, radius
                            f.write(f'    cgo_obj.extend([COLOR, {r:.3f}, {g:.3f}, {b:.3f}, ')
                            f.write(f'SPHERE, {c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}, {radius:.3f}])\n')

                    # 3. Load into PyMOL and set aesthetics
                    f.write(f'\n    cmd.load_cgo(cgo_obj, "Network_{mut_name}")\n')
                    f.write('    cmd.bg_color("white")\n')
                    f.write('    cmd.show("cartoon")\n')
                    f.write('    cmd.color("gray80", "polymer")\n')
                    f.write('    cmd.set("cartoon_transparency", 0.7)\n')
                    f.write('    cmd.set("ray_trace_mode", 1)\n')
                    f.write('    cmd.zoom()\n')
                    f.write(f'    print("--> MIST Network {mut_name} successfully loaded.")\n\n')

                    # Auto-execute function when the script is run
                    f.write(f'build_network_{mut_name}()\n')

                logging.info(f"PyMOL executable generated: {out_script}")

        logging.info("\n--- PHASE 5 SUCCESSFULLY COMPLETED ---")


if __name__ == "__main__":
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

    generator = PyMolCGOGenerator()
    generator.execute_cgo_pipeline(experimental_targets)