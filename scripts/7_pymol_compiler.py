"""
Module 7: PyMOL Virtual Environment Compiler
============================================
This module automates the generation of PyMOL execution scripts (.pml).
It translates the statistical topological data embedded in the B-factor columns
into publication-ready 3D renders, configuring the thermodynamic color spectrum
and anisotropic spatial representations.

Author: TFG Bioinformatics Pipeline
"""

import os
import glob
import logging

# Scientific logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class PyMolCompiler:
    """
    Automates the generation of highly optimized rendering scripts for PyMOL.
    Configures ray-tracing parameters, background occlusion, and topological gradients.
    """

    def __init__(self):
        """
        Initializes the compilation environment and dynamically resolves
        absolute paths to ensure cross-platform compatibility.
        """
        # Directory Management
        self.script_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.project_root: str = os.path.dirname(self.script_dir)

        self.pdb_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "mapped_pdb")
        self.pml_dir: str = os.path.join(self.project_root, "data_dynamic_allostery", "pymol_scripts")

        os.makedirs(self.pml_dir, exist_ok=True)

    def _generate_pml_content(self, pdb_path: str, obj_name: str) -> str:
        """
        Generates the raw PyMOL command language string, imposing strict
        biophysical visualization standards.

        :param pdb_path: str, Absolute path to the mapped PDB file.
        :param obj_name: str, The designated identifier for the PyMOL object.
        :return: str, The multiline string containing the PyMOL script.
        """
        # Normalize path for PyMOL cross-platform compatibility (requires forward slashes)
        normalized_path = pdb_path.replace("\\", "/")

        # PyMOL Script Definition
        pml_script = f"""# =========================================================
# TFG DYNAMIC ALLOSTERY RENDER SCRIPT
# System: {obj_name}
# =========================================================

# 1. Environment Initialization
reinitialize
bg_color white
set ray_trace_mode, 1
set antialias, 2
set depth_cue, 1

# 2. Topology Loading
load {normalized_path}, {obj_name}
hide everything, {obj_name}

# 3. Anisotropic Putty Representation
show cartoon, {obj_name}
cartoon putty, {obj_name}
set cartoon_putty_radius, 0.2
set cartoon_putty_scale_min, 0.7
set cartoon_putty_scale_max, 2.5
set cartoon_putty_transform, 0

# 4. Thermodynamic Gradient (Z-Score)
# Blue: Z < -1.5 (Loss of Epistatic Coupling)
# White: Z ~ 0.0 (Thermodynamic Invariance)
# Red: Z > +1.5 (Gain of Epistatic Coupling)
spectrum b, blue_white_red, {obj_name}, minimum=-2.0, maximum=2.0

# 5. Aesthetic Post-Processing
orient {obj_name}
center {obj_name}
zoom {obj_name}, 2.0

# Execute high-quality ray tracing (Uncomment for automated saving)
# ray 1200, 1200
# png {obj_name}_render.png, dpi=300
"""
        return pml_script

    def execute_compilation(self) -> None:
        """
        Master orchestration routine. Scans the target directory for topologically
        mapped PDB files and compiles individual `.pml` scripts for each structural microstate.

        :return: None
        """
        logging.info("===================================================")
        logging.info("INITIATING PYMOL VIRTUAL ENVIRONMENT COMPILER")
        logging.info("===================================================")

        pdb_files = glob.glob(os.path.join(self.pdb_dir, "*_Mapped.pdb"))
        if not pdb_files:
            logging.error(f"No mapped PDB files found in {self.pdb_dir}.")
            return

        for pdb_path in pdb_files:
            filename = os.path.basename(pdb_path)
            # Remove the extension to use as object name
            obj_name = filename.replace(".pdb", "")

            logging.info(f"Compiling rendering environment for: {obj_name}")

            # Generate script content
            pml_content = self._generate_pml_content(pdb_path, obj_name)

            # Serialize .pml file
            out_pml_path = os.path.join(self.pml_dir, f"Render_{obj_name}.pml")
            with open(out_pml_path, "w") as f:
                f.write(pml_content)

            logging.info(f"   -> Script successfully compiled: Render_{obj_name}.pml")

        logging.info("PYMOL COMPILATION COMPLETED.")
        logging.info("===================================================")


if __name__ == "__main__":
    compiler = PyMolCompiler()
    compiler.execute_compilation()