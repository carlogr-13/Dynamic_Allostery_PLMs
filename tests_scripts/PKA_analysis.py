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
from Bio.Data import IUPACData
from dynamic_allostery_esm2 import AllostericNetworkAnalyzer


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
    Curation PDB.
    Rejects solvent (HOH, WAT), ligands, and non-target chains.
    """

    def __init__(self, target_chain: str) -> None:
        self.target_chain = target_chain

    def accept_chain(self, chain: Any) -> int:
        return 1 if chain.get_id() == self.target_chain else 0

    def accept_residue(self, residue: Any) -> int:
        return 1 if residue.id[0] == " " and residue.resname not in ["HOH", "WAT"] else 0


if __name__ == "__main__":
    analyzer = AllostericNetworkAnalyzer()

    # 1. PKA Catalytic Subunit (PDB: 1ATP) - Biological Reference Data
    # It focuses exclusively on the external dynamic sensors: P+1 loop (204) (Madan et al., 2023),
    # alpha-D sensors (133,134) (Madan et al., 2023), GHI tether (280) (Taylor & Kornev, 2010),
    # and C-tail motif FDDY (327-330) (Madan et al., 2023).
    target_allosteric_site = [133, 134, 204, 280, 327, 328, 329, 330]

    pka_sequence = "VKEFLAKAKEDFLKKWETPSQNTAQLDQFDRIKTLGTGSFGRVMLVKHKESGNHYAMKILDKQKVVKLKQIEHTLNEKRILQAVNFPFLVKLEFSFKDNSNLYMVMEYVAGGEMFSHLRRIGRFSEPHARFYAAQIVLTFEYLHSLDLIYRDLKPENLLIDQQGYIQVTDFGFAKRVKGRTWTLCGTPEYLAPEIILSKGYNKAVDWWALGVLIYEMAAGYPPFFADQPIQIYEKIVSGKVRFPSHFSSDLKDLLRNLLQVDLTKRFGNLKNGVNDIKNHKWFATTDWIAIYQRKVEAPFIPKFKGPGDTSNFDDYEEEEIRVSINEKCGKEFTE"

    # 2. Mutational Probing Setup
    mutations = {
        "I150A": [["I", 150, "A"]]
    }

    # 3. Pipeline Ignition
    analyzer.execute_pipeline(
        project_name="PKA_Allostery",
        pdb_id="1ATP",
        chain="E",
        canonical_sequence=pka_sequence,
        offset=14,
        target_residues=target_allosteric_site,
        mutational_dict=mutations,
        seed=7355608
    )