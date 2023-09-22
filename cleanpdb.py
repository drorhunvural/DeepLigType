'''
Thanks to Rishal Aggarwal for original function (https://github.com/devalab/DeepPocket/blob/main/clean_pdb.py)
'''

from Bio.PDB import PDBParser, PDBIO, Select
import Bio
import os
import sys
import time
import collections
import struct
import numpy as np

import warnings
warnings.filterwarnings("ignore")

class AminoAcidSelect(Bio.PDB.Select):
    def accept_residue(self, residue):
        return 1 if Bio.PDB.Polypeptide.is_aa(residue, standard=True) else 0

def pdb_clean(input_file, output_file):
    try:
        pdb_parser = Bio.PDB.PDBParser()
        pdb_structure = pdb_parser.get_structure("protein", input_file)
    
        pdb_io = Bio.PDB.PDBIO()
        pdb_io.set_structure(pdb_structure)
        pdb_io.save(output_file, AminoAcidSelect())
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")