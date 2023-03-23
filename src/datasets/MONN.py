import time
import pickle
import json
import logging
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
from common.src.datasets.utils import ELEM_LIST, AA_LIST, BatchSeq

from src.datasets.CARADataset import CARADataset

import rdkit
from rdkit import Chem

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}

def batch_data_process_MONN(data):
    # re-organize
    data = list(zip(*data))
    comp, prot, aff, assay_id, value_type = data
    comp_batch = Batch.from_data_list(comp)
    prot_batch = BatchSeq(prot)
    aff = torch.Tensor(aff).reshape(-1, 1)

    return (comp_batch, prot_batch), \
        {"Affinity": aff, "assay_id": assay_id, "value_type":value_type}
    


class MONNData(object):
    def __init__(self, removeHs=True):
        self.removeHs = removeHs

    def get_compound(self, smiles):
        mol = self.get_mol(smiles)
        compound = self.get_graph(mol)

        return compound
        
    def get_protein(self, seq):
        protein = self.get_seq_features(seq)

        return protein

    def prepare_constant(self):
        # Feature lists
        self.list_element = np.array(ELEM_LIST)
        self.list_amino_acid = np.array(AA_LIST)
        self.list_degree = np.array(list(range(6)))
        self.list_explicit_valence = np.array(list(range(1, 7)))
        self.list_implicit_valence = np.array(list(range(6)))
        self.list_bond_type = np.array([rdkit.Chem.rdchem.BondType.SINGLE,
                                        rdkit.Chem.rdchem.BondType.DOUBLE,
                                        rdkit.Chem.rdchem.BondType.TRIPLE,
                                        rdkit.Chem.rdchem.BondType.AROMATIC])
        # Constant parameters
        self.atom_feature_dim = len(self.list_element) + \
                                len(self.list_degree) + \
                                len(self.list_explicit_valence) + \
                                len(self.list_implicit_valence) + 1

        self.bond_feature_dim = len(self.list_bond_type) + 2

        self.seq_feature_dim = len(self.list_amino_acid)

    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if self.removeHs:
            mol = Chem.RemoveHs(mol)
        # if ExactMolWt(mol) > 1000:
        #     raise ValueError
        # AllChem.EmbedMultipleConfs(mol, 1)
        # if mol.GetNumConformers() == 0:
        #     raise ValueError
        return mol
    
    def get_graph(self, mol):
        atom_features = self.get_atom_features(mol)
        bond_features = self.get_bond_features(mol)
        bond_neighbors = self.get_bond_neighbors(mol)

        mol_graph = Data(
            x=atom_features, 
            edge_index=bond_neighbors, 
            edge_attr=bond_features, 
        )
        
        return mol_graph

    def get_atom_features(self, mol):
        """
        obtain features for each atom of the molecule
        shape: [num_atom, atom_feature_dim]
        """
        num_atom = mol.GetNumAtoms()
        get_one_atom_feature = lambda atom: self.onehot(atom.GetSymbol(), self.list_element) + \
                                            self.onehot(atom.GetDegree(), self.list_degree) + \
                                            self.onehot(atom.GetExplicitValence(), self.list_explicit_valence) + \
                                            self.onehot(atom.GetImplicitValence(), self.list_implicit_valence) + \
                                            [atom.GetIsAromatic()]

        atom_features = list(map(get_one_atom_feature, [mol.GetAtomWithIdx(idx) for idx in range(num_atom)]))
        atom_features = torch.FloatTensor(atom_features)
        atom_features[torch.isnan(atom_features)] = 0.0
        atom_features[torch.isinf(atom_features)] = 0.0

        return atom_features

    def get_bond_features(self, mol):
        """
        obtain features for each bond
        shape: [num_bond, bond_feature_dim]
        """
        num_bond = mol.GetNumBonds()
        get_one_bond_feature = lambda bond: self.onehot(bond.GetBondType(), self.list_bond_type) + \
                                            [bond.GetIsConjugated(), bond.IsInRing()]

        bond_features = list(map(get_one_bond_feature, [mol.GetBondWithIdx(idx) for idx in range(num_bond)]))
        bond_features = torch.FloatTensor(bond_features)
        bond_features = torch.cat([bond_features, bond_features], dim=0)
        bond_features[torch.isnan(bond_features)] = 0.0
        bond_features[torch.isinf(bond_features)] = 0.0

        return bond_features

    def get_bond_neighbors(self, mol):
        """
        obtain neighbors for each bond
        shape: [2, num_bond*2]
        """
        num_bond = mol.GetNumBonds()
        get_one_bond_neighbors = lambda bond: [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        get_one_bond_neighbors_reverse = lambda bond: [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
        bond_neighbors = list(map(get_one_bond_neighbors, [mol.GetBondWithIdx(idx) for idx in range(num_bond)])) + \
                         list(map(get_one_bond_neighbors_reverse, [mol.GetBondWithIdx(idx) for idx in range(num_bond)]))
        bond_neighbors = torch.LongTensor(bond_neighbors).T
        
        return bond_neighbors

    def get_seq_features(self, seq):
        """
        obtain features for each residuke of the protein sequence
        shape: [num_seq, seq_feature_dim]
        """
        # get_one_seq_feature = lambda aa: self.onehot(aa, self.list_amino_acid)
        get_one_seq_feature = lambda aa: self.blosum_dict[aa] if aa in self.blosum_dict else np.zeros(self.seq_feature_dim)
        seq_features = np.array(list(map(get_one_seq_feature, seq)))
        seq_features = torch.FloatTensor(seq_features)

        return seq_features

    def load_blosum62(self):
        blosum_dict = {}
        lines = [
            "   A  C  D  E  F  G  H  I  K  L  M  N  P  Q  R  S  T  V  W  Y",
            "A  4  0 -2 -1 -2  0 -2 -1 -1 -1 -1 -2 -1 -1 -1  1  0  0 -3 -2",
            "C  0  9 -3 -4 -2 -3 -3 -1 -3 -1 -1 -3 -3 -3 -3 -1 -1 -1 -2 -2",
            "D -2 -3  6  2 -3 -1 -1 -3 -1 -4 -3  1 -1  0 -2  0 -1 -3 -4 -3",
            "E -1 -4  2  5 -3 -2  0 -3  1 -3 -2  0 -1  2  0  0 -1 -2 -3 -2",
            "F -2 -2 -3 -3  6 -3 -1  0 -3  0  0 -3 -4 -3 -3 -2 -2 -1  1  3",
            "G  0 -3 -1 -2 -3  6 -2 -4 -2 -4 -3  0 -2 -2 -2  0 -2 -3 -2 -3",
            "H -2 -3 -1  0 -1 -2  8 -3 -1 -3 -2  1 -2  0  0 -1 -2 -3 -2  2",
            "I -1 -1 -3 -3  0 -4 -3  4 -3  2  1 -3 -3 -3 -3 -2 -1  3 -3 -1",
            "K -1 -3 -1  1 -3 -2 -1 -3  5 -2 -1  0 -1  1  2  0 -1 -2 -3 -2",
            "L -1 -1 -4 -3  0 -4 -3  2 -2  4  2 -3 -3 -2 -2 -2 -1  1 -2 -1",
            "M -1 -1 -3 -2  0 -3 -2  1 -1  2  5 -2 -2  0 -1 -1 -1  1 -1 -1",
            "N -2 -3  1  0 -3  0  1 -3  0 -3 -2  6 -2  0  0  1  0 -3 -4 -2",
            "P -1 -3 -1 -1 -4 -2 -2 -3 -1 -3 -2 -2  7 -1 -2 -1 -1 -2 -4 -3",
            "Q -1 -3  0  2 -3 -2  0 -3  1 -2  0  0 -1  5  1  0 -1 -2 -2 -1",
            "R -1 -3 -2  0 -3 -2  0 -3  2 -2 -1  0 -2  1  5 -1 -1 -3 -3 -2",
            "S  1 -1  0  0 -2  0 -1 -2  0 -2 -1  1 -1  0 -1  4  1 -2 -3 -2",
            "T  0 -1 -1 -1 -2 -2 -2 -1 -1 -1 -1  0 -1 -1 -1  1  5  0 -2 -2",
            "V  0 -1 -3 -2 -1 -3 -3  3 -2  1  1 -3 -2 -2 -3 -2  0  4 -3 -1",
            "W -3 -2 -4 -3  1 -2 -2 -3 -3 -2 -1 -4 -4 -2 -3 -3 -2 -3 11  2",
            "Y -2 -2 -3 -2  3 -3  2 -1 -2 -1 -1 -2 -3 -1 -2 -2 -2 -1  2  7",
        ]
        skip =1 
        for i in lines:
            if skip == 1:
                skip = 0
                continue
            parsed = i.strip('\n').split()
            blosum_dict[parsed[0]] = np.array(parsed[1:]).astype(float)
        return blosum_dict

    def onehot(self, code, list_code):
        # if code not in list_code and "unknown" in list_code:
        #     return list(list_code=="unknown")

        # assert code in list_code, "{} not in list {}".format(code, list_code)
        return list(list_code==code)


class DataSet(CARADataset, MONNData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADataset.__init__(self, path, kwargs, remove_Hs)
        MONNData.__init__(self, remove_Hs)

        self.prepare_constant()
        self.blosum_dict = self.load_blosum62()


    def get_one_sample(self, idx):
        smiles = self.table.loc[idx, 'Smiles']
        seq = self.table.loc[idx, 'Target Sequence']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']
        
        compound = self.get_compound(smiles)
        protein = self.get_protein(seq)
        return compound, protein, affinity, assay_id, value_type
                
