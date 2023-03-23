import torch
import numpy as np
import networkx as nx
import rdkit.Chem as Chem
from torch_geometric.data import Data
from torch_geometric.data import Batch

from src.datasets.CARADataset import CARADataset

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}

def batch_data_process_GNN(datalist):

    datalist = list(zip(*datalist))
    assay_id, value_type, compound, affinity = datalist
    compound_batch = Batch.from_data_list(compound)
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)
    return (compound_batch, ), {'Affinity': affinity_batch, "assay_id":assay_id, "value_type":value_type}


class GraphDTAData(object):

    def get_compound(self, smiles):
        mol = self.get_mol(smiles)
        compound = self.mol_to_graph(mol)

        return compound
        
    def get_protein(self, seq):
        protein = torch.LongTensor([self.dict_char_seq[i] for i in seq[:self.max_seq_len]])
        
        return protein

    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if self.remove_Hs:
            mol = Chem.RemoveHs(mol)
        # if ExactMolWt(mol) > 1000:
        #     raise ValueError
        # AllChem.EmbedMultipleConfs(mol, 1)
        # if mol.GetNumConformers() == 0:
        #     raise ValueError
        return mol

    def register_init_feature(self):
        #load intial atom and bond features (i.e., embeddings)        
        seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"        
        self.dict_char_seq = {v:(i+1) for i,v in enumerate(seq_voc)}
        
        self.max_seq_len = 1000
        self.max_mol_len = 10 

    def atom_features(self, atom):
        return np.array(self.one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        self.one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        self.one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        self.one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                        [atom.GetIsAromatic()])

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
        
    def mol_to_graph(self, mol):              
        features = []
        for atom in mol.GetAtoms():
            feature = self.atom_features(atom).reshape(1, -1)
            features.append( feature / (sum(feature)+1e-8 ))
        features = torch.FloatTensor(np.concatenate(features))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])
        edge_index = torch.LongTensor(edge_index).T
        
        graph = Data(x=features, edge_index=edge_index)

        return graph


class DataSet(CARADataset, GraphDTAData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADataset.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.register_init_feature()

    def get_one_sample(self, idx):
        # get affinity
        smiles = self.table.loc[idx, 'Smiles']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']

        compound = self.get_compound(smiles)

        return assay_id, value_type, compound, affinity
