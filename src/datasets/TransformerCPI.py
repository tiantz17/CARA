import torch
import numpy as np 
from rdkit import Chem
from src.datasets.CARADataset import CARADataset
from src.datasets.word2vec_transformerCPI import seq_to_kmers, get_protein_embedding
from gensim.models import Word2Vec

DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
    "test_assay":""
}


num_atom_feat = 34
def batch_data_process_TransformerCPI(datalist):
    #def pack(atoms, adjs, proteins, labels, device):

    datalist = list(zip(*datalist))
    assay_id, value_type, atoms, adjs, proteins, labels = datalist
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
 
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    atoms_new = torch.zeros((N,atoms_len,34))
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        atoms_new[i, :a_len, :] = torch.Tensor(atom)
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len))
    i = 0
    for adj in adjs:
        adj = torch.Tensor(adj)
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        adjs_new[i, :a_len, :a_len] = adj
        i += 1
    proteins_new = torch.zeros((N, proteins_len, 100))
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = torch.Tensor(protein)
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long)
    # i = 0
    # for label in labels:
    #     print(label)
    #     labels_new[i] = torch.Tensor([label])
    #     i += 1
    # print(labels_new)
    # print(labels_new)
    affinity_batch  = torch.Tensor(labels).reshape(-1, 1)
    #return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num)

    atom_num = torch.Tensor(atom_num).long()
    protein_num = torch.Tensor(protein_num).long()

    return (atoms_new, adjs_new, proteins_new, atom_num, protein_num),  {"Affinity": torch.FloatTensor(affinity_batch), "assay_id": assay_id, "value_type":value_type}


class TransformerCPIData(object):

    def load_dict(self):
        path = self.root_path + "transformercpi/"
        self.w2vmodel = Word2Vec.load(path + "word2vec_30.model")

        self.dict_seq = {}
        self.dict_atom_feat = {}
        self.dict_adj_matrix = {}
        

    def get_compound(self, smiles):
        if smiles not in self.dict_atom_feat:
            mol = self.get_mol(smiles)
            atom_feat, adj_matrix = self.get_mol_features(mol)
            
            self.dict_atom_feat[smiles] = atom_feat
            self.dict_adj_matrix[smiles] = adj_matrix

        return self.dict_atom_feat[smiles], self.dict_adj_matrix[smiles]
        
    def get_protein(self, seq):
        if seq not in self.dict_seq:
            protein = self.get_seq_features(seq)
            self.dict_seq[seq] = protein
        
        return self.dict_seq[seq]

    def get_mol(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # if self.remove_Hs:
        #     mol = Chem.RemoveHs(mol)
        
        return mol

    def one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
        return [x == s for s in allowable_set]


    def one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]


    def atom_features(self, atom,explicit_H=False,use_chirality=True):
        """Generate atom features including atom symbol(10),degree(7),formal charge,
        radical electrons,hybridization(6),aromatic(1),Chirality(3)
        """
        symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  # 10-dim
        degree = [0, 1, 2, 3, 4, 5, 6]  # 7-dim
        hybridizationType = [Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2,
                                'other']   # 6-dim
        results = self.one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                    self.one_of_k_encoding(atom.GetDegree(),degree) + \
                    [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                    self.one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]  # 10+7+2+6+1=26

        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + self.one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                        [0, 1, 2, 3, 4])   # 26+5=31
        if use_chirality:
            try:
                results = results + self.one_of_k_encoding_unk(
                        atom.GetProp('_CIPCode'),
                        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
        return results


    def adjacent_matrix(self, mol):
        adjacency = Chem.GetAdjacencyMatrix(mol)
        return np.array(adjacency,dtype=np.float32)



    def get_mol_features(self, mol):
        """
        Input:
            mol: mol object
        Output:
            if the compound is valid, return its 200-dimension feautre;
            else return string "error"
        """
        
        #mol = Chem.AddHs(mol)
        atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = self.atom_features(atom)
        adj_matrix = self.adjacent_matrix(mol)

        return atom_feat, adj_matrix


    def get_seq_features(self, seq):
        """
        For TransformerCPI
        get protein embedding,infer a list of 3-mers to (num_word,100) matrix
        """
        
        protein_embedding = get_protein_embedding(self.w2vmodel, seq_to_kmers(seq))
        return protein_embedding


class DataSet(CARADataset, TransformerCPIData):
    """
    Class of Container for chemical compounds
    """
    def __init__(self, path, kwargs, remove_Hs=True):
        CARADataset.__init__(self, path, kwargs, remove_Hs)

        # load data
        self.load_dict()  

    def get_one_sample(self, idx):
        # get affinity
        smiles = self.table.loc[idx, 'Smiles']
        seq = self.table.loc[idx, 'Target Sequence']
        affinity = self.table.loc[idx, self.kwargs['label']]
        assay_id = self.table.loc[idx, 'Task ID']
        value_type = self.table.loc[idx, 'Value Type']

        atom_feat, adj_matrix = self.get_compound(smiles)
        protein = self.get_protein(seq)

        return assay_id, value_type, atom_feat, adj_matrix, protein, affinity
