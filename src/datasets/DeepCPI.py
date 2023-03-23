import numpy as np 
import os
import pickle
from collections import defaultdict
from gensim.models.word2vec import Word2Vec
from gensim import corpora, models
from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from src.datasets.CARADataset import CARADataset


DATASET_PARAMS = {
    "task": "LO_Kinase",
    "subset": "train",
}


def batch_data_process_DeepCPI(datalist):
    # bug when batch_size == 1
    if len(datalist) == 1:
        datalist = datalist * 2
    datalist = list(zip(*datalist))
    assay_id, value_type, compound, protein, affinity = datalist
    compound_batch = torch.Tensor(np.array(compound))
    protein_batch = torch.Tensor(np.array(protein))
    affinity_batch = torch.Tensor(affinity).reshape(-1, 1)

    return (compound_batch, protein_batch),  {"Affinity": torch.FloatTensor(affinity_batch), 
                                              "assay_id": assay_id,
                                              "value_type": value_type,
                                             }


class DeepCPIData(object):

    def load_dict(self):
        path = self.root_path + "deepcpi/"
        self.dictionary = corpora.Dictionary.load(path + 'dict_for_1_nofeatureinvariant2.dict')
        self.tfidf = models.tfidfmodel.TfidfModel.load(path + 'tfidf_for_1_nofeatureinvariant2.tfidf')
        self.lsi = models.lsimodel.LsiModel.load(path + 'lsi_for_1_nofeatureinvariant2.lsi')
        self.model = Word2Vec.load(path + 'new_word2vec_model_30_new')

        self.dict_seq = {}
        self.dict_mol = {}
        if os.path.exists(path + self.kwargs['task'] + "_dict_mol"):
            self.dict_mol = pickle.load(open(path + self.kwargs['task'] + "_dict_mol", 'rb'))
        if os.path.exists(path + self.kwargs['task'] + "_dict_seq"):
            self.dict_seq = pickle.load(open(path + self.kwargs['task'] + "_dict_seq", 'rb'))

    def get_compound(self, smiles):
        if smiles not in self.dict_mol:
            mol = self.get_mol(smiles)
            compound = self.get_mol_features(mol)
            self.dict_mol[smiles] = compound

        return self.dict_mol[smiles]

    def get_protein(self, seq):
        if seq not in self.dict_seq:
            protein = self.get_seq_features(seq)
            self.dict_seq[seq] = protein

        return self.dict_seq[seq]

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

    def get_mol_features(self, mol):
        """
        Input:
            mol: mol object
        Output:
            if the compound is valid, return its 200-dimension feautre;
            else return string "error"
        """
        # m = Chem.MolFromInchi(a_compound)
        # try:
        sentence1 = ''
        flag1 = 0
        for atom in range(mol.GetNumAtoms()):
            info = {}
            fp = AllChem.GetMorganFingerprint(mol, 2, useFeatures=False, fromAtoms=[atom], bitInfo=info)
            bits = list(fp.GetNonzeroElements())	
            for i in bits:
                position = info[i]
                if position[0][1] == 1:
                    if flag1 == 0:
                        flag1 = 1
                        sentence1 += str(i)
                    else:
                        sentence1 += ' '
                        sentence1 += str(i)
        # except:
        #     return 'error'
        text = [word for word in sentence1.split()]
        frequency = defaultdict(int)
        for token in text:
            frequency[token] += 1
        corpus = self.dictionary.doc2bow(text) 
        corpus_tfidf = self.tfidf[corpus]
        corpus_lsi = self.lsi[corpus_tfidf]
        lv = np.zeros(200)
        for i in range(len(corpus_lsi)):
            lv[corpus_lsi[i][0]] = corpus_lsi[i][1]

        return lv

    def get_seq_features(self, seq):
        """
        Input:
            a_protein: a protein sequence (string format)
        Output:
            return its 100-dimension feautre
        """
        value = seq.lower()

        count1 = 0		
        features = np.zeros(100)

        begin = 0
        step = 3
        while True:
            if begin+step > len(value):
                break
            else:
                try:
                    features += self.model.wv[value[begin:begin+step]]
                    begin += step
                    count1 += 1
                except:
                    begin += step
                    continue
        begin = 1
        step = 3
        while True:
            if begin+step > len(value):
                break
            else:
                try:
                    features += self.model.wv[value[begin:begin+step]]
                    begin += step
                    count1 += 1
                except:
                    begin += step
                    continue
        begin = 2
        step = 3
        while True:
            if begin+step > len(value):
                break
            else:
                try:
                    features += self.model.wv[value[begin:begin+step]]
                    begin += step
                    count1 += 1
                except:
                    begin += step
                    continue
        features = features/float(count1)

        return features


class DataSet(CARADataset, DeepCPIData):
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

        compound = self.get_compound(smiles)
        protein = self.get_protein(seq)
        return assay_id, value_type, compound, protein, affinity
