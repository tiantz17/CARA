import torch
import numpy as np

from torch_geometric.data import Batch

#some predefined parameters
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 6 + 6 + 1
BOND_FDIM = 6
MAX_NB = 6
MAX_MASIF_NB = 6
AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',]

# padding functions
def pad_1d(arr_list, dim=None, padding_value=0):
    if dim is None:
        dim = max([len(x) for x in arr_list])
    a = np.ones((len(arr_list), dim))*padding_value
    for i, arr in enumerate(arr_list):
        a[i, 0:len(arr)] = arr
    return a

def pad_2d(arr_list, dim1=None, dim2=None, padding_value=0, pad1d=False):
    if pad1d:
        arr_list = [pad_1d(item) for item in arr_list]
    if dim1 is None:
        dim1 = max([x.shape[0] for x in arr_list])
    if dim2 is None:
        dim2 = max([x.shape[1] for x in arr_list])
    a = np.ones((len(arr_list), dim1, dim2))*padding_value
    for i, arr in enumerate(arr_list):
        a[i, :arr.shape[0], :arr.shape[1]] = arr
    return a

def get_mask_1d(arr_list):  ## for vertex mask
    if type(arr_list[0]) == list:
        arr_list = [np.array(item) for item in arr_list]
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i,:arr.shape[0]] = 1
    return a

def get_mask_2d(arr_list, dim1=None, dim2=None, mask1d=False):
    if mask1d:
        arr_list = [get_mask_1d(item) for item in arr_list]
    if dim1 is None:
        dim1 = max([x.shape[0] for x in arr_list])
    if dim2 is None:
        dim2 = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), dim1, dim2))
    for i, arr in enumerate(arr_list):
        a[i, :arr.shape[0], :arr.shape[1]] = 1
    return a

# padding functions
def pad_label_2d(label, vertex, sequence):
    dim1 = vertex.shape[1]
    dim2 = sequence.shape[1]
    a = np.zeros((len(label), dim1, dim2))
    for i, arr in enumerate(label):
        a[i, :arr.shape[0], :arr.shape[1]] = arr
    return a

def pack1D(arr_list):
    N = max([len(x) for x in arr_list]) # x.shape[0]
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i, 0:len(arr)] = arr  # arr.shape[0]
    return a

def pack2D(arr_list, padding_value=0):
    N = max([x.shape[0] for x in arr_list])
    M = MAX_NB #max([x.shape[1] for x in arr_list])
    a = np.ones((len(arr_list), N, M))*padding_value
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def pack2Dhahaha(arr_list, padding_value=0):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.ones((len(arr_list), N, M))*padding_value
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def add_index_1d(input_array, ebd_size):  # for pocket idx
    batch_size, n_node = np.shape(input_array)
    add_idx = np.array(list(range(0, (ebd_size)*batch_size, ebd_size))*(n_node))
    add_idx = np.transpose(add_idx.reshape(-1, batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1)+add_idx
    return new_array

def add_index_2d(input_array, ebd_size):
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    add_idx = np.array(list(range(0,(ebd_size)*batch_size,ebd_size))*(n_nbs*n_vertex))
    add_idx = np.transpose(add_idx.reshape(-1,batch_size))
    add_idx = add_idx.reshape(-1)
    new_array = input_array.reshape(-1)+add_idx
    return new_array



class BatchSeq(object):
    def __init__(self, list_seq):
        self.batch_size = len(list_seq)
        self.length = torch.LongTensor([len(seq) for seq in list_seq])
        self.max_len = max(self.length)
        self.dim = list_seq[0].shape[1]
        self.x = torch.zeros(self.batch_size, self.max_len, self.dim)
        self.batch = []
        for i, seq in enumerate(list_seq):
            l = self.length[i]
            self.x[i, :l] = seq
            self.batch.extend([i]*l)
        self.batch = torch.LongTensor(self.batch)
        self.num_nodes = len(self.batch)

    def to_data_list(self):
        list_seq = []
        for i in range(self.batch_size):
            l = self.length[i]
            list_seq.append(self.x[i, :l])
        
        return list_seq

    def to_compact(self):
        list_seq = self.to_data_list()
        self.y = torch.cat(list_seq)
        assert len(self.y) == len(self.batch)

    def to(self, device):
        self.x = self.x.to(device)
        self.length = self.length.to(device)
        self.batch = self.batch.to(device)

        return self

    def _to_compact(self, x):
        list_seq = []
        for i in range(self.batch_size):
            l = self.length[i]
            list_seq.append(x[i, :l])
        y =  torch.cat(list_seq)
        return y

