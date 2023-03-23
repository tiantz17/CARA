import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from common.src.datasets.utils import ATOM_FDIM, BOND_FDIM

from torch_scatter import scatter_sum, scatter_max, scatter_mean


def scatter_softmax(a, index, dim=0):
    """
    softmax for scatter data structure
    """
    a_max, _ = scatter_max(a, index, dim)
    a_exp = torch.exp(a - a_max.index_select(0, index))
    a_sum = scatter_sum(a_exp, index, dim) + 1e-6
    a_softmax = a_exp / a_sum.index_select(0, index)
    return a_softmax


def fill_feature(site_feature, anch_feature, siteGraphBatch, anchGraphBatch, protGraphBatch):
    """
    fill graph with features
    atom->anchor or masif->anchor
    """
    siteGraphBatch.x = site_feature
    anchGraphBatch.x = anch_feature
    list_site = siteGraphBatch.to_data_list()
    list_anch = anchGraphBatch.to_data_list()
    list_prot = protGraphBatch.to_data_list()
    for site, anch, prot in zip(list_site, list_anch, list_prot):
        prot.x = torch.cat([site.x, anch.x])
    protGraphBatch = protGraphBatch.from_data_list(list_prot)
    return protGraphBatch


def get_split_feature(protGraphBatch):
    """
    get feature of source nodes and target nodes
    """
    site_feature = [prot.x[:prot.kwargs['num_site']] for prot in protGraphBatch.to_data_list()]
    anch_feature = [prot.x[prot.kwargs['num_site']:] for prot in protGraphBatch.to_data_list()]
    site_feature = torch.cat(site_feature, dim=0)
    anch_feature = torch.cat(anch_feature, dim=0)

    return site_feature, anch_feature


class WLNConv(MessagePassing):
    """
    Weisfeiler Lehman relabelling layer
    """
    def __init__(self, in_channels, out_channels):
        super(WLNConv, self).__init__(aggr='add')
        # WLN parameters
        self.label_U2 = nn.Sequential( #assume no edge feature transformation
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.label_U1 = nn.Linear(out_channels*2, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
    def message(self, x_j, edge_attr=None):
        if edge_attr is None:
            z = x_j
        else:
            z = torch.cat([x_j, edge_attr], dim=-1)
        return self.label_U2(z)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.label_U1(z)


class WLNkConv(MessagePassing):
    """
    Weisfeiler Lehman relabelling layer with k=3
    """
    def __init__(self, in_channels, out_channels, k=2):
        super(WLNkConv, self).__init__(aggr='add')
        # WLN parameters
        self.lin = nn.Linear(out_channels * (k+1), out_channels)
        self.K = k

        self.label_U2 = nn.Sequential( #assume no edge feature transformation
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.label_U1 = nn.Linear(out_channels*2, out_channels)
        
    def forward(self, x, edge_index, edge_attr):

        xs = [x]
        for _ in range(self.K):
            out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            xs.append(out)
        return self.lin(torch.cat(xs, dim=-1))
        
    def message(self, x_j, edge_attr=None):
        if edge_attr is None:
            z = x_j
        else:
            z = torch.cat([x_j, edge_attr], dim=-1)
        return self.label_U2(z)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.label_U1(z)


class GWM(nn.Module):
    """
    Graph Warp Module, see paper for more detail
    """
    def __init__(self, hidden_comp, GRU_main, GRU_super, k_head=1):
        super(GWM, self).__init__()
        self.hidden_comp = hidden_comp
        self.k_head = k_head
        
        # Transmitter parameters
        self.W_a_main = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) for i in range(self.k_head)]) 
        self.W_a_super = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) for i in range(self.k_head)]) 
        self.W_main = nn.ModuleList([
            nn.Linear(self.hidden_comp, self.hidden_comp) for i in range(self.k_head)]) 
        self.W_bmm = nn.ModuleList([
            nn.Linear(self.hidden_comp, 1) for i in range(self.k_head)]) 
        
        self.W_super = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) 
        self.W_main_to_super = nn.Sequential(
            nn.Linear(self.hidden_comp*self.k_head, self.hidden_comp),
            nn.Tanh(),
            ) 
        self.W_super_to_main = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            )
        
        # Warp gate
        self.W_zm1 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zm2 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zs1 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zs2 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.GRU_main = GRU_main
        self.GRU_super = GRU_super
        
        # WLN parameters
        self.WLN = WLNConv(self.hidden_comp+BOND_FDIM, self.hidden_comp)
    
    def forward(self, vertex_feature, super_feature, molGraphBatch):
        edge_initial = molGraphBatch.edge_attr
        edge_index = molGraphBatch.edge_index
        batch = molGraphBatch.batch
        
        # prepare main node features
        for k in range(self.k_head):
            a_main = self.W_a_main[k](vertex_feature)
            a_super = self.W_a_super[k](super_feature)
            a = self.W_bmm[k](a_main * a_super.index_select(0, batch))
            attn = scatter_softmax(a.view(-1), batch).view(-1, 1)
            k_main_to_super = scatter_sum(attn * self.W_main[k](vertex_feature), batch, dim=0)
            if k == 0:
                m_main_to_super = k_main_to_super
            else:
                m_main_to_super = torch.cat([m_main_to_super, k_main_to_super], dim=-1)  # concat k-head
        main_to_super = self.W_main_to_super(m_main_to_super)
        super_to_main = self.W_super_to_main(super_feature)

        main_self = self.WLN(x=vertex_feature, edge_index=edge_index, edge_attr=edge_initial)  
        super_self = self.W_super(super_feature)

        # warp gate and GRU for update main node features, use main_self and super_to_main
        z_main = torch.sigmoid(self.W_zm1(main_self) + self.W_zm2(super_to_main).index_select(0, batch)) 
        hidden_main = (1-z_main)*main_self + z_main*super_to_main.index_select(0, batch)
        vertex_feature = self.GRU_main(hidden_main, vertex_feature)
        # warp gate and GRU for update super node features
        z_supper = torch.sigmoid(self.W_zs1(super_self) + self.W_zs2(main_to_super))  
        hidden_super = (1-z_supper)*super_self + z_supper*main_to_super  
        super_feature = self.GRU_super(hidden_super, super_feature)

        return vertex_feature, super_feature


class ParallelGNN(nn.Module):
    """
    Parallel Graph Neural Network, modified from GWM
    """
    def __init__(self, hidden_comp, GRU_main, GRU_supe, GNNConv="WLN", edge_dim=0, k_head=1, RemoveSCl='0'):
        super(ParallelGNN, self).__init__()
        self.hidden_comp = hidden_comp
        self.k_head = k_head
        
        # GWM parameters
        self.Transmitter = TransmitterConv(self.hidden_comp, self.k_head, RemoveSCl=RemoveSCl)
                
        # GNN parameters
        if GNNConv == "WLN":
            self.W_main = WLNConv(self.hidden_comp+edge_dim, self.hidden_comp)
        elif GNNConv == "WLN1":
            self.W_main = WLNkConv(self.hidden_comp+edge_dim, self.hidden_comp, k=1)
        elif GNNConv == "WLN2":
            self.W_main = WLNkConv(self.hidden_comp+edge_dim, self.hidden_comp, k=2)
        elif GNNConv == "WLN3":
            self.W_main = WLNkConv(self.hidden_comp+edge_dim, self.hidden_comp, k=3)
        else:
            raise NotImplementedError
        self.W_supe = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_comp),
            nn.Tanh(),
            ) 

        self.W_main_to_supe = nn.Sequential(
            nn.Linear(self.hidden_comp*self.k_head, self.hidden_comp),
            nn.Tanh(),
            ) 
        self.W_supe_to_main = nn.Sequential(
            nn.Linear(self.hidden_comp*self.k_head, self.hidden_comp),
            nn.Tanh(),
            )
        
        self.W_zm1 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zm2 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zs1 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.W_zs2 = nn.Linear(self.hidden_comp, self.hidden_comp)
        self.GRU_main = GRU_main
        self.GRU_supe = GRU_supe
        
    def forward(self, main_feature, supe_feature, mainGraph, supeGraph, wholGraph):
        wholGraph = fill_feature(main_feature, supe_feature, mainGraph, supeGraph, wholGraph)
        supe_to_main, main_to_supe = self.Transmitter(mainGraph, supeGraph, wholGraph)

        main_self = self.W_main(x=main_feature, edge_index=mainGraph.edge_index, edge_attr=mainGraph.edge_attr)  
        supe_self = self.W_supe(supe_feature)

        # warp gate and GRU for update main node features, use main_self and super_to_main
        z_main = torch.sigmoid(self.W_zm1(main_self) + self.W_zm2(supe_to_main)) 
        hidden_main = (1 - z_main) * main_self + z_main * supe_to_main
        main_feature = self.GRU_main(hidden_main, main_feature)
        # warp gate and GRU for update super node features
        z_supe = torch.sigmoid(self.W_zs1(supe_self) + self.W_zs2(main_to_supe))  
        hidden_supe = (1 - z_supe) * supe_self + z_supe * main_to_supe  
        supe_feature = self.GRU_supe(hidden_supe, supe_feature)

        return main_feature, supe_feature


class AtomConv(MessagePassing):
    """
    Message Passing for atom
    """
    def __init__(self, in_channels, out_channels):
        super(AtomConv, self).__init__(aggr='add')
        # parameters
        self.U2 = nn.Sequential(
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.U1 = nn.Linear(in_channels+out_channels, out_channels)
        
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        
    def message(self, x_j):
        return self.U2(x_j)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.U1(z)


class AnchorConv(MessagePassing):
    """
    Message Passing for anchor
    """
    def __init__(self, in_channels, out_channels):
        super(AnchorConv, self).__init__(aggr='add')
        # parameters
        self.U2 = nn.Sequential(
            nn.Linear(in_channels, out_channels), 
            nn.LeakyReLU(0.1),
        )
        self.U1 = nn.Linear(in_channels+out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_j, edge_weight):
        if edge_weight is not None:
            return self.U2(x_j) * edge_weight
        else:
            return self.U2(x_j)

    def update(self, message, x):
        z = torch.cat([x, message], dim=-1)
        return self.U1(z)


class UpwardConv(MessagePassing):
    """
    Message Passing from bottom level to top level
    e.g., atom   -> anchor
          masif  -> anchor
          vertex -> group
    """
    def __init__(self):
        super(UpwardConv, self).__init__(aggr='add')
        # parameters
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_j, edge_weight):
        # shape of edge
        if edge_weight is not None:
            return x_j * edge_weight
        else:
            return x_j

    def update(self, message):
        # shape of node
        return message


class UpwardGatedConv(MessagePassing):
    """
    Message Passing from bottom level to top level, add gated unit
    e.g., atom   -> anchor
          masif  -> anchor
          vertex -> group
    """
    def __init__(self, in_channels, out_channels):
        super(UpwardGatedConv, self).__init__(aggr='add')
        # parameters
        self.w_alpha = nn.Sequential(
            nn.Linear(in_channels*2, out_channels),
            nn.Tanh(),
        )
        self.wx = nn.Linear(in_channels, out_channels)

        self.wz1 = nn.Linear(out_channels, out_channels)
        self.wz2 = nn.Linear(in_channels, out_channels)

        self.gru = nn.GRUCell(out_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
        
    def message(self, x_j, x_i, edge_weight):
        # shape of edge
        a = self.w_alpha(torch.cat([x_j, x_i], dim=-1))
        hidden = self.wx(x_j) * a

        if edge_weight is not None:
            return hidden * edge_weight
        else:
            return hidden

    def update(self, message, x):
        # shape of node
        z = torch.sigmoid(self.wz1(message) + self.wz2(x))  
        hidden = (1 - z) * x + z * message  
        feature = self.gru(hidden, x)

        return feature


class TransmitterConv(nn.Module):
    """
    Message Passing from main level to super level, see GWM paper for more detail
    """
    def __init__(self, hidden_dim, k_head=1, RemoveSCl='0'):
        super(TransmitterConv, self).__init__()
        # parameters        
        self.hidden_dim = hidden_dim
        self.k_head = k_head
        
        # GWM Transmitter parameters
        self.W_a_main = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            ) for i in range(self.k_head)]) 
        self.W_a_supe = nn.ModuleList([nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            ) for i in range(self.k_head)]) 

        self.W_main = nn.ModuleList([SoftmaxConv(self.hidden_dim, RemoveSCl=RemoveSCl) for i in range(self.k_head)])
        self.W_supe = nn.ModuleList([SoftmaxConv(self.hidden_dim, RemoveSCl=RemoveSCl) for i in range(self.k_head)]) 
                
        self.W_main_to_supe = nn.Sequential(
            nn.Linear(self.hidden_dim*self.k_head, self.hidden_dim),
            nn.Tanh(),
            ) 
        self.W_supe_to_main = nn.Sequential(
            nn.Linear(self.hidden_dim*self.k_head, self.hidden_dim),
            nn.Tanh(),
            )

    def forward(self, mainGraph, supeGraph, wholGraph):
        main_feature, supe_feature = get_split_feature(wholGraph)
        for k in range(self.k_head):
            a_main = self.W_a_main[k](main_feature)
            a_supe = self.W_a_supe[k](supe_feature)
            wholeGraph = fill_feature(a_main, a_supe, mainGraph, supeGraph, wholGraph)
            wholeGraph.x = self.W_main[k](wholeGraph.x, wholeGraph.edge_index)
            _, k_main_to_supe = get_split_feature(wholeGraph)
            wholeGraph = fill_feature(a_main, a_supe, mainGraph, supeGraph, wholGraph)
            wholeGraph.x = self.W_supe[k](wholeGraph.x, wholeGraph.edge_index[[1,0]])
            k_supe_to_main, _ = get_split_feature(wholeGraph)
            if k == 0:
                m_main_to_supe = k_main_to_supe
                m_supe_to_main = k_supe_to_main
            else:
                m_main_to_supe = torch.cat([m_main_to_supe, k_main_to_supe], dim=-1)  # concat k-head
                m_supe_to_main = torch.cat([m_supe_to_main, k_supe_to_main], dim=-1)  # concat k-head
        main_to_supe = self.W_main_to_supe(m_main_to_supe)
        supe_to_main = self.W_supe_to_main(m_supe_to_main)

        return supe_to_main, main_to_supe


class SoftmaxConv(MessagePassing):
    """
    Softmax attention layer
    """
    def __init__(self, hidden_dim, RemoveSCl='0'):
        super(SoftmaxConv, self).__init__(aggr='add')
        # parameters        
        self.hidden_dim = hidden_dim
        self.RemoveSCl = RemoveSCl
        if self.RemoveSCl == '0':
            self.W_supe = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_main_attn = nn.Linear(self.hidden_dim, 1)
        self.W_main = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)
        
    def message(self, x_i, x_j, edge_index):
        # shape of edge
        if self.RemoveSCl == '0':
            a = self.W_main_attn(self.W_supe(x_i) * x_j)
        else:
            a = self.W_main_attn(x_i * x_j)
        attn = scatter_softmax(a, edge_index[1], 0)
        main_to_supe = attn * self.W_main(x_j)
        return main_to_supe

    def update(self, message):
        # shape of node
        return message


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_comp, hidden_bond, k_head, depth):
        super(GraphNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_comp = hidden_comp
        self.hidden_bond = hidden_bond
        self.k_head = k_head
        self.depth = depth

        self.proj = nn.Linear(self.input_dim, self.hidden_comp)
        self.GRU_main = nn.GRUCell(self.hidden_comp, self.hidden_comp)
        self.GRU_super = nn.GRUCell(self.hidden_comp, self.hidden_comp)
        self.GNN = nn.ModuleList()
        for _ in range(self.depth):
            self.GNN.append(GWM(self.hidden_comp, self.GRU_main, self.GRU_super, self.k_head))

    def forward(self, compound):
        x = compound.x

        x = self.proj(x)
        g = scatter_sum(x, compound.batch, dim=0)

        for GNN in self.GNN:
            x, g = GNN(x, g, compound)
        
        return x, g


class GraphNeuralNetworkSimple(nn.Module):
    def __init__(self, input_dim, hidden_comp, hidden_bond, depth):
        super(GraphNeuralNetworkSimple, self).__init__()
        self.input_dim = input_dim
        self.hidden_comp = hidden_comp
        self.hidden_bond = hidden_bond
        self.depth = depth

        self.proj = nn.Linear(self.input_dim, self.hidden_comp)
        self.GNN = nn.ModuleList()
        for _ in range(self.depth):
            self.GNN.append(WLNConv(self.hidden_comp+self.hidden_bond, self.hidden_comp))

    def forward(self, compound):
        x = compound.x

        x = self.proj(x)

        for GNN in self.GNN:
            x = GNN(x, compound.edge_index, compound.edge_attr)
        
        return x


class GraphNeuralNetworkCliffNetB(nn.Module):
    def __init__(self, input_dim, hidden_vert, hidden_bond, hidden_frag, hidden_comp, k_head_vert, k_head_frag, depth_vert, depth_frag):
        super(GraphNeuralNetworkCliffNetB, self).__init__()
        self.input_dim = input_dim
        self.hidden_vert = hidden_vert
        self.hidden_bond = hidden_bond
        self.hidden_frag = hidden_frag
        self.hidden_comp = hidden_comp
        self.k_head_vert = k_head_vert
        self.k_head_frag = k_head_frag
        self.depth_vert = depth_vert
        self.depth_frag = depth_frag

        self.GNN_vert = GraphNeuralNetworkSimple(self.input_dim, self.hidden_vert, self.hidden_bond, self.depth_vert)
        self.ANN = UpwardGatedConv(self.hidden_vert, self.hidden_frag)
        self.GNN_frag = GraphNeuralNetworkSimple(self.hidden_frag, self.hidden_comp, 0, self.depth_frag)

    def forward(self, vert, frag, link):
        x = self.GNN_vert(vert)
        link.x = torch.zeros(link.num_nodes, x.shape[1], device=x.device)
        link.x[link.mask==False] = x
        h = self.ANN(link.x, link.edge_index)
        frag.x = h[link.mask]
        y = self.GNN_frag(frag)

        return y
        

class ConvNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_prot, kernel_size, depth):
        super(ConvNeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_prot = hidden_prot
        self.kernel_size = kernel_size
        self.depth = depth

        padding_size = int((self.kernel_size-1)/2)
        self.CNN = nn.ModuleList()
        input_dim = self.input_dim
        for _ in range(self.depth):
            self.CNN.append(nn.Conv1d(input_dim, self.hidden_prot, kernel_size=self.kernel_size, padding=padding_size))
            input_dim = self.hidden_prot

    def forward(self, protein):
        x = protein.x

        x = x.transpose(1, 2)
        for CNN in self.CNN:
            x = CNN(x)
            x = torch.relu(x)
        x = x.transpose(1, 2)

        return x


class AffinityNeuralNetworkCliffNetF(nn.Module):
    def __init__(self, hidden_comp, hidden_prot, hidden_aff):
        super(AffinityNeuralNetworkCliffNetF, self).__init__()
        self.hidden_comp = hidden_comp
        self.hidden_prot = hidden_prot
        self.hidden_aff = hidden_aff
        
        # Affinity module
        self.c_aff = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.c_sup = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.p_aff = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )

        self.W_out_raw = nn.Sequential(
            nn.Linear(self.hidden_aff*3, self.hidden_aff),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_aff, self.hidden_aff)
        )
        self.W_out_alpha = nn.Sequential(
            nn.Linear(self.hidden_aff*3, self.hidden_aff),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_aff, 1),
        )
        self.W_out = nn.Sequential(
            nn.Linear(self.hidden_aff, self.hidden_aff),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_aff, 1),
        )

    def get_affinity(self, comp_feature, prot_feature, batch_comp, batch_prot):
        comp_embedding = self.c_aff(comp_feature)
        prot_embedding = self.p_aff(prot_feature)
        prot_embedding = scatter_max(prot_embedding, batch_prot, dim=0)[0]
        supe_embedding = scatter_sum(self.c_sup(comp_feature), batch_comp, dim=0)

        feature = torch.cat([comp_embedding, supe_embedding.index_select(0, batch_comp), prot_embedding.index_select(0, batch_comp)], dim=1)
        raw = self.W_out_raw(feature)
        prealpha = self.W_out_alpha(feature)
        alpha = scatter_softmax(prealpha, batch_comp, dim=0)
        
        vector = scatter_sum(raw * alpha, batch_comp, dim=0)
        affinity_pred = self.W_out(vector)

        return vector, alpha, affinity_pred

    def forward(self, *input):
        if len(input) == 4:
            comp_feature, prot_feature, batch_comp, batch_prot = input
            vector, alpha, affinity_pred = self.get_affinity(comp_feature, prot_feature, batch_comp, batch_prot)

            return vector, alpha, affinity_pred


class AffinityNeuralNetworkMONN(nn.Module):
    def __init__(self, hidden_comp, hidden_prot, hidden_aff, DMA_depth):
        super(AffinityNeuralNetworkMONN, self).__init__()
        self.hidden_comp = hidden_comp
        self.hidden_prot = hidden_prot
        self.hidden_aff = hidden_aff
        self.DMA_depth = DMA_depth
        
        # Pairwise module
        self.pairwise_comp = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.pairwise_prot = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )

        # DMA module
        self.mc0 = nn.Linear(self.hidden_aff, self.hidden_aff)
        self.mp0 = nn.Linear(self.hidden_aff, self.hidden_aff)
        self.mc1 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.mp1 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hc0 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hp0 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hc1 = nn.ModuleList([nn.Linear(self.hidden_aff, 1) for i in range(self.DMA_depth)])
        self.hp1 = nn.ModuleList([nn.Linear(self.hidden_aff, 1) for i in range(self.DMA_depth)])
        self.c_to_p_transform = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.p_to_c_transform = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.GRU_dma = nn.GRUCell(self.hidden_aff, self.hidden_aff)

        # Affinity module
        self.c_aff = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.p_aff = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.super_aff = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )

        self.W_out = nn.Linear(self.hidden_aff*self.hidden_aff*2, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def dma_gru(self, comp_embedding, prot_embedding, pairwise_pred, batch_comp, batch_prot, list_num):
        c0 = scatter_mean(comp_embedding, batch_comp, dim=0)
        p0 = scatter_mean(prot_embedding, batch_prot, dim=0)
        m = c0*p0
        batch_size = len(pairwise_pred)

        for DMA_iter in range(self.DMA_depth):
            c_pre = torch.tanh(self.c_to_p_transform[DMA_iter](comp_embedding))
            p_pre = torch.tanh(self.p_to_c_transform[DMA_iter](prot_embedding))

            get_c2p = lambda i: torch.matmul(pairwise_pred[i].T, c_pre[list_num[i, 0]:list_num[i+1, 0]])
            get_p2c = lambda i: torch.matmul(pairwise_pred[i], p_pre[list_num[i, 1]:list_num[i+1, 1]])

            p_to_c = torch.cat(list(map(get_p2c, range(batch_size))))
            c_to_p = torch.cat(list(map(get_c2p, range(batch_size))))

            c_tmp = torch.tanh(self.hc0[DMA_iter](comp_embedding)) * \
                    torch.tanh(self.mc1[DMA_iter](m)).index_select(0, batch_comp) * \
                    p_to_c
            p_tmp = torch.tanh(self.hp0[DMA_iter](prot_embedding)) * \
                    torch.tanh(self.mp1[DMA_iter](m)).index_select(0, batch_prot) * \
                    c_to_p

            cf = scatter_sum(comp_embedding * scatter_softmax(self.hc1[DMA_iter](c_tmp), batch_comp, dim=0), batch_comp, dim=0)
            pf = scatter_sum(prot_embedding * scatter_softmax(self.hp1[DMA_iter](p_tmp), batch_prot, dim=0), batch_prot, dim=0)
            
            m = self.GRU_dma(m, cf*pf)
            
        return cf, pf

    def get_affinity(self, comp_feature, gomp_feature, prot_feature, batch_comp, batch_prot):
        pairwise_comp_feature = self.pairwise_comp(comp_feature)
        pairwise_prot_feature = self.pairwise_prot(prot_feature)
        batch_size = max(batch_comp) + 1
        len_comp = torch.bincount(batch_comp+1, minlength=batch_size+1).reshape(-1, 1)
        len_prot = torch.bincount(batch_prot+1, minlength=batch_size+1).reshape(-1, 1)
        list_num = torch.cat([len_comp, len_prot], dim=1)
        for i in range(batch_size): 
            list_num[i+1] = list_num[i] + list_num[i+1]
        get_pairwise = lambda i: torch.sigmoid(torch.matmul(pairwise_comp_feature[list_num[i, 0]:list_num[i+1, 0]], 
                                                            pairwise_prot_feature[list_num[i, 1]:list_num[i+1, 1]].T))
        pairwise_pred = list(map(get_pairwise, range(batch_size)))

        comp_embedding = self.c_aff(comp_feature)
        prot_embedding = self.p_aff(prot_feature)
        aff_comp_feature, aff_prot_feature = self.dma_gru(comp_embedding, prot_embedding, pairwise_pred, batch_comp, batch_prot, list_num)

        super_feature = self.super_aff(gomp_feature)
        aff_comp_feature = torch.cat([aff_comp_feature, super_feature], dim=1)
        kroneck = torch.matmul(aff_comp_feature.unsqueeze(2), aff_prot_feature.unsqueeze(1))
        kroneck = self.leaky_relu(kroneck.view(kroneck.shape[0], -1))
        
        affinity_pred = self.W_out(kroneck)

        return affinity_pred

    def forward(self, *input):
        if len(input) == 5:
            comp_feature, gomp_feature, prot_feature, batch_comp, batch_prot = input
            affinity_pred = self.get_affinity(comp_feature, gomp_feature, prot_feature, batch_comp, batch_prot)

            return affinity_pred

        elif len(input) == 7:
            comp_feature, gomp_feature, prot1_feature, prot2_feature, batch_comp, batch_prot1, batch_prot2 = input
            affinity1_pred = self.get_affinity(comp_feature, gomp_feature, prot1_feature, batch_comp, batch_prot1)
            affinity2_pred = self.get_affinity(comp_feature, gomp_feature, prot2_feature, batch_comp, batch_prot2)

            return affinity1_pred - affinity2_pred 
        
        else:
            raise NotImplementedError


class AffinityNeuralNetworkCliffNetMONN(nn.Module):
    def __init__(self, hidden_comp, hidden_prot, hidden_aff, DMA_depth):
        super(AffinityNeuralNetworkCliffNetMONN, self).__init__()
        self.hidden_comp = hidden_comp
        self.hidden_prot = hidden_prot
        self.hidden_aff = hidden_aff
        self.DMA_depth = DMA_depth
        
        # Pairwise module
        self.pairwise_comp = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.pairwise_prot = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )

        # DMA module
        self.mc0 = nn.Linear(self.hidden_aff, self.hidden_aff)
        self.mp0 = nn.Linear(self.hidden_aff, self.hidden_aff)
        self.mc1 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.mp1 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hc0 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hp0 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hc1 = nn.ModuleList([nn.Linear(self.hidden_aff, 1) for i in range(self.DMA_depth)])
        self.hp1 = nn.ModuleList([nn.Linear(self.hidden_aff, 1) for i in range(self.DMA_depth)])
        self.c_to_p_transform = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.p_to_c_transform = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.GRU_dma = nn.GRUCell(self.hidden_aff, self.hidden_aff)

        # Affinity module
        self.c_aff = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.p_aff = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )

        self.W_out = nn.Linear(self.hidden_aff*self.hidden_aff, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def dma_gru(self, comp_embedding, prot_embedding, pairwise_pred, batch_comp, batch_prot, list_num):
        c0 = scatter_mean(comp_embedding, batch_comp, dim=0)
        p0 = scatter_mean(prot_embedding, batch_prot, dim=0)
        m = c0*p0
        batch_size = len(pairwise_pred)

        for DMA_iter in range(self.DMA_depth):
            c_pre = torch.tanh(self.c_to_p_transform[DMA_iter](comp_embedding))
            p_pre = torch.tanh(self.p_to_c_transform[DMA_iter](prot_embedding))

            get_c2p = lambda i: torch.matmul(pairwise_pred[i].T, c_pre[list_num[i, 0]:list_num[i+1, 0]])
            get_p2c = lambda i: torch.matmul(pairwise_pred[i], p_pre[list_num[i, 1]:list_num[i+1, 1]])

            p_to_c = torch.cat(list(map(get_p2c, range(batch_size))))
            c_to_p = torch.cat(list(map(get_c2p, range(batch_size))))

            c_tmp = torch.tanh(self.hc0[DMA_iter](comp_embedding)) * \
                    torch.tanh(self.mc1[DMA_iter](m)).index_select(0, batch_comp) * \
                    p_to_c
            p_tmp = torch.tanh(self.hp0[DMA_iter](prot_embedding)) * \
                    torch.tanh(self.mp1[DMA_iter](m)).index_select(0, batch_prot) * \
                    c_to_p

            cf = scatter_sum(comp_embedding * scatter_softmax(self.hc1[DMA_iter](c_tmp), batch_comp, dim=0), batch_comp, dim=0)
            pf = scatter_sum(prot_embedding * scatter_softmax(self.hp1[DMA_iter](p_tmp), batch_prot, dim=0), batch_prot, dim=0)
            
            m = self.GRU_dma(m, cf*pf)
            
        return cf, pf

    def get_affinity(self, comp_feature, prot_feature, batch_comp, batch_prot):
        pairwise_comp_feature = self.pairwise_comp(comp_feature)
        pairwise_prot_feature = self.pairwise_prot(prot_feature)
        batch_size = max(batch_comp) + 1
        len_comp = torch.bincount(batch_comp+1, minlength=batch_size+1).reshape(-1, 1)
        len_prot = torch.bincount(batch_prot+1, minlength=batch_size+1).reshape(-1, 1)
        list_num = torch.cat([len_comp, len_prot], dim=1)
        for i in range(batch_size): 
            list_num[i+1] = list_num[i] + list_num[i+1]
        get_pairwise = lambda i: torch.sigmoid(torch.matmul(pairwise_comp_feature[list_num[i, 0]:list_num[i+1, 0]], 
                                                            pairwise_prot_feature[list_num[i, 1]:list_num[i+1, 1]].T))
        pairwise_pred = list(map(get_pairwise, range(batch_size)))

        comp_embedding = self.c_aff(comp_feature)
        prot_embedding = self.p_aff(prot_feature)
        aff_comp_feature, aff_prot_feature = self.dma_gru(comp_embedding, prot_embedding, pairwise_pred, batch_comp, batch_prot, list_num)

        kroneck = torch.matmul(aff_comp_feature.unsqueeze(2), aff_prot_feature.unsqueeze(1))
        kroneck = self.leaky_relu(kroneck.view(kroneck.shape[0], -1))
        
        affinity_pred = self.W_out(kroneck)

        return affinity_pred

    def forward(self, *input):
        if len(input) == 4:
            comp_feature, prot_feature, batch_comp, batch_prot = input
            affinity_pred = self.get_affinity(comp_feature, prot_feature, batch_comp, batch_prot)

            return affinity_pred

        elif len(input) == 6:
            comp_feature, prot1_feature, prot2_feature, batch_comp, batch_prot1, batch_prot2 = input
            affinity1_pred = self.get_affinity(comp_feature, prot1_feature, batch_comp, batch_prot1)
            affinity2_pred = self.get_affinity(comp_feature, prot2_feature, batch_comp, batch_prot2)

            return affinity1_pred - affinity2_pred 
        
        else:
            raise NotImplementedError


class AffinityNeuralNetworkCliffNetDMA(nn.Module):
    def __init__(self, hidden_comp, hidden_prot, hidden_aff, DMA_depth):
        super(AffinityNeuralNetworkCliffNetDMA, self).__init__()
        self.hidden_comp = hidden_comp
        self.hidden_prot = hidden_prot
        self.hidden_aff = hidden_aff
        self.DMA_depth = DMA_depth
        
        # Pairwise module
        self.pairwise_comp = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.pairwise_prot = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )

        # DMA module
        self.mc0 = nn.Linear(self.hidden_aff, self.hidden_aff)
        self.mp0 = nn.Linear(self.hidden_aff, self.hidden_aff)
        self.mc1 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.mp1 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hc0 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hp0 = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.hc1 = nn.ModuleList([nn.Linear(self.hidden_aff, 1) for i in range(self.DMA_depth)])
        self.hp1 = nn.ModuleList([nn.Linear(self.hidden_aff, 1) for i in range(self.DMA_depth)])
        self.c_to_p_transform = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.p_to_c_transform = nn.ModuleList([nn.Linear(self.hidden_aff, self.hidden_aff) for i in range(self.DMA_depth)])
        self.GRU_dma = nn.GRUCell(self.hidden_aff, self.hidden_aff)

        # Affinity module
        self.c_aff = nn.Sequential(
            nn.Linear(self.hidden_comp, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.p_aff = nn.Sequential(
            nn.Linear(self.hidden_prot, self.hidden_aff),
            nn.LeakyReLU(0.1),
        )
        self.W_out_raw = nn.Sequential(
            nn.Linear(self.hidden_aff*self.hidden_aff, self.hidden_aff),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_aff, self.hidden_aff)
        )
        self.W_out_alpha = nn.Sequential(
            nn.Linear(self.hidden_aff*self.hidden_aff, self.hidden_aff),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_aff, 1),
        )
        self.W_out = nn.Sequential(
            nn.Linear(self.hidden_aff, self.hidden_aff),
            nn.LeakyReLU(0.1),
            nn.Linear(self.hidden_aff, 1),
        )
    
    def dma_gru(self, comp_embedding, prot_embedding, pairwise_pred, batch_comp, batch_prot, list_num):
        c0 = scatter_mean(comp_embedding, batch_comp, dim=0)
        p0 = scatter_mean(prot_embedding, batch_prot, dim=0)
        m = c0*p0
        batch_size = len(pairwise_pred)

        for DMA_iter in range(self.DMA_depth):
            c_pre = torch.tanh(self.c_to_p_transform[DMA_iter](comp_embedding))
            p_pre = torch.tanh(self.p_to_c_transform[DMA_iter](prot_embedding))

            get_c2p = lambda i: torch.matmul(pairwise_pred[i].T, c_pre[list_num[i, 0]:list_num[i+1, 0]])
            get_p2c = lambda i: torch.matmul(pairwise_pred[i], p_pre[list_num[i, 1]:list_num[i+1, 1]])

            p_to_c = torch.cat(list(map(get_p2c, range(batch_size))))
            c_to_p = torch.cat(list(map(get_c2p, range(batch_size))))

            c_tmp = torch.tanh(self.hc0[DMA_iter](comp_embedding)) * \
                    torch.tanh(self.mc1[DMA_iter](m)).index_select(0, batch_comp) * \
                    p_to_c
            p_tmp = torch.tanh(self.hp0[DMA_iter](prot_embedding)) * \
                    torch.tanh(self.mp1[DMA_iter](m)).index_select(0, batch_prot) * \
                    c_to_p

            cf = scatter_sum(comp_embedding * scatter_softmax(self.hc1[DMA_iter](c_tmp), batch_comp, dim=0), batch_comp, dim=0)
            pf = scatter_sum(prot_embedding * scatter_softmax(self.hp1[DMA_iter](p_tmp), batch_prot, dim=0), batch_prot, dim=0)
            
            m = self.GRU_dma(m, cf*pf)
            
        return cf, pf

    def get_affinity(self, comp_feature, prot_feature, batch_comp, batch_prot):
        pairwise_comp_feature = self.pairwise_comp(comp_feature)
        pairwise_prot_feature = self.pairwise_prot(prot_feature)
        batch_size = max(batch_comp) + 1
        len_comp = torch.bincount(batch_comp+1, minlength=batch_size+1).reshape(-1, 1)
        len_prot = torch.bincount(batch_prot+1, minlength=batch_size+1).reshape(-1, 1)
        list_num = torch.cat([len_comp, len_prot], dim=1)
        for i in range(batch_size): 
            list_num[i+1] = list_num[i] + list_num[i+1]
        get_pairwise = lambda i: torch.sigmoid(torch.matmul(pairwise_comp_feature[list_num[i, 0]:list_num[i+1, 0]], 
                                                            pairwise_prot_feature[list_num[i, 1]:list_num[i+1, 1]].T))
        pairwise_pred = list(map(get_pairwise, range(batch_size)))

        comp_embedding = self.c_aff(comp_feature)
        prot_embedding = self.p_aff(prot_feature)
        aff_comp_feature, aff_prot_feature = self.dma_gru(comp_embedding, prot_embedding, pairwise_pred, batch_comp, batch_prot, list_num)

        kroneck = torch.matmul(comp_embedding.unsqueeze(2), aff_prot_feature.index_select(0, batch_comp).unsqueeze(1))
        feature = kroneck.view(kroneck.shape[0], -1)

        raw = self.W_out_raw(feature)
        prealpha = self.W_out_alpha(feature)
        alpha = scatter_softmax(prealpha, batch_comp, dim=0)
        
        vector = scatter_sum(raw * alpha, batch_comp, dim=0)
        affinity_pred = self.W_out(vector)
        
        return affinity_pred

    def forward(self, *input):
        if len(input) == 4:
            comp_feature, prot_feature, batch_comp, batch_prot = input
            affinity_pred = self.get_affinity(comp_feature, prot_feature, batch_comp, batch_prot)

            return affinity_pred

        elif len(input) == 6:
            comp_feature, prot1_feature, prot2_feature, batch_comp, batch_prot1, batch_prot2 = input
            affinity1_pred = self.get_affinity(comp_feature, prot1_feature, batch_comp, batch_prot1)
            affinity2_pred = self.get_affinity(comp_feature, prot2_feature, batch_comp, batch_prot2)

            return affinity1_pred - affinity2_pred 
        
        else:
            raise NotImplementedError



