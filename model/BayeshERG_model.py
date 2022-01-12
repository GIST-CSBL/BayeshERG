import numpy as np
import torch

import torch.nn as nn
from dgl.backend import pytorch as DF
from dgl import function as fn


class RegularizationAccumulator:
    def __init__(self):
        self.i = 0
        self.size = 0

    def notify_loss(self, depth):
        self.size += depth

    def initialize(self, cuda):
        self.arr = torch.empty(self.size)
        if cuda:
            self.arr = self.arr.cuda()

    def add_loss(self, loss):
        self.arr[self.i] = loss
        self.i += 1

    def get_sum(self):
        sum = torch.sum(self.arr)
        self.i = 0
        self.arr = self.arr.detach()
        return sum


class ConcreteDropout(nn.Module):
    def __init__(self, layer, reg_acc, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.2, init_max=0.2, depth=1):
        super(ConcreteDropout, self).__init__()
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.layer = layer
        self.reg_acc = reg_acc
        self.reg_acc.notify_loss(depth)
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x):
        p = torch.sigmoid(self.p_logit)
        out = self.layer(self._concrete_dropout(x, p))
        index = self.reg_acc.size - 1
        if self.training:
            sum_of_square = 0
            for param in self.layer.parameters():
                sum_of_square += torch.sum(torch.pow(param, 2))
            weights_regularizer = self.weight_regularizer * sum_of_square * (1 - p)
            dropout_regularizer = p * torch.log(p)
            dropout_regularizer += (1. - p) * torch.log(1. - p)
            input_dimensionality = x[0].numel()  # Number of elements of first item in batch
            dropout_regularizer *= self.dropout_regularizer * input_dimensionality
            regularization = weights_regularizer + dropout_regularizer
            self.reg_acc.add_loss(regularization)
        return out

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x = torch.mul(x, random_tensor)
        x /= retain_prob
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_head, reg_acc, wr, dr):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.proj_q = ConcreteDropout(layer=nn.Linear(d_model, num_heads * d_head, bias=False), reg_acc=reg_acc,
                                      weight_regularizer=wr, dropout_regularizer=dr)
        self.proj_k = ConcreteDropout(layer=nn.Linear(d_model, num_heads * d_head, bias=False), reg_acc=reg_acc,
                                      weight_regularizer=wr, dropout_regularizer=dr)
        self.proj_v = ConcreteDropout(layer=nn.Linear(d_model, num_heads * d_head, bias=False), reg_acc=reg_acc,
                                      weight_regularizer=wr, dropout_regularizer=dr)
        self.proj_o = ConcreteDropout(layer=nn.Linear(num_heads * d_head, d_model, bias=False), reg_acc=reg_acc,
                                      weight_regularizer=wr, dropout_regularizer=dr)
        self.relu = nn.ReLU()

    def forward(self, x, lengths_x):
        device = x.device
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        queries = self.proj_q(x).view(batch_size, -1, self.num_heads, self.d_head)
        keys = self.proj_k(x).view(batch_size, -1, self.num_heads, self.d_head)
        values = self.proj_v(x).view(batch_size, -1, self.num_heads, self.d_head)


        e = torch.einsum('bxhd,byhd->bhxy', queries, keys)
        e = e / np.sqrt(self.d_head)
        mask = torch.zeros(batch_size, max_len_x + 1, max_len_x + 1).to(e.device)
        for i in range(batch_size):
            mask[i, :lengths_x[i] + 1, :lengths_x[i] + 1].fill_(1)
        mask = mask.unsqueeze(1)
        e.masked_fill_(mask == 0, -1e10)
        alpha = torch.softmax(e, dim=-1)
        out = torch.einsum('bhxy,byhd->bxhd', alpha, values)
        out = self.proj_o(
            out.contiguous().view(batch_size, (max_len_x + 1), self.num_heads * self.d_head))
        x = self.relu(out)
        return x, alpha




class MultiHeadAttention_readout(nn.Module):
    def __init__(self, d_model, num_heads, d_head, reg_acc, wr, dr, num_mha=1):
        super(MultiHeadAttention_readout, self).__init__()
        self.num_mha = num_mha
        self.attblock = nn.ModuleList([MultiHeadAttentionBlock(d_model=d_model, num_heads=num_heads,
                                                               d_head=d_head,
                                                               reg_acc=reg_acc, wr=wr, dr=dr) for x in range(num_mha)])
        self.class_emb = torch.nn.Embedding(1, d_model)
    def transform_feat(self, x, lengths_x):
        device = x.device
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        cls_token = self.class_emb(torch.LongTensor([0]).to(device))
        x = DF.pad_packed_tensor(x, lengths_x, 0, l_min=max_len_x + 1)
        for i in range(batch_size):
            x[i, lengths_x[i], :] = cls_token
        return x

    def forward(self, x, lengths_x):
        device = x.device
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        x = self.transform_feat(x, lengths_x)
        for l in range(self.num_mha):
            x, alpha = self.attblock[l](x, lengths_x)

        index = []
        bef = 0
        for i in lengths_x:
            bef += i
            index.append(bef)
            bef += ((max_len_x + 1) - i)
        index = torch.tensor(index).to(device)
        x = x.view(batch_size * (max_len_x + 1), -1)
        x = torch.index_select(x, 0, index.long())

        return x, alpha
class DMPNN(nn.Module):

    def __init__(self,
                 node_dim,
                 edge_dim,
                 aggregator_type):
        super(DMPNN, self).__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type

    def udf_sub(self, edges):
        return {'e_res': (edges.data['t'] - edges.data['rm'])}

    def forward(self, graph, nfeat, efeat):
        with graph.local_scope():
            graph.ndata['h'] = nfeat.view(-1, self.node_dim)
            graph.edata['w'] = efeat.view(-1, self.edge_dim)
            edge_index = np.array(range(len(efeat)))
            edge_index[list(range(0, len(efeat), 2))] += 1
            edge_index[list(range(1, len(efeat), 2))] -= 1
            edge_index = torch.LongTensor(list(edge_index))
            rev_efeat = efeat[edge_index]
            graph.edata['rev_w'] = rev_efeat.view(-1, self.edge_dim)
            graph.update_all(fn.copy_edge('w', 'm'), self.reducer('m', 'neigh'))
            graph.apply_edges(fn.copy_edge('rev_w', 'rm'))
            graph.apply_edges(fn.copy_src('neigh', 't'))
            graph.apply_edges(self.udf_sub)
            edg_n = graph.edata['e_res']
            return edg_n






class MPN_Featurizer(nn.Module):
    def __init__(self, reg_acc, wr, dr,
                 node_input_dim=15,
                 edge_input_dim=5,
                 node_hidden_dim=64,
                 edge_hidden_dim=32,
                 num_step_message_passing=3):
        super(MPN_Featurizer, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.conv_list = nn.ModuleList(
            [DMPNN(node_dim=node_input_dim, edge_dim=edge_hidden_dim, aggregator_type='sum')])
        self.relu = nn.ReLU()
        self.init_message = ConcreteDropout(
            layer=nn.Sequential(nn.Linear(node_input_dim + edge_input_dim, edge_hidden_dim, bias=False), nn.ReLU()), \
            reg_acc=reg_acc,
            weight_regularizer=wr, dropout_regularizer=dr)
        self.e_update = ConcreteDropout(layer=nn.Sequential(nn.Linear(edge_hidden_dim, edge_hidden_dim, bias=False)),
                                        reg_acc=reg_acc,
                                        weight_regularizer=wr, dropout_regularizer=dr, depth=num_step_message_passing)
        self.last_update = ConcreteDropout(
            layer=nn.Sequential(nn.Linear(node_input_dim + edge_hidden_dim, node_hidden_dim, bias=False)),
            reg_acc=reg_acc,
            weight_regularizer=wr, dropout_regularizer=dr)
    def udf_init_m(self, edges):
        return {'im': self.init_message(torch.cat((edges.src['ih'], edges.data['iw']), dim=1))}
    def forward(self, g, n_feat, e_feat):
        h0 = n_feat
        e0 = e_feat
        g.ndata['ih'] = h0.view(-1, self.node_input_dim)
        g.edata['iw'] = e0.view(-1, self.edge_input_dim)
        g.apply_edges(self.udf_init_m)
        e_t0 = g.edata['im']
        e0 = e_t0
        for i in range(self.num_step_message_passing):
            m_e = self.conv_list[0](g, h0, e_t0)

            e_t0 = self.relu(e0 + self.e_update(m_e))
        g.edata['fe'] = e_t0
        g.ndata['fn'] = h0
        g.update_all(fn.copy_edge('fe', 'fm'), fn.sum('fm', 'ff'))
        out = self.relu(self.last_update(torch.cat((g.ndata['fn'], g.ndata['ff']), dim=1)))
        return out

class Readout(nn.Module):
    def __init__(self, reg_acc, wr, dr,
                 node_hidden_dim=64, num_mha=2):
        super(Readout, self).__init__()
        self.mha_readout = MultiHeadAttention_readout(d_model=node_hidden_dim, num_heads=8, d_head=node_hidden_dim // 8,
                                                      num_mha=num_mha, reg_acc=reg_acc, wr=wr, dr=dr)
    def forward(self, g, feat):
        lengths = g.batch_num_nodes
        feat, w = self.mha_readout(feat, lengths)
        return feat, w


class BayeshERG(nn.Module):
    def __init__(self, reg_acc,
                 node_input_dim=74,
                 edge_input_dim=12,
                 node_hidden_dim=128,
                 edge_hidden_dim=128,
                 num_step_message_passing=7, num_step_mha=1,
                 wr=1e-6, dr=1e-5):
        super(BayeshERG, self).__init__()
        self.featurizer = MPN_Featurizer(node_input_dim=node_input_dim,
                                         edge_input_dim=edge_input_dim,
                                         node_hidden_dim=node_hidden_dim,
                                         edge_hidden_dim=edge_hidden_dim,
                                         num_step_message_passing=num_step_message_passing, reg_acc=reg_acc,
                                         wr=wr, dr=dr)
        self.readout = Readout(node_hidden_dim=node_hidden_dim, num_mha=num_step_mha, reg_acc=reg_acc,
                               wr=wr, dr=dr)
        self.lin1 = ConcreteDropout(layer=nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim // 2), nn.ReLU())
                                    , reg_acc=reg_acc,
                                    weight_regularizer=wr, dropout_regularizer=dr)
        self.lin2 = ConcreteDropout(layer=nn.Linear(node_hidden_dim // 2, 2), reg_acc=reg_acc,
                                    weight_regularizer=wr, dropout_regularizer=dr)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, g, n_feat, e_feat):
        b_length = g.batch_num_nodes
        out = self.featurizer(g, n_feat, e_feat)
        out, w = self.readout(g, out)
        out = self.lin1(out)
        blocker_logit = self.lin2(out)
        blocker = self.softmax(blocker_logit)
        w_tensors = []
        for c, bl in enumerate(b_length):
            b_w = w[c, :, :, :]
            b_w = b_w[:, bl, 0:bl + 1]
            w_tensors.append(b_w)
        return (blocker_logit, blocker), w_tensors








