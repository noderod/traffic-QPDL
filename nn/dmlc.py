#!/usr/bin/env python3
import math

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import dgl
from dgl.dataloading import GraphDataLoader
import dgl.function as fn
from dgl import DGLGraph
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


class MWEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 bias=True,
                 num_channels=1,
                 aggr_mode='sum'):
        super(MWEConv, self).__init__()
        self.num_channels = num_channels
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats, num_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats, num_channels))
        else:
            self.bias = None
        self.reset_parameters()
        self.activation = activation

        if (aggr_mode == 'concat'):
            self.aggr_mode = 'concat'
            self.final = nn.Linear(out_feats * self.num_channels, out_feats)
        elif (aggr_mode == 'sum'):
            self.aggr_mode = 'sum'
            self.final = nn.Linear(out_feats, out_feats)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, node_state_prev):
        node_state = node_state_prev

        # if self.dropout:
        #     node_states = self.dropout(node_state)

        g = g.local_var()

        new_node_states = []

        ## perform weighted convolution for every channel of edge weight
        for c in range(self.num_channels):
            node_state_c = node_state
            if self._out_feats < self._in_feats:
                g.ndata['feat_' + str(c)] = torch.mm(node_state_c, self.weight[:, :, c])
            else:
                g.ndata['feat_' + str(c)] = node_state_c
            g.update_all(fn.src_mul_edge('feat_' + str(c), 'feat_' + str(c), 'm'), fn.sum('m', 'feat_' + str(c) + '_new'))
            node_state_c = g.ndata.pop('feat_' + str(c) + '_new')
            if self._out_feats >= self._in_feats:
                node_state_c = torch.mm(node_state_c, self.weight[:, :, c])          
            if self.bias is not None:
                node_state_c = node_state_c + self.bias[:, c]
            node_state_c = self.activation(node_state_c)   
            new_node_states.append(node_state_c) 
        if (self.aggr_mode == 'sum'):
            node_states = torch.stack(new_node_states, dim=1).sum(1)
        elif (self.aggr_mode == 'concat'):
            node_states = torch.cat(new_node_states, dim=1)

        node_states = self.final(node_states)

        return node_states

def normalize_edge_weights(graph, num_ew_channels, device="cpu"):
    degs = graph.in_degrees().float()
    degs = torch.clamp(degs, min=1)
    norm = torch.pow(degs, 0.5)
    norm = norm.to(device)
    graph.ndata['norm'] = norm.unsqueeze(1)
    graph.apply_edges(fn.e_div_u('feat', 'norm', 'feat'))
    graph.apply_edges(fn.e_div_v('feat', 'norm', 'feat'))
    for channel in range(num_ew_channels):
        graph.edata['feat_' + str(channel)] = graph.edata['feat'][:, channel:channel+1]


class MWE_GCN(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 n_layers,
                 activation,
                 dropout,
                 aggr_mode='concat',
                 device='cpu'):
        super(MWE_GCN, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.layers = nn.ModuleList()

        self.layers.append(MWEConv(n_input, n_hidden, activation=activation, \
            aggr_mode=aggr_mode))
        for i in range(n_layers - 1):
            self.layers.append(MWEConv(n_hidden, n_hidden, activation=activation, \
                aggr_mode=aggr_mode))

        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device

    def forward(self, g, node_state=None):
        node_state = torch.ones(g.number_of_nodes(), 1).float().to(self.device)

        for layer in self.layers:
            node_state = F.dropout(node_state, p=self.dropout, training=self.training)
            node_state = layer(g, node_state)
            node_state = self.activation(node_state)
        out = self.pred_out(node_state)
        return out


class MWE_DGCN(nn.Module):
    def __init__(self,
                 n_input,
                 n_hidden,
                 n_output,
                 n_layers,
                 activation,
                 dropout,
                 residual=False,
                 aggr_mode='concat',
                 device='cpu'):
        super(MWE_DGCN, self).__init__()
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.residual = residual

        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        self.layers.append(MWEConv(n_input, n_hidden, activation=activation, \
            aggr_mode=aggr_mode))
        
        for i in range(n_layers - 1):
            self.layers.append(MWEConv(n_hidden, n_hidden, activation=activation, \
                aggr_mode=aggr_mode))

        for i in range(n_layers):
            self.layer_norms.append(nn.LayerNorm(n_hidden, elementwise_affine=True))


        self.pred_out = nn.Linear(n_hidden, n_output)
        self.device = device


    def forward(self, g, node_state=None):
        node_state = torch.ones(g.number_of_nodes(), 1).float().to(self.device)

        node_state = self.layers[0](g, node_state)

        for layer in range(1, self.n_layers):
            node_state_new = self.layer_norms[layer-1](node_state)
            node_state_new = self.activation(node_state_new)
            node_state_new = F.dropout(node_state_new, p=self.dropout, training=self.training)

            if (self.residual == 'true'):
                node_state = node_state + self.layers[layer](g, node_state_new)
            else:
                node_state = self.layers[layer](g, node_state_new)

        node_state = self.layer_norms[self.n_layers-1](node_state)
        node_state = self.activation(node_state)
        node_state = F.dropout(node_state, p=self.dropout, training=self.training)
        out = self.pred_out(node_state)

        return out


def convert(sample, adj):
    sample /= sample.max()
    gr =  dgl.from_scipy(adj.copy())
    gr.edges[sample.nonzero()].data["feat_0"] = torch.FloatTensor(sample[sample.nonzero()]).squeeze()
    return gr

def train_test(model, X, y, adj):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

    print("Reformatting data")
    train_zipped = [(convert(x_samp, adj), convert(y_samp, adj)) for x_samp, y_samp in zip(X_train, y_train)]
    test_zipped = [(convert(x_samp, adj), convert(y_samp, adj)) for x_samp, y_samp in zip(X_test, y_test)]

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.09)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #, momentum=0.9)

    print("Setting up data loader")
    batch_size=1
    loader = GraphDataLoader(train_zipped, batch_size=batch_size,
                        shuffle=True)

    test_loader = GraphDataLoader(test_zipped, batch_size=batch_size,
                                         shuffle=True)

    criterion = lambda outputs, labels: torch.sum(torch.abs(outputs - labels))

    for epoch in range(10):  # loop over the dataset multiple times
        print("epoch")
        running_loss = 0.0
        for i, batch in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            # zero the parameter gradients
            optimizer.zero_grad()
            model.train()
            # forward + backward + optimize
            outputs = model(inputs)
            #import pdb
            labels_adj = labels.adj().coalesce()

            outputs = outputs * labels_adj.to_dense()
            labels = torch.sparse.FloatTensor(labels_adj.indices(), labels_adj.values()*labels.edata["feat_0"], size=labels_adj.shape)

            loss = criterion(outputs, labels)
            if i%50==49:
                import pdb
                pdb.set_trace()

            # print statistics
            once_loss = loss.item()
            running_loss += once_loss
            if i % 1 == 1 or True:
                print('[%d, %3d]      loss: %.3f' %
                      (epoch + 1, i + 1, once_loss))

            if i%10==9:
                test_loss = 0.0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        outputs = model(images)

                        labels_adj = labels.adj().coalesce()
                        outputs = outputs * labels_adj.to_dense()
                        labels = torch.sparse.FloatTensor(labels_adj.indices(), labels_adj.values()*labels.edata["feat_0"], size=labels_adj.shape)

                        test_once_loss = criterion(outputs, labels).item()
                        test_loss += test_once_loss
                print('[%d, %3d] test loss: %.3f' %
                      (epoch + 1, i + 1, test_loss/len(test_loader)))

            loss.backward()
            optimizer.step()


    print("Finished training")
    return model
