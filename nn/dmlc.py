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
from scipy import sparse

from sklearn.model_selection import train_test_split

# From https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-proteins
# Author: Zhengdao Chen
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

# From https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-proteins
# Author: Zhengdao Chen
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


# From https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-proteins
# Author: Zhengdao Chen
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

### THE FOLLOWING IS ALL /MY/ CODE (Jack Steilberg), and is not sourced like the above
def convert(sample, adj, samp_max):
    sample /= samp_max
    gr =  dgl.from_scipy(adj.copy())
    gr.edges[sample.nonzero()].data["feat_0"] = torch.FloatTensor(sample[sample.nonzero()]).squeeze()
    return gr

def train_test(model, X_train, X_test, y_train, y_test, adj, epochs):
    print("Reformatting data")
    #train_y_maxes = [samp.max() for samp in y_train]
    test_y_max = max([samp.max() for samp in y_test])
    
    train_zipped = [(convert(x_samp, adj, test_y_max),
                     convert(y_samp, adj, test_y_max)) for x_samp, y_samp in zip(X_train, y_train)]
    test_zipped = [(convert(x_samp, adj, test_y_max),
                    convert(y_samp, adj, test_y_max)) for x_samp, y_samp in zip(X_test, y_test)]

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.09)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    #, momentum=0.9)

    print("Setting up data loader")
    batch_size=1
    loader = GraphDataLoader(train_zipped, batch_size=batch_size,
                        shuffle=True)

    test_loader = GraphDataLoader(test_zipped, batch_size=batch_size,
                                        shuffle=False)

    criterion = lambda outputs, labels: torch.sum((outputs - labels)**2)
    intermittent_testing = True

    train_losses = []
    test_losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(f"Epoch {epoch}")
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
            if i%25==24:
                pass
                #import pdb
                #pdb.set_trace()

            # print statistics
            once_loss = loss.item()
            train_losses.append(once_loss)
            running_loss += once_loss
            if i % 1 == 1 or True:
                print('[%d, %3d]      loss: %.3f' %
                      (epoch + 1, i + 1, once_loss))

            if i%25==24 and intermittent_testing:
                test_loss = 0.0
                with torch.no_grad():
                    for idx, data in enumerate(test_loader):
                        images, labels = data
                        outputs = model(images)

                        labels_adj = labels.adj().coalesce()
                        outputs = outputs * labels_adj.to_dense() #* test_y_max
                        labels = torch.sparse.FloatTensor(labels_adj.indices(), labels_adj.values()*labels.edata["feat_0"], size=labels_adj.shape) #* test_y_max
                        test_once_loss = criterion(outputs, labels).item()
                        test_loss += test_once_loss
                    test_losses.append(test_loss/len(test_loader))
                print('[%d, %3d] test loss: %.3f' %
                      (epoch + 1, i + 1, test_loss/len(test_loader)))

            loss.backward()
            optimizer.step()

    import numpy as np
    train_losses = np.array(train_losses)/40
    test_losses = np.array(test_losses)/40 # Scale because of lack of batch size
    train_losses = [np.mean(train_losses[i:i+25]) for i in range(0, len(train_losses), 25)]
    import matplotlib.pyplot as plt
    plt.plot(train_losses[1:], label="Train loss")
    plt.plot(test_losses[:-1], label="Test loss")
    plt.legend()
    plt.title("Loss for normalized data, convolutional network")
    plt.show()

    true_test = []
    pred_test = []
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            images, labels = data
            outputs = model(images)

            labels_adj = labels.adj().coalesce()
            outputs = outputs * labels_adj.to_dense() * test_y_max
            labels = torch.sparse.FloatTensor(labels_adj.indices(), labels_adj.values()*labels.edata["feat_0"], size=labels_adj.shape) * test_y_max
            pred_test += [sparse.csr_matrix(outputs)]
            true_test += [sparse.csr_matrix(labels.to_dense())]

            test_once_loss = criterion(outputs, labels).item() / len(test_loader)
            test_loss += test_once_loss
    print(f"Final test loss: {test_loss/len(test_loader)}")
    print("Finished training")
    
    return model, pred_test, true_test
