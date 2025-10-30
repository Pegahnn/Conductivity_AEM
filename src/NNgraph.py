# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:27:52 2020

Modified on Mon Oct 02 11:01:36 2023

@author: sqin34
@modified by: Teslim
"""

# https://docs.dgl.ai/en/0.4.x/tutorials/basics/4_batch.html

import dgl
import torch
from dgl.nn.pytorch import GraphConv, GATConv

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

###############################################################################

class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.

        if torch.cuda.is_available():
            h = g.ndata['h'].float().cuda()
        else:
            h = g.ndata['h'].float()

        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))

        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output



def collate(samples):
    graphs, descriptors, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(descriptors), torch.tensor(labels)

###############################################################################

class GCNReg_add(nn.Module):
    def __init__(self, in_dim, extra_in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_add, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.classify1 = nn.Linear(hidden_dim+extra_in_dim, hidden_dim+extra_in_dim)
        self.classify2 = nn.Linear(hidden_dim+extra_in_dim, hidden_dim+extra_in_dim)
        self.classify3 = nn.Linear(hidden_dim+extra_in_dim, n_classes)
        self.saliency = saliency

    def forward(self, g, descriptors):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.

        if torch.cuda.is_available():
            h = g.ndata['h'].float().cuda()
        else:
            h = g.ndata['h'].float()
        #print(f"h: {h}; h.shape: {h.shape}")
        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))
        #print(f"h1: {h1}; h1.shape: {h1.shape}")

        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        #print(f"hg: {hg}; hg.shape: {hg.shape}")
        # Now concatenate along dimension 1 (columns)

        #hg = torch.cat((hg, descriptors.to(torch.float32)), dim=1)
        # Check if descriptors is a tensor, and ensure it has the same dtype as hg
        if torch.is_tensor(descriptors):
            hg = torch.cat((hg, descriptors.to(torch.float32)), dim=1)
        else:
            descriptors = torch.tensor(descriptors, dtype=torch.float32)
            hg = torch.cat((hg, descriptors), dim=1)

        # Calculate the final prediction
        # print(hg.dtype)
        # print(self.classify1.weight.dtype, self.classify1.weight.shape)
        # print(self.classify1.bias.dtype, self.classify1.bias.shape)    
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:

            return output
          
          
          
          

# GNN for multi-molecular graphs
class GCNReg_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.classify1 = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.classify2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, hidden_dim)
        self.classify4 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.

        if torch.cuda.is_available():
            h1 = g[0].ndata['h'].float().cuda()
            h2 = g[1].ndata['h'].float().cuda()
        else:
            h1 = g[0].ndata['h'].float()
            h2 = g[1].ndata['h'].float()

        if self.saliency == True:
            h1.requires_grad = True
            h2.requires_grad = True

        h1 = F.relu(self.conv1(g[0], h1))
        h1 = F.relu(self.conv2(g[0], h1))
        h2 = F.relu(self.conv1(g[1], h2))
        h2 = F.relu(self.conv2(g[1], h2))

        g[0].ndata['h'] = h1
        g[1].ndata['h'] = h2
        # Calculate graph representation by averaging all the node representations.
        hg1 = dgl.mean_nodes(g[0], 'h')
        hg2 = dgl.mean_nodes(g[1], 'h')

        # Now concatenate along dimension 1 (columns)
        hg = torch.cat((hg1, hg2), dim=1)

        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = F.relu(self.classify3(output))
        output = self.classify4(output)

        if self.saliency == True:
            output.backward()
            return output, h1.grad, h2.grad
        else:
            return output
        
# GNN for multi-molecular graphs with additional node features
class GCNReg_binary_add(nn.Module):
    def __init__(self, in_dim, extra_in_dim, hidden_dim, n_classes, rdkit_features=False, saliency=False):
        super(GCNReg_binary_add, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.rdkit_features = rdkit_features
        if self.rdkit_features:
            self.classify1 = nn.Linear(hidden_dim*2+extra_in_dim*2, hidden_dim*2+extra_in_dim*2)
            self.classify2 = nn.Linear(hidden_dim*2+extra_in_dim*2, hidden_dim)
        else:
            self.classify1 = nn.Linear(hidden_dim*2+extra_in_dim, hidden_dim*2+extra_in_dim)
            self.classify2 = nn.Linear(hidden_dim*2+extra_in_dim, hidden_dim)
        
        self.classify3 = nn.Linear(hidden_dim, hidden_dim)
        self.classify4 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency
    
    def forward(self, g, descriptors):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.

        if torch.cuda.is_available():
            h1 = g[0].ndata['h'].float().cuda()
            h2 = g[1].ndata['h'].float().cuda()
        else:
            h1 = g[0].ndata['h'].float()
            h2 = g[1].ndata['h'].float()

        if self.saliency == True:
            h1.requires_grad = True
            h2.requires_grad = True

        h1 = F.relu(self.conv1(g[0], h1))
        h1 = F.relu(self.conv2(g[0], h1))
        h2 = F.relu(self.conv1(g[1], h2))
        h2 = F.relu(self.conv2(g[1], h2))

        g[0].ndata['h'] = h1
        g[1].ndata['h'] = h2
        # Calculate graph representation by averaging all the node representations.
        hg1 = dgl.mean_nodes(g[0], 'h')
        hg2 = dgl.mean_nodes(g[1], 'h')

        # Now concatenate along dimension 1 (columns)
        hg = torch.cat((hg1, hg2), dim=1)

        #hg = torch.cat((hg, descriptors.to(torch.float32)), dim=1)
        # Check if descriptors is a tensor, and ensure it has the same dtype as hg
        if torch.is_tensor(descriptors):
            hg = torch.cat((hg, descriptors.to(torch.float32)), dim=1)
            #hg = torch.cat((hg, descriptors[0].to(torch.float32), descriptors[1].to(torch.float32)), dim=1)
        else:
            # descriptors = torch.tensor(descriptors, dtype=torch.float32)
            # hg = torch.cat((hg, descriptors), dim=1)
            hg = torch.cat((hg, torch.tensor(descriptors[0], dtype=torch.float32), torch.tensor(descriptors[1], dtype=torch.float32)), dim=1)


        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = F.relu(self.classify3(output))
        output = self.classify4(output)

        if self.saliency == True:
            hg1.retain_grad()
            hg2.retain_grad()
            output.backward()
            return output, hg1.grad, hg2.grad
        else:
            return output
class GATReg_add(nn.Module):
    def __init__(self, in_dim, extra_in_dim, hidden_dim, n_classes, num_heads=4, saliency=False, num_layers=2, dropout=0.0):
        super(GATReg_add, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(dglnn.GATConv(in_dim, hidden_dim, num_heads, activation=F.relu))
        for _ in range(num_layers - 2):
            self.gat_layers.append(dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads, activation=F.relu))
        self.gat_layers.append(dglnn.GATConv(hidden_dim * num_heads, hidden_dim, num_heads))

        final_dim = hidden_dim * num_heads + extra_in_dim
        self.fc1 = nn.Linear(final_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, g, desc):
        h = g.ndata['h'].float()
        for layer in self.gat_layers:
            h = layer(g, h)
            if isinstance(h, tuple):
                h = h[0]
            if h.dim() == 3:
                h = h.flatten(1)
            h = F.relu(h)
            h = self.dropout(h)

        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        combined = torch.cat([hg, desc], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.out(x)

# -------------------------
# ---- Small-batch-safe norm ----
def safe_norm(dim: int):
    # LayerNorm is stable for small batches
    return nn.LayerNorm(dim)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            safe_norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class GINReg_add(nn.Module):
    def __init__(self, in_dim, extra_in_dim, hidden_dim, n_classes,
                 num_layers=3, dropout=0.1, learn_eps=True, readout="sum"):
        super().__init__()
        assert num_layers >= 2
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GINConv
        self.convs.append(dglnn.GINConv(
            apply_func=MLP(in_dim, hidden_dim, hidden_dim, dropout),
            learn_eps=learn_eps
        ))
        self.norms.append(safe_norm(hidden_dim))

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(dglnn.GINConv(
                apply_func=MLP(hidden_dim, hidden_dim, hidden_dim, dropout),
                learn_eps=learn_eps
            ))
            self.norms.append(safe_norm(hidden_dim))

        # Last GINConv
        self.convs.append(dglnn.GINConv(
            apply_func=MLP(hidden_dim, hidden_dim, hidden_dim, dropout),
            learn_eps=learn_eps
        ))
        self.norms.append(safe_norm(hidden_dim))

        # Readout
        if readout == "sum":
            self.readout = dglnn.SumPooling()
        elif readout == "max":
            self.readout = dglnn.MaxPooling()
        else:
            self.readout = dglnn.AvgPooling()

        # MLP head (concat descriptors)
        self.fc1 = nn.Linear(hidden_dim + extra_in_dim, hidden_dim)
        self.norm1 = safe_norm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, desc):
        h = g.ndata["h"].float()
        for conv, norm in zip(self.convs, self.norms):
            h = conv(g, h)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        hg = self.readout(g, h)

        # concat descriptors
        x = torch.cat([hg, desc], dim=1)
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out
    # ===============================================================
# MPNNReg_add (Message Passing Neural Network for Regression)
# Compatible with GCN/GAT/GIN unified training & Optuna script
# ===============================================================
class MPNNLayer(nn.Module):
    """One message passing layer with edge-conditioned aggregation"""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.GRUCell(hidden_dim, node_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            g.ndata['h'] = node_feats
            g.edata['e'] = edge_feats

            # Compute messages
            g.apply_edges(lambda edges: {
                'm': self.message_net(torch.cat([edges.src['h'], edges.dst['h'], edges.data['e']], dim=1))
            })

            # Aggregate messages
            g.update_all(fn.copy_e('m', 'm'), fn.mean('m', 'agg_msg'))
            agg_msg = g.ndata['agg_msg']

            # Update node features
            h_new = self.update_net(agg_msg, node_feats)
            return h_new


class MPNNReg_add(nn.Module):
    """MPNN model with descriptor concatenation for regression"""
    def __init__(self, in_dim, extra_in_dim, hidden_dim, n_classes,
                 num_layers=3, edge_dim=10, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # Node and edge feature encoders
        self.node_embed = nn.Linear(in_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, hidden_dim)

        # Message passing layers
        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_layers)
        ])

        # Readout layer (mean pooling)
        self.readout = dglnn.AvgPooling()

        # Fully connected head (with descriptors)
        self.fc1 = nn.Linear(hidden_dim + extra_in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.out = nn.Linear(hidden_dim // 2, n_classes)

    def forward(self, g, desc):
        h = g.ndata['h'].float()
        e = g.edata['e'].float() if 'e' in g.edata else torch.zeros(
            (g.num_edges(), 10), device=h.device
        )

        # Encode node and edge features
        h = self.node_embed(h)
        e = self.edge_embed(e)

        # Message passing
        for layer in self.layers:
            h = layer(g, h, e)
            h = F.relu(h)
            h = self.dropout(h)

        # Graph-level embedding
        hg = self.readout(g, h)

        # Concatenate descriptors
        x = torch.cat([hg, desc], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        out = self.out(x)
        return out
