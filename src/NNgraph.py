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
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

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