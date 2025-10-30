
import torch
import dgl
from dgllife.utils import BaseAtomFeaturizer, CanonicalAtomFeaturizer, CanonicalBondFeaturizer 
from dgllife.utils import mol_to_graph, mol_to_bigraph, mol_to_complete_graph, smiles_to_complete_graph
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from sklearn.preprocessing import MinMaxScaler


__all__ = ['graph_dataset', 'multi_graph_dataset']


def summarize_graph_data(g):
    node_data = g.ndata['h'].numpy()
    print("node data:\n", node_data)
    edge_data = g.edata
    print("edge data:", edge_data)
    adj_mat = g.adjacency_matrix_scipy(transpose=True, return_edge_ids=False)
    adj_mat = adj_mat.todense().astype(np.float32)
    print("adjacency matrix:\n", adj_mat)


# ******************************************************* Single graph dataset *********************************************
class graph_dataset(object):

    def __init__(self, smiles, y, add_features=False, rdkit_descriptor=False,
                 node_enc=CanonicalAtomFeaturizer(), edge_enc=None,
                 graph_type=mol_to_bigraph, canonical_atom_order=False):
        super(graph_dataset, self).__init__()
        self.smiles = smiles
        self.y = y
        self.add_features = add_features
        self.rdkit_descriptor = rdkit_descriptor
        self.graph_type = graph_type
        self.node_enc = node_enc
        self.edge_enc = edge_enc
        self.canonical_atom_order = canonical_atom_order
        self.graphs = []
        self.descriptors = []
        self.labels = []

        self.scaler = MinMaxScaler()

        self._generate()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.add_features:
            return self.graphs[idx], self.descriptors[idx], self.labels[idx]
        else:
            return self.graphs[idx], self.labels[idx]

    def getsmiles(self, idx):
        if len(self.smiles[idx]) == 1:
            return self.smiles[idx]
        else:
            return self.smiles[idx][0]
    
    def node_to_atom(self, idx):
        g = self.graphs[idx]
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        node_feat = g.ndata['h'].numpy()[:, 0:len(allowable_set)]
        atom_list = []
        for i in range(g.number_of_nodes()):
            atom_list.append(allowable_set[np.where(node_feat[i] == 1)[0][0]])
        return atom_list
    
    def get_descriptor(self, mol):
        return (Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol), Descriptors.Ipc(mol), Descriptors.LabuteASA(mol))
    
    def _generate(self):
        if self.graph_type == mol_to_bigraph:
            for i, j in enumerate(self.smiles):
                if len(j) == 1:
                    m = Chem.MolFromSmiles(j)
                else:
                    m = Chem.MolFromSmiles(j[0])
                g = self.graph_type(m, True, self.node_enc, self.edge_enc,
                                    self.canonical_atom_order)
                # Normalize node features to 0-1 #added by Pegah
                h = g.ndata['h'].float()
                h_min, _ = torch.min(h, dim=0, keepdim=True)
                h_max, _ = torch.max(h, dim=0, keepdim=True)
                g.ndata['h'] = (h - h_min) / (h_max - h_min + 1e-9)
                ############
                self.graphs.append(g)
                if self.add_features:
                    if self.rdkit_descriptor:
                        self.descriptors.append(self.get_descriptor(m))
                    else:
                        self.descriptors.append(tuple([float(prop) for prop in j[1:]]))

                self.labels.append(torch.tensor(float(self.y[i]), dtype=torch.float))

        elif self.graph_type == smiles_to_complete_graph:
            for i, j in enumerate(self.smiles):
                if len(j) == 1:
                    m = Chem.MolFromSmiles(j)
                else:
                    m = Chem.MolFromSmiles(j[0])
                g = self.graph_type(j, True, self.node_enc, self.edge_enc,
                                    self.canonical_atom_order)
                self.graphs.append(g)
                if self.add_features:
                    if self.rdkit_descriptor:
                        self.descriptors.append(self.get_descriptor(Chem.MolFromSmiles(j)))
                    else:
                        self.descriptors.append(tuple([float(prop) for prop in j[1:]]))
                self.labels.append(torch.tensor(float(self.y[i]), dtype=torch.float))
                
        if self.add_features:
            self.scaler.fit(self.descriptors)
            self.descriptors = self.scaler.transform(self.descriptors)


def collates(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.float).unsqueeze(-1)


def collate_add(samples):
    graphs, descriptors, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    # Ensure descriptors is a single numpy array (fast conversion)
    if not isinstance(descriptors, np.ndarray):
        descriptors = np.array(descriptors)

    return (
        batched_graph,
        torch.tensor(descriptors, dtype=torch.float),
        torch.tensor(labels, dtype=torch.float).unsqueeze(-1)
    )


# ******************************************** Multi-graph dataset ********************************************
class multi_graph_dataset(object):

    def __init__(self, smiles, y, add_features=False, rdkit_descriptor=False,
                 node_enc=CanonicalAtomFeaturizer(), edge_enc=None,
                 graph_type=mol_to_bigraph, canonical_atom_order=False):
        super(multi_graph_dataset, self).__init__()

        self.smiles = smiles
        self.y = y
        self.add_features = add_features
        self.rdkit_descriptor = rdkit_descriptor
        self.graph_type = graph_type
        self.node_enc = node_enc
        self.edge_enc = edge_enc
        self.canonical_atom_order = canonical_atom_order
        self.graphs = []
        self.descriptors = []
        self.labels = []

        self.scaler = [MinMaxScaler(), MinMaxScaler()]

        self._generate()

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if self.add_features:
            if len(self.descriptors[idx]) == 2:
                return (self.graphs[idx][0], self.graphs[idx][1],
                        self.descriptors[idx][0], self.descriptors[idx][1], self.labels[idx])
            else:
                return (self.graphs[idx][0], self.graphs[idx][1],
                        self.descriptors[idx], self.labels[idx])
        else:
            return self.graphs[idx][0], self.graphs[idx][1], self.labels[idx]

    def getsmiles(self, idx):
        return self.smiles[idx][:2]

    def node_to_atom(self, idx):
        g1 = self.graphs[idx][0]
        g2 = self.graphs[idx][1]

        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        
        node_feat1 = g1.ndata['h'].numpy()[:, 0:len(allowable_set)]
        node_feat2 = g2.ndata['h'].numpy()[:, 0:len(allowable_set)]

        atom_list1 = []
        atom_list2 = []
        for i, j in zip(range(g1.number_of_nodes()), range(g2.number_of_nodes())):
            atom_list1.append(allowable_set[np.where(node_feat1[i] == 1)[0][0]])
            atom_list2.append(allowable_set[np.where(node_feat2[j] == 1)[0][0]])
        return atom_list1, atom_list2
    
    def get_graph_from_mol(self, mol):
        return self.graph_type(mol, True, self.node_enc, self.edge_enc, self.canonical_atom_order)
    
    def get_descriptor(self, mol):
        return (Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumRotatableBonds(mol), Descriptors.Ipc(mol), Descriptors.LabuteASA(mol))
    
    def _generate(self):
        if self.graph_type == mol_to_bigraph:   
            for loc, data in enumerate(self.smiles):
                (i, j) = data[:2]
                m1, m2 = Chem.MolFromSmiles(i), Chem.MolFromSmiles(j)
                g1, g2 = self.get_graph_from_mol(m1), self.get_graph_from_mol(m2)
                self.graphs.append((g1, g2))

                if self.add_features:
                    if self.rdkit_descriptor:
                        self.descriptors.append((self.get_descriptor(m1), self.get_descriptor(m2)))
                    else:
                        self.descriptors.append(tuple([float(prop) for prop in data[2:]]))

                self.labels.append(torch.tensor(float(self.y[loc]), dtype=torch.float))

        elif self.graph_type == smiles_to_complete_graph:
            for loc, data in enumerate(self.smiles):
                (i, j) = data[:2]
                g1, g2 = self.get_graph_from_mol(i), self.get_graph_from_mol(j)
                self.graphs.append((g1, g2))

                if self.add_features:
                    if self.rdkit_descriptor:
                        self.descriptors.append((self.get_descriptor(Chem.MolFromSmiles(i)), 
                                                 self.get_descriptor(Chem.MolFromSmiles(j))))
                    else:
                        self.descriptors.append(tuple([float(prop) for prop in data[2:]]))
                self.labels.append(torch.tensor(float(self.y[loc]), dtype=torch.float))

        if self.add_features:
            self.descriptors = np.array(self.descriptors)
            
            if self.rdkit_descriptor:
                self.scaler[0].fit(self.descriptors[:, 0, :])
                self.scaler[1].fit(self.descriptors[:, 1, :])
                self.descriptors[:, 0, :] = self.scaler[0].transform(self.descriptors[:, 0, :])
                self.descriptors[:, 1, :] = self.scaler[1].transform(self.descriptors[:, 1, :])
            else:
                self.scaler[0].fit(self.descriptors)
                self.descriptors = self.scaler[0].transform(self.descriptors)

            self.descriptors = self.descriptors.tolist()


def collate_multi(samples):
    graphs1, graphs2, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    return batched_graph1, batched_graph2, torch.tensor(labels, dtype=torch.float).unsqueeze(-1)


def collate_multi_non_rdkit(samples):
    graphs1, graphs2, descriptors, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    return (batched_graph1, batched_graph2,
            torch.tensor(descriptors, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float).unsqueeze(-1))


def collate_multi_rdkit(samples):
    graphs1, graphs2, descriptors1, descriptors2, labels = map(list, zip(*samples))
    batched_graph1 = dgl.batch(graphs1)
    batched_graph2 = dgl.batch(graphs2)
    return (batched_graph1, batched_graph2,
            torch.tensor(descriptors1, dtype=torch.float),
            torch.tensor(descriptors2, dtype=torch.float),
            torch.tensor(labels, dtype=torch.float).unsqueeze(-1))

