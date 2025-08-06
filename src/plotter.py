import matplotlib.pyplot as plt
import pickle

from NNgraph import GCNReg, GCNReg_add, GCNReg_binary, GCNReg_binary_add
from createGraph import collates, collate_add, collate_multi, collate_multi_rdkit, collate_multi_non_rdkit
from GNN_functions import AccumulationMeter, print_result, print_final_result, write_result, write_final_result
from GNN_functions import save_prediction_result, save_saliency_result
from GNN_functions import train, predict, validate, save_checkpoint, to_loader

class Plotter():

    def __init__(self, path):
        super(Plotter, self).__init__()

        self.path = path

        self.plot()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)
    
    def load_data(self, id):

        return {'train_sm': f'{self.path}/smlstr_train.p',
                 'test_sm': f'{self.path}/smlstr_test.p',
                    'train_CMC': f'{self.path}/logCMC_train.p',
                    'test_CMC': f'{self.path}/logCMC_test.p'}
    
    
        


    


    path = '../models/EarlyStop_rd4592' # or '../models/GCN'

smlstr_train = pickle.load(open(f"{path}/smlstr_train.p","rb"))
smlstr_test = pickle.load(open(f"{path}/smlstr_test.p","rb"))

logCMC_train = pickle.load(open(f"{path}/logCMC_train.p","rb"))
logCMC_test = pickle.load(open(f"{path}/logCMC_test.p","rb"))