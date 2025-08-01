# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:26:24 2020
Modified on Mon Oct 02 11:01:36 2023

@author: sqin34
@modified by: Teslim
"""


from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import argparse, os, time, random, pickle, csv
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch.optim
import torch.utils.data

from dgllife.utils import EarlyStopping
from dgllife.utils import BaseAtomFeaturizer,atomic_number
from sklearn.model_selection import KFold, train_test_split

from NNgraph import GCNReg, GCNReg_add, GCNReg_binary, GCNReg_binary_add
from createGraph import collates, collate_add, collate_multi, collate_multi_rdkit, collate_multi_non_rdkit
from GNN_functions import AccumulationMeter, print_result, print_final_result, write_result, write_final_result
from GNN_functions import save_prediction_result, save_saliency_result
from GNN_functions import train, predict, validate, save_checkpoint, to_loader

# argument parser
parser = argparse.ArgumentParser(description='GCN with descriptors')
parser.add_argument('-e', '--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=5, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('-l', '--lr', default=0.005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU ID to use.')
parser.add_argument('-c', '--cv', default=11, type=int,
                    help='k-fold cross validation')
parser.add_argument('-i', '--dim_input', default=74, type=int,
                    help='dimension of input')
parser.add_argument('-u', '--unit_per_layer', default=256, type=int,
                    help='unit per layer')
parser.add_argument('--train', action='store_true',
                    help='if train')
parser.add_argument('--randSplit', action='store_true',
                    help='if random split')
parser.add_argument('--seed', default=2020, type=int, metavar='N',
                    help='seed number')
parser.add_argument('--test_size', default=0.1, type=float,
                    help='test size')
parser.add_argument('--gnn_model', default=GCNReg_add,
                    help='gnn model')
parser.add_argument('--single_feat', action='store_true',
                    help='if atomic number node featurizer')
parser.add_argument('--early_stop', action='store_true',
                    help='if early stopping')
parser.add_argument('--patience', default=30, type=int,
                    help='early stop patience')
parser.add_argument('--dataset', default='nonionic',
                    help='nonionic or all')
parser.add_argument('--skip_cv', action='store_true',
                    help='if skip cross validation')
parser.add_argument('--path', default='../models/',
                                help='path to model')
parser.add_argument('--data', default='../data/dataset_122.csv',
                                help='path to model')
parser.add_argument('--num_feat', default=6, type=int,
                    help='number of additional features')
parser.add_argument('--add_features', action='store_true', 
                    help='if additional features')
parser.add_argument('--binary_system', action='store_true',
                    help='if two molecules in file')
parser.add_argument('--rdkit_descriptor', action='store_true',
                    help='if rdkit descriptor else use others in the file')


# main functions
def main(args):
    # Ensure the model path exists
    os.makedirs(args.path, exist_ok=True)
    
    # select model
    if args.binary_system:
        from createGraph import multi_graph_dataset as  graph_dataset

        if args.gnn_model == 'GCNReg_binary':
            args.gnn_model = GCNReg_binary
            collate = collate_multi
        elif args.gnn_model == 'GCNReg_binary_add':
            args.gnn_model = GCNReg_binary_add
            if args.rdkit_descriptor:
                collate = collate_multi_rdkit
            else:
                collate = collate_multi_non_rdkit
        else:
            raise ValueError('Invalid model name.')
    else:
        from createGraph import graph_dataset
     
        if args.gnn_model == 'GCNReg':
            args.gnn_model = GCNReg
            collate = collates
        elif args.gnn_model == 'GCNReg_add':
            args.gnn_model = GCNReg_add
            collate = collate_add
        else:
            raise ValueError('Invalid model name.')
    
    # tensorboard writer
    SWriter = SummaryWriter(f'{args.path}/')
    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load CSV dataset
    smlstr = []
    logCMC = []
    with open(f"{args.data}") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            if args.binary_system:
                smlstr.append(row[0:-1])
                logCMC.append(row[-1])
            else:
                if args.rdkit_descriptor:    
                    smlstr.append(row[0])
                    logCMC.append(row[1])
                else:
                    smlstr.append(row[0:-1])
                    logCMC.append(row[-1])

    smlstr = np.asarray(smlstr)
    logCMC = np.asarray(logCMC, dtype="float")
    dataset_size = len(logCMC)
    all_ind = np.arange(dataset_size)

    # split into training and testing
    if args.randSplit:
        print("Using Random Splits")
        train_full_ind, test_ind, \
        smlstr_train, smlstr_test, \
        logCMC_train, logCMC_test = train_test_split(all_ind, smlstr, logCMC,
                                                     test_size=args.test_size,
                                                     random_state=args.seed)

    # save train/test data and index corresponding to the original dataset
    pickle.dump(smlstr_train,open(f"{args.path}/smlstr_train.p","wb"))
    pickle.dump(smlstr_test,open(f"{args.path}/smlstr_test.p","wb"))
    pickle.dump(logCMC_train,open(f"{args.path}/logCMC_train.p","wb"))
    pickle.dump(logCMC_test,open(f"{args.path}/logCMC_test.p","wb"))
    pickle.dump(train_full_ind,open(f"{args.path}/original_ind_train_full.p","wb"))
    pickle.dump(test_ind,open(f"{args.path}/original_ind_test.p","wb"))
    rows = zip(train_full_ind,smlstr_train,logCMC_train)
    with open(f"{args.path}/dataset_train.csv",'w',newline='') as f:
        writer = csv.writer(f,delimiter=',')
        for row in rows:
            writer.writerow(row)
    rows = zip(test_ind,smlstr_test,logCMC_test)
    with open(f"{args.path}/dataset_test.csv",'w',newline='') as f:
        writer = csv.writer(f,delimiter=',')
        for row in rows:
            writer.writerow(row)

    train_size = len(logCMC_train)
    indices = list(range(train_size))

    if args.skip_cv == False:
        # K-fold CV setup
        kf = KFold(n_splits=args.cv, random_state=args.seed, shuffle=True)
        cv_index = 0
        index_list_train = []
        index_list_valid = []
        for train_indices, valid_indices in kf.split(indices):
            index_list_train.append(train_indices)
            index_list_valid.append(valid_indices)
            if args.gnn_model == GCNReg:
                model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=False)
                model_arch = 'GCNReg' 

            elif args.gnn_model == GCNReg_add:
                model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, n_classes=1,saliency=False)
                model_arch = 'GCNReg_add'

            elif args.gnn_model == GCNReg_binary:
                model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=False)
                model_arch = 'GCNReg_binary'

            elif args.gnn_model == GCNReg_binary_add:
                model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, rdkit_features=args.rdkit_descriptor, n_classes=1,saliency=False)
                model_arch = 'GCNReg_binary_add'

            else:
                raise ValueError('Invalid GNN model name')
            
            loss_fn = nn.MSELoss()

            # check gpu availability
            if args.gpu >= 0:
                model = model.cuda(args.gpu)
                loss_fn = loss_fn.cuda(args.gpu)
                cudnn.enabled = True
                cudnn.benchmark = True
                cudnn.deterministic = False
            optimizer = torch.optim.Adam(model.parameters(), args.lr)
            # training


            if args.single_feat:
                
                train_full_dataset = graph_dataset(smlstr_train,logCMC_train, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor, node_enc=BaseAtomFeaturizer({'h': atomic_number}))
                test_dataset = graph_dataset(smlstr_test,logCMC_test, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor, node_enc=BaseAtomFeaturizer({'h': atomic_number}))
                args.dim_input = 1
            else:
                train_full_dataset = graph_dataset(smlstr_train,logCMC_train, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
                test_dataset = graph_dataset(smlstr_test,logCMC_test, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
            train_sampler = SubsetRandomSampler(train_indices)
            valid_sampler = SubsetRandomSampler(valid_indices)
            train_loader = torch.utils.data.DataLoader(train_full_dataset, batch_size=args.batch_size,
                                                       sampler=train_sampler,
                                                       collate_fn=collate,
                                                       shuffle=False)
            val_loader = torch.utils.data.DataLoader(train_full_dataset, batch_size=args.batch_size,
                                                     sampler=valid_sampler,
                                                     collate_fn=collate,
                                                     shuffle=False)
            train_dataset = graph_dataset(smlstr_train[train_indices],logCMC_train[train_indices], add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
            valid_dataset = graph_dataset(smlstr_train[valid_indices],logCMC_train[valid_indices], add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)

            fname = r"ep{}bs{}lr{}kf{}hu{}cvid{}".format(args.epochs, args.batch_size,
                                                               args.lr,
                                                               args.cv,
                                                               args.unit_per_layer, cv_index)

            best_rmse = 1000
            if args.train:
                print("Training the model ...")
                stopper = EarlyStopping(mode='lower', patience=args.patience, filename=f'{args.path}/{fname}es.pth.tar') # early stop model
                for epoch in range(args.start_epoch, args.epochs):
                    train_loss = train(train_loader, model, loss_fn, optimizer, epoch, args, fname, SWriter)
                    rmse = validate(val_loader, model, epoch, args, fname, SWriter)
                    is_best = rmse < best_rmse
                    best_rmse = min(rmse, best_rmse)
                    if is_best:
                        save_checkpoint({
                            'epoch': epoch + 1,
                            'model_arch': model_arch,
                            'state_dict': model.state_dict(),
                            'best_rmse': best_rmse,
                            'optimizer': optimizer.state_dict(),
                        }, fname, args.path)
                    if args.early_stop:
                        early_stop = stopper.step(train_loss, model)
                        if early_stop:
                            print("**********Early Stopping!")
                            break


            # test
            print("Testing the model ...")
            checkpoint = torch.load(r"{}/{}.pth.tar".format(args.path, fname))
            args.start_epoch = 0
            best_rmse = checkpoint['best_rmse']
            
            if args.gnn_model == GCNReg:
                model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
                model_arch = 'GCNReg' 

            elif args.gnn_model == GCNReg_add:
                model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
                model_arch = 'GCNReg_add'

            elif args.gnn_model == GCNReg_binary:
                model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
                model_arch = 'GCNReg_binary'

            elif args.gnn_model == GCNReg_binary_add:
                model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, rdkit_features=args.rdkit_descriptor, n_classes=1,saliency=True)
                model_arch = 'GCNReg_binary_add'

            else:
                raise ValueError('Invalid GNN model name')

            if args.gpu >= 0:
                model = model.cuda(args.gpu)

            model.load_state_dict(checkpoint['state_dict'])
           
            print("=> loaded checkpoint '{}' (epoch {}, rmse {})"
                  .format(fname, checkpoint['epoch'], best_rmse))
            cudnn.deterministic = True
            stage = 'testtest'
            predict(to_loader(test_dataset, collate), model, -1, args, fname, stage, SWriter, args.path)
            stage = 'testtrain'
            predict(to_loader(train_dataset, collate), model, -1, args, fname, stage, SWriter, args.path)
            stage = 'testval'
            predict(to_loader(valid_dataset, collate), model, -1, args, fname, stage, SWriter, args.path)
            cv_index += 1
        pickle.dump(index_list_train,open(f"{args.path}/ind_train_list.p","wb"))
        pickle.dump(index_list_valid,open(f"{args.path}/ind_val_list.p","wb"))
        cv_index += 1

    else:
        if args.gnn_model == GCNReg:
            model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=False)
            model_arch = 'GCNReg' 

        elif args.gnn_model == GCNReg_add:
            model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, n_classes=1,saliency=False)
            model_arch = 'GCNReg_add'

        elif args.gnn_model == GCNReg_binary:
            model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=False)
            model_arch = 'GCNReg_binary'

        elif args.gnn_model == GCNReg_binary_add:
            model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, rdkit_features=args.rdkit_descriptor, n_classes=1,saliency=False)
            model_arch = 'GCNReg_binary_add'

        else:
            raise ValueError('Invalid GNN model name')
        loss_fn = nn.MSELoss()

        # check gpu availability
        if args.gpu >= 0:
            model = model.cuda(args.gpu)
            loss_fn = loss_fn.cuda(args.gpu)
            cudnn.enabled = True
            cudnn.benchmark = True
            cudnn.deterministic = False
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        # training


        if args.single_feat:
    
            train_full_dataset = graph_dataset(smlstr_train,logCMC_train,add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor, node_enc=BaseAtomFeaturizer({'h': atomic_number}))
            test_dataset = graph_dataset(smlstr_test,logCMC_test,add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor, node_enc=BaseAtomFeaturizer({'h': atomic_number}))
            args.dim_input = 1
        else:
            train_full_dataset = graph_dataset(smlstr_train,logCMC_train, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
            test_dataset = graph_dataset(smlstr_test,logCMC_test, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
        train_loader = torch.utils.data.DataLoader(train_full_dataset, batch_size=args.batch_size,
                                                   collate_fn=collate,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  collate_fn=collate,
                                                  shuffle=False)
        train_dataset = graph_dataset(smlstr_train,logCMC_train,add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
        fname = r"ep{}bs{}lr{}hu{}".format(args.epochs, args.batch_size,
                                                           args.lr,
                                                           args.unit_per_layer)

        best_rmse = 1000
        if args.train:
            print("Training the model ...")
            stopper = EarlyStopping(mode='lower', patience=args.patience, filename=f'{args.path}/{fname}es.pth.tar') # early stop model
            for epoch in range(args.start_epoch, args.epochs):
                train_loss = train(train_loader, model, loss_fn, optimizer, epoch, args, fname, SWriter)
                rmse = validate(test_loader, model, epoch, args, fname, SWriter)
                is_best = rmse < best_rmse
                best_rmse = min(rmse, best_rmse)
                if is_best:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'model_arch': model_arch,
                        'state_dict': model.state_dict(),
                        'best_rmse': best_rmse,
                        'optimizer': optimizer.state_dict(),
                    }, fname, args.path)
                if args.early_stop:
                    early_stop = stopper.step(train_loss, model)
                    if early_stop:
                        print("**********Early Stopping!")
                        break


        # test
        print("Testing the model ...")
        checkpoint = torch.load(r"{}/{}.pth.tar".format(args.path, fname))
        args.start_epoch = 0
        best_rmse = checkpoint['best_rmse']
        
        if args.gnn_model == GCNReg:
            model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
            model_arch = 'GCNReg' 

        elif args.gnn_model == GCNReg_add:
            model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
            model_arch = 'GCNReg_add'

        elif args.gnn_model == GCNReg_binary:
            model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
            model_arch = 'GCNReg_binary'

        elif args.gnn_model == GCNReg_binary_add:
            model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, rdkit_features=args.rdkit_descriptor, n_classes=1,saliency=True)
            model_arch = 'GCNReg_binary_add'

        else:
            raise ValueError('Invalid GNN model name')

        if args.gpu >= 0:
            model = model.cuda(args.gpu)
        model.load_state_dict(checkpoint['state_dict'])
        # if args.gpu < 0:
        #     model = model.cpu()
        # else:
        #     model = model.cuda(args.gpu)
        print("=> loaded checkpoint '{}' (epoch {}, rmse {})"
              .format(fname, checkpoint['epoch'], best_rmse))
        cudnn.deterministic = True
        stage = 'testtest'
        predict(test_dataset, model, -1, args, fname, stage, SWriter, args.path)
        stage = 'testtrain'
        predict(train_dataset, model, -1, args, fname, stage, SWriter, args.path)
        if args.early_stop:
            checkpoint = torch.load(r"{}/{}es.pth.tar".format(args.path, fname))
            args.start_epoch = 0
            #model = args.gnn_model(args.dim_input, args.add_feat, args.unit_per_layer,1,True)

            if args.gnn_model == GCNReg:
                model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
                model_arch = 'GCNReg' 

            elif args.gnn_model == GCNReg_add:
                model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
                model_arch = 'GCNReg_add'

            elif args.gnn_model == GCNReg_binary:
                model = args.gnn_model(in_dim=args.dim_input, hidden_dim = args.unit_per_layer, n_classes=1,saliency=True)
                model_arch = 'GCNReg_binary'

            elif args.gnn_model == GCNReg_binary_add:
                model = args.gnn_model(in_dim=args.dim_input, extra_in_dim=args.num_feat, hidden_dim = args.unit_per_layer, rdkit_features=args.rdkit_descriptor, n_classes=1,saliency=True)
                model_arch = 'GCNReg_binary_add'

            else:
                raise ValueError('Invalid GNN model name')

            if args.gpu >= 0:
                model = model.cuda(args.gpu)
            model.load_state_dict(checkpoint['model_state_dict'])
            train_dataset = graph_dataset(smlstr_train,logCMC_train,add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
            test_dataset = graph_dataset(smlstr_test,logCMC_test,add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
            cudnn.deterministic = True
            stage = 'testtest'
            predict(to_loader(test_dataset, collate), model, -1, args, r"{}es".format(fname), stage, SWriter, args.path)
            stage = 'testtrain'
            predict(to_loader(train_dataset, collate), model, -1, args, r"{}es".format(fname), stage, SWriter, args.path)


    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)