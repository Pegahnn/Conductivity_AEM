import optuna
import torch
import os
import csv
import numpy as np
from torch.utils.data import DataLoader, random_split
from createGraph_PN import graph_dataset, collates, collate_add, multi_graph_dataset, collate_multi, collate_multi_rdkit, collate_multi_non_rdkit
from NNgraph import GCNReg, GATReg_add, GCNReg_binary, GCNReg_binary_add
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from dgllife.utils import EarlyStopping
import time
import json
import pickle
import pandas as pd
import traceback

# Borrowing utility classes from GNN_functions.py
class AccumulationMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.avg = 0.0
        self.sum = 0
        self.count = 0.0
        self.predictions = []
        self.true_labels = []

    def update(self, value, n=1, preds=None, labels=None):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.sqrt = self.value ** 0.5
        self.rmse = self.avg ** 0.5
        if preds is not None and labels is not None:
            self.predictions.extend(preds.cpu().detach().numpy())
            self.true_labels.extend(labels.cpu().numpy())


def load_data(data_path, binary_system=False, num_feat=0, log_file=None):
    """
    Load data from CSV file with robust error handling
    
    Args:
        data_path (str): Path to CSV file
        binary_system (bool): Whether this is a binary molecular system
        num_feat (int): Number of expected additional features
        log_file (str): Path to log file for errors
        
    Returns:
        tuple: Numpy arrays of SMILES strings and target values
    """
    try:
        # First check if file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Use pandas for more robust CSV parsing
        df = pd.read_csv(data_path, header=None)
        
        # Log the first 5 rows for debugging
        print(f"Data preview (first 5 rows):")
        print(df.head())
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Data preview (first 5 rows):\n")
                f.write(str(df.head()) + "\n")
        
        # Check for NULL/NaN values
        if df.isnull().values.any():
            print(f"Warning: Dataset contains NULL/NaN values")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"Warning: Dataset contains NULL/NaN values\n")
        
        if binary_system:
            # For binary systems, we expect SMILES1, SMILES2, [features...], target
            if len(df.columns) < 3:
                raise ValueError(f"Binary system requires at least 3 columns (2 SMILES + target), found {len(df.columns)}")
            
            # All columns except the last one are inputs
            data = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values.astype(float)
            
            print(f"Binary system data shape: {data.shape}, target shape: {y.shape}")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"Binary system data shape: {data.shape}, target shape: {y.shape}\n")
            
        else:
            # For single molecule systems with features:
            # If we expect additional features, format is SMILES, [features...], target
            # If no features, format is just SMILES, target
            if num_feat > 0:
                # Check if enough columns
                if len(df.columns) < num_feat + 2:  # SMILES + features + target
                    raise ValueError(f"Expected {num_feat} features but only found {len(df.columns)-2} feature columns")
                
                # Extract SMILES and convert features to proper format
                smiles = df.iloc[:, 0].values
                features = df.iloc[:, 1:-1].values
                
                # Clean features and ensure they're numeric
                for col in range(features.shape[1]):
                    # Replace any non-numeric values with 0
                    for i in range(features.shape[0]):
                        try:
                            float(features[i, col])
                        except (ValueError, TypeError):
                            print(f"Warning: Non-numeric value '{features[i, col]}' in feature column {col+1}, row {i+1}. Replacing with 0.")
                            features[i, col] = 0
                
                # For createGraph.py compatibility, combine SMILES with features
                data = []
                for i in range(len(smiles)):
                    row = [smiles[i]] + features[i].tolist()
                    data.append(row)
                data = np.array(data)
                
            else:
                # No features, just SMILES and target
                if len(df.columns) < 2:
                    raise ValueError(f"Expected at least 2 columns (SMILES + target), found {len(df.columns)}")
                
                smiles = df.iloc[:, 0].values
                data = smiles
            
            y = df.iloc[:, -1].values.astype(float)
            
            print(f"Data shape: {data.shape if isinstance(data, np.ndarray) else len(data)}, target shape: {y.shape}")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"Data shape: {data.shape if isinstance(data, np.ndarray) else len(data)}, target shape: {y.shape}\n")
        
        return data, y
        
    except Exception as e:
        error_msg = f"Error loading data: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(error_msg + "\n")
        raise


def to_2d_tensor_on_device(x, device):
    """
    Convert input to 2D tensor on specified device
    
    Args:
        x: Input data (numpy array or tensor)
        device: Target device
        
    Returns:
        torch.Tensor: 2D tensor on specified device
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x).float()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x.to(device, non_blocking=True)


# Custom training function
def train(train_loader, model, loss_fn, optimizer, device, is_binary=False, has_add_features=False, rdkit_descriptor=False):
    """
    Train the network on the training set
    
    Args:
        train_loader: DataLoader for training set
        model: Model to train
        loss_fn: Loss function
        optimizer: Optimizer
        device: Device to use
        is_binary: Whether using binary molecular system
        has_add_features: Whether using additional features
        rdkit_descriptor: Whether using RDKit descriptors
        
    Returns:
        tuple: Training loss, RMSE, and R²
    """
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    model.train()
    end = time.time()

    for data in train_loader:
        # Handle different data formats based on model type
        if is_binary:
            if has_add_features:
                if rdkit_descriptor:
                    (graph1, graph2, descriptor1, descriptor2, label) = data
                    graph1 = graph1.to(device)
                    graph2 = graph2.to(device)
                    label = label.to(device, non_blocking=True)
                    descriptor1 = to_2d_tensor_on_device(descriptor1, device)
                    descriptor2 = to_2d_tensor_on_device(descriptor2, device)
                    output = model((graph1, graph2), (descriptor1, descriptor2))
                else:
                    (graph1, graph2, descriptor, label) = data
                    graph1 = graph1.to(device)
                    graph2 = graph2.to(device)
                    label = label.to(device, non_blocking=True)
                    descriptor = to_2d_tensor_on_device(descriptor, device)
                    output = model((graph1, graph2), descriptor)
            else:    
                (graph1, graph2, label) = data
                graph1 = graph1.to(device)
                graph2 = graph2.to(device)
                label = label.to(device, non_blocking=True)    
                output = model((graph1, graph2))
        else:
            if has_add_features:
                graph, descriptor, label = data
                graph = graph.to(device)
                label = label.to(device, non_blocking=True)
                descriptor = to_2d_tensor_on_device(descriptor, device)
                output = model(graph, descriptor)
            else:
                graph, label = data
                graph = graph.to(device)
                label = label.to(device, non_blocking=True)
                output = model(graph)

        # Calculate loss
        loss = loss_fn(output, label.float())
        loss_accum.update(loss.item(), label.size(0), output, label)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()

    # Calculate R² score
    r2 = r2_score(loss_accum.true_labels, loss_accum.predictions)
    return loss_accum.avg, loss_accum.rmse, r2


# Custom validation function
def validate(val_loader, model, device, is_binary=False, has_add_features=False, rdkit_descriptor=False):
    """
    Evaluate the network on the validation set
    
    Args:
        val_loader: DataLoader for validation set
        model: Model to evaluate
        device: Device to use
        is_binary: Whether using binary molecular system
        has_add_features: Whether using additional features
        rdkit_descriptor: Whether using RDKit descriptors
        
    Returns:
        tuple: RMSE, validation loss, and R²
    """
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for data in val_loader:
            # Handle different data formats based on model type
            if is_binary:
                if has_add_features:
                    if rdkit_descriptor:
                        (graph1, graph2, descriptor1, descriptor2, label) = data
                        graph1 = graph1.to(device)
                        graph2 = graph2.to(device)
                        label = label.to(device, non_blocking=True)
                        descriptor1 = to_2d_tensor_on_device(descriptor1, device)
                        descriptor2 = to_2d_tensor_on_device(descriptor2, device)
                        output = model((graph1, graph2), (descriptor1, descriptor2))
                    else:
                        (graph1, graph2, descriptor, label) = data
                        graph1 = graph1.to(device)
                        graph2 = graph2.to(device)
                        label = label.to(device, non_blocking=True)
                        descriptor = to_2d_tensor_on_device(descriptor, device)
                        output = model((graph1, graph2), descriptor)
                else:    
                    (graph1, graph2, label) = data
                    graph1 = graph1.to(device)
                    graph2 = graph2.to(device)
                    label = label.to(device, non_blocking=True)    
                    output = model((graph1, graph2))
            else:
                if has_add_features:
                    graph, descriptor, label = data
                    graph = graph.to(device)
                    label = label.to(device, non_blocking=True)
                    descriptor = to_2d_tensor_on_device(descriptor, device)
                    output = model(graph, descriptor)
                else:
                    graph, label = data
                    graph = graph.to(device)
                    label = label.to(device, non_blocking=True)
                    output = model(graph)
            
            # Calculate loss
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, label.float())
            loss_accum.update(loss.item(), label.size(0), output, label)
            
            batch_time.update(time.time() - end)
            end = time.time()

    # Calculate R² score
    r2 = r2_score(loss_accum.true_labels, loss_accum.predictions)
    return loss_accum.rmse, loss_accum.avg, r2


def objective(trial, model_name, split, data_path, save_dir, device="cuda:0", n_extra_features=6, binary_system=False, rdkit_descriptor=False):
    """
    Optuna objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        model_name: Name of model to use
        split: Validation split ratio
        data_path: Path to data file
        save_dir: Directory to save results
        device: Device to use
        n_extra_features: Number of additional features
        binary_system: Whether using binary molecular system
        rdkit_descriptor: Whether using RDKit descriptors
        
    Returns:
        float: Best validation RMSE
    """
    # Setup logging
    log_file = os.path.join(save_dir, f"trial_{trial.number}.log")
    
    # Parameters to optimize
    params = {
        'batch_size': trial.suggest_categorical('batch_size', [4, 5, 6, 8, 16, 20, 32, 64]),
        'lr': trial.suggest_float('lr', 0.0005, 0.01, log=True),
        'unit_per_layer': trial.suggest_categorical('unit_per_layer', [128, 256, 384, 512]),
        'epochs': trial.suggest_categorical('epochs', [500, 1000, 1500]),
        'patience': trial.suggest_categorical('patience', [20, 30, 40]),
        'seed': trial.suggest_categorical('seed', [42]),#Using a single seed for faster trials
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
        'scheduler': trial.suggest_categorical('scheduler', ['none', 'reduce_on_plateau', 'cosine']),
    }
    
    # Log parameters
    with open(log_file, 'w') as f:
        f.write(f"Trial {trial.number} parameters:\n")
        f.write(json.dumps(params, indent=2) + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Extra features: {n_extra_features}\n")
        f.write(f"Binary system: {binary_system}\n")
        f.write(f"RDKit descriptor: {rdkit_descriptor}\n\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])
    
    try:
        # Determine if we're using additional features
        has_add_features = model_name in ['GATReg_add', 'GCNReg_binary_add']
        
        # Load data with robust error handling
        smiles, y = load_data(
            data_path, 
            binary_system=binary_system, 
            num_feat=n_extra_features if has_add_features else 0,
            log_file=log_file
        )
        
        # Log data loading success
        with open(log_file, 'a') as f:
            f.write(f"Successfully loaded data: {len(y)} samples\n")
        
        # Create dataset
        if binary_system:
            dataset_class = multi_graph_dataset
        else:
            dataset_class = graph_dataset
            
        try:
            dataset = dataset_class(
                smiles, y, 
                add_features=has_add_features, 
                rdkit_descriptor=rdkit_descriptor
            )
            
            # Log dataset creation success
            with open(log_file, 'a') as f:
                f.write(f"Successfully created dataset\n")
                
        except Exception as e:
            error_msg = f"Error creating dataset: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open(log_file, 'a') as f:
                f.write(error_msg + "\n")
            raise
        
        # Split data
        train_size = int((1 - split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(params['seed'])
        )
        
        # Create data loaders
        if binary_system:
            if has_add_features:
                if rdkit_descriptor:
                    collate_fn = collate_multi_rdkit
                else:
                    collate_fn = collate_multi_non_rdkit
            else:
                collate_fn = collate_multi
        else:
            collate_fn = collate_add if has_add_features else collates
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Model parameters
        model_params = {
            'in_dim': 74,  # Standard atom feature dimension
            'hidden_dim': params['unit_per_layer'],
            'n_classes': 1
        }
        
        # Create model
        if model_name == 'GCNReg':
            model = GCNReg(**model_params)
        elif model_name == 'GATReg_add':
            model = GATReg_add(
                in_dim=model_params['in_dim'],
                extra_in_dim=n_extra_features,
                hidden_dim=model_params['hidden_dim'],
                n_classes=model_params['n_classes']
            )
        elif model_name == 'GCNReg_binary':
            model = GCNReg_binary(**model_params)
        elif model_name == 'GCNReg_binary_add':
            model = GCNReg_binary_add(
                in_dim=model_params['in_dim'],
                extra_in_dim=n_extra_features,
                hidden_dim=model_params['hidden_dim'],
                n_classes=model_params['n_classes'],
                rdkit_features=rdkit_descriptor
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Move model to device
        model = model.to(device)
        
        # Loss function and optimizer
        loss_fn = nn.MSELoss()
        loss_fn = loss_fn.to(device)
        
        # Create optimizer
        if params['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay']
            )
        elif params['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
        
        # Setup scheduler
        if params['scheduler'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10, verbose=True
            )
        elif params['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=params['epochs'], eta_min=1e-6
            )
        else:
            scheduler = None
        
        # Setup early stopping
        stopper = EarlyStopping(
            mode='lower',
            patience=params['patience'],
            filename=os.path.join(save_dir, f"trial_{trial.number}_best.pth.tar")
        )
        
        # Training loop
        best_val_rmse = float('inf')
        trial_dir = os.path.join(save_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)
        
        # Create a tensorboard writer for this trial
        writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs", f"trial_{trial.number}"))
        
        for epoch in range(params['epochs']):
            # Train for one epoch
            train_loss, train_rmse, train_r2 = train(
                train_loader, model, loss_fn, optimizer, device, 
                is_binary=binary_system, has_add_features=has_add_features, 
                rdkit_descriptor=rdkit_descriptor
            )
            
            # Evaluate on validation set
            val_rmse, val_loss, val_r2 = validate(
                val_loader, model, device, 
                is_binary=binary_system, has_add_features=has_add_features, 
                rdkit_descriptor=rdkit_descriptor
            )
            
            # Update scheduler if applicable
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_rmse)
                else:
                    scheduler.step()
            
            # Log metrics
            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('val_loss', val_loss, epoch)
            writer.add_scalar('train_rmse', train_rmse, epoch)
            writer.add_scalar('val_rmse', val_rmse, epoch)
            writer.add_scalar('train_r2', train_r2, epoch)
            writer.add_scalar('val_r2', val_r2, epoch)
            
            # Progress reporting for Optuna
            trial.report(val_rmse, epoch)
            
            # Handle pruning based on intermediate results
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            # Check if this is the best model so far
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                # Save the best model
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_rmse': val_rmse,
                    'optimizer': optimizer.state_dict(),
                    'params': params
                }, os.path.join(trial_dir, "best_model.pth.tar"))
            
            # Early stopping
            early_stop = stopper.step(val_loss, model)
            if early_stop:
                print(f"Trial {trial.number}: Early stopping at epoch {epoch}")
                with open(log_file, 'a') as f:
                    f.write(f"Early stopping at epoch {epoch}\n")
                break
        
        writer.close()
        
        # Save trial results
        with open(os.path.join(save_dir, "trials_results.csv"), "a") as f:
            writer = csv.writer(f)
            writer.writerow([
                trial.number,
                best_val_rmse,
                params['batch_size'],
                params['lr'],
                params['unit_per_layer'],
                params['epochs'],
                params['patience'],
                params['seed'],
                params['weight_decay'],
                params['optimizer'],
                params['scheduler']
            ])
        
        # Log final results
        with open(log_file, 'a') as f:
            f.write(f"Trial completed with best RMSE: {best_val_rmse}\n")
        
        return best_val_rmse
        
    except Exception as e:
        error_msg = f"Trial {trial.number} failed with error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(log_file, 'a') as f:
            f.write(error_msg + "\n")
        return float('inf')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='GNN Hyperparameter Optimization')
    parser.add_argument('--model', type=str, choices=['GCNReg', 'GATReg_add', 'GCNReg_binary', 'GCNReg_binary_add'], required=True,
                        help='Model type to optimize')
    parser.add_argument('--split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset CSV')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--num_feat', type=int, default=6, 
                        help='Number of additional features (for _add models)')
    parser.add_argument('--binary_system', action='store_true',
                        help='Whether using binary molecular system')
    parser.add_argument('--rdkit_descriptor', action='store_true',
                        help='Whether using RDKit descriptors')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout for optimization in seconds')
    parser.add_argument('--random_seed', type=int, default=42, help='Fixed seed for reproducibility')

    args = parser.parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save parameters
    with open(os.path.join(args.save_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Create pruner for Optuna (stops unpromising trials early)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=20,
        interval_steps=10
    )
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name=f"gnn_optimization_{args.model}"
    )
    
    # Create results file header
    results_file = os.path.join(args.save_dir, "trials_results.csv")
    if not os.path.exists(results_file):
        with open(results_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial_number",
                "val_rmse",
                "batch_size",
                "learning_rate",
                "hidden_dim",
                "epochs",
                "patience",
                "seed",
                "weight_decay",
                "optimizer",
                "scheduler"
            ])
    
    # Main log file
    main_log = os.path.join(args.save_dir, "optimization.log")
    with open(main_log, "w") as f:
        f.write(f"Starting optimization for model {args.model} with {args.n_trials} trials\n")
        f.write(f"Data path: {args.data_path}\n")
        f.write(f"Save directory: {args.save_dir}\n\n")
        f.write(f"Command line arguments:\n")
        f.write(json.dumps(vars(args), indent=2) + "\n\n")
    
    # Run optimization
    print(f"Starting optimization for model {args.model} with {args.n_trials} trials")
    print(f"Data path: {args.data_path}")
    print(f"Save directory: {args.save_dir}")
    
    # Define callback to save study after each trial
    def save_study_callback(study, trial):
        pickle.dump(study, open(os.path.join(args.save_dir, "study.pkl"), "wb"))
        
        # Also save current best trial info
        best_trial = study.best_trial
        with open(os.path.join(args.save_dir, "current_best.txt"), "w") as f:
            f.write(f"Best trial so far: #{best_trial.number}\n")
            f.write(f"Best RMSE: {best_trial.value}\n\n")
            f.write("Parameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
    
    try:
        # First check if the data can be loaded correctly
        print("Checking data loading...")
        try:
            has_add_features = args.model in ['GATReg_add', 'GCNReg_binary_add']
            smiles, y = load_data(
                args.data_path, 
                binary_system=args.binary_system, 
                num_feat=args.num_feat if has_add_features else 0,
                log_file=main_log
            )
            print(f"Data check successful. Loaded {len(y)} samples.")
            with open(main_log, "a") as f:
                f.write(f"Data check successful. Loaded {len(y)} samples.\n")
        except Exception as e:
            error_msg = f"Data check failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            with open(main_log, "a") as f:
                f.write(error_msg + "\n")
            raise
        
        # Start optimization
        study.optimize(
            lambda trial: objective(
                trial,
                args.model,
                args.split,
                args.data_path,
                args.save_dir,
                device,
                args.num_feat,
                args.binary_system,
                args.rdkit_descriptor
            ),
            n_trials=args.n_trials,
            timeout=args.timeout,
            callbacks=[save_study_callback],
            gc_after_trial=True
        )
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
        with open(main_log, "a") as f:
            f.write("Optimization interrupted by user.\n")
    except Exception as e:
        error_msg = f"Optimization failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        with open(main_log, "a") as f:
            f.write(error_msg + "\n")
    
    # Print and save results
    try:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best parameters
        with open(os.path.join(args.save_dir, "best_params.json"), "w") as f:
            results = {
                "best_value": trial.value,
                "best_params": trial.params
            }
            json.dump(results, f, indent=2)
        
        # Final log entry
        with open(main_log, "a") as f:
            f.write("\nOptimization completed\n")
            f.write(f"Best trial: #{trial.number}\n")
            f.write(f"Best RMSE: {trial.value}\n\n")
            f.write("Parameters:\n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")
        
        # Plot optimization history
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(os.path.join(args.save_dir, "optimization_history.html"))
            
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(os.path.join(args.save_dir, "param_importances.html"))
            
            fig = optuna.visualization.plot_slice(study)
            fig.write_html(os.path.join(args.save_dir, "slice_plot.html"))
        except Exception as e:
            print(f"Error creating visualization: {e}")
            with open(main_log, "a") as f:
                f.write(f"Error creating visualization: {e}\n")
    except Exception as e:
        print(f"Error processing results: {e}")
        with open(main_log, "a") as f:
            f.write(f"Error processing results: {e}\n")


if __name__ == "__main__":
    main()

# import optuna
# import torch
# import os
# import csv
# import numpy as np
# from torch.utils.data import DataLoader, random_split
# from createGraph import graph_dataset, collates, collate_add, multi_graph_dataset, collate_multi, collate_multi_rdkit, collate_multi_non_rdkit
# from NNgraph import GCNReg, GCNReg_add, GCNReg_binary, GCNReg_binary_add
# import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# from dgllife.utils import EarlyStopping
# import time
# import json
# import pickle

# # Borrowing utility classes from GNN_functions.py
# class AccumulationMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.value = 0.0
#         self.avg = 0.0
#         self.sum = 0
#         self.count = 0.0
#         self.predictions = []
#         self.true_labels = []

#     def update(self, value, n=1, preds=None, labels=None):
#         self.value = value
#         self.sum += value * n
#         self.count += n
#         self.avg = self.sum / self.count
#         self.sqrt = self.value ** 0.5
#         self.rmse = self.avg ** 0.5
#         if preds is not None and labels is not None:
#             self.predictions.extend(preds.cpu().detach().numpy())
#             self.true_labels.extend(labels.cpu().numpy())


# def load_data(data_path):
#     """
#     Load data from CSV file
    
#     Args:
#         data_path (str): Path to CSV file
        
#     Returns:
#         tuple: Numpy arrays of SMILES strings and target values
#     """
#     smiles = []
#     y = []
#     with open(data_path, 'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             smiles.append(row[0])
#             y.append(float(row[1]))
#     return np.array(smiles), np.array(y)


# def load_binary_data(data_path):
#     """
#     Load binary molecular system data from CSV file
    
#     Args:
#         data_path (str): Path to CSV file
        
#     Returns:
#         tuple: Numpy arrays of SMILES strings and target values
#     """
#     data = []
#     y = []
#     with open(data_path, 'r') as f:
#         reader = csv.reader(f)
#         for row in reader:
#             data.append(row[:-1])
#             y.append(float(row[-1]))
#     return np.array(data), np.array(y)


# def to_2d_tensor_on_device(x, device):
#     """
#     Convert input to 2D tensor on specified device
    
#     Args:
#         x: Input data (numpy array or tensor)
#         device: Target device
        
#     Returns:
#         torch.Tensor: 2D tensor on specified device
#     """
#     if isinstance(x, np.ndarray):
#         x = torch.tensor(x).float()
#     if x.dim() == 1:
#         x = x.unsqueeze(0)
#     return x.to(device, non_blocking=True)


# # Custom training function
# def train(train_loader, model, loss_fn, optimizer, device, is_binary=False, has_add_features=False, rdkit_descriptor=False):
#     """
#     Train the network on the training set
    
#     Args:
#         train_loader: DataLoader for training set
#         model: Model to train
#         loss_fn: Loss function
#         optimizer: Optimizer
#         device: Device to use
#         is_binary: Whether using binary molecular system
#         has_add_features: Whether using additional features
#         rdkit_descriptor: Whether using RDKit descriptors
        
#     Returns:
#         tuple: Training loss, RMSE, and R²
#     """
#     batch_time = AccumulationMeter()
#     loss_accum = AccumulationMeter()
#     model.train()
#     end = time.time()

#     for data in train_loader:
#         # Handle different data formats based on model type
#         if is_binary:
#             if has_add_features:
#                 if rdkit_descriptor:
#                     (graph1, graph2, descriptor1, descriptor2, label) = data
#                     graph1 = graph1.to(device)
#                     graph2 = graph2.to(device)
#                     label = label.to(device, non_blocking=True)
#                     descriptor1 = to_2d_tensor_on_device(descriptor1, device)
#                     descriptor2 = to_2d_tensor_on_device(descriptor2, device)
#                     output = model((graph1, graph2), (descriptor1, descriptor2))
#                 else:
#                     (graph1, graph2, descriptor, label) = data
#                     graph1 = graph1.to(device)
#                     graph2 = graph2.to(device)
#                     label = label.to(device, non_blocking=True)
#                     descriptor = to_2d_tensor_on_device(descriptor, device)
#                     output = model((graph1, graph2), descriptor)
#             else:    
#                 (graph1, graph2, label) = data
#                 graph1 = graph1.to(device)
#                 graph2 = graph2.to(device)
#                 label = label.to(device, non_blocking=True)    
#                 output = model((graph1, graph2))
#         else:
#             if has_add_features:
#                 graph, descriptor, label = data
#                 graph = graph.to(device)
#                 label = label.to(device, non_blocking=True)
#                 descriptor = to_2d_tensor_on_device(descriptor, device)
#                 output = model(graph, descriptor)
#             else:
#                 graph, label = data
#                 graph = graph.to(device)
#                 label = label.to(device, non_blocking=True)
#                 output = model(graph)

#         # Calculate loss
#         loss = loss_fn(output, label.float())
#         loss_accum.update(loss.item(), label.size(0), output, label)
        
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         batch_time.update(time.time() - end)
#         end = time.time()

#     # Calculate R² score
#     r2 = r2_score(loss_accum.true_labels, loss_accum.predictions)
#     return loss_accum.avg, loss_accum.rmse, r2


# # Custom validation function
# def validate(val_loader, model, device, is_binary=False, has_add_features=False, rdkit_descriptor=False):
#     """
#     Evaluate the network on the validation set
    
#     Args:
#         val_loader: DataLoader for validation set
#         model: Model to evaluate
#         device: Device to use
#         is_binary: Whether using binary molecular system
#         has_add_features: Whether using additional features
#         rdkit_descriptor: Whether using RDKit descriptors
        
#     Returns:
#         tuple: RMSE, validation loss, and R²
#     """
#     batch_time = AccumulationMeter()
#     loss_accum = AccumulationMeter()
#     model.eval()
    
#     with torch.no_grad():
#         end = time.time()
#         for data in val_loader:
#             # Handle different data formats based on model type
#             if is_binary:
#                 if has_add_features:
#                     if rdkit_descriptor:
#                         (graph1, graph2, descriptor1, descriptor2, label) = data
#                         graph1 = graph1.to(device)
#                         graph2 = graph2.to(device)
#                         label = label.to(device, non_blocking=True)
#                         descriptor1 = to_2d_tensor_on_device(descriptor1, device)
#                         descriptor2 = to_2d_tensor_on_device(descriptor2, device)
#                         output = model((graph1, graph2), (descriptor1, descriptor2))
#                     else:
#                         (graph1, graph2, descriptor, label) = data
#                         graph1 = graph1.to(device)
#                         graph2 = graph2.to(device)
#                         label = label.to(device, non_blocking=True)
#                         descriptor = to_2d_tensor_on_device(descriptor, device)
#                         output = model((graph1, graph2), descriptor)
#                 else:    
#                     (graph1, graph2, label) = data
#                     graph1 = graph1.to(device)
#                     graph2 = graph2.to(device)
#                     label = label.to(device, non_blocking=True)    
#                     output = model((graph1, graph2))
#             else:
#                 if has_add_features:
#                     graph, descriptor, label = data
#                     graph = graph.to(device)
#                     label = label.to(device, non_blocking=True)
#                     descriptor = to_2d_tensor_on_device(descriptor, device)
#                     output = model(graph, descriptor)
#                 else:
#                     graph, label = data
#                     graph = graph.to(device)
#                     label = label.to(device, non_blocking=True)
#                     output = model(graph)
            
#             # Calculate loss
#             loss_fn = nn.MSELoss()
#             loss = loss_fn(output, label.float())
#             loss_accum.update(loss.item(), label.size(0), output, label)
            
#             batch_time.update(time.time() - end)
#             end = time.time()

#     # Calculate R² score
#     r2 = r2_score(loss_accum.true_labels, loss_accum.predictions)
#     return loss_accum.rmse, loss_accum.avg, r2


# def objective(trial, model_name, split, data_path, save_dir, device="cuda:0", n_extra_features=6, binary_system=False, rdkit_descriptor=False):
#     """
#     Optuna objective function for hyperparameter optimization
    
#     Args:
#         trial: Optuna trial object
#         model_name: Name of model to use
#         split: Validation split ratio
#         data_path: Path to data file
#         save_dir: Directory to save results
#         device: Device to use
#         n_extra_features: Number of additional features
#         binary_system: Whether using binary molecular system
#         rdkit_descriptor: Whether using RDKit descriptors
        
#     Returns:
#         float: Best validation RMSE
#     """
#     # Parameters to optimize
#     params = {
#         'batch_size': trial.suggest_categorical('batch_size', [4, 5, 6, 8, 16, 20, 32, 64]),
#         'lr': trial.suggest_float('lr', 0.0005, 0.01, log=True),
#         'unit_per_layer': trial.suggest_categorical('unit_per_layer', [128, 256, 384, 512]),
#         'epochs': trial.suggest_categorical('epochs', [500, 1000, 1500]),
#         'patience': trial.suggest_categorical('patience', [20, 30, 40]),
#         'seed': trial.suggest_categorical('seed', [2021]),  # Using a single seed for faster trials
#         'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
#         'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw']),
#         'scheduler': trial.suggest_categorical('scheduler', ['none', 'reduce_on_plateau', 'cosine']),
#     }
    
#     # Set random seed for reproducibility
#     torch.manual_seed(params['seed'])
#     np.random.seed(params['seed'])
    
#     # Load data
#     if binary_system:
#         smiles, y = load_binary_data(data_path)
#     else:
#         smiles, y = load_data(data_path)
    
#     # Create dataset
#     has_add_features = model_name in ['GCNReg_add', 'GCNReg_binary_add']
    
#     if binary_system:
#         dataset_class = multi_graph_dataset
#     else:
#         dataset_class = graph_dataset
        
#     dataset = dataset_class(smiles, y, add_features=has_add_features, rdkit_descriptor=rdkit_descriptor)
    
#     # Split data
#     train_size = int((1 - split) * len(dataset))
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = random_split(
#         dataset, 
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(params['seed'])
#     )
    
#     # Create data loaders
#     if binary_system:
#         if has_add_features:
#             if rdkit_descriptor:
#                 collate_fn = collate_multi_rdkit
#             else:
#                 collate_fn = collate_multi_non_rdkit
#         else:
#             collate_fn = collate_multi
#     else:
#         collate_fn = collate_add if has_add_features else collates
    
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=params['batch_size'],
#         shuffle=True,
#         collate_fn=collate_fn
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=params['batch_size'],
#         shuffle=False,
#         collate_fn=collate_fn
#     )
    
#     # Model parameters
#     model_params = {
#         'in_dim': 74,  # Standard atom feature dimension
#         'hidden_dim': params['unit_per_layer'],
#         'n_classes': 1
#     }
    
#     # Create model
#     if model_name == 'GCNReg':
#         model = GCNReg(**model_params)
#     elif model_name == 'GCNReg_add':
#         model = GCNReg_add(
#             in_dim=model_params['in_dim'],
#             extra_in_dim=n_extra_features,
#             hidden_dim=model_params['hidden_dim'],
#             n_classes=model_params['n_classes']
#         )
#     elif model_name == 'GCNReg_binary':
#         model = GCNReg_binary(**model_params)
#     elif model_name == 'GCNReg_binary_add':
#         model = GCNReg_binary_add(
#             in_dim=model_params['in_dim'],
#             extra_in_dim=n_extra_features,
#             hidden_dim=model_params['hidden_dim'],
#             n_classes=model_params['n_classes'],
#             rdkit_features=rdkit_descriptor
#         )
#     else:
#         raise ValueError(f"Unknown model name: {model_name}")
    
#     # Move model to device
#     model = model.to(device)
    
#     # Loss function and optimizer
#     loss_fn = nn.MSELoss()
#     loss_fn = loss_fn.to(device)
    
#     # Create optimizer
#     if params['optimizer'] == 'adam':
#         optimizer = torch.optim.Adam(
#             model.parameters(), 
#             lr=params['lr'], 
#             weight_decay=params['weight_decay']
#         )
#     elif params['optimizer'] == 'adamw':
#         optimizer = torch.optim.AdamW(
#             model.parameters(), 
#             lr=params['lr'], 
#             weight_decay=params['weight_decay']
#         )
#     else:
#         optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
#     # Setup scheduler
#     if params['scheduler'] == 'reduce_on_plateau':
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode='min', factor=0.5, patience=10, verbose=True
#         )
#     elif params['scheduler'] == 'cosine':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=params['epochs'], eta_min=1e-6
#         )
#     else:
#         scheduler = None
    
#     # Setup early stopping
#     stopper = EarlyStopping(
#         mode='lower',
#         patience=params['patience'],
#         filename=os.path.join(save_dir, f"trial_{trial.number}_best.pth.tar")
#     )
    
#     # Training loop
#     best_val_rmse = float('inf')
#     trial_dir = os.path.join(save_dir, f"trial_{trial.number}")
#     os.makedirs(trial_dir, exist_ok=True)
    
#     # Create a tensorboard writer for this trial
#     writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs", f"trial_{trial.number}"))
    
#     try:
#         for epoch in range(params['epochs']):
#             # Train for one epoch
#             train_loss, train_rmse, train_r2 = train(
#                 train_loader, model, loss_fn, optimizer, device, 
#                 is_binary=binary_system, has_add_features=has_add_features, 
#                 rdkit_descriptor=rdkit_descriptor
#             )
            
#             # Evaluate on validation set
#             val_rmse, val_loss, val_r2 = validate(
#                 val_loader, model, device, 
#                 is_binary=binary_system, has_add_features=has_add_features, 
#                 rdkit_descriptor=rdkit_descriptor
#             )
            
#             # Update scheduler if applicable
#             if scheduler is not None:
#                 if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#                     scheduler.step(val_rmse)
#                 else:
#                     scheduler.step()
            
#             # Log metrics
#             writer.add_scalar('train_loss', train_loss, epoch)
#             writer.add_scalar('val_loss', val_loss, epoch)
#             writer.add_scalar('train_rmse', train_rmse, epoch)
#             writer.add_scalar('val_rmse', val_rmse, epoch)
#             writer.add_scalar('train_r2', train_r2, epoch)
#             writer.add_scalar('val_r2', val_r2, epoch)
            
#             # Progress reporting for Optuna
#             trial.report(val_rmse, epoch)
            
#             # Handle pruning based on intermediate results
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()
            
#             # Check if this is the best model so far
#             if val_rmse < best_val_rmse:
#                 best_val_rmse = val_rmse
#                 # Save the best model
#                 torch.save({
#                     'epoch': epoch + 1,
#                     'state_dict': model.state_dict(),
#                     'best_rmse': val_rmse,
#                     'optimizer': optimizer.state_dict(),
#                     'params': params
#                 }, os.path.join(trial_dir, "best_model.pth.tar"))
            
#             # Early stopping
#             early_stop = stopper.step(val_loss, model)
#             if early_stop:
#                 print(f"Trial {trial.number}: Early stopping at epoch {epoch}")
#                 break
        
#         writer.close()
        
#         # Save trial results
#         with open(os.path.join(save_dir, "trials_results.csv"), "a") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 trial.number,
#                 best_val_rmse,
#                 params['batch_size'],
#                 params['lr'],
#                 params['unit_per_layer'],
#                 params['epochs'],
#                 params['patience'],
#                 params['seed'],
#                 params['weight_decay'],
#                 params['optimizer'],
#                 params['scheduler']
#             ])
        
#         return best_val_rmse
        
#     except Exception as e:
#         print(f"Trial {trial.number} failed with error: {e}")
#         return float('inf')


# def main():
#     import argparse
#     parser = argparse.ArgumentParser(description='GNN Hyperparameter Optimization')
#     parser.add_argument('--model', type=str, choices=['GCNReg', 'GCNReg_add', 'GCNReg_binary', 'GCNReg_binary_add'], required=True,
#                         help='Model type to optimize')
#     parser.add_argument('--split', type=float, default=0.1,
#                         help='Validation split ratio')
#     parser.add_argument('--n_trials', type=int, default=100,
#                         help='Number of optimization trials')
#     parser.add_argument('--data_path', type=str, required=True,
#                         help='Path to dataset CSV')
#     parser.add_argument('--save_dir', type=str, required=True,
#                         help='Directory to save results')
#     parser.add_argument('--gpu', type=int, default=0,
#                         help='GPU ID to use (-1 for CPU)')
#     parser.add_argument('--num_feat', type=int, default=6, 
#                         help='Number of additional features (for _add models)')
#     parser.add_argument('--binary_system', action='store_true',
#                         help='Whether using binary molecular system')
#     parser.add_argument('--rdkit_descriptor', action='store_true',
#                         help='Whether using RDKit descriptors')
#     parser.add_argument('--timeout', type=int, default=None,
#                         help='Timeout for optimization in seconds')
#     args = parser.parse_args()
    
#     # Set device
#     if args.gpu >= 0 and torch.cuda.is_available():
#         device = f"cuda:{args.gpu}"
#     else:
#         device = "cpu"
#     print(f"Using device: {device}")
    
#     # Create save directory if it doesn't exist
#     os.makedirs(args.save_dir, exist_ok=True)
    
#     # Save parameters
#     with open(os.path.join(args.save_dir, "params.json"), "w") as f:
#         json.dump(vars(args), f, indent=2)
    
#     # Create pruner for Optuna (stops unpromising trials early)
#     pruner = optuna.pruners.MedianPruner(
#         n_startup_trials=5,
#         n_warmup_steps=20,
#         interval_steps=10
#     )
    
#     # Create study
#     study = optuna.create_study(
#         direction="minimize",
#         pruner=pruner,
#         study_name=f"gnn_optimization_{args.model}"
#     )
    
#     # Create results file header
#     results_file = os.path.join(args.save_dir, "trials_results.csv")
#     if not os.path.exists(results_file):
#         with open(results_file, "w") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "trial_number",
#                 "val_rmse",
#                 "batch_size",
#                 "learning_rate",
#                 "hidden_dim",
#                 "epochs",
#                 "patience",
#                 "seed",
#                 "weight_decay",
#                 "optimizer",
#                 "scheduler"
#             ])
    
#     # Run optimization
#     print(f"Starting optimization for model {args.model} with {args.n_trials} trials")
#     print(f"Data path: {args.data_path}")
#     print(f"Save directory: {args.save_dir}")
    
#     # Define callback to save study after each trial
#     def save_study_callback(study, trial):
#         pickle.dump(study, open(os.path.join(args.save_dir, "study.pkl"), "wb"))
    
#     try:
#         study.optimize(
#             lambda trial: objective(
#                 trial,
#                 args.model,
#                 args.split,
#                 args.data_path,
#                 args.save_dir,
#                 device,
#                 args.num_feat,
#                 args.binary_system,
#                 args.rdkit_descriptor
#             ),
#             n_trials=args.n_trials,
#             timeout=args.timeout,
#             callbacks=[save_study_callback],
#             gc_after_trial=True
#         )
#     except KeyboardInterrupt:
#         print("Optimization interrupted by user.")
    
#     # Print and save results
#     print("\nBest trial:")
#     trial = study.best_trial
#     print(f"  Value: {trial.value}")
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")
    
#     # Save best parameters
#     with open(os.path.join(args.save_dir, "best_params.json"), "w") as f:
#         results = {
#             "best_value": trial.value,
#             "best_params": trial.params
#         }
#         json.dump(results, f, indent=2)
    
#     # Plot optimization history
#     try:
#         fig = optuna.visualization.plot_optimization_history(study)
#         fig.write_html(os.path.join(args.save_dir, "optimization_history.html"))
        
#         fig = optuna.visualization.plot_param_importances(study)
#         fig.write_html(os.path.join(args.save_dir, "param_importances.html"))
        
#         fig = optuna.visualization.plot_slice(study)
#         fig.write_html(os.path.join(args.save_dir, "slice_plot.html"))
#     except Exception as e:
#         print(f"Error creating visualization: {e}")


# if __name__ == "__main__":
#     main()