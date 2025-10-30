# ===========================================================
# Graph Joint Autoencoder + Regressor + Ensemble + HPO (PyG)
# ===========================================================
# - PyTorch 2.6 / PyG 2.5 compatible
# - R²-first hyperparameter search with tie-break on MAE
# - Denoising AE pretrain (node feature recon: BCE on binary, MSE on cont.)
# - Joint fine-tune with supervised + light reconstruction
# - CosineAnnealingWarmRestarts + SWA, ensemble across diverse seeds
# - Saves HPO log and final artifacts
# ===========================================================

import os, json, random, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem
RDLogger.DisableLog('rdApp.*')

from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, BatchNorm

# ---------------------------
# Reproducibility / Device
# ---------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Paths
# ---------------------------
CSV_PATH = "./modified_data_final.csv"
HPO_LOG_PATH = "graph_hpo_log.json"
ARTIFACT_PATH = "option4b_graph_joint_ae_regressor_ensemble.pt"

# ---------------------------
# Base Config (can be overridden by HPO)
# ---------------------------
BASE_CFG = dict(
    # Featurization
    atom_nums=[1,6,7,8,9,15,16,17,35,53],
    max_degree=5,
    charges=[-2,-1,0,1,2],
    hybs=[rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3,
          rdchem.HybridizationType.SP3D, rdchem.HybridizationType.SP3D2],
    bond_types=[rdchem.BondType.SINGLE, rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE, rdchem.BondType.AROMATIC],
    force_reprocess=False,

    # Encoder (GNN)
    enc_hidden=128,
    enc_layers=3,
    enc_dropout=0.10,

    # AE pretrain
    ae_lr=1e-3,
    ae_weight_decay=1e-5,
    ae_epochs=150,
    ae_batch_size=64,
    ae_feat_drop=0.10,

    # Regressor (Dense + SE)
    growth=256,
    n_blocks=4,
    dropout=0.28,
    head_hidden=384,

    # Joint training
    lr=1.05e-3,
    weight_decay=9e-4,
    batch_size=32,
    epochs=600,
    T0=70,
    Tmult=1,
    swa_ratio=0.33,

    # Loss mix
    alpha=0.65,        # supervised mix: alpha * SmoothL1 + (1-alpha) * MSE
    sup_weight=0.90,   # total = sup_weight * sup + recon_weight * recon
    recon_weight=0.10,

    # Data split / ensemble
    val_split=0.10,
    ensemble_size=10
)

# ---------------------------
# Atom / Bond Features
# ---------------------------
def atom_features(atom, cfg):
    nums, hybs, charges, max_deg = cfg["atom_nums"], cfg["hybs"], cfg["charges"], cfg["max_degree"]
    f_num = [1.0 if atom.GetAtomicNum()==z else 0.0 for z in nums] + [0.0]
    if sum(f_num[:-1]) == 0: f_num[-1] = 1.0
    d = atom.GetTotalDegree()
    f_deg = [1.0 if d==k else 0.0 for k in range(max_deg+1)] + [1.0 if d>max_deg else 0.0]
    ch = atom.GetFormalCharge()
    f_ch = [1.0 if ch==c else 0.0 for c in charges] + [1.0 if ch not in charges else 0.0]
    hyb = atom.GetHybridization()
    f_hy = [1.0 if hyb==h else 0.0 for h in hybs] + [1.0 if hyb not in hybs else 0.0]
    f_bin_misc = [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    f_cont = [float(atom.GetTotalNumHs(includeNeighbors=True)), float(atom.GetImplicitValence())]
    return np.array(f_num + f_deg + f_ch + f_hy + f_bin_misc + f_cont, dtype=np.float32)

def bond_features(bond, cfg):
    btypes = cfg["bond_types"]
    f_bt = [1.0 if bond.GetBondType()==t else 0.0 for t in btypes] + [0.0]
    if sum(f_bt[:-1]) == 0: f_bt[-1] = 1.0
    f_misc = [float(bond.GetIsConjugated()), float(bond.IsInRing())]
    return np.array(f_bt + f_misc, dtype=np.float32)

def get_atom_feature_dims(cfg):
    bin_dim = (len(cfg["atom_nums"])+1) + (cfg["max_degree"]+1+1) + (len(cfg["charges"])+1) + (len(cfg["hybs"])+1) + 2
    cont_dim = 2
    return bin_dim, cont_dim

def smiles_to_pyg(smiles, y, cfg):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    Chem.Kekulize(mol, clearAromaticFlags=False)
    x = torch.tensor(np.vstack([atom_features(a, cfg) for a in mol.GetAtoms()]), dtype=torch.float32)

    ei_src, ei_dst, eattr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bf = bond_features(b, cfg)
        ei_src += [i, j]; ei_dst += [j, i]
        eattr.append(bf); eattr.append(bf)

    if len(ei_src) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0, len(cfg["bond_types"]) + 2 + 1), dtype=torch.float32)  # +1 fallback included above
    else:
        edge_index = torch.tensor([ei_src, ei_dst], dtype=torch.long)
        edge_attr  = torch.tensor(np.vstack(eattr), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([float(y)], dtype=torch.float32))

# ---------------------------
# Dataset + Target Transform
# ---------------------------
class GraphDataset(InMemoryDataset):
    def __init__(self, csv_path, cfg, force_reprocess=False):
        self.csv_path = csv_path
        self.cfg = cfg
        self._procdir = ".proc_graph_pt26"
        os.makedirs(self._procdir, exist_ok=True)
        super().__init__(self._procdir)
        fname = self.processed_paths[0]
        if force_reprocess or (not os.path.exists(fname)):
            self.process()
        try:
            self.data, self.slices = torch.load(fname, weights_only=False)
        except Exception:
            self.process()
            self.data, self.slices = torch.load(fname, weights_only=False)

    @property
    def processed_file_names(self):
        base = os.path.splitext(os.path.basename(self.csv_path))[0]
        return [f"{base}_graphs.pt"]

    def process(self):
        df = pd.read_csv(self.csv_path)
        smiles = df.iloc[:, 0].astype(str).tolist()
        num_feats = df.iloc[:, 1:-1].astype(float).values   # 4 numeric feature columns
        targets = df.iloc[:, -1].astype(float).tolist()

        data_list = []
        for s, nf, t in zip(smiles, num_feats, targets):
            d = smiles_to_pyg(s, t, self.cfg)
            if d is not None:
                d.num_feats = torch.tensor(nf, dtype=torch.float32)  # attach 4 numeric features
                data_list.append(d)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def fit_target_transform(dataset):
    y = torch.stack([dataset.get(i).y for i in range(len(dataset))]).cpu().numpy().reshape(-1,1)
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    y_t = pt.fit_transform(y).astype(np.float32)
    for i in range(len(dataset)):
        dataset.data.y[dataset.slices['y'][i]:dataset.slices['y'][i+1]] = torch.tensor(y_t[i], dtype=torch.float32)
    return pt

def inverse_target(pt, y_np):
    y_np = np.asarray(y_np, dtype=np.float64)
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=10.0, neginf=-10.0)
    y_np = np.clip(y_np, -10, 10)
    try:
        inv = pt.inverse_transform(y_np.reshape(-1,1)).ravel()
    except Exception:
        inv = y_np
    inv = np.nan_to_num(inv, nan=0.0, posinf=np.finfo(np.float32).max/2, neginf=np.finfo(np.float32).min/2)
    return inv.astype(np.float32)

# ---------------------------
# Model Definitions
# ---------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, dropout, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        last = in_dim
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(last, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
            self.bns.append(BatchNorm(hidden))
            last = hidden
        self.dropout = dropout
        self.out_dim = hidden

    def forward(self, x, edge_index, edge_attr):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class NodeFeatDecoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, h_nodes):
        return self.net(h_nodes)

class DenseBlock(nn.Module):
    def __init__(self, in_dim, growth, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, growth),
            nn.BatchNorm1d(growth),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class SEBlock(nn.Module):
    def __init__(self, dim, reduction=8):
        super().__init__()
        hidden = max(4, dim // reduction)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * w

class DenseSERegressor(nn.Module):
    """
    Consistent-width regressor that starts from the actual input dimension
    (latent + numeric features) instead of the growth value.
    """
    def __init__(self, in_dim, growth, n_blocks, dropout, head_hidden):
        super().__init__()
        layers = []

        # First block takes encoder latent size as input
        layers.append(nn.Sequential(
            nn.Linear(in_dim, growth),
            nn.BatchNorm1d(growth),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))

        # Subsequent blocks keep same width (growth)
        for _ in range(n_blocks - 1):
            layers.append(nn.Sequential(
                nn.Linear(growth, growth),
                nn.BatchNorm1d(growth),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))

        self.blocks = nn.Sequential(*layers)
        self.se = SEBlock(growth, reduction=8)
        self.head = nn.Sequential(
            nn.Linear(growth, head_hidden),
            nn.BatchNorm1d(head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def forward(self, z):
        h = self.blocks(z)
        h = self.se(h)
        return self.head(h)



class JointGraphAER(nn.Module):
    def __init__(self, node_in, edge_in, cfg, num_extra_feats=4):
        super().__init__()
        self.encoder = GNNEncoder(node_in, cfg["enc_hidden"], cfg["enc_layers"], cfg["enc_dropout"], edge_in)
        self.decoder = NodeFeatDecoder(cfg["enc_hidden"], node_in)
        self.num_extra_feats = num_extra_feats
        # 🔹 Regressor input = GNN latent + 4 numeric features
        self.regressor = DenseSERegressor(
            in_dim=cfg["enc_hidden"] + num_extra_feats,
            growth=cfg["growth"],
            n_blocks=cfg["n_blocks"],
            dropout=cfg["dropout"],
            head_hidden=cfg["head_hidden"]
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        h_nodes = self.encoder(x, edge_index, edge_attr)
        z_graph = global_mean_pool(h_nodes, batch)

        # 🔹 concatenate extra numeric features per graph ONLY if this model expects them
        if self.num_extra_feats > 0:
            if hasattr(data, "num_feats"):
                num_feats = data.num_feats.to(z_graph.device)
                if num_feats.dim() == 1:
                    num_feats = num_feats.unsqueeze(0)
                # ensure right width and batch match
                if z_graph.size(0) == num_feats.size(0):
                    if num_feats.size(1) > self.num_extra_feats:
                        num_feats = num_feats[:, :self.num_extra_feats]
                    elif num_feats.size(1) < self.num_extra_feats:
                        pad = torch.zeros(z_graph.size(0), self.num_extra_feats - num_feats.size(1), device=z_graph.device)
                        num_feats = torch.cat([num_feats, pad], dim=1)
                    z_graph = torch.cat([z_graph, num_feats], dim=1)
            else:
                # if we expect extras but they’re missing in the batch, pad zeros
                pad = torch.zeros(z_graph.size(0), self.num_extra_feats, device=z_graph.device)
                z_graph = torch.cat([z_graph, pad], dim=1)



        # Debug moved to training loop (once per epoch)



        y_hat = self.regressor(z_graph)
        x_logits = self.decoder(h_nodes)
        return y_hat, x_logits, h_nodes, z_graph


# ---------------------------
# Data Loaders / Metrics
# ---------------------------
def make_split_loaders(dataset, cfg, seed):
    n = len(dataset); n_val = max(1, int(cfg["val_split"] * n)); n_tr = max(1, n - n_val)
    gen = torch.Generator().manual_seed(seed)
    tr_ds, val_ds = random_split(dataset, [n_tr, n_val], generator=gen)
    bs_tr = min(cfg["batch_size"], len(tr_ds)); bs_tr = bs_tr if bs_tr > 0 else 1
    bs_va = min(cfg["batch_size"], len(val_ds)); bs_va = bs_va if bs_va > 0 else 1
    tr_loader = DataLoader(tr_ds, batch_size=bs_tr, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=bs_va, shuffle=False, drop_last=False)
    return tr_loader, val_loader

def eval_ensemble(models, loader, pt):
    yhat_all, ytrue_all = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            preds = []
            for m in models:
                y_hat, _, _, _ = m(data)
                preds.append(y_hat.squeeze(1).cpu().numpy())
            p_mean = np.mean(preds, axis=0)
            yhat_all.append(p_mean)
            ytrue_all.append(data.y.cpu().numpy().ravel())
    y_pred = np.concatenate(yhat_all)
    y_true = np.concatenate(ytrue_all)
    y_pred_inv = inverse_target(pt, y_pred)
    y_true_inv = inverse_target(pt, y_true)
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true_inv, y_pred_inv)
    return dict(mae=mae, mse=mse, rmse=rmse, r2=r2)

# ---------------------------
# Training Loops
# ---------------------------
def pretrain_encoder(dataset, cfg, bin_dim, cont_dim, epochs_override=None, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    sample = dataset.get(0)
    node_in = sample.x.shape[1]
    edge_in = sample.edge_attr.shape[1] if sample.edge_attr is not None and sample.edge_attr.numel()>0 else 0

    model = JointGraphAER(node_in, edge_in, cfg, num_extra_feats=0).to(device)


    params = list(model.encoder.parameters()) + list(model.decoder.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["ae_lr"], weight_decay=cfg["ae_weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=40, T_mult=1)
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()

    tr_loader, _ = make_split_loaders(dataset, cfg, seed)
    E = epochs_override if epochs_override is not None else cfg["ae_epochs"]
    for epoch in range(E):
        model.train()
        for data in tr_loader:
            data = data.to(device)
            opt.zero_grad()
            _, x_logits, _, _ = model(data)
            x_bin = data.x[:, :bin_dim]; x_cont = data.x[:, bin_dim:bin_dim+cont_dim]
            xl_bin = x_logits[:, :bin_dim]; xl_cont = x_logits[:, bin_dim:bin_dim+cont_dim]
            loss = bce(xl_bin, x_bin) + mse(xl_cont, x_cont)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step(epoch)

    return {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
            "node_in": node_in, "edge_in": edge_in}

def train_single_joint(dataset, pt, ae_init, cfg, seed_offset, bin_dim, cont_dim, epochs_override=None):
    seed = SEED + seed_offset
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    tr_loader, val_loader = make_split_loaders(dataset, cfg, seed)
    model = JointGraphAER(ae_init["node_in"], ae_init["edge_in"], cfg, num_extra_feats=4).to(device)

    model.encoder.load_state_dict(ae_init["encoder"])
    model.decoder.load_state_dict(ae_init["decoder"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg["T0"], T_mult=cfg["Tmult"])

    swa_start_epoch = int((1.0 - cfg["swa_ratio"]) * (epochs_override if epochs_override is not None else cfg["epochs"]))
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=cfg["lr"] * 0.1)

    sl1 = nn.SmoothL1Loss(); mse = nn.MSELoss(); bce = nn.BCEWithLogitsLoss()
    E = epochs_override if epochs_override is not None else cfg["epochs"]

    for epoch in range(E):
        model.train()
        
        # ✅ Print model dimension info only once at the start of training
        if epoch == 0:
            sample_data = next(iter(tr_loader)).to(device)
            with torch.no_grad():
                _, _, _, z_graph_dbg = model(sample_data)
            print(f"[DEBUG] Epoch {epoch} — z_graph: {z_graph_dbg.shape}, "
                f"Regressor input dim: {model.regressor.blocks[0][0].in_features}")

        for data in tr_loader:
            data = data.to(device)
            optimizer.zero_grad()
            y_hat, x_logits, _, _ = model(data)
            sup = cfg["alpha"] * sl1(y_hat, data.y.unsqueeze(1)) + (1.0 - cfg["alpha"]) * mse(y_hat, data.y.unsqueeze(1))
            x_bin = data.x[:, :bin_dim]; x_cont = data.x[:, bin_dim:bin_dim+cont_dim]
            xl_bin = x_logits[:, :bin_dim]; xl_cont = x_logits[:, bin_dim:bin_dim+cont_dim]
            recon = bce(xl_bin, x_bin) + mse(xl_cont, x_cont)

            loss = cfg["sup_weight"] * sup + cfg["recon_weight"] * recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch < swa_start_epoch:
            scheduler.step(epoch)
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    update_bn(tr_loader, swa_model, device=device)
    return swa_model, val_loader
from sklearn.model_selection import KFold

def run_cv_pipeline(cfg, n_splits=5):
    print(f"\n=== {n_splits}-Fold Cross-Validation ===")

    ds = GraphDataset(CSV_PATH, cfg, force_reprocess=cfg.get("force_reprocess", False))
    bin_dim, cont_dim = get_atom_feature_dims(cfg)

    # Fit global PowerTransformer on entire dataset targets once
    pt = fit_target_transform(ds)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    indices = np.arange(len(ds))
    metrics_list = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        tr_subset = ds[torch.tensor(train_idx)]
        val_subset = ds[torch.tensor(val_idx)]

        tr_loader = DataLoader(tr_subset, batch_size=cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=cfg["batch_size"], shuffle=False)

        # Pretrain encoder on train subset
        ae_init = pretrain_encoder(tr_subset, cfg, bin_dim, cont_dim, epochs_override=cfg["ae_epochs"], seed=SEED+fold)

        # Train joint model on this fold
        model, _ = train_single_joint(tr_subset, pt, ae_init, cfg, seed_offset=fold*11,
                                      bin_dim=bin_dim, cont_dim=cont_dim, epochs_override=cfg["epochs"])

        # Evaluate on val fold
        metrics = eval_ensemble([model], val_loader, pt)
        metrics_list.append(metrics)

        print(f"[Fold {fold}] R²={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")

    # Aggregate results
    r2s = [m["r2"] for m in metrics_list]
    maes = [m["mae"] for m in metrics_list]
    rmses = [m["rmse"] for m in metrics_list]

    summary = {
        "R2_mean": np.mean(r2s),
        "R2_std": np.std(r2s),
        "MAE_mean": np.mean(maes),
        "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses),
        "RMSE_std": np.std(rmses)
    }

    print("\n=== CV Summary ===")
    print(f"R²:   {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}")
    print(f"MAE:  {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f}")
    print(f"RMSE: {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f}")

    return summary, metrics_list

# ---------------------------
# HPO: R²-first Random Search
# ---------------------------
def sample_cfg(base):
    cfg = dict(base)

    # Increase search space around capacity / regularization
    cfg["enc_hidden"]   = random.choice([128, 160, 192, 224, 256])
    cfg["enc_layers"]   = random.choice([3, 4, 5])
    cfg["enc_dropout"]  = random.choice([0.05, 0.08, 0.10, 0.12, 0.15])

    cfg["growth"]       = random.choice([192, 224, 256, 320])
    cfg["n_blocks"]     = random.choice([3, 4, 5])
    cfg["dropout"]      = random.choice([0.10, 0.15, 0.20, 0.25])
    cfg["head_hidden"]  = random.choice([256, 320, 384, 512])

    cfg["lr"]           = random.choice([7e-4, 9e-4, 1e-3, 1.2e-3])
    cfg["weight_decay"] = random.choice([1e-4, 5e-4, 9e-4, 1.2e-3])
    cfg["batch_size"]   = random.choice([24, 32, 48])

    cfg["T0"]           = random.choice([70, 100, 120, 150])
    cfg["Tmult"]        = random.choice([1, 2])
    cfg["swa_ratio"]    = random.choice([0.25, 0.30, 0.33, 0.40])

    cfg["alpha"]        = random.choice([0.60, 0.65, 0.70, 0.80])
    sup_w               = random.choice([0.90, 0.92, 0.95])
    cfg["sup_weight"]   = sup_w
    cfg["recon_weight"] = 1.0 - sup_w

    cfg["ae_epochs"]    = random.choice([80, 100, 120])
    cfg["ae_feat_drop"] = random.choice([0.05, 0.10, 0.15])

    return cfg


def quick_eval(cfg, dataset, bin_dim, cont_dim, hpo_ae_epochs=80, hpo_joint_epochs=220, seed_offset=0):
    # Fit target transform fresh for each trial (keeps consistency per split)
    pt = fit_target_transform(dataset)

    # AE pretrain (short)
    ae_init = pretrain_encoder(dataset, cfg, bin_dim, cont_dim,
                           epochs_override=hpo_ae_epochs, seed=SEED + seed_offset)

    model, val_loader = train_single_joint(dataset, pt, ae_init, cfg,
                                       seed_offset=seed_offset,
                                       bin_dim=bin_dim,
                                       cont_dim=cont_dim,
                                       epochs_override=hpo_joint_epochs)

    # Score on validation
    metrics = eval_ensemble([model], val_loader, pt)
    return metrics, model

def hpo_random_search(base_cfg, n_trials=20, hpo_ae_epochs=80, hpo_joint_epochs=220):
    # Prepare dataset once for HPO
    ds = GraphDataset(CSV_PATH, base_cfg, force_reprocess=base_cfg.get("force_reprocess", False))
    bin_dim, cont_dim = get_atom_feature_dims(base_cfg)

    logs = []
    best_cfg = None
    best_met = None

    for t in range(n_trials):
        cfg = sample_cfg(base_cfg)
        metrics, _ = quick_eval(cfg, ds, bin_dim, cont_dim,
                                hpo_ae_epochs=hpo_ae_epochs,
                                hpo_joint_epochs=hpo_joint_epochs,
                                seed_offset=t*97)

        logs.append({"trial": t, "cfg": cfg, "metrics": metrics})
        print(f"[HPO {t:02d}] R2={metrics['r2']:.4f}  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}")

        # R²-first selection, tie-break by MAE
        if (best_met is None) or (metrics["r2"] > best_met["r2"] + 1e-6) or \
           (abs(metrics["r2"] - best_met["r2"]) < 1e-6 and metrics["mae"] < best_met["mae"]):
            best_cfg = cfg
            best_met = metrics

    with open(HPO_LOG_PATH, "w") as f:
        json.dump({"logs": logs, "best_cfg": best_cfg, "best_metrics": best_met}, f, indent=2)

    print(f"[HPO] Best R2={best_met['r2']:.4f}, MAE={best_met['mae']:.4f}, RMSE={best_met['rmse']:.4f}")
    return best_cfg, best_met


# ---------------------------
# Final Training (Full) + Ensemble
# ---------------------------
def run_pipeline(cfg):
    print("Loading dataset...")
    ds = GraphDataset(CSV_PATH, cfg, force_reprocess=cfg.get("force_reprocess", False))
    pt = fit_target_transform(ds)
    bin_dim, cont_dim = get_atom_feature_dims(cfg)

    print("Pretraining Graph Encoder (denoising recon)...")
    ae_init = pretrain_encoder(ds, cfg, bin_dim, cont_dim, epochs_override=cfg["ae_epochs"], seed=SEED)

    print("Joint fine-tuning + Ensemble...")
    models = []
    val_loader_ref = None
    for i in range(cfg["ensemble_size"]):
        m, v = train_single_joint(ds, pt, ae_init, cfg, seed_offset=i*123, bin_dim=bin_dim, cont_dim=cont_dim)
        models.append(m)
        val_loader_ref = v

    metrics = eval_ensemble(models, val_loader_ref, pt)
    return metrics, models, ae_init, pt

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    print("=== GRAPH JOINT AE + REGRESSOR + ENSEMBLE (with HPO) ===")
    print("Base config:")
    for k, v in BASE_CFG.items():
        print(f"  {k}: {v}")

    # ---------------------------
    # Hyperparameter Tuning (R²-first)
    # ---------------------------
    print("\n=== HPO: Random Search ===")
    best_cfg, best_hpo_metrics = hpo_random_search(
        BASE_CFG,
        n_trials=20,          # increase for deeper search
        hpo_ae_epochs=80,     # short AE pretrain during HPO
        hpo_joint_epochs=220  # short joint train during HPO
    )

    # Merge best cfg with full training lengths and ensemble size
    FINAL_CFG = dict(BASE_CFG)
    FINAL_CFG.update(best_cfg)
    FINAL_CFG["ae_epochs"] = max(120, FINAL_CFG.get("ae_epochs", BASE_CFG["ae_epochs"]))  # ensure decent pretrain
    FINAL_CFG["epochs"] = max(600, FINAL_CFG.get("epochs", BASE_CFG["epochs"]))           # full train
    FINAL_CFG["ensemble_size"] = BASE_CFG["ensemble_size"]                                # full ensemble

    print("\n=== Final Training with Best Config (Full) ===")
    for k, v in FINAL_CFG.items():
        print(f"  {k}: {v}")
    print("\n=== 5-Fold Cross-Validation with Best Config ===")
    cv_summary, fold_metrics = run_cv_pipeline(FINAL_CFG, n_splits=5)

    metrics, models, ae_init, pt = run_pipeline(FINAL_CFG)

    print("\nFinal Ensemble Metrics (Validation):")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MSE:  {metrics['mse']:.6f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    torch.save({
        "config": FINAL_CFG,
        "best_hpo_metrics": best_hpo_metrics,
        "target_transform_lambdas": getattr(pt, "lambdas_", None),
        "ae_encoder_state": ae_init["encoder"],
        "ae_decoder_state": ae_init["decoder"],
        "joint_states": [m.state_dict() for m in models],
    }, ARTIFACT_PATH)

    print(f"\nSaved artifacts: {ARTIFACT_PATH}")
    print(f"HPO log: {HPO_LOG_PATH}")
