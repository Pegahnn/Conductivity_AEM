# GNN_optuna_cv.py
import argparse, os, csv, json, numpy as np, optuna, torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from NNgraph_2 import GCNReg_add, GCNReg, GATReg_add
from createGraph_edited import graph_dataset, collates, collate_add
import random

# Verbose logging so we see errors
optuna.logging.set_verbosity(optuna.logging.DEBUG)


def _use_gpu(args):
    return args.gpu >= 0 and torch.cuda.is_available()


def objective_with_cv(trial, args, smlstr_train, logCMC_train, results_file, y_scaler):
    # ---- hyperparams ----
    lr           = trial.suggest_float("lr", 1e-5, 3e-2, log=True)
    batch_size   = trial.suggest_categorical("batch_size", [4, 8, 16, 32])
    hidden_dim   = trial.suggest_categorical("unit_per_layer", [128, 256, 384, 512])
    num_heads    = trial.suggest_categorical("num_heads", [2, 4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    num_layers   = trial.suggest_int("num_layers", 2, 5)
    dropout      = trial.suggest_float("dropout", 0.0, 0.5)   # ✅ added

    # ---- dataset & folds ----
    dataset = graph_dataset(
        smlstr_train, logCMC_train,
        add_features=args.add_features,
        rdkit_descriptor=args.rdkit_descriptor
    )
    print(f"[Trial {trial.number}] Train samples: {len(dataset)}", flush=True)

    kf = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    cv_r2_scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(smlstr_train), start=1):
        y_val_fold = logCMC_train[va_idx]
        print(f"[Trial {trial.number} | Fold {fold}] Var(y_val)={np.var(y_val_fold):.6f}, Mean={np.mean(y_val_fold):.6f}")

        train_ds = torch.utils.data.Subset(dataset, tr_idx)
        val_ds   = torch.utils.data.Subset(dataset, va_idx)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_add if args.gnn_model.endswith("_add") else collates, num_workers=0
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_add if args.gnn_model.endswith("_add") else collates, num_workers=0
        )

        # ---- build model ----
        if args.gnn_model == "GCNReg_add":
            model = GCNReg_add(args.dim_input, args.num_feat, hidden_dim, 1,
                               False, num_layers=num_layers, dropout=dropout)
        elif args.gnn_model == "GATReg_add":
            model = GATReg_add(args.dim_input, args.num_feat, hidden_dim, 1,
                               num_heads=num_heads, saliency=False,
                               num_layers=num_layers, dropout=dropout)
        elif args.gnn_model == "GCNReg":
            model = GCNReg(args.dim_input, hidden_dim, 1,
                           False, num_layers=num_layers, dropout=dropout)
        else:
            raise ValueError(f"Unknown gnn_model {args.gnn_model}")

        if _use_gpu(args):
            model = model.cuda(args.gpu)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        loss_fn = nn.HuberLoss(delta=1.0)

        best_val_loss, best_val_r2 = float("inf"), -999.0
        patience, patience_counter = 30, 0

        try:
            for epoch in range(args.epochs):
                # ---- train ----
                model.train()
                for batch in train_loader:
                    optimizer.zero_grad()
                    if args.gnn_model.endswith("_add"):
                        g, desc, y = batch
                        if _use_gpu(args):
                            g, desc, y = g.to(f"cuda:{args.gpu}"), desc.cuda(args.gpu), y.cuda(args.gpu)
                        pred = model(g, desc)
                    else:
                        g, y = batch
                        if _use_gpu(args):
                            g, y = g.to(f"cuda:{args.gpu}"), y.cuda(args.gpu)
                        pred = model(g)
                    loss = loss_fn(pred, y.float())
                    loss.backward()
                    optimizer.step()

                # ---- validate ----
                model.eval()
                val_losses, preds, trues = [], [], []
                with torch.no_grad():
                    for batch in val_loader:
                        if args.gnn_model.endswith("_add"):
                            g, desc, y = batch
                            if _use_gpu(args):
                                g, desc, y = g.to(f"cuda:{args.gpu}"), desc.cuda(args.gpu), y.cuda(args.gpu)
                            pred = model(g, desc)
                        else:
                            g, y = batch
                            if _use_gpu(args):
                                g, y = g.to(f"cuda:{args.gpu}"), y.cuda(args.gpu)
                            pred = model(g)

                        val_losses.append(loss_fn(pred, y.float()).item())
                        preds.append(pred.detach().cpu().numpy())
                        trues.append(y.detach().cpu().numpy())

                mean_val_loss = float(np.mean(val_losses)) if val_losses else np.inf
                scheduler.step(mean_val_loss)

                preds = np.concatenate(preds).ravel() if preds else np.array([])
                trues = np.concatenate(trues).ravel() if trues else np.array([])
                if preds.size and trues.size:
                    preds = y_scaler.inverse_transform(preds.reshape(-1,1)).ravel()
                    trues = y_scaler.inverse_transform(trues.reshape(-1,1)).ravel()
                    val_r2 = r2_score(trues, preds)
                else:
                    val_r2 = -999.0

                print(f"[T{trial.number} F{fold}] Epoch {epoch:03d} | Val Loss: {mean_val_loss:.6f} | Val R²: {val_r2:.6f}")

                if mean_val_loss < best_val_loss:
                    best_val_loss, best_val_r2 = mean_val_loss, val_r2
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"[T{trial.number} F{fold}] Early stopping at epoch {epoch}")
                        break

            cv_r2_scores.append(best_val_r2)

        except Exception as e:
            import traceback
            print(f"[T{trial.number} F{fold}] ERROR:\n{traceback.format_exc()}", flush=True)
            cv_r2_scores.append(-999.0)

    mean_r2 = float(np.mean(cv_r2_scores)) if cv_r2_scores else -999.0
    print(f"\n✅ Trial {trial.number} | Mean CV R² across {args.cv} folds: {mean_r2:.6f}")

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([trial.number, mean_r2, lr, batch_size, hidden_dim, weight_decay, num_heads, num_layers, dropout])

    return mean_r2


def optuna_cv_runner(args, smlstr, logCMC, smlstr_test, logCMC_test, y_scaler):
    os.makedirs(args.save_dir, exist_ok=True)
    results_file = os.path.join(args.save_dir, "trials_results.csv")
    if not os.path.exists(results_file):
        with open(results_file, "w", newline="") as f:
            csv.writer(f).writerow(["trial_number", "cv_r2", "lr", "batch_size", "unit_per_layer",
                                    "weight_decay", "num_heads", "num_layers", "dropout"])

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda tr: objective_with_cv(tr, args, smlstr, logCMC, results_file, y_scaler),
                   n_trials=args.n_trials)

    with open(os.path.join(args.save_dir, "best_params.json"), "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)

    df = study.trials_dataframe()
    df.to_csv(os.path.join(args.save_dir, "all_trials.csv"), index=False)

    print("Best hyperparameters:", study.best_params, flush=True)
    print("Best CV R²:", study.best_value, flush=True)

    # ---- retrain on full train set with best params ----
    best = study.best_params
    batch_size   = best["batch_size"]
    hidden_dim   = best["unit_per_layer"]
    lr           = best["lr"]
    weight_decay = best["weight_decay"]
    num_heads    = best["num_heads"]
    num_layers   = best["num_layers"]
    dropout      = best["dropout"]

    if args.gnn_model == "GCNReg_add":
        model, collate_fn = GCNReg_add(args.dim_input, args.num_feat, hidden_dim, 1,
                                       False, num_layers=num_layers, dropout=dropout), collate_add
    elif args.gnn_model == "GATReg_add":
        model, collate_fn = GATReg_add(args.dim_input, args.num_feat, hidden_dim, 1,
                                       num_heads=num_heads, saliency=False,
                                       num_layers=num_layers, dropout=dropout), collate_add
    else:
        model, collate_fn = GCNReg(args.dim_input, hidden_dim, 1,
                                   False, num_layers=num_layers, dropout=dropout), collates

    if _use_gpu(args):
        model = model.cuda(args.gpu)

    train_dataset = graph_dataset(smlstr, logCMC, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=collate_fn, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    print("=== Retraining Best Model on Full Training Data ===", flush=True)
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            if args.gnn_model.endswith("_add"):
                g, desc, y = batch
                if _use_gpu(args):
                    g, desc, y = g.to(f"cuda:{args.gpu}"), desc.cuda(args.gpu), y.cuda(args.gpu)
                pred = model(g, desc)
            else:
                g, y = batch
                if _use_gpu(args):
                    g, y = g.to(f"cuda:{args.gpu}"), y.cuda(args.gpu)
                pred = model(g)
            loss = loss_fn(pred, y.float())
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {np.mean(epoch_losses):.6f}", flush=True)

    torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    # ---- Test ----
    test_dataset = graph_dataset(smlstr_test, logCMC_test, add_features=args.add_features, rdkit_descriptor=args.rdkit_descriptor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                              collate_fn=collate_fn, num_workers=0)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if args.gnn_model.endswith("_add"):
                g, desc, y = batch
                if _use_gpu(args):
                    g, desc, y = g.to(f"cuda:{args.gpu}"), desc.cuda(args.gpu), y.cuda(args.gpu)
                pred = model(g, desc)
            else:
                g, y = batch
                if _use_gpu(args):
                    g, y = g.to(f"cuda:{args.gpu}"), y.cuda(args.gpu)
                pred = model(g)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_scaler.inverse_transform(y_true.reshape(-1,1)).ravel()
    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).ravel()

    print("=== Final Test Evaluation ===", flush=True)
    print("Test R²:", r2_score(y_true, y_pred), flush=True)
    print("Test MAE:", mean_absolute_error(y_true, y_pred), flush=True)
    print("Test RMSE:", mean_squared_error(y_true, y_pred, squared=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./results")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--dim_input", type=int, default=74)
    parser.add_argument("--num_feat", type=int, default=4)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gnn_model", type=str, default="GCNReg_add")
    parser.add_argument("--add_features", action="store_true")
    parser.add_argument("--rdkit_descriptor", action="store_true")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset
    smlstr, logCMC = [], []
    with open(args.data_path) as f:
        reader = csv.reader(f)
        for row in reader:
            smlstr.append(row[0:-1])
            logCMC.append(row[-1])
    smlstr = np.asarray(smlstr)
    logCMC = np.asarray(logCMC, dtype=float)

    # Split train/test
    tr_idx, te_idx = train_test_split(np.arange(len(logCMC)), test_size=args.test_size, random_state=args.seed)
    smlstr_train, smlstr_test = smlstr[tr_idx], smlstr[te_idx]
    logCMC_train, logCMC_test = logCMC[tr_idx], logCMC[te_idx]

    from sklearn.preprocessing import StandardScaler
    y_scaler = StandardScaler().fit(logCMC_train.reshape(-1, 1))
    logCMC_train_scaled = y_scaler.transform(logCMC_train.reshape(-1, 1)).ravel()
    logCMC_test_scaled  = y_scaler.transform(logCMC_test.reshape(-1, 1)).ravel()

    print(f"Train size: {len(smlstr_train)} | Test size: {len(smlstr_test)}", flush=True)

    optuna_cv_runner(args, smlstr_train, logCMC_train_scaled, smlstr_test, logCMC_test_scaled, y_scaler)
