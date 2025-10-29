import argparse
import os
import pickle
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from datetime import datetime


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_traj(dataset: str, label_source: str = "actions") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/test trajectories and construct features/labels.

    X_t = [delta_close_t] from traj['observations'] (only delta feature)
    Y_t = argmax(one_hot_action_t) from traj['actions'] (classes: 0,1,2)
    """
    base = os.path.join("trajectory")
    train_pkl = os.path.join(base, f"{dataset}_train_traj.pkl")
    test_pkl = os.path.join(base, f"{dataset}_test_traj.pkl")

    if not os.path.exists(train_pkl) or not os.path.exists(test_pkl):
        raise FileNotFoundError(
            f"Missing trajectory files: {train_pkl} or {test_pkl}. Run create_data.py first."
        )

    with open(train_pkl, "rb") as f:
        train_paths = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_paths = pickle.load(f)

    # We use the first path (list of length 1 by default)
    train_traj = train_paths[0]
    test_traj = test_paths[0]

    X_train = np.asarray(train_traj["observations"], dtype=np.float32)
    Y_train_oh = np.asarray(train_traj["actions"], dtype=np.float32)
    X_test = np.asarray(test_traj["observations"], dtype=np.float32)
    Y_test_oh = np.asarray(test_traj["actions"], dtype=np.float32)

    # Always use only delta_close feature (drop previous position)
    X_train = X_train[:, 0:1]
    X_test = X_test[:, 0:1]

    if label_source == "actions":
        # Convert one-hot labels to class indices for CrossEntropyLoss
        y_train = np.argmax(Y_train_oh, axis=1).astype(np.int64)
        y_test = np.argmax(Y_test_oh, axis=1).astype(np.int64)
    elif label_source == "positions":
        # Use positions -1/0/1 and map to classes 0/1/2 via +1
        pos_train = np.asarray(train_traj["positions"], dtype=np.int64)
        pos_test = np.asarray(test_traj["positions"], dtype=np.int64)
        if not np.isin(pos_train, [-1, 0, 1]).all() or not np.isin(pos_test, [-1, 0, 1]).all():
            raise ValueError("positions must be in {-1,0,1} to map to class indices")
        y_train = (pos_train + 1).astype(np.int64)
        y_test = (pos_test + 1).astype(np.int64)
    else:
        raise ValueError(f"Unknown label_source: {label_source}")

    # Safety: align lengths if any mismatch
    n_train = min(len(X_train), len(y_train))
    n_test = min(len(X_test), len(y_test))
    X_train, y_train = X_train[:n_train], y_train[:n_train]
    X_test, y_test = X_test[:n_test], y_test[:n_test]

    # Expect input_dim=1 (delta only)
    assert X_train.shape[1] == 1 and X_test.shape[1] == 1, (
        f"Expected input_dim=1, got {X_train.shape[1]} and {X_test.shape[1]}"
    )

    return X_train, y_train, X_test, y_test


def load_raw_traj_dicts(dataset: str):
    base = os.path.join("trajectory")
    train_pkl = os.path.join(base, f"{dataset}_train_traj.pkl")
    test_pkl = os.path.join(base, f"{dataset}_test_traj.pkl")
    with open(train_pkl, "rb") as f:
        train_paths = pickle.load(f)
    with open(test_pkl, "rb") as f:
        test_paths = pickle.load(f)
    return train_paths[0], test_paths[0]

class Logit(nn.Module):
    def __init__(self, input_dim: int = 2, num_classes: int = 3) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def accuracy(pred_logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = pred_logits.argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / max(1, y.numel())


def macro_f1(y_true, y_pred):
    """Compute macro F1; if sklearn is unavailable, fall back to numpy implementation."""
    try:
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred, average="macro")
    except Exception:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classes = np.unique(y_true)
        f1s = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        if len(f1s) == 0:
            return float("nan")
        return float(np.mean(f1s))

def transfer_accuracy(y_true, y_pred):
    # 仅统计状态变化日
    idx = np.where(y_true[1:] != y_true[:-1])[0] + 1
    if len(idx) == 0: return np.nan
    return np.mean(y_pred[idx] == y_true[idx])


def threshold_predict(delta: np.ndarray, m: float = 0.001, mode: str = "sign") -> np.ndarray:
    """
    Baseline threshold predictor using only delta_close.
    - mode == 'sign':       cls_raw = sign(delta - m) ∈ {-1,0,1}
    - mode == 'band':       three-zone: [-inf,-m)->-1, [-m,m]->0, (m,inf)->1
    Returns class indices in {0,1,2} via mapping (-1,0,1) -> (0,1,2)
    """
    delta = np.asarray(delta, dtype=np.float32)
    if mode == "sign":
        cls = np.sign(delta - m)
    elif mode == "band":
        cls = np.zeros_like(delta)
        cls[delta > m] = 1
        cls[delta < -m] = -1
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")
    return (cls.astype(np.int64) + 1)


def class_distribution(y: np.ndarray, num_classes: int = 3):
    y = np.asarray(y)
    counts = np.bincount(y, minlength=num_classes)
    ratios = counts / max(1, y.size)
    return counts, ratios


def confusion_matrix_safe(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    try:
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    except Exception:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        return cm


def load_split_close_series(dataset: str, split: str = "test") -> np.ndarray:
    """Load close price series for a split and return as 1-D float array.

    split: "train" -> datasets/{dataset}_train.csv, "test" -> datasets/{dataset}_trade.csv
    The series is expected to be single-ticker; values are ordered as in the CSV.
    """
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    csv_name = f"{dataset}_{'train' if split=='train' else 'trade'}.csv"
    csv_path = os.path.join("datasets", csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing data file: {csv_path}")
    df = pd.read_csv(csv_path)
    if "close" not in df.columns:
        raise ValueError(f"'close' column not found in {csv_path}")
    # Keep single ticker if multiple exist (current workflow uses single ticker like 000300.SH)
    if "tic" in df.columns:
        tics = df["tic"].unique().tolist()
        if len(tics) > 1:
            # Pick the first ticker deterministically
            df = df[df["tic"] == tics[0]].reset_index(drop=True)
    # Ensure chronological order if date column exists
    if "date" in df.columns:
        try:
            df["_date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values(["_date_parsed"]).reset_index(drop=True)
            df = df.drop(columns=["_date_parsed"])  # cleanup
        except Exception:
            pass
    close = df["close"].astype(float).to_numpy()
    return close


def evaluate_trading_from_predictions(
    y_pred_classes: np.ndarray,
    close: np.ndarray,
    transaction_cost: float = 0.001,
    initial_amount: float = 1_000_000.0,
) -> dict:
    """Compute compounded PnL given predicted action classes and close prices.

    - y_pred_classes: class indices in {0,1,2} mapping to positions {-1,0,1} via (cls-1)
    - close: price series length >= len(y_pred_classes)+1 (needs next-day price)
    - transaction_cost: fee per unit position change (abs delta in position)
    - Returns dict with trade_count, total_return, final_amount, total_fee
    """
    y_pred_classes = np.asarray(y_pred_classes, dtype=np.int64)
    close = np.asarray(close, dtype=np.float64)
    if close.size < y_pred_classes.size + 1:
        # Align to the minimal feasible length
        max_T = max(0, min(y_pred_classes.size, close.size - 1))
        y_pred_classes = y_pred_classes[:max_T]
    positions = (y_pred_classes - 1).astype(np.int64)  # -1/0/1

    amount = float(initial_amount)
    amount_nofee = float(initial_amount)
    last_pos = 0
    trade_events = 0
    total_fee = 0.0

    for t in range(positions.size):
        pos = int(positions[t])
        # price return from t->t+1
        r = (close[t + 1] / close[t]) - 1.0
        # fee proportional to absolute change in position
        flag = abs(pos - last_pos)
        fee = flag * transaction_cost
        # apply returns
        step_ret_with_fee = (r * pos) - fee
        step_ret_nofee = (r * pos)
        amount *= (1.0 + step_ret_with_fee)
        amount_nofee *= (1.0 + step_ret_nofee)
        total_fee += amount_nofee * 0.0  # not tracking dynamic cash; fee already deducted in returns
        if flag > 0:
            trade_events += 1
        last_pos = pos

    total_return = (amount - initial_amount) / initial_amount
    return {
        "trade_count": int(trade_events),
        "total_return": float(total_return),
        "final_amount": float(amount),
        "total_fee": float(initial_amount * ((amount_nofee / initial_amount) - (amount / initial_amount))),
    }


def evaluate_model_on_test(
    model: nn.Module,
    dataset: str,
    batch_size: int = 256,
    label_source: str = "actions",
) -> dict:
    """Run inference on Test split and print trading metrics. Returns metrics dict."""
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    # Load test features/labels
    _, _, X_test, y_test = load_traj(dataset, label_source=label_source)
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    with torch.no_grad():
        all_y_pred = []
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            all_y_pred.append(logits.argmax(dim=1).cpu().numpy())
    y_pred_np = np.concatenate(all_y_pred, axis=0) if len(all_y_pred) > 0 else np.array([], dtype=np.int64)

    close_test = load_split_close_series(dataset, split="test")
    trade_metrics = evaluate_trading_from_predictions(y_pred_np, close_test, transaction_cost=0.001)
    print("Test Trading:")
    print(f"  trade_count  : {trade_metrics['trade_count']}")
    print(f"  total_return : {trade_metrics['total_return']:.4f}")
    print(f"  final_amount : {trade_metrics['final_amount']:.2f}")
    print(f"  total_fee    : {trade_metrics['total_fee']:.2f}")

    return trade_metrics


def load_saved_model(model_path: str, device: torch.device) -> Tuple[nn.Module, dict]:
    """Load a saved Logit model checkpoint and rebuild the model."""
    ckpt = torch.load(model_path, map_location=device)
    input_dim = int(ckpt.get("input_dim", 1))
    num_classes = int(ckpt.get("num_classes", 3))
    model = Logit(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def find_latest_model(dataset: str, search_dir: str = "results") -> str:
    """Find latest saved model for a dataset in search_dir."""
    if not os.path.isdir(search_dir):
        return ""
    candidates = []
    for name in os.listdir(search_dir):
        if name.startswith(f"logit_{dataset}_") and name.endswith(".pt"):
            path = os.path.join(search_dir, name)
            try:
                mtime = os.path.getmtime(path)
            except Exception:
                mtime = 0
            candidates.append((mtime, path))
    if not candidates:
        return ""
    candidates.sort()
    return candidates[-1][1]


def train_and_eval(
    dataset: str = "csi",
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-2,
    l2: float = 1e-4,
    seed: int = 42,
    label_source: str = "actions",
) -> None:
    set_seed(seed)

    X_train, y_train, X_test, y_test = load_traj(dataset, label_source=label_source)

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = X_train.shape[1]
    print(f"Input dim: {input_dim}")
    model = Logit(input_dim=input_dim, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            total_acc += accuracy(logits.detach(), yb) * bs
            total_count += bs

        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_acc / max(1, total_count)

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_acc = 0.0
            test_count = 0
            all_y_true = []
            all_y_pred = []
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                bs = yb.size(0)
                test_loss += loss.item() * bs
                test_acc += accuracy(logits, yb) * bs
                test_count += bs
                all_y_true.append(yb.cpu().numpy())
                all_y_pred.append(logits.argmax(dim=1).cpu().numpy())

        y_true_np = np.concatenate(all_y_true, axis=0) if len(all_y_true) > 0 else np.array([], dtype=np.int64)
        y_pred_np = np.concatenate(all_y_pred, axis=0) if len(all_y_pred) > 0 else np.array([], dtype=np.int64)
        mf1 = macro_f1(y_true_np, y_pred_np) if y_true_np.size > 0 else float("nan")
        ta = transfer_accuracy(y_true_np, y_pred_np) if y_true_np.size > 1 else float("nan")

        print(
            f"Epoch {epoch:03d} | Train Loss {avg_loss:.4f} Acc {avg_acc:.4f} | "
            f"Test Loss {test_loss/max(1,test_count):.4f} Acc {test_acc/max(1,test_count):.4f} "
            f"MacroF1 {mf1:.4f} TransferAcc {ta:.4f}"
        )

    # === After training, compute final Test predictions and evaluate trading metrics ===
    model.eval()
    with torch.no_grad():
        all_y_true = []
        all_y_pred = []
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            all_y_true.append(yb.cpu().numpy())
            all_y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true_np = np.concatenate(all_y_true, axis=0) if len(all_y_true) > 0 else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(all_y_pred, axis=0) if len(all_y_pred) > 0 else np.array([], dtype=np.int64)

    # Load Test split close series and compute trading metrics from predictions
    close_test = load_split_close_series(dataset, split="test")
    print(f"Test split close series length: {len(close_test)}")
    print(f"Test split y_true length: {len(y_true_np)}")
    print(f"Test split y_pred length: {len(y_pred_np)}")
    trade_metrics = evaluate_trading_from_predictions(y_pred_np, close_test, transaction_cost=0.001)
    print("Test Trading:")
    print(f"  trade_count  : {trade_metrics['trade_count']}")
    print(f"  total_return : {trade_metrics['total_return']:.4f}")
    print(f"  final_amount : {trade_metrics['final_amount']:.2f}")
    print(f"  total_fee    : {trade_metrics['total_fee']:.2f}")

    # Save trained model
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join("results", f"logit_{dataset}_{timestamp}.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": 3,
            "dataset": dataset,
            "created_at": timestamp,
            "epochs": epochs,
            "lr": lr,
            "l2": l2,
            "seed": seed,
        },
        model_path,
    )
    print(f"Saved model to: {model_path}")

    return model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="csi", help="Dataset key used in trajectory file names")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 weight decay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-source", type=str, default="actions", choices=["actions", "positions"],
                        help="Choose labels from one-hot actions or positions (-1/0/1 -> 0/1/2)")
    # Removed prev position feature entirely; input now uses only delta_close
    parser.add_argument("--baseline", action="store_true", help="Run threshold baseline instead of training")
    parser.add_argument("--baseline-mode", type=str, default="band", choices=["sign", "band"], help="Threshold mode")
    parser.add_argument("--m", type=float, default=0.001, help="Threshold m for delta_close")
    parser.add_argument("--baseline-split", type=str, default="test", choices=["train", "test", "both"],
                        help="Which split to evaluate for baseline")
    parser.add_argument("--debug-transfers", type=int, default=0,
                        help="Print first N transfer events with deltas and preds (baseline mode)")
    parser.add_argument("--test-only", action="store_true", help="Only run test evaluation using a saved model")
    parser.add_argument("--model-path", type=str, default="", help="Path to saved model .pt; if empty, auto-pick latest")
    args = parser.parse_args()

    if args.baseline: 
        # Load data (only need X and y for evaluation)
        X_train, y_train, X_test, y_test = load_traj(
            args.dataset, label_source=args.label_source
        )
        # Use only delta feature for baseline
        delta_train = X_train[:, 0]
        delta_test = X_test[:, 0]

        def eval_split(name: str, delta: np.ndarray, y_true: np.ndarray) -> None:
            """
            Evaluate the threshold baseline on a specified split.

            Steps:
            - Generate predictions from delta via the chosen threshold mode (sign/band).
            - Compute metrics: overall accuracy, macro-F1, and transfer accuracy
              (accuracy restricted to position-change days).
            - Compute and print class distributions for true and predicted labels
              (counts and ratios) and the confusion matrix (rows=true, cols=pred).

            Args:
                name: Split name to display in logs (e.g., "Train" or "Test").
                delta: 1-D array of delta_close values for the split.
                y_true: 1-D array of integer class labels (0/1/2) for the split.
            """
            y_pred = threshold_predict(delta, m=args.m, mode=args.baseline_mode)
            acc = float(np.mean(y_pred == y_true)) if y_true.size > 0 else float("nan")
            mf1 = macro_f1(y_true, y_pred)
            ta = transfer_accuracy(y_true, y_pred)
            # distributions
            t_cnt, t_rat = class_distribution(y_true)
            p_cnt, p_rat = class_distribution(y_pred)
            cm = confusion_matrix_safe(y_true, y_pred)
            print(
                f"Baseline[{args.baseline_mode}] m={args.m} | {name} Acc {acc:.4f} MacroF1 {mf1:.4f} TransferAcc {ta:.4f}"
            )
            print(f"{name} True dist counts={t_cnt.tolist()} ratios={[float(x) for x in t_rat]}")
            print(f"{name} Pred dist counts={p_cnt.tolist()} ratios={[float(x) for x in p_rat]}")
            print(f"{name} Confusion Matrix (rows=true, cols=pred):\n{cm}")

            # Trading metrics on corresponding split
            split_key = "train" if name.lower().startswith("train") else "test"
            close_series = load_split_close_series(args.dataset, split=split_key)
            trade_metrics = evaluate_trading_from_predictions(y_pred, close_series, transaction_cost=0.001)
            print(
                f"Baseline[{args.baseline_mode}] {name} Trading | trades={trade_metrics['trade_count']} "
                f"total_return={trade_metrics['total_return']:.4f} final_amount={trade_metrics['final_amount']:.2f} "
                f"total_fee={trade_metrics['total_fee']:.2f}"
            )

        if args.baseline_split in ("train", "both"):
            eval_split("Train", delta_train, y_train)
        if args.baseline_split in ("test", "both"):
            eval_split("Test", delta_test, y_test)

        # Optional transfer debug printouts using true positions
        if args.debug_transfers > 0:
            train_traj, test_traj = load_raw_traj_dicts(args.dataset)

            def debug_split(name: str, delta: np.ndarray, traj: dict, limit: int):
                pos = np.asarray(traj["positions"], dtype=np.int64)  # -1/0/1
                y_true_cls = pos + 1  # 0/1/2
                y_pred_cls = threshold_predict(delta, m=args.m, mode=args.baseline_mode)
                idx = np.where(y_true_cls[1:] != y_true_cls[:-1])[0] + 1
                print(f"DEBUG transfers [{name}] total={len(idx)} (show up to {limit})")
                for t in idx[:limit]:
                    pred_pos = int(y_pred_cls[t]) - 1
                    hit = (y_pred_cls[t] == y_true_cls[t])
                    d_prev = float(delta[t-1]) if t-1 >= 0 else float('nan')
                    d_curr = float(delta[t])
                    print(
                        f"[t={t}] pos {int(pos[t-1])}->{int(pos[t])} | "
                        f"delta {d_prev:+.6f}->{d_curr:+.6f} | pred_pos={pred_pos} | hit={hit}"
                    )

            if args.baseline_split in ("train", "both"):
                debug_split("Train", delta_train, train_traj, args.debug_transfers)
            if args.baseline_split in ("test", "both"):
                debug_split("Test", delta_test, test_traj, args.debug_transfers)
        return
    else:
        if args.test_only:
            device = torch.device("mps" if torch.mps.is_available() else "cpu")
            model_path = args.model_path or find_latest_model(args.dataset)
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError("No model found. Provide --model-path or train first.")
            print(f"Loading model from: {model_path}")
            model, _ = load_saved_model(model_path, device)
            evaluate_model_on_test(model, dataset=args.dataset, batch_size=args.batch_size, label_source=args.label_source)
        else:
            saved_path = train_and_eval(
                dataset=args.dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                l2=args.l2,
                seed=args.seed,
                label_source=args.label_source,
            )


if __name__ == "__main__":
    main()


