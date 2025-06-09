# attempt_1__focal_loss_pass.py
"""
CMI – Detect Behavior with Sensor Data – Attempt 1
Change Type: focal_loss
Description: Replace standard CrossEntropyLoss with FocalLoss to better handle class imbalance and potentially lift macro-F1.
"""

import os
import gc
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import polars as pl
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import kaggle_evaluation.cmi_inference_server

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 1. Paths
# ===============================
DATA_DIR = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_DEM = os.path.join(DATA_DIR, "test_demographics.csv")

# ===============================
# 2. Load data & encode labels
# ===============================
print("Loading train …")

df = pd.read_csv(TRAIN_CSV)
print(f"Rows: {len(df):,}")

le = LabelEncoder()
df["gesture"] = le.fit_transform(df["gesture"].astype(str))
np.save("gesture_classes.npy", le.classes_)
NUM_CLASSES = len(le.classes_)

# ===============================
# 3. Feature engineering helpers
# ===============================
EXCL = {
    "gesture",
    "sequence_type",
    "behavior",
    "orientation",
    "row_id",
    "subject",
    "phase",
    "sequence_id",
    "sequence_counter",
}
THERM_TOF = [c for c in df.columns if c.startswith("thm_") or c.startswith("tof_")]
EXCL.update(THERM_TOF)  # IMU-only baseline
FEATURE_COLS = [c for c in df.columns if c not in EXCL]
ACC_COLS = [c for c in FEATURE_COLS if c.startswith("acc_")]
GYR_COLS = [c for c in FEATURE_COLS if c.startswith("rot_")]
np.save("feature_cols.npy", np.array(FEATURE_COLS))
print(f"Using {len(FEATURE_COLS)} base features (IMU).")


def jerk(df_sub: pd.DataFrame) -> pd.DataFrame:
    d = df_sub[ACC_COLS + GYR_COLS].diff().fillna(0)
    d.columns = [f"d_{c}" for c in d.columns]
    return d


def zscore(mat: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(mat)


def stats(mat: np.ndarray) -> np.ndarray:
    mins, maxs = mat.min(0), mat.max(0)
    means, stds = mat.mean(0), mat.std(0)
    rng = maxs - mins
    return np.concatenate([mins, maxs, means, stds, rng]).astype("float32")


def preprocess_seq(df_seq: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    seq = df_seq.sort_values("sequence_counter")
    base = seq[FEATURE_COLS].ffill().bfill().fillna(0)
    full = pd.concat([base, jerk(base)], axis=1)
    arr = zscore(full)
    return arr.astype("float32"), stats(arr)


# ===============================
# 4. Build tensors
# ===============================
seqs, gl_stats, labels, lengths = [], [], [], []
for sid, grp in df.groupby("sequence_id"):
    a, s = preprocess_seq(grp)
    seqs.append(a)
    gl_stats.append(s)
    labels.append(grp["gesture"].iloc[0])
    lengths.append(len(a))

PAD_LEN = int(np.percentile(lengths, 90))
np.save("sequence_maxlen.npy", PAD_LEN)

# Pad / truncate sequences to PAD_LEN

def pad_sequences_torch(sequences, maxlen, padding="post", truncating="post", value=0.0):
    padded = []
    for seq in sequences:
        seq_tensor = torch.tensor(seq, dtype=torch.float32)
        if len(seq_tensor) > maxlen:
            if truncating == "post":
                seq_tensor = seq_tensor[:maxlen]
            else:
                seq_tensor = seq_tensor[-maxlen:]
        if len(seq_tensor) < maxlen:
            pad_size = maxlen - len(seq_tensor)
            if padding == "post":
                seq_tensor = F.pad(seq_tensor, (0, 0, 0, pad_size), value=value)
            else:
                seq_tensor = F.pad(seq_tensor, (0, 0, pad_size, 0), value=value)
        padded.append(seq_tensor)
    return torch.stack(padded)


seqs = pad_sequences_torch(seqs, maxlen=PAD_LEN, padding="post", truncating="post")
stat_arr = torch.tensor(np.stack(gl_stats), dtype=torch.float32)
Y = torch.tensor(labels, dtype=torch.long)

# Train-val split
indices = np.arange(len(seqs))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

X_seq_tr = seqs[train_idx]
X_seq_val = seqs[val_idx]
X_stat_tr = stat_arr[train_idx]
X_stat_val = stat_arr[val_idx]
y_tr = Y[train_idx]
y_val = Y[val_idx]

DELTA_F = seqs.shape[2]
STAT_F = stat_arr.shape[1]
print(f"Pad length {PAD_LEN}, sequence features {DELTA_F}, stats {STAT_F}")


# ===============================
# 5. Model definitions
# ===============================
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, ff: int = 256, drop: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=drop, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff),
            nn.ReLU(),
            nn.Linear(ff, d_model),
        )
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


class BFRBTransformer(nn.Module):
    def __init__(self, seq_len: int, delta_f: int, stat_f: int, num_classes: int, d_model: int = 128):
        super().__init__()
        self.proj = nn.Linear(delta_f, d_model)
        self.transformer1 = TransformerBlock(d_model)
        self.transformer2 = TransformerBlock(d_model)

        self.stat_feat = nn.Sequential(
            nn.Linear(stat_f, 128),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, seq_input: torch.Tensor, stat_input: torch.Tensor):
        mask = (seq_input == 0).all(dim=-1)
        x = self.proj(seq_input)
        x = self.transformer1(x, mask)
        x = self.transformer2(x, mask)
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = x.masked_fill(mask_expanded, 0)
        seq_feat = x.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float()
        stat_feat = self.stat_feat(stat_input)
        combined = torch.cat([seq_feat, stat_feat], dim=1)
        return self.classifier(combined)


# ===============================
# 6. Focal Loss implementation
# ===============================
class FocalLoss(nn.Module):
    """Multi-class focal loss implementation."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha] * NUM_CLASSES, dtype=torch.float32)
            elif isinstance(alpha, list):
                assert len(alpha) == NUM_CLASSES
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                raise ValueError("alpha must be float, int, or list")
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        ce_loss = F.nll_loss(log_probs, targets, reduction="none")
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_term = focal_term * alpha_t
        loss = focal_term * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# ===============================
# 7. Prepare loaders
# ===============================
train_dataset = TensorDataset(X_seq_tr, X_stat_tr, y_tr)
val_dataset = TensorDataset(X_seq_val, X_stat_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# ===============================
# 8. Model, loss, optimizer
# ===============================
model = BFRBTransformer(PAD_LEN, DELTA_F, STAT_F, NUM_CLASSES).to(device)
criterion = FocalLoss(gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ===============================
# 9. Train loop (early stop)
# ===============================
print("Training with FocalLoss …")

best_val_loss = float("inf")
patience_counter = 0
best_state = None

for epoch in range(60):
    model.train()
    train_loss = 0
    for seq_batch, stat_batch, lbl_batch in train_loader:
        seq_batch = seq_batch.to(device)
        stat_batch = stat_batch.to(device)
        lbl_batch = lbl_batch.to(device)

        optimizer.zero_grad()
        outputs = model(seq_batch, stat_batch)
        loss = criterion(outputs, lbl_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for seq_batch, stat_batch, lbl_batch in val_loader:
            seq_batch = seq_batch.to(device)
            stat_batch = stat_batch.to(device)
            lbl_batch = lbl_batch.to(device)
            outputs = model(seq_batch, stat_batch)
            loss = criterion(outputs, lbl_batch)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1:02d} – train_loss: {avg_train_loss:.4f} – val_loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 8:
            print("Early stopping!")
            break

# Restore best state
model.load_state_dict(best_state)

torch.save({
    "model_state_dict": model.state_dict(),
    "pad_len": PAD_LEN,
    "delta_f": DELTA_F,
    "stat_f": STAT_F,
    "num_classes": NUM_CLASSES,
}, "bfrb_transformer_focal.pth")

# ===============================
# 10. Inference utilities
# ===============================
CLASS_NAMES = np.load("gesture_classes.npy", allow_pickle=True)
FEATURE_COLS = np.load("feature_cols.npy", allow_pickle=True).tolist()
ACC_COLS = [c for c in FEATURE_COLS if c.startswith("acc_")]
GYR_COLS = [c for c in FEATURE_COLS if c.startswith("rot_")]

checkpoint = torch.load("bfrb_transformer_focal.pth", map_location=device)
_inf_model = BFRBTransformer(
    checkpoint["pad_len"],
    checkpoint["delta_f"],
    checkpoint["stat_f"],
    checkpoint["num_classes"],
).to(device)
_inf_model.load_state_dict(checkpoint["model_state_dict"])
_inf_model.eval()


def preprocess_for_inf(df_seq):
    s_arr, g_stat = preprocess_seq(df_seq)
    s_pad = pad_sequences_torch([s_arr], maxlen=PAD_LEN, padding="post", truncating="post")
    return s_pad, torch.tensor(g_stat[np.newaxis, :], dtype=torch.float32)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame):
    s_pad, g_stat = preprocess_for_inf(sequence.to_pandas())
    s_pad = s_pad.to(device)
    g_stat = g_stat.to(device)
    with torch.no_grad():
        outputs = _inf_model(s_pad, g_stat)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).cpu().numpy()[0]
    return CLASS_NAMES[int(pred_idx)]


# ===============================
# 11. Launch evaluation server
# ===============================
serv = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    serv.serve()
else:
    serv.run_local_gateway(data_paths=(TEST_CSV, TEST_DEM))

print("Attempt 1 complete.")