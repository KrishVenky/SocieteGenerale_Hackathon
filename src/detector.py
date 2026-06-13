"""
Anomaly detector: Bi-directional LSTM autoencoder on user event sequences.

Trained unsupervised on the full event log.
High reconstruction error on a sequence = anomalous pattern.
Score is normalised to [0, 1] using the 95th-percentile training error as ceiling.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ── Event encoding ─────────────────────────────────────────────────────────────

ACTIONS = ["login", "file_access", "sql_query", "api_call", "admin_operation", "export_data"]
RESOURCES = [
    "Customer_Vault", "GL_System", "HRIS", "PROD_DB", "Admin_Console",
    "SIEM", "Data_Lake", "BI_Tool", "File_Share", "Email_Archive",
]
TIME_CLASSES = ["business_hours", "weekend", "unusual_hours", "night"]
STATUSES = ["success", "failure"]
SENSITIVITY = {"low": 0.33, "medium": 0.67, "high": 1.0}

_ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}
_RESOURCE_IDX = {r: i for i, r in enumerate(RESOURCES)}
_TIME_IDX = {t: i for i, t in enumerate(TIME_CLASSES)}
_STATUS_IDX = {s: i for i, s in enumerate(STATUSES)}

# Total input dim: 6 + 10 + 4 + 2 + 1 = 23
EVENT_DIM = len(ACTIONS) + len(RESOURCES) + len(TIME_CLASSES) + len(STATUSES) + 1


def encode_event(row) -> np.ndarray:
    """One-hot encode a single event row into a float32 vector of size EVENT_DIM."""
    vec = np.zeros(EVENT_DIM, dtype=np.float32)
    offset = 0

    idx = _ACTION_IDX.get(str(row["action"]), 0)
    vec[offset + idx] = 1.0
    offset += len(ACTIONS)

    idx = _RESOURCE_IDX.get(str(row["resource"]), 0)
    vec[offset + idx] = 1.0
    offset += len(RESOURCES)

    idx = _TIME_IDX.get(str(row["time_classification"]), 0)
    vec[offset + idx] = 1.0
    offset += len(TIME_CLASSES)

    idx = _STATUS_IDX.get(str(row["status"]), 0)
    vec[offset + idx] = 1.0
    offset += len(STATUSES)

    vec[offset] = SENSITIVITY.get(str(row["resource_sensitivity"]), 0.33)

    return vec


# ── Model ──────────────────────────────────────────────────────────────────────

class BiLSTMAutoencoder(nn.Module):
    """
    BiLSTM encoder → latent → LSTM decoder → reconstruction.
    Reconstruction MSE is the anomaly score.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Bidirectional encoder
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.encoder_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, latent_dim),
            nn.ReLU(),
        )

        # Unidirectional decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=1, batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)

        # Encode
        _, (h_n, _) = self.encoder_lstm(x)
        h_fwd, h_bwd = h_n[0], h_n[1]
        latent = self.encoder_fc(torch.cat([h_fwd, h_bwd], dim=-1))

        # Decode
        dec_in = self.decoder_fc(latent).unsqueeze(1).expand(-1, seq_len, -1)
        dec_out, _ = self.decoder_lstm(dec_in)
        reconstruction = self.output_fc(dec_out)

        return reconstruction, latent


# ── Detector ───────────────────────────────────────────────────────────────────

class AnomalyDetector:
    def __init__(
        self,
        seq_len: int = 10,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        epochs: int = 60,
        lr: float = 1e-3,
        batch_size: int = 32,
    ):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: BiLSTMAutoencoder | None = None
        self._threshold: float = 1.0  # 95th percentile reconstruction error on training set

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, logs: pd.DataFrame, seed: int = 42) -> "AnomalyDetector":
        torch.manual_seed(seed)
        np.random.seed(seed)

        sequences = self._build_sequences(logs)
        if len(sequences) < 4:
            print("[detector] too few sequences to train — skipping LSTM")
            return self

        X = torch.tensor(np.array(sequences), dtype=torch.float32).to(self.device)
        self.model = BiLSTMAutoencoder(EVENT_DIM, self.hidden_dim, self.latent_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loader = DataLoader(TensorDataset(X), batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, _ = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                print(f"[detector] epoch {epoch+1}/{self.epochs}  loss={epoch_loss/len(loader):.4f}")

        # Set threshold at 95th percentile of training errors
        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X)
            errors = ((recon - X) ** 2).mean(dim=[1, 2]).cpu().numpy()
        self._threshold = float(np.percentile(errors, 95))
        print(f"[detector] threshold (95th pct): {self._threshold:.4f}")
        return self

    def save(self, path: str) -> None:
        if self.model is None:
            return
        torch.save(
            {"state_dict": self.model.state_dict(), "threshold": self._threshold,
             "seq_len": self.seq_len, "hidden_dim": self.hidden_dim,
             "latent_dim": self.latent_dim},
            path,
        )

    def load(self, path: str) -> "AnomalyDetector":
        ckpt = torch.load(path, map_location=self.device)
        self.seq_len = ckpt["seq_len"]
        self.hidden_dim = ckpt["hidden_dim"]
        self.latent_dim = ckpt["latent_dim"]
        self._threshold = ckpt["threshold"]
        self.model = BiLSTMAutoencoder(EVENT_DIM, self.hidden_dim, self.latent_dim).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        return self

    # ── Scoring ───────────────────────────────────────────────────────────────

    def score_sequence(self, events: list) -> float:
        """
        Anomaly score in [0, 1] for a list of event rows (pd.Series or dicts).
        0 = perfectly normal, 1 = maximally anomalous.
        """
        if self.model is None or not events:
            return 0.0

        arr = self._pad(events)
        X = torch.tensor(arr[np.newaxis], dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            recon, _ = self.model(X)
            error = float(((recon - X) ** 2).mean().cpu())

        return float(min(1.0, error / max(self._threshold, 1e-8)))

    def get_latent(self, events: list) -> np.ndarray:
        """Latent vector for a user's event window — useful for graph embedding concat."""
        if self.model is None:
            return np.zeros(self.latent_dim, dtype=np.float32)
        arr = self._pad(events)
        X = torch.tensor(arr[np.newaxis], dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, latent = self.model(X)
        return latent.squeeze(0).cpu().numpy()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_sequences(self, logs: pd.DataFrame) -> list:
        """Sliding window of seq_len events per user, encoded."""
        seqs = []
        logs = logs.sort_values(["user_id", "timestamp"])
        for _, user_events in logs.groupby("user_id"):
            rows = [row for _, row in user_events.iterrows()]
            for i in range(len(rows) - self.seq_len + 1):
                window = rows[i : i + self.seq_len]
                encoded = np.array([encode_event(e) for e in window], dtype=np.float32)
                seqs.append(encoded)
        return seqs

    def _pad(self, events: list) -> np.ndarray:
        """Encode a list of events, pad/truncate to seq_len."""
        encoded = [
            encode_event(e if hasattr(e, "__getitem__") else vars(e))
            for e in events[-self.seq_len:]
        ]
        arr = np.array(encoded, dtype=np.float32)
        if len(arr) < self.seq_len:
            pad = np.zeros((self.seq_len - len(arr), EVENT_DIM), dtype=np.float32)
            arr = np.vstack([pad, arr])
        return arr
