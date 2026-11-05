#!/usr/bin/env python3
"""Frozen external-rig check on HUSTbearing.

The outer split is by constant motor speed. Raw and VMD representations use an
identical classifier. Only the held-out test windows receive 0 dB AWGN.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from vmdpy import VMD


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\bearing_research\data")
PROTOCOL = ROOT / "HUST_EXTERNAL_PROTOCOL.json"
RUNS_PATH = ROOT / "hust_external_runs.csv"
STATUS_PATH = ROOT / "hust_external_status.json"
L = 2048
WINDOWS_PER_RECORDING = 6
ALPHA = 2000.0
TAU = 0.0
TOL = 1e-6
SPEEDS = [20, 25, 30, 35, 40, 60, 65, 70, 75, 80]
CLASSES = ["inner", "inner+outer", "normal", "outer", "rolling_element"]
SEEDS = [20261103, 20261104, 20261105]
TARGET_SNR_DB = 0.0
MAX_EPOCHS = 25
PATIENCE = 6
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4


def normalize(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float32)
    return (window - window.mean()) / (window.std() + 1e-8)


def add_noise(window: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scale = window.std() * 10 ** (-snr_db / 20.0)
    return (
        window + rng.normal(0.0, scale, size=window.shape)
    ).astype(np.float32)


def run_vmd(window: np.ndarray, k: int) -> np.ndarray:
    modes, _, _ = VMD(
        np.asarray(window, dtype=np.float64),
        ALPHA,
        TAU,
        k,
        0,
        1,
        TOL,
    )
    return np.asarray(modes, dtype=np.float32)


class Modes:
    def __init__(self, values: np.ndarray, labels: np.ndarray, groups: dict):
        self.values = torch.from_numpy(values.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.groups = groups

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.values[index], self.labels[index]


class ModeCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = len(CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def load_recordings() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    sys.path.insert(0, r"D:\bearing_research")
    import bearing_datasets as bd

    dataset = bd.open("hustbearing", root=str(DATA_ROOT))
    meta = dataset.metadata()
    meta = meta[
        (meta["channel"] == "X")
        & (meta["speed_profile"] == "constant")
        & (meta["fault_type"].isin(CLASSES))
    ].copy()
    meta["speed_hz"] = (
        meta["operating_condition"]
        .str.replace("Hz", "", regex=False)
        .astype(int)
    )
    meta = meta[meta["speed_hz"].isin(SPEEDS)].copy()
    signals = {
        row.signal_id: np.asarray(dataset.signal(row.signal_id), dtype=np.float32)
        for row in meta.itertuples()
    }
    return meta.reset_index(drop=True), signals


def build_window_data(
    meta: pd.DataFrame,
    signals: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    clean_raw = []
    test_raw = []
    clean_modes = {3: [], 4: []}
    test_modes = {3: [], 4: []}
    labels = []
    speeds = []
    recordings = []
    class_to_index = {name: index for index, name in enumerate(CLASSES)}
    start = time.time()
    for row_index, row in enumerate(meta.itertuples()):
        signal = signals[row.signal_id]
        max_start = len(signal) - L
        starts = np.unique(
            np.linspace(0, max_start, WINDOWS_PER_RECORDING, dtype=int)
        )
        for window_index, window_start in enumerate(starts):
            clean = normalize(signal[window_start : window_start + L])
            test = add_noise(
                clean,
                TARGET_SNR_DB,
                20261103
                + row_index * 1009
                + window_index * 37,
            )
            clean_raw.append(clean[None, :])
            test_raw.append(test[None, :])
            for k in (3, 4):
                clean_modes[k].append(run_vmd(clean, k))
                test_modes[k].append(run_vmd(test, k))
            labels.append(class_to_index[row.fault_type])
            speeds.append(int(row.speed_hz))
            recordings.append(row.signal_id)
        if (row_index + 1) % 10 == 0:
            print(
                f"prepared {row_index + 1}/{len(meta)} recordings "
                f"in {time.time() - start:.1f}s",
                flush=True,
            )
    arrays = {
        "clean_raw": np.stack(clean_raw),
        "test_raw": np.stack(test_raw),
        "clean_k3": np.stack(clean_modes[3]),
        "test_k3": np.stack(test_modes[3]),
        "clean_k4": np.stack(clean_modes[4]),
        "test_k4": np.stack(test_modes[4]),
    }
    return (
        arrays,
        np.asarray(labels, dtype=np.int64),
        np.asarray(speeds, dtype=np.int64),
        np.asarray(recordings, dtype=object),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def evaluate(
    model: nn.Module,
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, np.ndarray]:
    model.eval()
    loader = DataLoader(Modes(values, labels, {}), batch_size=128)
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            predictions.extend(model(x).argmax(dim=1).numpy().tolist())
    prediction = np.asarray(predictions, dtype=np.int64)
    return balanced_accuracy_score(labels, prediction), prediction


def train_one(
    representation: str,
    seed: int,
    arrays: dict[str, np.ndarray],
    labels: np.ndarray,
    speeds: np.ndarray,
    recordings: np.ndarray,
    target_speed: int,
    validation_speed: int,
) -> dict:
    seed_everything(seed)
    train_mask = ~np.isin(speeds, [target_speed, validation_speed])
    val_mask = speeds == validation_speed
    test_mask = speeds == target_speed
    train_values = arrays[f"clean_{representation}"][train_mask]
    val_values = arrays[f"clean_{representation}"][val_mask]
    test_values = arrays[f"test_{representation}"][test_mask]
    in_channels = train_values.shape[1]
    model = ModeCNN(in_channels)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
    )
    loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(
        Modes(train_values, labels[train_mask], {}),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    best_state = None
    best_val = -1.0
    bad_epochs = 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        val_score, _ = evaluate(model, val_values, labels[val_mask])
        if val_score > best_val + 1e-12:
            best_val = val_score
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break
    model.load_state_dict(best_state)
    test_score, prediction = evaluate(model, test_values, labels[test_mask])
    test_recordings = recordings[test_mask]
    test_truth = labels[test_mask]
    recording_scores = []
    for recording in sorted(set(test_recordings)):
        mask = test_recordings == recording
        recording_scores.append(
            float(np.mean(prediction[mask] == test_truth[mask]))
        )
    return {
        "representation": representation,
        "seed": seed,
        "target_speed_hz": target_speed,
        "validation_speed_hz": validation_speed,
        "best_validation_balanced_accuracy": float(best_val),
        "test_balanced_accuracy": float(test_score),
        "test_recording_mean_accuracy": float(np.mean(recording_scores)),
        "n_train_windows": int(train_mask.sum()),
        "n_validation_windows": int(val_mask.sum()),
        "n_test_windows": int(test_mask.sum()),
        "n_test_recordings": int(len(recording_scores)),
    }


def cluster_bootstrap(
    values: np.ndarray,
    seed: int,
    draws: int = 20_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def summarize_pair(
    results: pd.DataFrame,
    representation: str,
) -> tuple[pd.DataFrame, dict]:
    raw = results[results.representation == "raw"][
        [
            "seed",
            "target_speed_hz",
            "test_balanced_accuracy",
            "test_recording_mean_accuracy",
        ]
    ].rename(
        columns={
            "test_balanced_accuracy": "raw_balanced_accuracy",
            "test_recording_mean_accuracy": "raw_recording_mean_accuracy",
        }
    )
    candidate = results[results.representation == representation][
        [
            "seed",
            "target_speed_hz",
            "test_balanced_accuracy",
            "test_recording_mean_accuracy",
        ]
    ].rename(
        columns={
            "test_balanced_accuracy": f"{representation}_balanced_accuracy",
            "test_recording_mean_accuracy": (
                f"{representation}_recording_mean_accuracy"
            ),
        }
    )
    paired = raw.merge(
        candidate,
        on=["seed", "target_speed_hz"],
        how="inner",
        validate="one_to_one",
    )
    paired["delta_balanced_accuracy"] = (
        paired[f"{representation}_balanced_accuracy"]
        - paired["raw_balanced_accuracy"]
    )
    paired["delta_recording_mean_accuracy"] = (
        paired[f"{representation}_recording_mean_accuracy"]
        - paired["raw_recording_mean_accuracy"]
    )
    speed = (
        paired.groupby("target_speed_hz")
        .agg(
            raw_balanced_accuracy=("raw_balanced_accuracy", "mean"),
            candidate_balanced_accuracy=(
                f"{representation}_balanced_accuracy",
                "mean",
            ),
            mean_delta=("delta_balanced_accuracy", "mean"),
            mean_delta_recording=(
                "delta_recording_mean_accuracy",
                "mean",
            ),
        )
        .reset_index()
    )
    speed["representation"] = representation
    delta = speed["mean_delta"].to_numpy(dtype=float)
    ci_low, ci_high = cluster_bootstrap(
        delta,
        20261103 + len(representation),
    )
    loo_means = np.asarray(
        [
            float(np.delete(delta, index).mean())
            for index in range(len(delta))
        ]
    )
    summary = {
        "representation": representation,
        "n_speeds": int(len(speed)),
        "mean_delta_balanced_accuracy": float(delta.mean()),
        "median_delta_balanced_accuracy": float(np.median(delta)),
        "speed_bootstrap_ci_low": ci_low,
        "speed_bootstrap_ci_high": ci_high,
        "speeds_improved": int((delta > 1e-12).sum()),
        "speeds_unchanged": int((np.abs(delta) <= 1e-12).sum()),
        "speeds_worsened": int((delta < -1e-12).sum()),
        "loo_speed_mean_min": float(loo_means.min()),
        "loo_speed_mean_max": float(loo_means.max()),
        "mean_delta_recording_accuracy": float(
            speed["mean_delta_recording"].mean()
        ),
    }
    return paired, speed, summary


def append_run(row: dict) -> None:
    frame = pd.DataFrame([row])
    frame.to_csv(
        RUNS_PATH,
        mode="a",
        header=not RUNS_PATH.exists(),
        index=False,
    )


def write_status(
    state: str,
    completed_runs: int,
    total_runs: int,
    last_target_speed: int | None = None,
    last_representation: str | None = None,
    last_seed: int | None = None,
    error: str | None = None,
) -> None:
    status = {
        "state": state,
        "completed_runs": int(completed_runs),
        "total_runs": int(total_runs),
        "fraction_complete": float(completed_runs / total_runs),
        "last_target_speed_hz": last_target_speed,
        "last_representation": last_representation,
        "last_seed": last_seed,
        "error": error,
    }
    STATUS_PATH.write_text(
        json.dumps(status, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    meta, signals = load_recordings()
    arrays, labels, speeds, recordings = build_window_data(meta, signals)
    if args.smoke:
        row = train_one(
            "k3",
            SEEDS[0],
            arrays,
            labels,
            speeds,
            recordings,
            20,
            25,
        )
        print(json.dumps(row, indent=2))
        return

    existing = (
        pd.read_csv(RUNS_PATH)
        if RUNS_PATH.exists() and RUNS_PATH.stat().st_size > 0
        else pd.DataFrame()
    )
    rows = existing.to_dict(orient="records")
    completed = {
        (
            int(row["target_speed_hz"]),
            str(row["representation"]),
            int(row["seed"]),
        )
        for row in rows
    }
    total_runs = len(SPEEDS) * 3 * len(SEEDS)
    write_status("running", len(completed), total_runs)
    start = time.time()
    for speed_index, target_speed in enumerate(SPEEDS):
        remaining = [speed for speed in SPEEDS if speed != target_speed]
        validation_speed = remaining[speed_index % len(remaining)]
        for representation in ("raw", "k3", "k4"):
            for seed in SEEDS:
                key = (target_speed, representation, seed)
                if key in completed:
                    continue
                row = train_one(
                    representation,
                    seed,
                    arrays,
                    labels,
                    speeds,
                    recordings,
                    target_speed,
                    validation_speed,
                )
                rows.append(row)
                append_run(row)
                completed.add(key)
                write_status(
                    "running",
                    len(completed),
                    total_runs,
                    target_speed,
                    representation,
                    seed,
                )
                print(
                    f"[{time.time() - start:7.1f}s] "
                    f"target={target_speed:>2} val={validation_speed:>2} "
                    f"rep={representation:>3} seed={seed} "
                    f"test_ba={row['test_balanced_accuracy']:.4f}",
                    flush=True,
                )
    results = pd.DataFrame(rows)
    if len(results) != total_runs:
        raise RuntimeError(
            f"Incomplete run matrix: {len(results)} of {total_runs}"
        )
    results = results.sort_values(
        ["target_speed_hz", "representation", "seed"]
    ).reset_index(drop=True)
    results.to_csv(RUNS_PATH, index=False)

    paired_frames = []
    summaries = []
    for representation in ("k3", "k4"):
        paired, speed, summary = summarize_pair(results, representation)
        paired["representation"] = representation
        paired.to_csv(
            ROOT / f"hust_external_paired_{representation}.csv",
            index=False,
        )
        speed.to_csv(
            ROOT / f"hust_external_by_speed_{representation}.csv",
            index=False,
        )
        paired_frames.append(paired)
        summaries.append(summary)
    output = {
        "registration": protocol["registration"],
        "dataset": "HUSTbearing",
        "scope": (
            "Independent external rig with leave-one-speed-out validation. "
            "The same physical bearings recur across speeds, so the unit of "
            "inference is the held-out speed, not an independent bearing."
        ),
        "n_recordings": int(len(meta)),
        "n_windows": int(len(labels)),
        "test_snr_db": TARGET_SNR_DB,
        "summaries": summaries,
        "runs_per_representation": int(len(results) / 3),
        "runtime_seconds_preparation_and_training": float(time.time() - start),
    }
    (ROOT / "hust_external_results.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    write_status("completed", len(results), total_runs)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
