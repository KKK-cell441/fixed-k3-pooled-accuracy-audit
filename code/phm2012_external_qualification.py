#!/usr/bin/env python3
"""Locked external qualification audit on the complete PHM2012 bearing lives."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from hust_external_validation import (
    BATCH_SIZE,
    CLASSES as HUST_CLASSES,
    LR,
    ModeCNN,
    Modes,
    add_noise,
    normalize,
    run_vmd,
    seed_everything,
)


ROOT = Path(__file__).resolve().parent
SOURCE = Path(
    r"D:\EvidenceAware_20261011\data\kaggle_phm2012"
    r"\ieee-phm-2012-data-challenge-dataset-master"
)
PROTOCOL = ROOT / "PHM2012_LOCKED_PROTOCOL.json"
CACHE = ROOT / "phm2012_external_cache.npz"
RUNS = ROOT / "phm2012_external_runs.csv"
RESULTS = ROOT / "phm2012_external_results.json"

L = 2048
SNR_DB = 0.0
ACQUISITIONS_PER_CLASS = 12
SEEDS = (20261103, 20261104, 20261105)
EPOCHS = 20
WEIGHT_DECAY = 1e-4
DELTA = 0.01
ETA = 0.05
MIN_SUPPORT = 6


def bearing_files() -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for split in ("Learning_set", "Full_Test_Set"):
        root = SOURCE / split
        for bearing_dir in sorted(root.iterdir()):
            if not bearing_dir.is_dir():
                continue
            files = sorted(bearing_dir.glob("acc_*.csv"))
            if files:
                result[bearing_dir.name] = files
    return result


def select_acquisitions(files: list[Path]) -> tuple[list[Path], list[Path]]:
    n = len(files)
    if n < 10:
        raise ValueError(f"Too few acquisitions for a stable early/late split: {n}")
    early_end = max(1, int(np.floor(0.20 * n)))
    late_start = min(n - 1, int(np.ceil(0.80 * n)))
    early = files[:early_end]
    late = files[late_start:]
    early_idx = np.linspace(
        0,
        len(early) - 1,
        min(ACQUISITIONS_PER_CLASS, len(early)),
        dtype=int,
    )
    late_idx = np.linspace(
        0,
        len(late) - 1,
        min(ACQUISITIONS_PER_CLASS, len(late)),
        dtype=int,
    )
    return (
        [early[index] for index in np.unique(early_idx)],
        [late[index] for index in np.unique(late_idx)],
    )


def read_acceleration(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    delimiter = "," if first_line.count(",") >= 5 else ";"
    values = np.loadtxt(
        path,
        delimiter=delimiter,
        usecols=4,
        dtype=np.float32,
    )
    if len(values) < L:
        raise ValueError(f"Acquisition too short: {path}")
    return normalize(values[:L])


def build_cache() -> dict[str, np.ndarray]:
    if CACHE.exists():
        loaded = np.load(CACHE, allow_pickle=False)
        return {name: loaded[name] for name in loaded.files}

    bearings = bearing_files()
    clean_raw: list[np.ndarray] = []
    test_raw: list[np.ndarray] = []
    train_noisy_raw: list[np.ndarray] = []
    clean_k3: list[np.ndarray] = []
    test_k3: list[np.ndarray] = []
    train_noisy_k3: list[np.ndarray] = []
    labels: list[int] = []
    unit_ids: list[str] = []
    acquisition_ids: list[str] = []

    for bearing_index, (bearing, files) in enumerate(bearings.items()):
        early, late = select_acquisitions(files)
        for class_index, selected in enumerate((early, late)):
            for local_index, path in enumerate(selected):
                clean = read_acceleration(path)
                test = add_noise(
                    clean,
                    SNR_DB,
                    20261105
                    + bearing_index * 100003
                    + local_index * 1009,
                )
                train_noisy = add_noise(
                    clean,
                    SNR_DB,
                    20261113
                    + bearing_index * 100003
                    + local_index * 1009,
                )
                clean_raw.append(clean[None, :])
                test_raw.append(test[None, :])
                train_noisy_raw.append(train_noisy[None, :])
                clean_k3.append(run_vmd(clean, 3))
                test_k3.append(run_vmd(test, 3))
                train_noisy_k3.append(run_vmd(train_noisy, 3))
                labels.append(class_index)
                unit_ids.append(bearing)
                acquisition_ids.append(path.name)

    arrays = {
        "clean_raw": np.stack(clean_raw),
        "test_raw": np.stack(test_raw),
        "train_noisy_raw": np.stack(train_noisy_raw),
        "clean_k3": np.stack(clean_k3),
        "test_k3": np.stack(test_k3),
        "train_noisy_k3": np.stack(train_noisy_k3),
        "labels": np.asarray(labels, dtype=np.int64),
        "units": np.asarray(unit_ids),
        "acquisitions": np.asarray(acquisition_ids),
    }
    np.savez(CACHE, **arrays)
    return arrays


def recall_from_confusion(confusion: np.ndarray) -> np.ndarray:
    support = confusion.sum(axis=1)
    return np.divide(
        np.diag(confusion),
        support,
        out=np.full(len(support), np.nan),
        where=support > 0,
    )


def evaluate(
    model: nn.Module,
    values: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for x, _ in DataLoader(Modes(values, labels, {}), batch_size=128):
            predictions.extend(
                model(x.to(device)).argmax(dim=1).cpu().numpy().tolist()
            )
    return np.asarray(predictions, dtype=np.int64)


def train_one(
    representation: str,
    protocol: str,
    seed: int,
    data: dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    clean_key = "clean_raw" if representation == "raw" else "clean_k3"
    noisy_key = "train_noisy_raw" if representation == "raw" else "train_noisy_k3"
    test_key = "test_raw" if representation == "raw" else "test_k3"
    labels = data["labels"].astype(np.int64)
    if protocol == "clean":
        train_values = data[clean_key][train_mask]
        train_labels = labels[train_mask]
    else:
        train_values = np.concatenate(
            [data[clean_key][train_mask], data[noisy_key][train_mask]],
            axis=0,
        )
        train_labels = np.tile(labels[train_mask], 2)

    seed_everything(seed)
    model = ModeCNN(train_values.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )
    loader = DataLoader(
        Modes(train_values, train_labels, {}),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    prediction = evaluate(
        model,
        data[test_key][test_mask],
        labels[test_mask],
        device,
    )
    return confusion_matrix(
        labels[test_mask],
        prediction,
        labels=[0, 1],
    )


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def bootstrap_class(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = np.nanmean(values[indices], axis=1)
    return (
        np.nanquantile(means, 0.005, axis=0),
        np.nanquantile(means, 0.995, axis=0),
    )


def decide(
    mean_interval: list[float],
    class_low: np.ndarray,
    class_high: np.ndarray,
    support: np.ndarray,
) -> tuple[str, str]:
    if np.any(support == 0):
        return "Insufficient", "missing class evidence"
    if np.any(support < MIN_SUPPORT):
        return "Insufficient", "fragile class evidence"
    if mean_interval[0] > DELTA and np.all(class_low >= -ETA):
        return "Qualified", "mean effect and class safety"
    if mean_interval[1] <= DELTA:
        return "Not qualified", "mean effect"
    if np.any(class_high < -ETA):
        return "Not qualified", "class harm"
    return "Insufficient", "mean or class evidence inconclusive"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = build_cache()
    labels = data["labels"].astype(np.int64)
    units = data["units"].astype(str)
    unit_values = sorted(set(units.tolist()))

    existing = []
    if RUNS.exists():
        with RUNS.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    completed = {
        (
            row["protocol"],
            row["bearing"],
            row["representation"],
            int(row["seed"]),
        )
        for row in existing
    }
    rows = existing

    for protocol in ("clean", "matched"):
        for bearing in unit_values:
            train_mask = units != bearing
            test_mask = units == bearing
            for representation in ("raw", "k3"):
                for seed in SEEDS:
                    key = (protocol, bearing, representation, seed)
                    if key in completed:
                        continue
                    confusion = train_one(
                        representation,
                        protocol,
                        seed,
                        data,
                        train_mask,
                        test_mask,
                        device,
                    )
                    rows.append(
                        {
                            "protocol": protocol,
                            "bearing": bearing,
                            "representation": representation,
                            "seed": seed,
                            "confusion": json.dumps(confusion.tolist()),
                        }
                    )
                    with RUNS.open("w", encoding="utf-8", newline="") as handle:
                        writer = csv.DictWriter(
                            handle,
                            fieldnames=[
                                "protocol",
                                "bearing",
                                "representation",
                                "seed",
                                "confusion",
                            ],
                        )
                        writer.writeheader()
                        writer.writerows(rows)
                    print(
                        f"{protocol} {bearing} {representation} seed={seed}",
                        flush=True,
                    )

    frame = {(row["protocol"], row["bearing"], row["representation"], int(row["seed"])): np.asarray(json.loads(row["confusion"]), dtype=float) for row in rows}
    results: dict[str, object] = {}
    for protocol_index, protocol in enumerate(("clean", "matched")):
        unit_class_deltas = []
        unit_mean_deltas = []
        support = np.zeros(2, dtype=int)
        for bearing in unit_values:
            seed_deltas = []
            valid_unit = np.ones(2, dtype=bool)
            for seed in SEEDS:
                raw = frame[(protocol, bearing, "raw", seed)]
                k3 = frame[(protocol, bearing, "k3", seed)]
                raw_recall = recall_from_confusion(raw)
                k3_recall = recall_from_confusion(k3)
                delta = k3_recall - raw_recall
                seed_deltas.append(delta)
                valid_unit &= ~np.isnan(delta)
            support += valid_unit.astype(int)
            unit_delta = np.nanmean(np.stack(seed_deltas), axis=0)
            unit_class_deltas.append(unit_delta)
            unit_mean_deltas.append(float(np.nanmean(unit_delta)))
        unit_class = np.stack(unit_class_deltas)
        unit_means = np.asarray(unit_mean_deltas)
        mean_interval = bootstrap_mean(
            unit_means,
            np.random.default_rng(20261105 + protocol_index),
        )
        class_low, class_high = bootstrap_class(
            unit_class,
            np.random.default_rng(20261205 + protocol_index),
        )
        status, basis = decide(mean_interval, class_low, class_high, support)
        results[protocol] = {
            "n_units": len(unit_values),
            "mean_effect_pp": float(unit_means.mean() * 100),
            "unit_95_pp": [float(value * 100) for value in mean_interval],
            "class_means_pp": [
                float(value * 100) for value in np.nanmean(unit_class, axis=0)
            ],
            "class_simultaneous_intervals_pp": [
                [float(low * 100), float(high * 100)]
                for low, high in zip(class_low, class_high)
            ],
            "support_units_by_class": support.tolist(),
            "status": status,
            "status_basis": basis,
        }
    output = {
        "registration": "PHM2012-EXTERNAL-QUALIFICATION-20261105",
        "candidate": "fixed K3",
        "delta_pp": DELTA * 100,
        "eta_pp": ETA * 100,
        "protocols": results,
        "claim_boundary": (
            "Locked external run-to-failure audit; not a prospective acquisition "
            "and not a reproduction of the five-class HUSTbearing endpoint."
        ),
    }
    RESULTS.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
