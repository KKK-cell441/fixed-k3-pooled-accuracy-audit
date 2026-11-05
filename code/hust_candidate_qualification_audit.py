#!/usr/bin/env python3
"""Pre-specified candidate audit for real Qualified reachability on HUSTbearing.

The candidate set is fixed to K=2,3,4,5. Both clean-trained and matched
noise-augmentation protocols are reported. The decision margins are fixed to
delta=1 pp and eta=5 pp. No candidate is added or removed after inspection.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from hust_external_validation import (
    BATCH_SIZE,
    CLASSES,
    LR,
    SEEDS,
    SPEEDS,
    ModeCNN,
    Modes,
    seed_everything,
)


ROOT = Path(__file__).resolve().parent
MODE_CACHE = ROOT / "hust_independent_k_modes.npz"
DUAL_RUNS = ROOT / "hust_dual_validation_runs.csv"
AUGMENTED_RUNS = ROOT / "hust_augmented_split_runs.csv"
RUNS = ROOT / "hust_candidate_qualification_runs.csv"
RESULTS = ROOT / "hust_candidate_qualification_results.json"

CANDIDATES = (2, 3, 4, 5)
PROTOCOLS = ("clean", "matched")
DELTA = 0.01
ETA = 0.05
MIN_NONFRAGILE_SUPPORT = 6
EPOCHS = 20
WEIGHT_DECAY = 1e-4


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
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, _ in DataLoader(Modes(values, labels, {}), batch_size=128):
            predictions.extend(
                model(x.to(device)).argmax(dim=1).cpu().numpy().tolist()
            )
    return np.asarray(predictions, dtype=np.int64)


def train_candidate(
    k: int,
    seed: int,
    train_values: np.ndarray,
    test_values: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    seed_everything(seed)
    model = ModeCNN(k, len(CLASSES)).to(device)
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
        Modes(train_values, labels, {}),
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
    predictions = evaluate(
        model,
        test_values,
        labels[test_mask],
        device,
    )
    return confusion_matrix(
        labels[test_mask],
        predictions,
        labels=list(range(len(CLASSES))),
    )


def load_existing_confusions() -> dict[tuple[str, int, str, int], np.ndarray]:
    result: dict[tuple[str, int, str, int], np.ndarray] = {}
    dual = pd.read_csv(DUAL_RUNS)
    dual = dual[(dual.scheme == "speed") & (dual.representation == "k3")]
    for row in dual.itertuples():
        result[("clean", int(row.target_unit), "k3", int(row.seed))] = (
            np.asarray(json.loads(row.confusion), dtype=float)
        )
    augmented = pd.read_csv(AUGMENTED_RUNS)
    augmented = augmented[
        (augmented.scheme == "speed")
        & (augmented.representation == "k3")
    ]
    for row in augmented.itertuples():
        result[("matched", int(row.target_unit), "k3", int(row.seed))] = (
            np.asarray(json.loads(row.confusion), dtype=float)
        )
    return result


def load_raw_confusions() -> dict[tuple[str, int, int], np.ndarray]:
    result: dict[tuple[str, int, int], np.ndarray] = {}
    dual = pd.read_csv(DUAL_RUNS)
    dual = dual[(dual.scheme == "speed") & (dual.representation == "raw")]
    for row in dual.itertuples():
        result[("clean", int(row.target_unit), int(row.seed))] = np.asarray(
            json.loads(row.confusion),
            dtype=float,
        )
    augmented = pd.read_csv(AUGMENTED_RUNS)
    augmented = augmented[
        (augmented.scheme == "speed")
        & (augmented.representation == "raw")
    ]
    for row in augmented.itertuples():
        result[("matched", int(row.target_unit), int(row.seed))] = np.asarray(
            json.loads(row.confusion),
            dtype=float,
        )
    return result


def bootstrap_mean_interval(
    effects: np.ndarray,
    rng: np.random.Generator,
) -> list[float]:
    indices = rng.integers(
        0,
        len(effects),
        size=(20_000, len(effects)),
    )
    means = effects[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def bootstrap_class_intervals(
    unit_class_deltas: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_units = unit_class_deltas.shape[0]
    indices = rng.integers(0, n_units, size=(20_000, n_units))
    sampled = unit_class_deltas[indices]
    means = np.nanmean(sampled, axis=1)
    return (
        np.nanquantile(means, 0.005, axis=0),
        np.nanquantile(means, 0.995, axis=0),
    )


def status_from_intervals(
    mean_low: float,
    mean_high: float,
    class_low: np.ndarray,
    class_high: np.ndarray,
    support: np.ndarray,
) -> tuple[str, str]:
    has_missing = bool(np.any(support == 0))
    has_fragile = bool(
        np.any((support > 0) & (support < MIN_NONFRAGILE_SUPPORT))
    )
    if has_missing or has_fragile:
        if mean_high <= DELTA:
            return "Not qualified", "mean effect"
        if np.any(class_high < -ETA):
            return "Not qualified", "class harm"
        return "Insufficient", "missing or fragile class evidence"
    if mean_low > DELTA and np.all(class_low >= -ETA):
        return "Qualified", "mean effect and class safety"
    if mean_high <= DELTA:
        return "Not qualified", "mean effect"
    if np.any(class_high < -ETA):
        return "Not qualified", "class harm"
    return "Insufficient", "mean or class evidence inconclusive"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = np.load(MODE_CACHE, allow_pickle=False)
    labels = cache["labels"].astype(np.int64)
    speeds = cache["speeds"].astype(np.int64)

    raw_confusions = load_raw_confusions()
    candidate_confusions = load_existing_confusions()
    existing = pd.read_csv(RUNS) if RUNS.exists() else pd.DataFrame()
    rows = existing.to_dict(orient="records") if len(existing) else []
    for row in rows:
        candidate_confusions[
            (
                str(row["protocol"]),
                int(row["target_speed_hz"]),
                f"k{int(row['k'])}",
                int(row["seed"]),
            )
        ] = np.asarray(json.loads(row["confusion"]), dtype=float)

    for protocol in PROTOCOLS:
        for speed in SPEEDS:
            train_mask = speeds != speed
            test_mask = speeds == speed
            for k in CANDIDATES:
                representation = f"k{k}"
                clean_values = cache[f"clean_{representation}"]
                noisy_values = cache[f"test_{representation}"]
                if protocol == "clean":
                    train_values = clean_values[train_mask]
                else:
                    train_values = np.concatenate(
                        [
                            clean_values[train_mask],
                            noisy_values[train_mask],
                        ],
                        axis=0,
                    )
                train_labels = (
                    labels[train_mask]
                    if protocol == "clean"
                    else np.tile(labels[train_mask], 2)
                )
                for seed in SEEDS:
                    key = (protocol, speed, representation, seed)
                    if key in candidate_confusions:
                        continue
                    seed_everything(seed)
                    model = ModeCNN(k, len(CLASSES)).to(device)
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
                    predictions = evaluate(
                        model,
                        noisy_values[test_mask],
                        labels[test_mask],
                        device,
                    )
                    confusion = confusion_matrix(
                        labels[test_mask],
                        predictions,
                        labels=list(range(len(CLASSES))),
                    )
                    candidate_confusions[key] = confusion.astype(float)
                    rows.append(
                        {
                            "protocol": protocol,
                            "target_speed_hz": speed,
                            "k": k,
                            "seed": seed,
                            "confusion": json.dumps(confusion.tolist()),
                        }
                    )
                    pd.DataFrame(rows).to_csv(RUNS, index=False)
                    print(
                        f"{protocol} speed={speed} K={k} seed={seed}",
                        flush=True,
                    )

    summaries = {}
    rng = np.random.default_rng(20261105)
    for protocol_index, protocol in enumerate(PROTOCOLS):
        summaries[protocol] = {}
        for k in CANDIDATES:
            unit_class_deltas = []
            unit_mean_deltas = []
            support = np.zeros(len(CLASSES), dtype=int)
            for speed in SPEEDS:
                seed_deltas = []
                unit_valid = np.zeros(len(CLASSES), dtype=bool)
                for seed in SEEDS:
                    raw = raw_confusions[(protocol, speed, seed)]
                    candidate = candidate_confusions[
                        (protocol, speed, f"k{k}", seed)
                    ]
                    raw_recall = recall_from_confusion(raw)
                    candidate_recall = recall_from_confusion(candidate)
                    seed_deltas.append(candidate_recall - raw_recall)
                    valid = ~np.isnan(raw_recall) & ~np.isnan(candidate_recall)
                    unit_valid |= valid
                support += unit_valid.astype(int)
                unit_delta = np.nanmean(np.stack(seed_deltas, axis=0), axis=0)
                unit_class_deltas.append(unit_delta)
                unit_mean_deltas.append(float(np.nanmean(unit_delta)))
            unit_class = np.stack(unit_class_deltas, axis=0)
            unit_means = np.asarray(unit_mean_deltas, dtype=float)
            mean_interval = bootstrap_mean_interval(
                unit_means,
                np.random.default_rng(20261105 + protocol_index * 10 + k),
            )
            class_low, class_high = bootstrap_class_intervals(
                unit_class,
                np.random.default_rng(20261205 + protocol_index * 10 + k),
            )
            status, basis = status_from_intervals(
                mean_low=mean_interval[0],
                mean_high=mean_interval[1],
                class_low=class_low,
                class_high=class_high,
                support=support,
            )
            summaries[protocol][f"k{k}"] = {
                "n_units": len(SPEEDS),
                "n_seeds": len(SEEDS),
                "support_units_by_class": support.tolist(),
                "fragile_by_class": (support < MIN_NONFRAGILE_SUPPORT).tolist(),
                "missing_by_class": (support == 0).tolist(),
                "mean_effect_pp": float(unit_means.mean() * 100),
                "mean_unit_interval_95_pp": [
                    float(mean_interval[0] * 100),
                    float(mean_interval[1] * 100),
                ],
                "class_means_pp": [
                    float(value * 100)
                    for value in np.nanmean(unit_class, axis=0)
                ],
                "class_simultaneous_intervals_pp": [
                    [float(low * 100), float(high * 100)]
                    for low, high in zip(class_low, class_high)
                ],
                "status": status,
                "status_basis": basis,
            }

    result = {
        "design": {
            "source": "HUSTbearing",
            "candidate_set": list(CANDIDATES),
            "protocols": list(PROTOCOLS),
            "speed_units": SPEEDS,
            "training_seeds": SEEDS,
            "noise_snr_db": 0.0,
            "epochs": EPOCHS,
            "delta_pp": DELTA * 100,
            "eta_pp": ETA * 100,
            "class_confidence": 0.99,
            "class_interval_method": "Bonferroni-adjusted percentile bootstrap",
            "unit_bootstrap_draws": 20_000,
            "min_nonfragile_support": MIN_NONFRAGILE_SUPPORT,
            "candidate_selection": "fixed before inspecting audit results",
        },
        "summaries": summaries,
    }
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
