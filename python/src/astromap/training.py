"""Scientific training pipeline, benchmark evaluation against BLS, and model card generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from astropy.timeseries import BoxLeastSquares
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor
from torch.nn import functional
from torch.utils.data import DataLoader

from astromap.dataset import (
    CatalogManifest,
    TessTransitDataset,
    generate_curve_for_record,
    get_curated_seed_records,
)
from astromap.model import PinnTransNet, physics_loss
from astromap.pipeline import preprocess, synthetic_transit_curve


@dataclass(frozen=True)
class EvaluationMetrics:
    """Scientific evaluation metrics for transit detection."""

    loss: float
    roc_auc: float
    pr_auc: float
    fpr_at_95_recall: float
    mae_period: float
    mae_duration: float
    mae_depth: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_fpr_at_recall(labels: np.ndarray, scores: np.ndarray, target_recall: float = 0.95) -> float:
    """Calculate the False Positive Rate when True Positive Rate (Recall) reaches target_recall."""

    positive_mask = labels == 1
    negative_mask = labels == 0
    if not np.any(positive_mask) or not np.any(negative_mask):
        return 0.0

    pos_scores = scores[positive_mask]
    neg_scores = scores[negative_mask]

    threshold = float(np.percentile(pos_scores, (1.0 - target_recall) * 100))
    fpr = float(np.mean(neg_scores >= threshold))
    return fpr


def evaluate_model(
    model: PinnTransNet,
    dataset: TessTransitDataset,
    batch_size: int = 16,
) -> EvaluationMetrics:
    """Evaluate model on a dataset partition and return scientific metrics."""

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels: list[float] = []
    all_probs: list[float] = []
    all_pred_params: list[np.ndarray] = []
    all_true_params: list[np.ndarray] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in loader:
            temporal = batch["temporal"]
            scalogram = batch["scalogram"]
            labels = batch["label"]
            targets = batch["parameters"]
            stellar_radius = batch["stellar_radius"].float()
            stellar_mass = batch["stellar_mass"].float()

            logits, pred_params = model(temporal, scalogram)
            probs = torch.sigmoid(logits)

            cls_loss = functional.binary_cross_entropy_with_logits(logits, labels)
            has_pos = labels.bool()
            reg_loss = (
                functional.huber_loss(pred_params[has_pos], targets[has_pos])
                if has_pos.any()
                else torch.tensor(0.0)
            )
            phys_loss = physics_loss(pred_params, stellar_radius, stellar_mass)
            total_loss = cls_loss + 0.2 * reg_loss + 0.05 * phys_loss

            losses.append(float(total_loss.detach()))
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_pred_params.append(pred_params.cpu().numpy())
            all_true_params.append(targets.cpu().numpy())

    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    pred_params_arr = np.vstack(all_pred_params)
    true_params_arr = np.vstack(all_true_params)

    # Classification metrics
    unique_labels = np.unique(labels_arr)
    if len(unique_labels) > 1:
        roc_auc = float(roc_auc_score(labels_arr, probs_arr))
        pr_auc = float(average_precision_score(labels_arr, probs_arr))
        fpr95 = calculate_fpr_at_recall(labels_arr, probs_arr, target_recall=0.95)
    else:
        roc_auc = 1.0
        pr_auc = 1.0
        fpr95 = 0.0

    # Regression MAE for positive samples
    pos_idx = labels_arr == 1
    if np.any(pos_idx):
        mae_depth = float(np.mean(np.abs(pred_params_arr[pos_idx, 0] - true_params_arr[pos_idx, 0])))
        mae_duration = float(np.mean(np.abs(pred_params_arr[pos_idx, 1] - true_params_arr[pos_idx, 1])))
        mae_period = float(np.mean(np.abs(pred_params_arr[pos_idx, 2] - true_params_arr[pos_idx, 2])))
    else:
        mae_depth = 0.0
        mae_duration = 0.0
        mae_period = 0.0

    return EvaluationMetrics(
        loss=float(np.mean(losses)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        fpr_at_95_recall=fpr95,
        mae_period=mae_period,
        mae_duration=mae_duration,
        mae_depth=mae_depth,
    )


def evaluate_bls_baseline(records: list[Any]) -> EvaluationMetrics:
    """Run classical Astropy Box Least Squares on the records to establish the official benchmark."""

    labels: list[int] = []
    scores: list[float] = []
    period_errors: list[float] = []
    duration_errors: list[float] = []
    depth_errors: list[float] = []

    for record in records:
        curve = generate_curve_for_record(record)
        time = curve.time[curve.mask]
        flux = curve.flux[curve.mask]
        baseline = float(time[-1] - time[0])
        max_period = min(13.0, baseline / 2.0)
        if max_period <= 0.5:
            continue

        periods = np.linspace(0.5, max_period, 500)
        durations = np.array([0.05, 0.1, 0.15, 0.2, 0.3])
        bls = BoxLeastSquares(time, flux).power(periods, durations)
        best_idx = int(np.nanargmax(bls.power))
        period_est = float(bls.period[best_idx])
        duration_est = float(bls.duration[best_idx])
        depth_est = max(float(bls.depth[best_idx]), 0.0)

        # Standard BLS signal-to-noise score
        point_noise = max(float(np.median(np.abs(np.diff(flux))) / 0.9539), 1e-8)
        in_transit = max(len(flux) * duration_est / period_est, 1.0)
        snr = depth_est / point_noise * np.sqrt(in_transit)
        score = float(1.0 / (1.0 + np.exp(-(snr - 7.0) / 2.0)))

        labels.append(record.label)
        scores.append(score)

        if record.label == 1:
            period_errors.append(abs(period_est - record.period_days))
            duration_errors.append(abs(duration_est - record.duration_days))
            depth_errors.append(abs(depth_est - record.depth))

    labels_arr = np.array(labels)
    scores_arr = np.array(scores)
    unique_labels = np.unique(labels_arr)
    if len(unique_labels) > 1:
        roc_auc = float(roc_auc_score(labels_arr, scores_arr))
        pr_auc = float(average_precision_score(labels_arr, scores_arr))
        fpr95 = calculate_fpr_at_recall(labels_arr, scores_arr, target_recall=0.95)
    else:
        roc_auc = 1.0
        pr_auc = 1.0
        fpr95 = 0.0

    return EvaluationMetrics(
        loss=0.0,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        fpr_at_95_recall=fpr95,
        mae_period=float(np.mean(period_errors)) if period_errors else 0.0,
        mae_duration=float(np.mean(duration_errors)) if duration_errors else 0.0,
        mae_depth=float(np.mean(depth_errors)) if depth_errors else 0.0,
    )


def train_manifest(
    manifest_path: Path,
    output_checkpoint: Path,
    model_card_path: Path | None = None,
    epochs: int = 15,
    learning_rate: float = 3e-4,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Train PinnTransNet on the versioned non-leakage catalog manifest and generate model card."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Always generate a fresh split if manifest does not exist or recreate
    manifest = CatalogManifest(get_curated_seed_records()).split_by_tic(seed=seed)
    manifest.save_json(manifest_path)

    train_records = manifest.filter_split("train")
    val_records = manifest.filter_split("validation")
    test_records = manifest.filter_split("test")

    train_dataset = TessTransitDataset(train_records, base_seed=seed)
    val_dataset = TessTransitDataset(val_records, base_seed=seed + 100)
    test_dataset = TessTransitDataset(test_records, base_seed=seed + 200)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = PinnTransNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    best_val_pr_auc = -1.0
    best_weights = model.state_dict()
    epoch_losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in train_loader:
            temporal = batch["temporal"]
            scalogram = batch["scalogram"]
            labels = batch["label"]
            targets = batch["parameters"]
            stellar_radius = batch["stellar_radius"].float()
            stellar_mass = batch["stellar_mass"].float()

            logits, pred_params = model(temporal, scalogram)
            cls_loss = functional.binary_cross_entropy_with_logits(logits, labels)
            has_pos = labels.bool()
            reg_loss = (
                functional.huber_loss(pred_params[has_pos], targets[has_pos])
                if has_pos.any()
                else torch.tensor(0.0)
            )
            phys_loss = physics_loss(pred_params, stellar_radius, stellar_mass)
            total_loss = cls_loss + 0.2 * reg_loss + 0.05 * phys_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(total_loss.detach()))

        avg_loss = float(np.mean(batch_losses))
        epoch_losses.append(avg_loss)

        # Validation check
        val_metrics = evaluate_model(model, val_dataset)
        if val_metrics.pr_auc >= best_val_pr_auc:
            best_val_pr_auc = val_metrics.pr_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    model.load_state_dict(best_weights)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, output_checkpoint)

    # Compute checksum of saved checkpoint
    with open(output_checkpoint, "rb") as f:
        checkpoint_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Final Locked Test Set Evaluation
    test_metrics = evaluate_model(model, test_dataset)
    bls_test_metrics = evaluate_bls_baseline(test_records)

    model_card = {
        "model_name": "PINN-TransNet-v1",
        "architecture": "Dual-Domain 1D CNN + 2D CWT Scalogram CNN with Physics-Informed Duration Regularization",
        "checkpoint_file": str(output_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": {
            "manifest_file": str(manifest_path),
            "manifest_checksum": manifest.compute_checksum(),
            "train_samples": len(train_records),
            "validation_samples": len(val_records),
            "test_samples": len(test_records),
            "partitioning": "Strict grouping by TIC ID (Zero target leakage)",
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "final_train_loss": epoch_losses[-1],
            "best_val_pr_auc": best_val_pr_auc,
        },
        "evaluation_on_locked_test_set": {
            "pinn_transnet": test_metrics.to_dict(),
            "astropy_bls_baseline": bls_test_metrics.to_dict(),
            "scientific_summary": {
                "pinn_pr_auc": test_metrics.pr_auc,
                "bls_pr_auc": bls_test_metrics.pr_auc,
                "pinn_roc_auc": test_metrics.roc_auc,
                "bls_roc_auc": bls_test_metrics.roc_auc,
                "pinn_fpr_at_95_recall": test_metrics.fpr_at_95_recall,
                "bls_fpr_at_95_recall": bls_test_metrics.fpr_at_95_recall,
            },
        },
    }

    if model_card_path is not None:
        model_card_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_card_path, "w", encoding="utf-8") as f:
            json.dump(model_card, f, indent=2)

    return model_card


def build_batch(seed: int, size: int = 8) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a balanced deterministic smoke-training batch."""

    temporal_inputs: list[Tensor] = []
    scalograms: list[Tensor] = []
    labels: list[float] = []
    parameters: list[list[float]] = []
    for index in range(size):
        positive = index % 2 == 0
        depth = 0.012 if positive else 1e-6
        processed = preprocess(synthetic_transit_curve(seed + index, depth=depth))
        flux = torch.from_numpy(processed.flux - 1.0).float()
        mask = torch.from_numpy(processed.mask.astype(np.float32))
        temporal_inputs.append(torch.stack((flux, mask)))
        scalograms.append(torch.from_numpy(processed.scalogram).float().unsqueeze(0))
        labels.append(float(positive))
        parameters.append([0.012, 0.16, 3.4])
    return (
        torch.stack(temporal_inputs),
        torch.stack(scalograms),
        torch.tensor(labels),
        torch.tensor(parameters),
    )


def train_smoke(output: Path, epochs: int = 2, seed: int = 42) -> list[float]:
    """Train a deterministic demonstration checkpoint and return epoch losses."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = PinnTransNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    temporal, scalogram, labels, targets = build_batch(seed)
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        logits, parameters = model(temporal, scalogram)
        classification = functional.binary_cross_entropy_with_logits(logits, labels)
        regression = functional.huber_loss(parameters[labels.bool()], targets[labels.bool()])
        physical = physics_loss(parameters, torch.ones(len(labels)), torch.ones(len(labels)))
        loss = classification + 0.2 * regression + 0.05 * physical
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AstroMap-AI scientific model with physics regularization")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pinn-transnet-v1.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--model-card", type=Path, default=Path("artifacts/model_card.json"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run fast smoke training on synthetic batch")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()

    if arguments.synthetic:
        epochs = arguments.epochs or 2
        losses = train_smoke(arguments.output, epochs=epochs, seed=arguments.seed)
        print(f"smoke_checkpoint={arguments.output} final_loss={losses[-1]:.6f}")
        return

    epochs = arguments.epochs
    lr = 3e-4
    batch_size = 8
    if arguments.config.exists():
        with open(arguments.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            t_cfg = cfg.get("training", {})
            if epochs is None:
                epochs = t_cfg.get("epochs", 15)
            lr = t_cfg.get("learning_rate", 3e-4)
            batch_size = min(t_cfg.get("batch_size", 8), 16)
    if epochs is None:
        epochs = 15

    print(f"Starting scientific training: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    card = train_manifest(
        manifest_path=arguments.manifest,
        output_checkpoint=arguments.output,
        model_card_path=arguments.model_card,
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        seed=arguments.seed,
    )
    print("\nTraining completed successfully!")
    print(f"Checkpoint saved: {card['checkpoint_file']} (SHA256: {card['checkpoint_sha256'][:16]}...)")
    print(f"Model Card saved: {arguments.model_card}")
    test_results = card["evaluation_on_locked_test_set"]["scientific_summary"]
    print(f"Test PR-AUC:  PINN={test_results['pinn_pr_auc']:.4f} vs BLS={test_results['bls_pr_auc']:.4f}")
    print(f"Test ROC-AUC: PINN={test_results['pinn_roc_auc']:.4f} vs BLS={test_results['bls_roc_auc']:.4f}")


if __name__ == "__main__":
    main()
