"""Colab-friendly training pipeline for Early Autism Detection Using Repetitive Movement.

Preferred model: VideoMAE (`MCG-NJU/videomae-base`).
Fallback model: lightweight 3D CNN.
"""

from __future__ import annotations

import csv
import json
import random
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from google.colab import drive  # type: ignore
except Exception:
    drive = None

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

from transformers import AutoImageProcessor, VideoMAEForVideoClassification

SEED = 42


@dataclass
class Config:
    dataset_dir: str = "/content/drive/MyDrive/dataset"
    output_dir: str = "/content/drive/MyDrive/autism_outputs"
    model_name: str = "MCG-NJU/videomae-base"
    num_frames: int = 16
    frame_size: int = 224
    train_split: float = 0.8
    batch_size: int = 4
    num_workers: int = 2
    lr: float = 1e-4
    num_epochs: int = 12
    freeze_backbone: bool = True
    onnx_export: bool = True


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mount_drive_if_available() -> None:
    if drive is not None:
        drive.mount("/content/drive", force_remount=False)


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def safe_sample_indices(total_frames: int, needed: int) -> Optional[np.ndarray]:
    if total_frames < needed:
        return None
    return np.linspace(0, total_frames - 1, needed).astype(int)


def read_frames_decord(path: str, num_frames: int) -> Optional[np.ndarray]:
    if VideoReader is None:
        return None
    try:
        vr = VideoReader(path, ctx=cpu(0))
        idx = safe_sample_indices(len(vr), num_frames)
        if idx is None:
            return None
        return vr.get_batch(idx).asnumpy()
    except Exception:
        return None


def read_frames_opencv(path: str, num_frames: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = safe_sample_indices(total_frames, num_frames)
        if idx is None:
            return None
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                return None
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.asarray(frames)
    finally:
        cap.release()


def load_video_frames(path: str, num_frames: int) -> Optional[np.ndarray]:
    frames = read_frames_decord(path, num_frames)
    if frames is None:
        frames = read_frames_opencv(path, num_frames)
    return frames


def collect_video_samples(dataset_dir: str) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_names = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: List[Tuple[str, int]] = []
    valid_ext = {".mp4", ".avi", ".mov", ".mkv"}
    for class_name in class_names:
        for p in (root / class_name).iterdir():
            if p.is_file() and p.suffix.lower() in valid_ext:
                samples.append((str(p), class_to_idx[class_name]))

    random.shuffle(samples)
    print(f"Discovered {len(samples)} videos across {len(class_names)} classes")
    return samples, class_to_idx


def filter_valid_samples(samples: List[Tuple[str, int]], num_frames: int) -> List[Tuple[str, int]]:
    valid: List[Tuple[str, int]] = []
    skipped = 0
    for path, label in tqdm(samples, desc="Validating videos"):
        if load_video_frames(path, num_frames) is not None:
            valid.append((path, label))
        else:
            skipped += 1
    print(f"Valid videos: {len(valid)} | Skipped (corrupt/short): {skipped}")
    return valid


class VideoDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], processor: AutoImageProcessor, num_frames: int, frame_size: int):
        self.samples = samples
        self.processor = processor
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path, label = self.samples[index]
        frames = load_video_frames(path, self.num_frames)
        if frames is None:
            raise ValueError(f"Invalid sample encountered after filtering: {path}")

        resized = [cv2.resize(f, (self.frame_size, self.frame_size)) for f in frames]
        encoded = self.processor(resized, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class Tiny3DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, pixel_values, labels=None):
        x = pixel_values.permute(0, 2, 1, 3, 4)
        logits = self.classifier(self.features(x).flatten(1))
        loss = nn.CrossEntropyLoss()(logits, labels) if labels is not None else None
        return type("Output", (), {"loss": loss, "logits": logits})


def build_model(class_names: List[str], model_name: str, freeze_backbone: bool):
    num_classes = len(class_names)
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {v: k for k, v in id2label.items()}
    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        print("Loaded VideoMAE model")
        return model, processor, "videomae"
    except Exception as exc:
        warnings.warn(f"VideoMAE failed to load ({exc}); using Tiny3DCNN fallback.")
        return Tiny3DCNN(num_classes), processor, "tiny3dcnn"


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    losses, preds, targets = [], [], []
    for batch in tqdm(loader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(**batch)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.item()))
        preds.extend(out.logits.argmax(-1).detach().cpu().numpy())
        targets.extend(batch["labels"].detach().cpu().numpy())
    return float(np.mean(losses)), accuracy_score(targets, preds)


def evaluate(model, loader, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(float(out.loss.item()))
            preds.extend(out.logits.argmax(-1).cpu().numpy())
            targets.extend(batch["labels"].cpu().numpy())
    return float(np.mean(losses)), accuracy_score(targets, preds), preds, targets


def save_training_log(path: Path, rows: List[Dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path: Path, y_true: List[int], y_pred: List[int], labels: List[str]) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def export_onnx_if_requested(model, output_path: Path, cfg: Config):
    if not cfg.onnx_export:
        return
    try:
        model.eval()
        dummy = torch.randn(1, cfg.num_frames, 3, cfg.frame_size, cfg.frame_size)
        if isinstance(model, Tiny3DCNN):
            torch.onnx.export(
                model.cpu(), (dummy,), str(output_path), input_names=["pixel_values"], output_names=["logits"], opset_version=17
            )
        else:
            torch.onnx.export(
                model.cpu(), (dummy,), str(output_path), input_names=["pixel_values"], output_names=["logits"], opset_version=17
            )
        print(f"ONNX exported: {output_path}")
    except Exception as exc:
        warnings.warn(f"ONNX export failed: {exc}")


def zip_results(output_dir: Path) -> Path:
    zip_path = output_dir / "results.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.iterdir():
            if p.is_file() and p.name != zip_path.name:
                zf.write(p, arcname=p.name)
    return zip_path


def write_run_metadata(output_dir: Path, cfg: Config, class_names: List[str], backend: str) -> None:
    payload = {
        "config": asdict(cfg),
        "class_names": class_names,
        "backend": backend,
    }
    with (output_dir / "run_metadata.json").open("w") as f:
        json.dump(payload, f, indent=2)


def main(cfg: Config):
    set_seed()
    mount_drive_if_available()
    device = get_device()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples, class_to_idx = collect_video_samples(cfg.dataset_dir)
    valid_samples = filter_valid_samples(all_samples, cfg.num_frames)
    if len(valid_samples) < 2:
        raise RuntimeError("Not enough valid videos to train.")

    labels = [label for _, label in valid_samples]
    train_samples, val_samples = train_test_split(
        valid_samples,
        train_size=cfg.train_split,
        random_state=SEED,
        stratify=labels if len(set(labels)) > 1 else None,
    )

    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    model, processor, backend = build_model(class_names, cfg.model_name, cfg.freeze_backbone)

    train_ds = VideoDataset(train_samples, processor, cfg.num_frames, cfg.frame_size)
    val_ds = VideoDataset(val_samples, processor, cfg.num_frames, cfg.frame_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = model.to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    logs = []
    best_val_acc = -1.0
    best_preds: List[int] = []
    best_targets: List[int] = []

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc, preds, targets = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{cfg.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        logs.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_preds, best_targets = preds, targets
            torch.save(model.state_dict(), output_dir / "model.pth")

    save_training_log(output_dir / "training_log.csv", logs)
    save_confusion_matrix(output_dir / "confusion_matrix.png", best_targets, best_preds, class_names)

    precision, recall, f1, _ = precision_recall_fscore_support(best_targets, best_preds, average="weighted", zero_division=0)
    acc = accuracy_score(best_targets, best_preds)
    report = classification_report(best_targets, best_preds, target_names=class_names, zero_division=0)

    with (output_dir / "classification_report.txt").open("w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write(report)

    write_run_metadata(output_dir, cfg, class_names, backend)
    export_onnx_if_requested(model, output_dir / "model.onnx", cfg)
    zip_file = zip_results(output_dir)

    print("\n=== Exported outputs ===")
    for name in [
        "model.pth",
        "training_log.csv",
        "confusion_matrix.png",
        "classification_report.txt",
        "run_metadata.json",
        "model.onnx",
        zip_file.name,
    ]:
        p = output_dir / name
        if p.exists():
            print(f"- {p}")


if __name__ == "__main__":
    main(Config())
