"""Inference utility for repetitive movement classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, VideoMAEForVideoClassification

DEFAULT_CLASSES = ["armflapping", "headbanging", "spinning"]


def safe_sample_indices(total_frames: int, needed: int) -> Optional[np.ndarray]:
    if total_frames < needed:
        return None
    return np.linspace(0, total_frames - 1, needed).astype(int)


def extract_frames(video_path: str, num_frames: int = 16, frame_size: int = 224) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = safe_sample_indices(total, num_frames)
        if idx is None:
            raise ValueError(f"Video too short (<{num_frames} frames): {video_path}")
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Corrupt frame in video: {video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (frame_size, frame_size)))
        return frames
    finally:
        cap.release()


def load_metadata(metadata_path: Optional[str]):
    if metadata_path is None:
        return {"class_names": DEFAULT_CLASSES, "config": {"num_frames": 16, "frame_size": 224}}
    p = Path(metadata_path)
    if not p.exists():
        return {"class_names": DEFAULT_CLASSES, "config": {"num_frames": 16, "frame_size": 224}}
    with p.open() as f:
        return json.load(f)


def load_model(model_path: str, class_names: List[str], model_name: str = "MCG-NJU/videomae-base"):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_name,
        num_labels=len(class_names),
        ignore_mismatched_sizes=True,
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, processor


def predict_video(video_path: str, model_path: str = "model.pth", metadata_path: Optional[str] = "run_metadata.json") -> Dict[str, float]:
    metadata = load_metadata(metadata_path)
    class_names = metadata.get("class_names", DEFAULT_CLASSES)
    cfg = metadata.get("config", {})
    num_frames = int(cfg.get("num_frames", 16))
    frame_size = int(cfg.get("frame_size", 224))

    model, processor = load_model(model_path, class_names)
    frames = extract_frames(video_path, num_frames=num_frames, frame_size=frame_size)
    inputs = processor(frames, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    result = {"predicted_class": class_names[pred_idx], "confidence": confidence}

    print(f"Predicted behavior: {result['predicted_class']}")
    print(f"Confidence score: {result['confidence']:.4f}")
    return result


if __name__ == "__main__":
    if Path("test.mp4").exists() and Path("model.pth").exists():
        predict_video("test.mp4", "model.pth", "run_metadata.json")
    else:
        print("Place test.mp4/model.pth (and optional run_metadata.json) in current directory.")
