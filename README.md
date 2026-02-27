# Early Autism Detection Using Repetitive Movement (VideoMAE)

Colab-ready training and inference pipeline for classifying repetitive behaviors into:

- `armflapping`
- `spinning`
- `headbanging`

## Files

- `train_videomae_colab.py` — full training + evaluation + output export.
- `inference.py` — `predict_video("test.mp4")` inference helper.
- `requirements.txt` — dependencies.

## Dataset structure

```text
dataset/
 ├── armflapping/
 ├── spinning/
 └── headbanging/
```

## Colab quick start

```bash
pip install -r requirements.txt
python train_videomae_colab.py
```

Before running, set `Config.dataset_dir` and `Config.output_dir` in `train_videomae_colab.py` to your Google Drive paths.

## What gets exported (output folder)

The script explicitly exports:

- `model.pth`
- `training_log.csv`
- `classification_report.txt`
- `confusion_matrix.png`
- `run_metadata.json`
- `model.onnx` (best effort)
- `results.zip` (zip bundle of outputs)

At the end of training, it prints the absolute output file paths.

## Inference

```python
from inference import predict_video
predict_video("test.mp4", model_path="/path/to/model.pth", metadata_path="/path/to/run_metadata.json")
```

## Optional GitHub push

```bash
git add .
git commit -m "Update VideoMAE training pipeline"
git push
```
