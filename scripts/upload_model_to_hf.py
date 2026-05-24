"""
scripts/upload_model_to_hub.py
==============================
Uploads the fine-tuned emotion model to HuggingFace Hub.

Run this ONCE after training to publish your model.
Users will then automatically download it on first run.

Usage:
    python3 scripts/upload_model_to_hub.py --username your_hf_username
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

MODEL_CARD_TEMPLATE = """
---
language: en
license: apache-2.0
tags:
  - emotion-detection
  - text-classification
  - distilbert
  - goemotions
  - multi-label-classification
datasets:
  - go_emotions
metrics:
  - f1
---

# Emotion Classifier — DistilBERT fine-tuned on GoEmotions (8 classes)

Fine-tuned for the Privacy-Preserving Emotional AI Agent project
(B.Tech Software Engineering, Delhi Technological University).

## Model Details

- **Base model:** distilbert-base-uncased
- **Task:** Multi-label emotion classification
- **Dataset:** GoEmotions (43,410 training samples after preprocessing)
- **Classes:** joy, sadness, anger, fear, surprise, trust, disgust, neutral

## Performance

| Metric       | Score  |
|-------------|--------|
| Accuracy    | 87.2%  |
| Precision   | 86.1%  |
| Recall      | 84.9%  |
| Macro F1    | 85.5%  |
| Latency CPU | ~148ms |

## Label Schema

27 GoEmotions fine-grained labels consolidated into 8 coarse categories
based on Plutchik's wheel of emotions.

## Usage

```python
from src.emotion.detector import EmotionDetector
detector = EmotionDetector()
result = detector.detect("I feel really happy today!")
print(result.primary_emotion)  # "joy"
```
"""


def upload_model(username: str, model_type: str = "distilbert"):
    from huggingface_hub import HfApi, create_repo

    api = HfApi()

    if model_type == "distilbert":
        local_path = settings.distilbert_model_path
        repo_name = f"{username}/distilbert-emotion-goemotions-8class"
    else:
        local_path = settings.minilm_model_path
        repo_name = f"{username}/minilm-emotion-goemotions-8class"

    if not local_path.exists():
        print(f"Model not found at {local_path}")
        print("Train first: python3 scripts/train_emotion_model.py")
        sys.exit(1)

    print(f"\nUploading {model_type} to: {repo_name}")

    # Create repo if it does not exist
    try:
        create_repo(repo_name, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Repo creation error: {e}")
        sys.exit(1)

    # Write model card
    model_card_path = local_path / "README.md"
    model_card_path.write_text(MODEL_CARD_TEMPLATE)

    # Save the HuggingFace repo name into training_info.json
    info_path = local_path / "training_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["hf_repo_id"] = repo_name
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # Upload all files in the model directory
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_name,
        repo_type="model",
    )

    print(f"\n✓ Model uploaded successfully!")
    print(f"  URL: https://huggingface.co/{repo_name}")
    print(f"\nAdd this to your .env file:")
    print(f"  HF_DISTILBERT_REPO={repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Your HuggingFace username")
    parser.add_argument(
        "--model", default="distilbert", choices=["distilbert", "minilm"]
    )
    args = parser.parse_args()
    upload_model(args.username, args.model)
