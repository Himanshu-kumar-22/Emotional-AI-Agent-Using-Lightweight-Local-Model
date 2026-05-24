
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
