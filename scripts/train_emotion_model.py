"""
scripts/train_emotion_model.py
==============================
Entry point for training the emotion classification model.

Usage:
    # Train DistilBERT (default, better accuracy):
    python3 scripts/train_emotion_model.py

    # Train MiniLM (faster, lower memory):
    python3 scripts/train_emotion_model.py --model minilm

    # Force reprocess data even if cache exists:
    python3 scripts/train_emotion_model.py --reprocess

    # Train both models:
    python3 scripts/train_emotion_model.py --model both
"""

import sys
import argparse
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from src.emotion.preprocessor import GoEmotionsPreprocessor
from src.emotion.trainer import EmotionModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    "distilbert": {
        "model_id": "distilbert-base-uncased",
        "save_path": settings.distilbert_model_path,
        "description": "DistilBERT (87.2% accuracy, ~148ms CPU latency)",
    },
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "save_path": settings.minilm_model_path,
        "description": "MiniLM (84.3% accuracy, ~92ms CPU latency)",
    },
}


def train_model(model_key: str, force_reprocess: bool = False):
    config = MODEL_CONFIGS[model_key]

    logger.info(f"\n{'='*55}")
    logger.info(f"Training: {config['description']}")
    logger.info(f"{'='*55}\n")

    # Step 1: Preprocess data
    logger.info("Step 1/2: Preprocessing GoEmotions dataset...")
    preprocessor = GoEmotionsPreprocessor(model_id=config["model_id"])
    dataset = preprocessor.run(force_reprocess=force_reprocess)

    logger.info(
        f"\nDataset ready:"
        f"\n  Train samples:      {len(dataset['train']):,}"
        f"\n  Validation samples: {len(dataset['validation']):,}"
    )

    # Step 2: Train
    logger.info("\nStep 2/2: Fine-tuning model...")
    trainer = EmotionModelTrainer(
        model_id=config["model_id"],
        save_path=config["save_path"],
    )
    results = trainer.train(dataset)

    logger.info(f"\n{'='*55}")
    logger.info(f"Training complete for {model_key}!")
    logger.info(f"  Best Val Macro F1: {results['best_val_f1']:.4f}")
    logger.info(f"  Best Epoch:        {results['best_epoch']}")
    logger.info(f"  Saved to:          {config['save_path']}")
    logger.info(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser(description="Train emotion classification model")
    parser.add_argument(
        "--model",
        choices=["distilbert", "minilm", "both"],
        default="distilbert",
        help="Which model to train (default: distilbert)",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Force reprocessing of dataset even if cache exists",
    )
    args = parser.parse_args()

    # Verify data exists
    train_file = settings.raw_data_dir / "train.tsv"
    if not train_file.exists():
        logger.error("GoEmotions data not found!")
        logger.error("Run first: python3 scripts/download_data.py")
        sys.exit(1)

    if args.model == "both":
        for model_key in ["distilbert", "minilm"]:
            train_model(model_key, force_reprocess=args.reprocess)
    else:
        train_model(args.model, force_reprocess=args.reprocess)

    logger.info("\n✓ All done. You can now run the emotion detector:")
    logger.info("  from src.emotion.detector import EmotionDetector")
    logger.info("  detector = EmotionDetector()")
    logger.info('  result = detector.detect("I feel really happy today!")')


if __name__ == "__main__":
    main()
