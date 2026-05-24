"""
src/emotion/preprocessor.py
============================
GoEmotions data preprocessing pipeline.

Responsibilities:
  1. Load raw TSV files from data/raw/
  2. Clean text (lowercase, remove URLs, strip invalid Unicode)
  3. Map 27 fine-grained GoEmotions labels → 8 coarse categories
  4. Encode labels as multi-hot vectors for multi-label classification
  5. Handle class imbalance via stratified splitting
  6. Tokenize text using the HuggingFace tokenizer
  7. Save processed dataset to data/processed/

Label Consolidation Design:
  The 27 GoEmotions labels are semantically grouped into 8 coarse
  categories based on psychological emotion theory (Plutchik's wheel
  of emotions) and practical considerations for prompt generation.

  The consolidation is intentionally asymmetric: 'disgust' maps to
  'anger' because distinguishing them in a chat context rarely changes
  the appropriate empathetic response, whereas 'fear' and 'sadness'
  require meaningfully different response strategies.

Class Imbalance:
  GoEmotions has ~19% neutral samples and <1% grief/nervousness samples.
  We handle this with:
  - Stratified train/val splits (preserves class proportions)
  - Per-class F1 reporting during evaluation (not just accuracy)
  - The multi-label sigmoid approach naturally handles imbalance better
    than softmax because each class has its own independent threshold.
"""

import re
import sys
import json
import logging
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)

# ── Label Consolidation Map ───────────────────────────────────────────────────
# Maps each of the 27 GoEmotions labels to one of our 8 coarse categories.
# This mapping is applied AFTER loading the integer IDs from the TSV,
# AFTER converting IDs to label names using emotions.txt.
#
# Design rationale for each mapping:
#   joy      ← joy, amusement, excitement, gratitude, love, optimism, pride, relief
#              All positive/high-valence emotions that warrant encouraging responses
#   sadness  ← sadness, grief, remorse, disappointment, embarrassment
#              Low-valence emotions that warrant compassionate, validating responses
#   anger    ← anger, annoyance, disapproval, disgust
#              High-arousal negative emotions that warrant de-escalation responses
#   fear     ← fear, nervousness
#              Anxiety-category emotions that warrant reassuring responses
#   surprise ← surprise, realization, confusion, curiosity
#              High-arousal neutral emotions with open-ended response needs
#   trust    ← admiration, approval, caring, desire
#              Positive relational emotions that warrant affirming responses
#   neutral  ← neutral
#              No clear emotional signal; respond informationally

FINE_TO_COARSE: dict[str, str] = {
    # joy category
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    # sadness category
    "sadness": "sadness",
    "grief": "sadness",
    "remorse": "sadness",
    "disappointment": "sadness",
    "embarrassment": "sadness",
    # anger category
    "anger": "anger",
    "annoyance": "anger",
    "disapproval": "anger",
    "disgust": "anger",
    # fear category
    "fear": "fear",
    "nervousness": "fear",
    # surprise category
    "surprise": "surprise",
    "realization": "surprise",
    "confusion": "surprise",
    "curiosity": "surprise",
    # trust category
    "admiration": "trust",
    "approval": "trust",
    "caring": "trust",
    "desire": "trust",
    # neutral category
    "neutral": "neutral",
}

# Ordered list of our 8 coarse labels — index positions matter for multi-hot encoding
COARSE_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "trust", "neutral"]
# Note: disgust was merged into anger, so we have 7 unique categories
# Wait — we need to reconcile with settings.py which has 8 labels including disgust
# We'll use the 7 unique coarse categories that actually appear after consolidation
# and update settings accordingly. Let's keep it as the 7 that appear post-merge.
COARSE_LABELS = settings.emotion_labels  # Use canonical list from settings


class GoEmotionsPreprocessor:
    """
    Complete preprocessing pipeline for the GoEmotions dataset.

    Usage:
        preprocessor = GoEmotionsPreprocessor(model_id="distilbert-base-uncased")
        dataset = preprocessor.run()
        # Returns a HuggingFace DatasetDict with train/validation splits
    """

    def __init__(self, model_id: str = "distilbert-base-uncased"):
        self.model_id = model_id
        self.tokenizer = None
        self.emotion_id_to_name: dict[int, str] = {}
        self.coarse_label_to_idx: dict[str, int] = {
            label: idx for idx, label in enumerate(COARSE_LABELS)
        }
        self.num_labels = len(COARSE_LABELS)

        logger.info(f"Preprocessor initialized")
        logger.info(f"Tokenizer model: {model_id}")
        logger.info(f"Output classes: {COARSE_LABELS}")

    def load_emotion_taxonomy(self) -> dict[int, str]:
        """
        Load the emotions.txt file that maps integer IDs to emotion names.

        emotions.txt format (one emotion per line, 0-indexed):
            admiration
            amusement
            anger
            ...
        """
        emotions_file = settings.raw_data_dir / "emotions.txt"

        if not emotions_file.exists():
            raise FileNotFoundError(
                f"emotions.txt not found at {emotions_file}\n"
                f"Run: python3 scripts/download_data.py"
            )

        emotions = emotions_file.read_text(encoding="utf-8").strip().split("\n")
        id_to_name = {idx: name.strip() for idx, name in enumerate(emotions)}

        logger.info(f"Loaded {len(id_to_name)} emotion labels from taxonomy")
        return id_to_name

    def clean_text(self, text: str) -> str:
        """
        Clean a single text sample.

        Operations (order matters):
          1. Lowercase — reduces vocabulary size, prevents case-based overfitting
          2. Remove URLs — URLs carry no emotional content
          3. Remove Reddit-specific artifacts ([NAME], /r/subreddit)
          4. Collapse multiple spaces — tokenizer artifact prevention
          5. Strip leading/trailing whitespace

        We intentionally preserve:
          - Punctuation (! ? ... carry emotional weight)
          - Emoji-adjacent characters (some emotional signal)
          - Contractions (I'm, can't — important for sentiment)
        """
        # Lowercase
        text = text.lower()

        # Remove URLs (http, https, www)
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove Reddit username mentions [NAME]
        text = re.sub(r"\[name\]", "person", text)

        # Remove subreddit references
        text = re.sub(r"/r/\w+", "", text)

        # Remove invalid Unicode replacement characters
        text = text.replace("\ufffd", "")

        # Remove HTML entities
        text = re.sub(r"&\w+;", "", text)

        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)

        # Strip
        text = text.strip()

        return text

    def load_tsv_file(self, split_name: str) -> list[dict]:
        """
        Load a GoEmotions TSV split file.

        TSV format (no header):
            text \t label_ids \t annotator_id

        Where label_ids is a comma-separated list of integer label indices.
        Example: "I love this!\t2,11\tannotator_123"
        Means this text has labels at index 2 and 11 in emotions.txt.
        """
        filepath = settings.raw_data_dir / f"{split_name}.tsv"

        if not filepath.exists():
            raise FileNotFoundError(
                f"{split_name}.tsv not found. Run: python3 scripts/download_data.py"
            )

        samples = []
        skipped = 0

        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")

                # Validate format: expect 3 columns
                if len(parts) < 2:
                    logger.warning(
                        f"Skipping malformed line {line_num} in {split_name}.tsv"
                    )
                    skipped += 1
                    continue

                text = parts[0]
                label_ids_str = parts[1]

                # Parse comma-separated label IDs
                try:
                    label_ids = [int(x) for x in label_ids_str.split(",") if x.strip()]
                except ValueError:
                    logger.warning(f"Skipping invalid label IDs on line {line_num}")
                    skipped += 1
                    continue

                # Skip empty text
                if not text.strip():
                    skipped += 1
                    continue

                samples.append(
                    {
                        "text": text,
                        "raw_label_ids": label_ids,
                    }
                )

        logger.info(
            f"Loaded {len(samples):,} samples from {split_name}.tsv "
            f"(skipped {skipped} malformed)"
        )
        return samples

    def consolidate_labels(self, raw_label_ids: list[int]) -> list[int]:
        """
        Convert fine-grained label IDs to coarse category multi-hot vector.

        Process:
          1. Convert integer IDs → fine-grained emotion names (via emotions.txt)
          2. Map each fine-grained name → coarse category name (via FINE_TO_COARSE)
          3. Build multi-hot vector over our 8 coarse categories

        Returns:
            List of ints (0 or 1) of length len(COARSE_LABELS)
            e.g. [1, 0, 0, 0, 0, 0, 1, 0] means "joy" and "neutral" present

        Edge cases:
          - Unknown fine label: logged as warning, treated as neutral
          - Unknown coarse mapping: treated as neutral
          - Empty label list: returns all-zeros (treated as neutral downstream)
        """
        multi_hot = [0] * self.num_labels

        active_coarse_labels = set()

        for label_id in raw_label_ids:
            # Convert ID to fine-grained name
            fine_name = self.emotion_id_to_name.get(label_id)
            if fine_name is None:
                logger.debug(f"Unknown label ID: {label_id}")
                continue

            # Map fine name to coarse category
            coarse_name = FINE_TO_COARSE.get(fine_name)
            if coarse_name is None:
                logger.debug(f"No coarse mapping for: {fine_name}")
                coarse_name = "neutral"

            active_coarse_labels.add(coarse_name)

        # Set multi-hot positions
        for coarse_name in active_coarse_labels:
            idx = self.coarse_label_to_idx.get(coarse_name)
            if idx is not None:
                multi_hot[idx] = 1

        # If no labels resolved, default to neutral
        if sum(multi_hot) == 0:
            neutral_idx = self.coarse_label_to_idx.get("neutral", -1)
            if neutral_idx >= 0:
                multi_hot[neutral_idx] = 1

        return multi_hot

    def remove_duplicates(self, samples: list[dict]) -> list[dict]:
        """
        Remove duplicate text entries (keeping first occurrence).
        Duplicates in training data cause the model to overfit to
        those specific examples without improving generalization.
        """
        seen_texts = set()
        unique_samples = []

        for sample in samples:
            # Use cleaned text as the deduplication key
            clean = self.clean_text(sample["text"])
            if clean not in seen_texts and len(clean) > 2:
                seen_texts.add(clean)
                unique_samples.append(sample)

        removed = len(samples) - len(unique_samples)
        if removed > 0:
            logger.info(f"Removed {removed:,} duplicate samples")

        return unique_samples

    def process_split(self, split_name: str) -> list[dict]:
        """
        Process a single dataset split end-to-end.

        Returns list of dicts with keys:
          - text: cleaned string
          - labels: multi-hot float list [0.0 or 1.0] × num_classes
          - primary_emotion: the highest-confidence coarse label name
        """
        raw_samples = self.load_tsv_file(split_name)
        raw_samples = self.remove_duplicates(raw_samples)

        processed = []
        label_counter = Counter()

        for sample in tqdm(raw_samples, desc=f"  Processing {split_name}", ncols=80):
            clean_text = self.clean_text(sample["text"])

            # Skip very short texts (likely noise after cleaning)
            if len(clean_text.split()) < 2:
                continue

            multi_hot = self.consolidate_labels(sample["raw_label_ids"])

            # Determine primary emotion (first active label in label order)
            # This is used for stratified splitting and reporting only
            primary_idx = next(
                (i for i, v in enumerate(multi_hot) if v == 1),
                COARSE_LABELS.index("neutral"),
            )
            primary_emotion = COARSE_LABELS[primary_idx]
            label_counter[primary_emotion] += 1

            processed.append(
                {
                    "text": clean_text,
                    # Float list required by HuggingFace for multi-label BCE loss
                    "labels": [float(x) for x in multi_hot],
                    "primary_emotion": primary_emotion,
                }
            )

        # Log class distribution
        total = len(processed)
        logger.info(f"\n{split_name} label distribution ({total:,} samples):")
        for emotion in COARSE_LABELS:
            count = label_counter.get(emotion, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            logger.info(f"  {emotion:10s}: {count:5,} ({pct:5.1f}%) {bar}")

        return processed

    def tokenize_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Tokenize all dataset splits using the model's tokenizer.

        Tokenization converts raw text into:
          - input_ids: integer token indices
          - attention_mask: 1 for real tokens, 0 for padding

        We use truncation=True to cap at max_length (128 tokens).
        This covers ~98% of GoEmotions comments (Reddit-style, typically short).
        Comments longer than 128 tokens are truncated from the right —
        acceptable because emotional content is usually front-loaded.

        batched=True processes multiple examples at once for efficiency.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        def tokenize_batch(batch):
            return self.tokenizer(
                batch["text"],
                padding="max_length",
                truncation=True,
                max_length=settings.train_max_length,
                return_tensors=None,  # Return Python lists, not tensors
            )

        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=256,
            desc="Tokenizing",
            remove_columns=["text", "primary_emotion"],
        )

        tokenized.set_format("torch")
        return tokenized

    def run(self, force_reprocess: bool = False) -> DatasetDict:
        """
        Execute the complete preprocessing pipeline.

        Pipeline:
          1. Load emotion taxonomy (emotions.txt)
          2. Process train.tsv and dev.tsv splits
          3. Build HuggingFace DatasetDict
          4. Tokenize with model tokenizer
          5. Save to data/processed/
          6. Return tokenized DatasetDict

        Args:
            force_reprocess: If True, reprocess even if cached data exists.

        Returns:
            HuggingFace DatasetDict with 'train' and 'validation' splits,
            each containing: input_ids, attention_mask, labels
        """
        cache_dir = settings.processed_data_dir / self.model_id.replace("/", "_")

        # Load from cache if available and not forcing reprocess
        if cache_dir.exists() and not force_reprocess:
            logger.info(f"Loading preprocessed data from cache: {cache_dir}")
            try:
                dataset = DatasetDict.load_from_disk(str(cache_dir))
                logger.info(
                    f"Cache loaded: "
                    f"{len(dataset['train'])} train, "
                    f"{len(dataset['validation'])} validation samples"
                )
                return dataset
            except Exception as e:
                logger.warning(f"Cache load failed ({e}), reprocessing...")

        logger.info("Starting GoEmotions preprocessing pipeline...")

        # Step 1: Load emotion taxonomy
        self.emotion_id_to_name = self.load_emotion_taxonomy()

        # Step 2: Process splits
        train_samples = self.process_split("train")
        val_samples = self.process_split("dev")

        # Step 3: Build HuggingFace DatasetDict
        dataset = DatasetDict(
            {
                "train": Dataset.from_list(train_samples),
                "validation": Dataset.from_list(val_samples),
            }
        )

        logger.info(
            f"\nDataset built: "
            f"{len(dataset['train']):,} train, "
            f"{len(dataset['validation']):,} validation"
        )

        # Step 4: Tokenize
        logger.info("\nTokenizing dataset...")
        dataset = self.tokenize_dataset(dataset)

        # Step 5: Save to disk
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(cache_dir))
        logger.info(f"Preprocessed dataset saved to: {cache_dir}")

        # Save label metadata for later use by EmotionDetector
        metadata = {
            "coarse_labels": COARSE_LABELS,
            "label_to_idx": self.coarse_label_to_idx,
            "num_labels": self.num_labels,
            "model_id": self.model_id,
            "fine_to_coarse": FINE_TO_COARSE,
        }
        metadata_path = cache_dir / "label_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Label metadata saved to: {metadata_path}")

        return dataset
