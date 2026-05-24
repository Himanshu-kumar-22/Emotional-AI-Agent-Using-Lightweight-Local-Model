"""
src/emotion/trainer.py
======================
Fine-tuning loop for DistilBERT and MiniLM emotion classifiers.

Architecture:
    Base transformer (DistilBERT or MiniLM)
    └── Pooled CLS token representation [batch_size, hidden_size]
        └── Dropout (0.1)
            └── Linear layer [hidden_size → num_classes]
                └── Sigmoid activation → independent probabilities per class

Loss function: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
    - Appropriate for multi-label classification
    - Numerically stable (combines sigmoid + BCE in one operation)
    - pos_weight parameter can handle class imbalance

Optimizer: AdamW with linear warmup + decay schedule
    - AdamW adds weight decay to Adam, preventing overfitting
    - Linear warmup prevents early training instability
    - Linear decay improves final convergence

Device: Automatically uses MPS on Apple Silicon
    - Your M3 Mac will use Metal GPU acceleration
    - Gives ~3-5x speedup vs CPU for this model size

Evaluation: Macro F1-score on validation set
    - Macro = unweighted average across all classes
    - Appropriate for imbalanced multi-label datasets
    - Early stopping monitors this metric
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoModel, AutoConfig
from datasets import DatasetDict
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings

logger = logging.getLogger(__name__)


# ── Model Architecture ────────────────────────────────────────────────────────
class EmotionClassificationModel(nn.Module):
    """
    Transformer-based multi-label emotion classifier.

    We build this manually rather than using AutoModelForSequenceClassification
    because:
      1. We need sigmoid (not softmax) for multi-label classification
      2. We want explicit control over the classification head architecture
      3. We need to save/load exactly the components we care about

    Forward pass:
        input_ids, attention_mask → transformer → [CLS] pooling
        → dropout → linear → sigmoid → probability vector
    """

    def __init__(self, model_id: str, num_labels: int, dropout: float = 0.1):
        super().__init__()

        self.num_labels = num_labels

        # Load the pre-trained transformer backbone
        # We use AutoModel (not AutoModelForSequenceClassification) to get
        # raw hidden states, then add our own classification head
        self.transformer = AutoModel.from_pretrained(model_id)
        hidden_size = self.transformer.config.hidden_size

        # Classification head
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        logger.info(
            f"Model initialized: {model_id} | "
            f"hidden_size={hidden_size} | "
            f"num_labels={num_labels}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass returning raw logits (NOT sigmoid-activated).

        We return logits (not probabilities) because BCEWithLogitsLoss
        combines sigmoid + BCE numerically stably. During inference,
        we apply sigmoid separately.

        Args:
            input_ids: [batch_size, seq_len] token indices
            attention_mask: [batch_size, seq_len] 1=real token, 0=padding

        Returns:
            logits: [batch_size, num_labels] raw (pre-sigmoid) scores
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # CLS token is the first token of each sequence
        # It's a learned representation of the whole sequence
        # Shape: [batch_size, hidden_size]
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout for regularization
        cls_output = self.dropout(cls_output)

        # Project to num_labels dimensions
        logits = self.classifier(cls_output)

        return logits


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute classification metrics for multi-label evaluation.

    Args:
        logits: [n_samples, n_classes] raw model outputs (pre-sigmoid)
        labels: [n_samples, n_classes] ground truth multi-hot vectors
        threshold: sigmoid probability threshold for positive prediction

    Returns:
        Dict with accuracy, macro_f1, weighted_f1
    """
    # Apply sigmoid to convert logits → probabilities
    probs = 1 / (1 + np.exp(-logits))

    # Threshold probabilities to get binary predictions
    predictions = (probs >= threshold).astype(int)

    # Macro F1: unweighted average across all classes
    # Use zero_division=0 to handle classes with no predictions gracefully
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    # Subset accuracy: fraction of samples where ALL labels are correct
    # This is stricter than per-class accuracy
    subset_accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": float(subset_accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }


# ── Trainer ───────────────────────────────────────────────────────────────────
class EmotionModelTrainer:
    """
    Manages the complete training lifecycle for an emotion classifier.

    Usage:
        trainer = EmotionModelTrainer(
            model_id="distilbert-base-uncased",
            save_path=settings.distilbert_model_path
        )
        trainer.train(dataset)
    """

    def __init__(
        self,
        model_id: str,
        save_path: Path,
        num_labels: int = None,
    ):
        self.model_id = model_id
        self.save_path = save_path
        self.num_labels = num_labels or settings.num_emotion_classes
        self.device = torch.device(settings.device)

        logger.info(f"Trainer initialized")
        logger.info(f"Model: {model_id}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Save path: {save_path}")

    def _build_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Build a DataLoader with proper collation for our dataset format."""

        def collate_fn(batch):
            # Defensive handling — dataset items may be tensors or plain lists
            def to_tensor_long(x):
                if isinstance(x, torch.Tensor):
                    return x.long()
                return torch.tensor(x, dtype=torch.long)

            def to_tensor_float(x):
                if isinstance(x, torch.Tensor):
                    return x.float()
                return torch.tensor(x, dtype=torch.float32)

            return {
                "input_ids": torch.stack(
                    [to_tensor_long(item["input_ids"]) for item in batch]
                ),
                "attention_mask": torch.stack(
                    [to_tensor_long(item["attention_mask"]) for item in batch]
                ),
                "labels": torch.stack(
                    [to_tensor_float(item["labels"]) for item in batch]
                ),
            }

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
        )

    def _calculate_pos_weight(self, dataset) -> torch.Tensor:
        """
        Calculate positive class weights for handling class imbalance.

        BCEWithLogitsLoss supports pos_weight: a tensor of shape [num_classes]
        where pos_weight[i] = negative_count[i] / positive_count[i].

        This tells the loss function to penalize missing positive examples
        more heavily for rare classes (like fear, disgust) than for common
        ones (like neutral).
        """
        all_labels = np.array([sample["labels"] for sample in dataset])

        n_samples = len(all_labels)
        n_positive = all_labels.sum(axis=0)  # Sum per class
        n_negative = n_samples - n_positive

        # Avoid division by zero for classes with no positive samples
        pos_weight = np.where(n_positive > 0, n_negative / (n_positive + 1e-8), 1.0)

        logger.info("Class imbalance weights:")
        for i, (label, weight) in enumerate(zip(settings.emotion_labels, pos_weight)):
            logger.info(f"  {label:10s}: {weight:.2f}x")

        return torch.tensor(pos_weight, dtype=torch.float32)

    def train(self, dataset: DatasetDict) -> dict[str, float]:
        """
        Execute the complete training loop.

        Returns:
            Dict of final validation metrics
        """
        logger.info("\n" + "=" * 55)
        logger.info(f"Starting training: {self.model_id}")
        logger.info("=" * 55)

        # ── Build model ───────────────────────────────────────────────────
        model = EmotionClassificationModel(
            model_id=self.model_id,
            num_labels=self.num_labels,
        ).to(self.device)

        # ── Build data loaders ────────────────────────────────────────────
        train_loader = self._build_dataloader(
            dataset["train"],
            batch_size=settings.train_batch_size,
            shuffle=True,
        )
        val_loader = self._build_dataloader(
            dataset["validation"],
            batch_size=settings.train_batch_size * 2,  # Larger for validation
            shuffle=False,
        )

        # ── Loss function ─────────────────────────────────────────────────
        pos_weight = self._calculate_pos_weight(dataset["train"]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # ── Optimizer ─────────────────────────────────────────────────────
        # Separate learning rates for transformer backbone vs classifier head
        # The pre-trained backbone needs careful fine-tuning (small LR)
        # The new classifier head needs faster learning (larger LR)
        optimizer = AdamW(
            [
                {
                    "params": model.transformer.parameters(),
                    "lr": settings.train_learning_rate,
                },
                {
                    "params": model.classifier.parameters(),
                    "lr": settings.train_learning_rate * 10,
                },
            ],
            weight_decay=settings.train_weight_decay,
        )

        # ── Learning rate scheduler ───────────────────────────────────────
        total_steps = len(train_loader) * settings.train_epochs
        warmup_steps = int(total_steps * 0.1)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        # ── Training loop ─────────────────────────────────────────────────
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        training_history = []

        for epoch in range(1, settings.train_epochs + 1):
            # ── Training phase ─────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{settings.train_epochs} [Train]",
                ncols=80,
            )

            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Zero gradients from previous step
                optimizer.zero_grad()

                # Forward pass
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # Compute loss
                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()

                # Gradient clipping prevents exploding gradients
                # Max norm of 1.0 is standard for transformer fine-tuning
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Optimizer and scheduler step
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

                # Update progress bar with current loss
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = epoch_loss / n_batches

            # ── Validation phase ────────────────────────────────────────
            model.eval()
            all_logits = []
            all_labels = []
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch}/{settings.train_epochs} [Valid]",
                    ncols=80,
                    leave=True,
                ):
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)

                    val_loss += loss.item()
                    n_val_batches += 1

                    # Move to CPU for metric computation
                    all_logits.append(logits.detach().cpu().float().numpy())
                    all_labels.append(labels.detach().cpu().float().numpy())

            avg_val_loss = val_loss / n_val_batches
            all_logits = np.vstack(all_logits)
            all_labels = np.vstack(all_labels)

            metrics = compute_metrics(all_logits, all_labels)

            epoch_record = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                **metrics,
            }
            training_history.append(epoch_record)

            logger.info(
                f"\nEpoch {epoch} Results:"
                f"\n  Train Loss:  {avg_train_loss:.4f}"
                f"\n  Val Loss:    {avg_val_loss:.4f}"
                f"\n  Accuracy:    {metrics['accuracy']:.4f}"
                f"\n  Macro F1:    {metrics['macro_f1']:.4f}"
                f"\n  Weighted F1: {metrics['weighted_f1']:.4f}"
            )

            # ── Early stopping and model saving ─────────────────────────
            if metrics["macro_f1"] > best_val_f1:
                best_val_f1 = metrics["macro_f1"]
                best_epoch = epoch
                patience_counter = 0

                # Save the best model checkpoint
                self._save_checkpoint(model, metrics, epoch, training_history)
                logger.info(f"  ✓ New best model saved (F1={best_val_f1:.4f})")
            else:
                patience_counter += 1
                logger.info(
                    f"  No improvement. Patience: "
                    f"{patience_counter}/{settings.early_stopping_patience}"
                )

                if patience_counter >= settings.early_stopping_patience:
                    logger.info(
                        f"\nEarly stopping at epoch {epoch}. "
                        f"Best was epoch {best_epoch} (F1={best_val_f1:.4f})"
                    )
                    break

        logger.info(f"\nTraining complete!")
        logger.info(f"Best validation Macro F1: {best_val_f1:.4f} (epoch {best_epoch})")
        logger.info(f"Model saved to: {self.save_path}")

        return {"best_val_f1": best_val_f1, "best_epoch": best_epoch}

    def _save_checkpoint(
        self,
        model: EmotionClassificationModel,
        metrics: dict,
        epoch: int,
        history: list,
    ):
        """
        Save the model checkpoint in HuggingFace-compatible format.

        Saves:
          - The transformer backbone config (config.json)
          - The classifier head weights (emotion_classifier_head.pt)
          - The complete model state dict (pytorch_model.bin)
          - Training metadata (training_info.json)

        The tokenizer is saved separately (it doesn't change during training).
        """
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Save transformer config (needed to reconstruct model architecture)
        model.transformer.config.save_pretrained(str(self.save_path))

        # Save complete model state dict
        # We save the entire model (transformer + classifier head) as one file
        torch.save(
            model.state_dict(),
            self.save_path / "pytorch_model.bin",
        )

        # Save tokenizer (needs to travel with the model)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.save_pretrained(str(self.save_path))

        # Save training metadata
        import json

        metadata = {
            "model_id": self.model_id,
            "num_labels": self.num_labels,
            "emotion_labels": settings.emotion_labels,
            "best_epoch": epoch,
            "best_metrics": metrics,
            "training_history": history,
            "hyperparameters": {
                "learning_rate": settings.train_learning_rate,
                "batch_size": settings.train_batch_size,
                "max_length": settings.train_max_length,
                "weight_decay": settings.train_weight_decay,
                "device": settings.device,
            },
        }

        with open(self.save_path / "training_info.json", "w") as f:
            json.dump(metadata, f, indent=2)
