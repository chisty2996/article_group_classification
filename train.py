"""
Training script for the hierarchical attention model and baselines
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import os


class Trainer:
    """Trainer class for model training and evaluation"""

    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001,
                 weight_decay=1e-5, patience=5, gradient_accumulation_steps=8, use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp and device.type == 'cuda'

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2
        )

        # Mixed precision training
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
            print("Using Automatic Mixed Precision (AMP) training")

        self.best_val_acc = 0
        self.patience_counter = 0
        self.train_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """Train for one epoch with gradient accumulation and mixed precision"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc="Training")
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            sentence_masks = batch['sentence_masks'].to(self.device)
            labels = batch['labels'].to(self.device)

            if self.use_amp:
                # Mixed precision training
                from torch.cuda.amp import autocast
                with autocast():
                    logits = self.model(input_ids, sentence_masks=sentence_masks)
                    loss = self.criterion(logits, labels) / self.gradient_accumulation_steps

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Update weights after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    # Clear CUDA cache periodically
                    if (batch_idx + 1) % 50 == 0:
                        torch.cuda.empty_cache()
            else:
                # Standard training
                logits = self.model(input_ids, sentence_masks=sentence_masks)
                loss = self.criterion(logits, labels) / self.gradient_accumulation_steps
                loss.backward()

                # Update weights after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps

            # Track predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

        # Handle remaining gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                sentence_masks = batch['sentence_masks'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(input_ids, sentence_masks=sentence_masks)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()

                # Track predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy, all_preds, all_labels

    def train(self, num_epochs, save_path='best_model.pt'):
        """Train the model for multiple epochs"""

        print(f"Training on device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc, _, _ = self.validate()
            self.val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            # Early stopping and model saving
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                }, save_path)
                print(f"Model saved with validation accuracy: {val_acc:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        print(f"\nTraining completed. Best validation accuracy: {self.best_val_acc:.4f}")
        return self.train_losses, self.val_accuracies


def evaluate_model(model, test_loader, device, label_names, save_path='evaluation_results.json'):
    """
    Comprehensive evaluation of the model

    Returns:
        metrics: dictionary containing accuracy, precision, recall, f1
        predictions: array of predictions
        true_labels: array of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    print("\nEvaluating model...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            sentence_masks = batch['sentence_masks'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, sentence_masks=sentence_masks)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    print(f"\nOverall Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=label_names, zero_division=0))

    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class_metrics': {
            label_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            } for i in range(len(label_names))
        }
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {save_path}")

    return results, all_preds, all_labels, np.array(all_logits)


def compare_models(results_dict, label_names, save_path='comparison.json'):
    """
    Compare multiple models

    Args:
        results_dict: dictionary mapping model names to their results
    """
    comparison = {}

    for model_name, results in results_dict.items():
        comparison[model_name] = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1']
        }

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)

    for model_name, metrics in comparison.items():
        print(f"{model_name:<30} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")

    print("=" * 80)

    # Save comparison
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    return comparison


def get_misclassified_examples(predictions, true_labels, test_loader, label_names, num_examples=5):
    """
    Extract misclassified examples for error analysis

    Returns:
        misclassified: list of dictionaries containing misclassified examples
    """
    misclassified = []

    # Get indices of misclassified samples
    misclassified_indices = np.where(predictions != true_labels)[0]

    print(f"\nTotal misclassified: {len(misclassified_indices)} / {len(predictions)}")

    # Organize by class
    class_errors = {}
    for idx in misclassified_indices:
        true_label = true_labels[idx]
        pred_label = predictions[idx]

        key = (true_label, pred_label)
        if key not in class_errors:
            class_errors[key] = []
        class_errors[key].append(idx)

    # Print confusion pairs
    print("\nMost common confusion pairs:")
    sorted_pairs = sorted(class_errors.items(), key=lambda x: len(x[1]), reverse=True)

    for (true_label, pred_label), indices in sorted_pairs[:10]:
        print(f"{label_names[true_label]} -> {label_names[pred_label]}: {len(indices)} errors")

    return class_errors, misclassified_indices
