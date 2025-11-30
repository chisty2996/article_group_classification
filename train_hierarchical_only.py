"""
Train only the Hierarchical Attention model (non-BERT) with optimal settings
"""

import torch
import os
from data_loader import load_20newsgroups_data
from model import HierarchicalAttentionClassifier
from train import Trainer, evaluate_model

def train_hierarchical_model():
    """Train the hierarchical attention model with optimal hyperparameters"""

    print("\n" + "=" * 80)
    print("TRAINING: Hierarchical Attention Classifier")
    print("=" * 80)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: {device} (NVIDIA CUDA GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: {device} (Apple Metal)")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device} (CPU)")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load data with NORMAL batch size (not reduced for BERT)
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_loader, val_loader, test_loader, vocab, label_names = load_20newsgroups_data(
        subset='all',
        max_vocab_size=20000,
        min_freq=2,
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15,
        batch_size=16  # Normal batch size for non-BERT model
    )

    print(f"\nDataset loaded successfully")
    print(f"  - Number of classes: {len(label_names)}")
    print(f"  - Vocabulary size: {len(vocab)}")
    print(f"  - Training batches: {len(train_loader)} (batch_size=16)")
    print(f"  - Validation batches: {len(val_loader)} (batch_size=32)")
    print(f"  - Test batches: {len(test_loader)} (batch_size=32)")

    # Initialize model with optimal hyperparameters
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)

    model = HierarchicalAttentionClassifier(
        vocab_size=len(vocab),
        num_classes=len(label_names),
        embedding_dim=200,
        hidden_dim=256,
        num_lstm_layers=2,
        num_attention_heads=4,
        filter_ratio=0.5,
        pooling_method='multi',
        dropout=0.3,
        max_sent_len=100,
        max_num_sent=30
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized:")
    print(f"  - Total parameters: {total_params:,}")

    # Initialize trainer with optimal settings
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    print("\nHyperparameters:")
    print(f"  - Learning rate: 0.0005 (reduced to prevent exploding gradients)")
    print(f"  - Weight decay: 1e-5")
    print(f"  - Batch size: 16")
    print(f"  - Gradient accumulation: 1 (disabled)")
    print(f"  - Label smoothing: 0.05 (reduced)")
    print(f"  - Gradient clipping: 5.0")
    print(f"  - Patience: 5 epochs")
    print(f"  - Max epochs: 20")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.0005,  # Reduced from 0.001
        weight_decay=1e-5,
        patience=5,
        gradient_accumulation_steps=1,  # No gradient accumulation for normal batch size
        use_amp=device.type == 'cuda',  # Use AMP only on CUDA
        label_smoothing=0.05  # Reduced from 0.1
    )

    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    train_losses, val_accuracies = trainer.train(
        num_epochs=20,
        save_path='models/hierarchical_attention_best.pt'
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Expected range: 0.75 - 0.82")

    if trainer.best_val_acc < 0.70:
        print(f"\n⚠️  WARNING: Accuracy below expected range!")
        print(f"   This may indicate a training issue.")

    # Load best model for evaluation
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    checkpoint = torch.load('models/hierarchical_attention_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    results, predictions, true_labels, logits = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        label_names=label_names,
        save_path='results/hierarchical_attention_results.json'
    )

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print("\nSaved files:")
    print("  - models/hierarchical_attention_best.pt")
    print("  - results/hierarchical_attention_results.json")

    print(f"\nFinal Test Accuracy: {results['accuracy']:.4f}")
    print(f"Final Test F1 Score: {results['f1']:.4f}")

    return model, results, train_losses, val_accuracies


if __name__ == '__main__':
    model, results, train_losses, val_accuracies = train_hierarchical_model()
