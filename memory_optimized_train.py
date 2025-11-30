"""
Memory-optimized training script for BERT model
Provides multiple memory optimization strategies
"""

import torch
import os
from data_loader import load_20newsgroups_data
from model import HierarchicalAttentionBERT
from train import Trainer

def memory_optimized_training():
    """
    Run training with memory optimizations for BERT model

    Applied optimizations:
    1. Reduced batch size (2 for training, 4 for validation)
    2. Gradient accumulation (8 steps) to simulate larger batches
    3. Mixed precision training (FP16) to reduce memory by ~50%
    4. Gradient checkpointing in BERT
    5. Frozen BERT layers (only last 2 layers trainable)
    6. Periodic CUDA cache clearing
    """

    print("\n" + "=" * 80)
    print("MEMORY-OPTIMIZED BERT TRAINING")
    print("=" * 80)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        # Print GPU memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.2f} GB")

        # Clear cache before starting
        torch.cuda.empty_cache()
        print("CUDA cache cleared")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device}")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load data with memory-optimized settings
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_loader, val_loader, test_loader, vocab, label_names = load_20newsgroups_data(
        subset='all',
        max_vocab_size=20000,
        min_freq=2,
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15
    )

    print(f"\nDataset loaded:")
    print(f"  - Training batches: {len(train_loader)} (batch_size=2)")
    print(f"  - Validation batches: {len(val_loader)} (batch_size=4)")
    print(f"  - Test batches: {len(test_loader)} (batch_size=4)")
    print(f"  - Effective batch size: 2 × 8 (grad accum) = 16")

    # Initialize BERT model with memory optimizations
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)

    model = HierarchicalAttentionBERT(
        num_classes=len(label_names),
        hidden_dim=256,
        num_attention_heads=4,
        filter_ratio=0.5,
        pooling_method='multi',
        dropout=0.3,
        max_sent_len=100,
        max_num_sent=30,
        freeze_bert_layers=True  # Freeze most BERT layers
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen parameters: {total_params - trainable_params:,}")
    print(f"  - Memory reduction: {(1 - trainable_params/total_params)*100:.1f}%")

    # Initialize trainer with memory optimizations
    print("\n" + "=" * 80)
    print("TRAINING CONFIGURATION")
    print("=" * 80)

    print("\nMemory optimizations enabled:")
    print("  ✓ Gradient accumulation (8 steps)")
    print("  ✓ Mixed precision training (FP16)")
    print("  ✓ Gradient checkpointing")
    print("  ✓ Frozen BERT layers (10/12)")
    print("  ✓ Reduced batch size (2)")
    print("  ✓ Periodic cache clearing")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5,
        patience=5,
        gradient_accumulation_steps=8,  # Simulate batch_size=16
        use_amp=True  # Mixed precision training
    )

    # Train the model
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    train_losses, val_accuracies = trainer.train(
        num_epochs=20,
        save_path='models/hierarchical_attention_bert_optimized.pt'
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"\nBest validation accuracy: {trainer.best_val_acc:.4f}")
    print(f"Model saved to: models/hierarchical_attention_bert_optimized.pt")

    # Print final GPU memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        memory_reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"\nGPU Memory Usage:")
        print(f"  - Allocated: {memory_allocated:.2f} GB")
        print(f"  - Reserved: {memory_reserved:.2f} GB")

    return trainer, train_losses, val_accuracies


if __name__ == '__main__':
    trainer, train_losses, val_accuracies = memory_optimized_training()
