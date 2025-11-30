"""
Train only the Hierarchical Attention BERT model
Use this if you've already trained the other models
"""

import torch
import os
from data_loader_bert import load_20newsgroups_data_bert
from model import HierarchicalAttentionBERT
from train import Trainer, evaluate_model

def train_bert_model():
    """Train only the BERT-based hierarchical attention model"""

    print("\n" + "=" * 80)
    print("TRAINING: Hierarchical Attention BERT")
    print("=" * 80)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: {device} (NVIDIA CUDA GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: {device} (Apple Metal)")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device} (CPU)")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_loader, val_loader, test_loader, tokenizer, label_names = load_20newsgroups_data_bert(
        subset='all',
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15
    )

    print(f"\nDataset loaded successfully")
    print(f"  - Number of classes: {len(label_names)}")
    print(f"  - BERT vocab size: {tokenizer.vocab_size}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # Initialize BERT model
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
        freeze_bert_layers=True
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel initialized:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen parameters: {total_params - trainable_params:,}")

    # Initialize trainer
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5,
        patience=5,
        gradient_accumulation_steps=8,
        use_amp=True
    )

    # Train
    train_losses, val_accuracies = trainer.train(
        num_epochs=20,
        save_path='models/hierarchical_attention_bert_best.pt'
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nBest validation accuracy: {trainer.best_val_acc:.4f}")

    # Load best model for evaluation
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)

    checkpoint = torch.load('models/hierarchical_attention_bert_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    results, predictions, true_labels, logits = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        label_names=label_names,
        save_path='results/hierarchical_attention_bert_results.json'
    )

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print("\nSaved files:")
    print("  - models/hierarchical_attention_bert_best.pt")
    print("  - results/hierarchical_attention_bert_results.json")

    if torch.cuda.is_available():
        print(f"\nFinal GPU memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

    return model, results, train_losses, val_accuracies


if __name__ == '__main__':
    model, results, train_losses, val_accuracies = train_bert_model()
