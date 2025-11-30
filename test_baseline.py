"""
Quick test of baseline models to verify data pipeline is correct
"""

import torch
from data_loader import load_20newsgroups_data
from model import BaselineLSTM, BaselineCNN
from train import Trainer

def test_baselines():
    """Test baseline models to verify data/training works"""

    print("\n" + "=" * 80)
    print("TESTING BASELINE MODELS")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, vocab, label_names = load_20newsgroups_data(
        subset='all',
        max_vocab_size=20000,
        min_freq=2,
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15,
        batch_size=16
    )

    print(f"Vocab size: {len(vocab)}")
    print(f"Num classes: {len(label_names)}")

    # Test Baseline LSTM
    print("\n" + "=" * 80)
    print("1. BASELINE LSTM")
    print("=" * 80)

    lstm_model = BaselineLSTM(
        vocab_size=len(vocab),
        num_classes=len(label_names),
        embedding_dim=200,
        hidden_dim=256,
        dropout=0.3
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    lstm_trainer = Trainer(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5,
        patience=3,
        gradient_accumulation_steps=1,
        use_amp=False,
        label_smoothing=0.0
    )

    print("\nTraining for 3 epochs...")
    train_losses, val_accs = lstm_trainer.train(
        num_epochs=3,
        save_path='models/baseline_lstm_test.pt'
    )

    print(f"\nBaseline LSTM Results:")
    print(f"  Final val acc: {lstm_trainer.best_val_acc:.4f}")
    print(f"  Expected: > 0.60 (at least 60% after 3 epochs)")

    if lstm_trainer.best_val_acc < 0.50:
        print(f"  ⚠️  LSTM FAILED - Data pipeline or training issue!")
        return False

    # Test Baseline CNN
    print("\n" + "=" * 80)
    print("2. BASELINE CNN")
    print("=" * 80)

    cnn_model = BaselineCNN(
        vocab_size=len(vocab),
        num_classes=len(label_names),
        embedding_dim=200,
        num_filters=128,
        dropout=0.3
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")

    cnn_trainer = Trainer(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.001,
        weight_decay=1e-5,
        patience=3,
        gradient_accumulation_steps=1,
        use_amp=False,
        label_smoothing=0.0
    )

    print("\nTraining for 3 epochs...")
    train_losses, val_accs = cnn_trainer.train(
        num_epochs=3,
        save_path='models/baseline_cnn_test.pt'
    )

    print(f"\nBaseline CNN Results:")
    print(f"  Final val acc: {cnn_trainer.best_val_acc:.4f}")
    print(f"  Expected: > 0.55 (at least 55% after 3 epochs)")

    if cnn_trainer.best_val_acc < 0.45:
        print(f"  ⚠️  CNN FAILED - Data pipeline or training issue!")
        return False

    print("\n" + "=" * 80)
    print("BASELINE TEST COMPLETE")
    print("=" * 80)

    if lstm_trainer.best_val_acc > 0.50 and cnn_trainer.best_val_acc > 0.45:
        print("\n✅ Baselines work - Issue is in Hierarchical Attention architecture")
    else:
        print("\n❌ Baselines failed - Issue is in data pipeline or training setup")

    return True

if __name__ == '__main__':
    test_baselines()
