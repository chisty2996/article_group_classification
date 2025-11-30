"""
Main script to train and evaluate the hierarchical attention model
and baseline models on 20 Newsgroups dataset
"""

import torch
import numpy as np
import random
import os
import argparse
from data_loader import load_20newsgroups_data
from model import HierarchicalAttentionClassifier, BaselineLSTM, BaselineCNN, HierarchicalAttentionBERT
from train import Trainer, evaluate_model, compare_models, get_misclassified_examples
from visualization import (
    visualize_word_attention, visualize_cross_attention, visualize_confusion_matrix,
    plot_training_curves, plot_model_comparison, analyze_per_class_performance,
    analyze_attention_distribution
)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main(args):
    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device - prioritize CUDA (VM/Cloud GPU) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} (NVIDIA CUDA GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: {device} (Apple Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device} (CPU only)")

    # Create output directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    train_loader, val_loader, test_loader, vocab, label_names = load_20newsgroups_data(
        subset='all',
        max_vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        max_sent_len=args.max_sent_len,
        max_num_sent=args.max_num_sent,
        test_size=0.15
    )

    print(f"\nDataset loaded successfully!")
    print(f"Number of classes: {len(label_names)}")
    print(f"Vocabulary size: {len(vocab)}")

    # Initialize models
    print("\n" + "=" * 80)
    print("INITIALIZING MODELS")
    print("=" * 80)

    model_configs = {}

    # Main hierarchical attention model
    if args.train_main:
        print("\n1. Hierarchical Attention Classifier")
        main_model = HierarchicalAttentionClassifier(
            vocab_size=len(vocab),
            num_classes=len(label_names),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_lstm_layers=args.num_lstm_layers,
            num_attention_heads=args.num_heads,
            filter_ratio=args.filter_ratio,
            pooling_method=args.pooling_method,
            dropout=args.dropout,
            max_sent_len=args.max_sent_len,
            max_num_sent=args.max_num_sent
        )
        model_configs['Hierarchical Attention'] = main_model

    # Baseline 1: Simple Bi-LSTM
    if args.train_baselines:
        print("\n2. Baseline Bi-LSTM")
        baseline_lstm = BaselineLSTM(
            vocab_size=len(vocab),
            num_classes=len(label_names),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        model_configs['Baseline LSTM'] = baseline_lstm

        # Baseline 2: CNN
        print("\n3. Baseline CNN")
        baseline_cnn = BaselineCNN(
            vocab_size=len(vocab),
            num_classes=len(label_names),
            embedding_dim=args.embedding_dim,
            num_filters=128,
            dropout=args.dropout
        )
        model_configs['Baseline CNN'] = baseline_cnn

    # BERT-based hierarchical attention model
    if args.train_bert:
        print("\n4. Hierarchical Attention with BERT")
        bert_model = HierarchicalAttentionBERT(
            num_classes=len(label_names),
            hidden_dim=args.hidden_dim,
            num_attention_heads=args.num_heads,
            filter_ratio=args.filter_ratio,
            pooling_method=args.pooling_method,
            dropout=args.dropout,
            max_sent_len=args.max_sent_len,
            max_num_sent=args.max_num_sent
        )
        model_configs['Hierarchical Attention BERT'] = bert_model

    # Train models
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)

    results_dict = {}
    training_histories = {}

    for model_name, model in model_configs.items():
        print(f"\n{'=' * 80}")
        print(f"Training {model_name}")
        print(f"{'=' * 80}")

        # Initialize trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            patience=args.patience
        )

        # Train
        train_losses, val_accuracies = trainer.train(
            num_epochs=args.num_epochs,
            save_path=f'models/{model_name.lower().replace(" ", "_")}_best.pt'
        )

        training_histories[model_name] = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }

        # Load best model
        checkpoint = torch.load(f'models/{model_name.lower().replace(" ", "_")}_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Evaluate on test set
        print(f"\n{'=' * 80}")
        print(f"Evaluating {model_name} on Test Set")
        print(f"{'=' * 80}")

        results, predictions, true_labels, logits = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            label_names=label_names,
            save_path=f'results/{model_name.lower().replace(" ", "_")}_results.json'
        )

        results_dict[model_name] = {
            'results': results,
            'predictions': predictions,
            'true_labels': true_labels,
            'logits': logits
        }

    # Compare models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    comparison = compare_models(
        {name: data['results'] for name, data in results_dict.items()},
        label_names,
        save_path='results/model_comparison.json'
    )

    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot model comparison
    plot_model_comparison(comparison, save_path='visualizations/model_comparison.png')

    # For each model, create visualizations
    for model_name, data in results_dict.items():
        print(f"\nGenerating visualizations for {model_name}...")

        # Confusion matrix
        visualize_confusion_matrix(
            data['true_labels'],
            data['predictions'],
            label_names,
            save_path=f'visualizations/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        )

        # Per-class performance
        analyze_per_class_performance(
            data['results'],
            label_names,
            save_path=f'visualizations/{model_name.lower().replace(" ", "_")}_per_class.png'
        )

        # Training curves
        if model_name in training_histories:
            plot_training_curves(
                training_histories[model_name]['train_losses'],
                training_histories[model_name]['val_accuracies'],
                save_path=f'visualizations/{model_name.lower().replace(" ", "_")}_training_curves.png'
            )

    # Attention visualizations for main model
    if 'Hierarchical Attention' in model_configs:
        print("\n" + "=" * 80)
        print("ANALYZING ATTENTION MECHANISMS")
        print("=" * 80)

        main_model = model_configs['Hierarchical Attention'].to(device)

        # Word-level attention
        print("\nVisualizing word-level attention...")
        visualize_word_attention(
            main_model, test_loader, device,
            num_samples=5, save_dir='visualizations/word_attention'
        )

        # Cross-attention
        print("\nVisualizing document-level cross-attention...")
        visualize_cross_attention(
            main_model, test_loader, device,
            num_samples=5, save_dir='visualizations/cross_attention'
        )

        # Attention distribution analysis
        print("\nAnalyzing attention distributions...")
        attn_stats = analyze_attention_distribution(
            main_model, test_loader, device,
            save_dir='visualizations'
        )

    # Error analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    for model_name, data in results_dict.items():
        print(f"\n{model_name}:")
        class_errors, misclassified_indices = get_misclassified_examples(
            data['predictions'],
            data['true_labels'],
            test_loader,
            label_names,
            num_examples=5
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAll results saved to:")
    print("  - models/: Trained model checkpoints")
    print("  - results/: Evaluation metrics and comparisons")
    print("  - visualizations/: Plots and attention visualizations")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hierarchical attention model for 20 Newsgroups')

    # Data parameters
    parser.add_argument('--vocab_size', type=int, default=20000, help='Maximum vocabulary size')
    parser.add_argument('--min_freq', type=int, default=2, help='Minimum word frequency')
    parser.add_argument('--max_sent_len', type=int, default=100, help='Maximum sentence length')
    parser.add_argument('--max_num_sent', type=int, default=30, help='Maximum number of sentences')

    # Model parameters
    parser.add_argument('--embedding_dim', type=int, default=200, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num_lstm_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--filter_ratio', type=float, default=0.5, help='Word filtering ratio')
    parser.add_argument('--pooling_method', type=str, default='multi',
                       choices=['max', 'mean', 'attention', 'multi'],
                       help='Sentence pooling method')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Experiment flags
    parser.add_argument('--train_main', action='store_true', default=True, help='Train main model')
    parser.add_argument('--train_baselines', action='store_true', default=True, help='Train baseline models')
    parser.add_argument('--train_bert', action='store_true', default=False, help='Train BERT-based hierarchical model')

    args = parser.parse_args()

    main(args)
