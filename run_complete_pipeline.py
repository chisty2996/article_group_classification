"""
Complete pipeline script that runs training, evaluation, and all analyses
"""

import torch
import os
import json
from data_loader import load_20newsgroups_data
from model import HierarchicalAttentionClassifier, BaselineLSTM, BaselineCNN, HierarchicalAttentionBERT
from train import Trainer, evaluate_model, compare_models, get_misclassified_examples
from visualization import (
    visualize_word_attention, visualize_cross_attention, visualize_confusion_matrix,
    plot_training_curves, plot_model_comparison, analyze_per_class_performance,
    analyze_attention_distribution
)
from error_analysis import FailureModeAnalyzer, propose_improvements, print_proposed_improvements


def run_complete_pipeline():
    """Run the complete training and analysis pipeline"""

    print("\n" + "=" * 80)
    print("HIERARCHICAL ATTENTION MODEL FOR 20 NEWSGROUPS CLASSIFICATION")
    print("Complete Pipeline Execution")
    print("=" * 80)

    # Configuration - prioritize CUDA (VM/Cloud GPU) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: {device} (NVIDIA CUDA GPU)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: {device} (Apple Metal Performance Shaders)")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: {device} (CPU only)")

    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('visualizations/word_attention', exist_ok=True)
    os.makedirs('visualizations/cross_attention', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING 20 NEWSGROUPS DATASET")
    print("=" * 80)

    train_loader, val_loader, test_loader, vocab, label_names = load_20newsgroups_data(
        subset='all',
        max_vocab_size=20000,
        min_freq=2,
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15
    )

    print(f"\n✓ Dataset loaded successfully")
    print(f"  - Number of classes: {len(label_names)}")
    print(f"  - Vocabulary size: {len(vocab)}")
    print(f"  - Training batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")
    print(f"  - Test batches: {len(test_loader)}")

    # ========================================================================
    # STEP 2: INITIALIZE MODELS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING MODELS")
    print("=" * 80)

    # Main model
    print("\n1. Hierarchical Attention Classifier")
    main_model = HierarchicalAttentionClassifier(
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
    print(f"   Parameters: {sum(p.numel() for p in main_model.parameters()):,}")

    # Baseline 1: LSTM
    print("\n2. Baseline Bi-LSTM")
    baseline_lstm = BaselineLSTM(
        vocab_size=len(vocab),
        num_classes=len(label_names),
        embedding_dim=200,
        hidden_dim=256,
        dropout=0.3
    )
    print(f"   Parameters: {sum(p.numel() for p in baseline_lstm.parameters()):,}")

    # Baseline 2: CNN
    print("\n3. Baseline CNN")
    baseline_cnn = BaselineCNN(
        vocab_size=len(vocab),
        num_classes=len(label_names),
        embedding_dim=200,
        num_filters=128,
        dropout=0.3
    )
    print(f"   Parameters: {sum(p.numel() for p in baseline_cnn.parameters()):,}")

    # BERT-based hierarchical model
    print("\n4. Hierarchical Attention with BERT")
    bert_model = HierarchicalAttentionBERT(
        num_classes=len(label_names),
        hidden_dim=256,
        num_attention_heads=4,
        filter_ratio=0.5,
        pooling_method='multi',
        dropout=0.3,
        max_sent_len=100,
        max_num_sent=30
    )
    print(f"   Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")

    models = {
        'Hierarchical Attention': main_model,
        'Baseline LSTM': baseline_lstm,
        'Baseline CNN': baseline_cnn,
        'Hierarchical Attention BERT': bert_model
    }

    # ========================================================================
    # STEP 3: TRAIN MODELS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING MODELS")
    print("=" * 80)

    results_dict = {}
    training_histories = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"Training: {model_name}")
        print(f"{'=' * 80}")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=0.001,
            weight_decay=1e-5,
            patience=5
        )

        train_losses, val_accuracies = trainer.train(
            num_epochs=20,
            save_path=f'models/{model_name.lower().replace(" ", "_")}_best.pt'
        )

        training_histories[model_name] = {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }

        # Load best model
        checkpoint = torch.load(f'models/{model_name.lower().replace(" ", "_")}_best.pt')
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"\n✓ {model_name} training complete")
        print(f"  Best validation accuracy: {checkpoint['val_acc']:.4f}")

    # ========================================================================
    # STEP 4: EVALUATE MODELS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: EVALUATING MODELS ON TEST SET")
    print("=" * 80)

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 40)

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

    # ========================================================================
    # STEP 5: MODEL COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: COMPARING MODELS")
    print("=" * 80)

    comparison = compare_models(
        {name: data['results'] for name, data in results_dict.items()},
        label_names,
        save_path='results/model_comparison.json'
    )

    plot_model_comparison(comparison, save_path='visualizations/model_comparison.png')
    print("\n✓ Model comparison chart saved")

    # ========================================================================
    # STEP 6: GENERATE VISUALIZATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: GENERATING VISUALIZATIONS")
    print("=" * 80)

    for model_name, data in results_dict.items():
        print(f"\n{model_name}:")

        # Confusion matrix
        visualize_confusion_matrix(
            data['true_labels'],
            data['predictions'],
            label_names,
            save_path=f'visualizations/{model_name.lower().replace(" ", "_")}_confusion_matrix.png'
        )
        print("  ✓ Confusion matrix")

        # Per-class performance
        analyze_per_class_performance(
            data['results'],
            label_names,
            save_path=f'visualizations/{model_name.lower().replace(" ", "_")}_per_class.png'
        )
        print("  ✓ Per-class performance")

        # Training curves
        if model_name in training_histories:
            plot_training_curves(
                training_histories[model_name]['train_losses'],
                training_histories[model_name]['val_accuracies'],
                save_path=f'visualizations/{model_name.lower().replace(" ", "_")}_training_curves.png'
            )
            print("  ✓ Training curves")

    # ========================================================================
    # STEP 7: ATTENTION ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: ANALYZING ATTENTION MECHANISMS")
    print("=" * 80)

    main_model = models['Hierarchical Attention'].to(device)

    print("\n1. Word-level attention patterns...")
    visualize_word_attention(
        main_model, test_loader, device,
        num_samples=5, save_dir='visualizations/word_attention'
    )

    print("\n2. Document-level cross-attention patterns...")
    visualize_cross_attention(
        main_model, test_loader, device,
        num_samples=5, save_dir='visualizations/cross_attention'
    )

    print("\n3. Attention distribution analysis...")
    attn_stats = analyze_attention_distribution(
        main_model, test_loader, device,
        save_dir='visualizations'
    )

    print("\n✓ Attention analysis complete")

    # ========================================================================
    # STEP 8: ERROR ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: ERROR ANALYSIS")
    print("=" * 80)

    for model_name, data in results_dict.items():
        print(f"\n{model_name}:")
        get_misclassified_examples(
            data['predictions'],
            data['true_labels'],
            test_loader,
            label_names,
            num_examples=5
        )

    # ========================================================================
    # STEP 9: FAILURE MODE ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: FAILURE MODE ANALYSIS")
    print("=" * 80)

    analyzer = FailureModeAnalyzer(main_model, device)

    print("\nAnalyzing attention cascade failures...")
    failure_modes, attention_metrics = analyzer.analyze_attention_cascade_failures(
        test_loader, label_names
    )

    print("\nAnalyzing encoder bias propagation...")
    encoder_analysis = analyzer.analyze_encoder_bias_propagation(
        test_loader, label_names, num_samples=100
    )

    print("\nGenerating failure mode report...")
    failure_report = analyzer.generate_failure_report(
        failure_modes, attention_metrics, encoder_analysis,
        save_path='results/failure_analysis.json'
    )

    analyzer.print_failure_summary(failure_report)

    # ========================================================================
    # STEP 10: PROPOSE IMPROVEMENTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 10: PROPOSED IMPROVEMENTS")
    print("=" * 80)

    improvements = propose_improvements(failure_report)
    print_proposed_improvements(improvements)

    # Save improvements
    with open('results/proposed_improvements.json', 'w') as f:
        json.dump(improvements, f, indent=2)

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 80)

    print("\nGenerated Outputs:")
    print("  📁 models/ - Trained model checkpoints")
    print("  📁 results/ - Evaluation metrics and analysis (JSON)")
    print("  📁 visualizations/ - Charts and attention heatmaps")
    print("\nKey Files:")
    print("  📄 REPORT.md - Comprehensive project report")
    print("  📄 results/model_comparison.json - Performance comparison")
    print("  📄 results/failure_analysis.json - Failure mode analysis")
    print("  📄 results/proposed_improvements.json - Architectural improvements")

    print("\n✅ All tasks completed successfully!")

    return {
        'models': models,
        'results': results_dict,
        'training_histories': training_histories,
        'failure_report': failure_report,
        'improvements': improvements
    }


if __name__ == '__main__':
    results = run_complete_pipeline()
