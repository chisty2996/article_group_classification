"""
Visualization and analysis tools for attention mechanisms
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def visualize_word_attention(model, data_loader, device, num_samples=5, save_dir='visualizations'):
    """
    Visualize word-level attention scores for sample documents

    Args:
        model: trained model
        data_loader: data loader
        num_samples: number of samples to visualize
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    sample_count = 0
    attention_stats = {
        'mean_scores': [],
        'std_scores': [],
        'sparsity': []  # percentage of words filtered out
    }

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            input_ids = batch['input_ids'].to(device)
            sentence_masks = batch['sentence_masks'].to(device)

            # Get attention weights
            logits, attention_dict = model(input_ids, sentence_masks, return_attention=True)

            word_scores = attention_dict['word_attention_scores']  # (batch, num_sent, sent_len)

            # Process first sample in batch
            sample_scores = word_scores[0].cpu().numpy()  # (num_sent, sent_len)
            sample_mask = sentence_masks[0].cpu().numpy()  # (num_sent, sent_len)

            # Calculate statistics
            valid_scores = sample_scores[sample_mask == 1]
            attention_stats['mean_scores'].append(valid_scores.mean())
            attention_stats['std_scores'].append(valid_scores.std())

            # Visualize
            fig, axes = plt.subplots(figsize=(15, 8))

            # Show heatmap of attention scores
            sns.heatmap(sample_scores, cmap='YlOrRd', cbar_kws={'label': 'Attention Score'},
                       mask=(sample_mask == 0), ax=axes)
            axes.set_xlabel('Word Position in Sentence')
            axes.set_ylabel('Sentence Index')
            axes.set_title(f'Word-Level Attention Scores - Sample {sample_count + 1}')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/word_attention_sample_{sample_count + 1}.png', dpi=150)
            plt.close()

            sample_count += 1

    # Plot statistics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(attention_stats['mean_scores'], bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Mean Attention Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Mean Word Attention Scores')

    axes[1].hist(attention_stats['std_scores'], bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Std Dev of Attention Scores')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of Attention Score Variability')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/word_attention_statistics.png', dpi=150)
    plt.close()

    print(f"\nWord-level attention visualizations saved to {save_dir}")

    return attention_stats


def visualize_cross_attention(model, data_loader, device, num_samples=5, save_dir='visualizations'):
    """
    Visualize document-level cross-attention patterns

    Args:
        model: trained model
        data_loader: data loader
        num_samples: number of samples to visualize
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    sample_count = 0

    with torch.no_grad():
        for batch in data_loader:
            if sample_count >= num_samples:
                break

            input_ids = batch['input_ids'].to(device)
            sentence_masks = batch['sentence_masks'].to(device)

            # Get attention weights
            logits, attention_dict = model(input_ids, sentence_masks, return_attention=True)

            # Cross attention: (batch, num_heads, num_filtered_words, num_sentences)
            cross_attn = attention_dict['cross_attention_weights']

            # Average over heads
            cross_attn_avg = cross_attn[0].mean(dim=0).cpu().numpy()

            # Visualize
            fig, ax = plt.subplots(figsize=(12, 10))

            sns.heatmap(cross_attn_avg, cmap='viridis', cbar_kws={'label': 'Attention Weight'},
                       ax=ax)
            ax.set_xlabel('Sentence Index')
            ax.set_ylabel('Filtered Word Index')
            ax.set_title(f'Document Cross-Attention Pattern - Sample {sample_count + 1}')

            plt.tight_layout()
            plt.savefig(f'{save_dir}/cross_attention_sample_{sample_count + 1}.png', dpi=150)
            plt.close()

            sample_count += 1

    print(f"Cross-attention visualizations saved to {save_dir}")


def visualize_confusion_matrix(true_labels, predictions, label_names, save_path='confusion_matrix.png'):
    """
    Visualize confusion matrix

    Args:
        true_labels: array of true labels
        predictions: array of predicted labels
        label_names: list of label names
        save_path: path to save the figure
    """
    cm = confusion_matrix(true_labels, predictions)

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names, cbar_kws={'label': 'Proportion'})

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to {save_path}")


def plot_training_curves(train_losses, val_accuracies, save_path='training_curves.png'):
    """
    Plot training loss and validation accuracy curves

    Args:
        train_losses: list of training losses
        val_accuracies: list of validation accuracies
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].plot(train_losses, linewidth=2, color='blue')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Validation accuracy
    axes[1].plot(val_accuracies, linewidth=2, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy over Epochs', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Training curves saved to {save_path}")


def plot_model_comparison(comparison_dict, save_path='model_comparison.png'):
    """
    Create bar chart comparing different models

    Args:
        comparison_dict: dictionary mapping model names to metrics
        save_path: path to save the figure
    """
    models = list(comparison_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        values = [comparison_dict[model][metric] for model in models]
        ax.bar(x + i * width, values, width, label=metric.capitalize())

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Model comparison chart saved to {save_path}")


def analyze_per_class_performance(results, label_names, save_path='per_class_performance.png'):
    """
    Visualize per-class performance metrics

    Args:
        results: results dictionary containing per_class_metrics
        label_names: list of label names
        save_path: path to save the figure
    """
    per_class = results['per_class_metrics']

    classes = label_names
    precision = [per_class[c]['precision'] for c in classes]
    recall = [per_class[c]['recall'] for c in classes]
    f1 = [per_class[c]['f1'] for c in classes]
    support = [per_class[c]['support'] for c in classes]

    # Sort by F1 score
    sorted_indices = np.argsort(f1)

    classes_sorted = [classes[i] for i in sorted_indices]
    precision_sorted = [precision[i] for i in sorted_indices]
    recall_sorted = [recall[i] for i in sorted_indices]
    f1_sorted = [f1[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, 10))

    y_pos = np.arange(len(classes_sorted))
    width = 0.25

    ax.barh(y_pos - width, precision_sorted, width, label='Precision', alpha=0.8)
    ax.barh(y_pos, recall_sorted, width, label='Recall', alpha=0.8)
    ax.barh(y_pos + width, f1_sorted, width, label='F1 Score', alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes_sorted, fontsize=9)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-class performance chart saved to {save_path}")


def analyze_attention_distribution(model, data_loader, device, save_dir='visualizations'):
    """
    Analyze and visualize the distribution of attention weights

    Args:
        model: trained model
        data_loader: data loader
        device: device
        save_dir: directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    all_word_scores = []
    all_cross_attn_entropy = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            sentence_masks = batch['sentence_masks'].to(device)

            logits, attention_dict = model(input_ids, sentence_masks, return_attention=True)

            # Word attention scores
            word_scores = attention_dict['word_attention_scores']
            masks = sentence_masks

            # Flatten and collect valid scores
            valid_scores = word_scores[masks == 1].cpu().numpy()
            all_word_scores.extend(valid_scores)

            # Cross-attention entropy (measure of attention concentration)
            cross_attn = attention_dict['cross_attention_weights']  # (batch, heads, words, sents)
            cross_attn_avg = cross_attn.mean(dim=1)  # Average over heads

            # Compute entropy for each word's attention distribution
            for b in range(cross_attn_avg.size(0)):
                attn_dist = cross_attn_avg[b]  # (num_words, num_sents)
                entropy = -(attn_dist * torch.log(attn_dist + 1e-9)).sum(dim=-1)
                all_cross_attn_entropy.extend(entropy.cpu().numpy())

    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Word attention distribution
    axes[0].hist(all_word_scores, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0].set_xlabel('Word Attention Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Word-Level Attention Scores', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')

    # Cross-attention entropy
    axes[1].hist(all_cross_attn_entropy, bins=50, edgecolor='black', alpha=0.7, color='coral')
    axes[1].set_xlabel('Attention Entropy', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Distribution of Cross-Attention Entropy\n(Higher = More Distributed)', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/attention_distributions.png', dpi=150)
    plt.close()

    print(f"Attention distribution analysis saved to {save_dir}")

    # Print statistics
    print(f"\nWord Attention Statistics:")
    print(f"  Mean: {np.mean(all_word_scores):.4f}")
    print(f"  Std: {np.std(all_word_scores):.4f}")
    print(f"  Min: {np.min(all_word_scores):.4f}")
    print(f"  Max: {np.max(all_word_scores):.4f}")

    print(f"\nCross-Attention Entropy Statistics:")
    print(f"  Mean: {np.mean(all_cross_attn_entropy):.4f}")
    print(f"  Std: {np.std(all_cross_attn_entropy):.4f}")

    return {
        'word_scores_mean': np.mean(all_word_scores),
        'word_scores_std': np.std(all_word_scores),
        'cross_attn_entropy_mean': np.mean(all_cross_attn_entropy),
        'cross_attn_entropy_std': np.std(all_cross_attn_entropy)
    }
