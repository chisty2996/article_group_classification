"""
Comprehensive error analysis and failure mode detection for the hierarchical attention model
"""

import torch
import numpy as np
from collections import defaultdict
import json


class FailureModeAnalyzer:
    """
    Analyzes failure modes in the hierarchical attention model, particularly
    focusing on how errors propagate from word-level to document-level attention
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def analyze_attention_cascade_failures(self, data_loader, label_names):
        """
        Identify failure modes where word-level attention errors cascade
        to document-level cross-attention errors

        Returns:
            failure_modes: dictionary containing different types of failures
        """
        failure_modes = {
            'low_quality_filtering': [],  # Word filter selects uninformative words
            'context_representation_collapse': [],  # LSTM produces poor representations
            'attention_sparsity_errors': [],  # Over-concentration on few words
            'cross_attention_misalignment': [],  # Word and sentence signals conflict
            'noisy_word_amplification': []  # High attention to context-poor words
        }

        attention_metrics = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                input_ids = batch['input_ids'].to(self.device)
                sentence_masks = batch['sentence_masks'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get predictions and attention
                logits, attention_dict = self.model(input_ids, sentence_masks, return_attention=True)
                predictions = torch.argmax(logits, dim=1)

                # Extract attention components
                word_scores = attention_dict['word_attention_scores']  # (batch, num_sent, sent_len)
                word_indices = attention_dict['word_indices']  # (batch, num_sent, num_filtered)
                cross_attn = attention_dict['cross_attention_weights']  # (batch, heads, words, sents)

                # Analyze each sample
                for i in range(input_ids.size(0)):
                    is_correct = predictions[i] == labels[i]
                    sample_info = {
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'true_label': label_names[labels[i].item()],
                        'pred_label': label_names[predictions[i].item()],
                        'is_correct': is_correct.item()
                    }

                    # Analyze word-level attention quality
                    word_attn = word_scores[i]  # (num_sent, sent_len)
                    mask = sentence_masks[i]  # (num_sent, sent_len)
                    valid_attn = word_attn[mask == 1]

                    # Metrics
                    attn_mean = valid_attn.mean().item()
                    attn_std = valid_attn.std().item()
                    attn_max = valid_attn.max().item()

                    # 1. Low-quality filtering: Low variance in attention scores
                    if attn_std < 0.1 and not is_correct:
                        failure_modes['low_quality_filtering'].append({
                            **sample_info,
                            'attn_std': attn_std,
                            'reason': 'Uniform attention - model cannot distinguish important words'
                        })

                    # 2. Attention sparsity errors: Over-concentration
                    top_10_percent_threshold = torch.quantile(valid_attn, 0.9)
                    top_10_mass = (valid_attn > top_10_percent_threshold).float().mean().item()
                    if top_10_mass > 0.8 and not is_correct:
                        failure_modes['attention_sparsity_errors'].append({
                            **sample_info,
                            'top_10_mass': top_10_mass,
                            'reason': 'Over-concentration on few words, missing diverse signals'
                        })

                    # 3. Noisy word amplification: High attention to low-information positions
                    # Detect if high-attention words are at document boundaries (often boilerplate)
                    top_k_indices = torch.topk(valid_attn, min(10, len(valid_attn)))[1]
                    boundary_positions = 0
                    total_positions = 0

                    sent_idx = 0
                    global_idx = 0
                    for s in range(mask.size(0)):
                        for w in range(mask.size(1)):
                            if mask[s, w] == 1:
                                if global_idx in top_k_indices:
                                    # Check if at sentence boundary
                                    if w < 3 or w > mask[s].sum() - 3:
                                        boundary_positions += 1
                                    total_positions += 1
                                global_idx += 1

                    if total_positions > 0 and boundary_positions / total_positions > 0.5 and not is_correct:
                        failure_modes['noisy_word_amplification'].append({
                            **sample_info,
                            'boundary_ratio': boundary_positions / total_positions,
                            'reason': 'High attention on sentence boundaries (likely boilerplate)'
                        })

                    # 4. Cross-attention analysis
                    cross_attn_sample = cross_attn[i].mean(dim=0)  # Average over heads: (words, sents)

                    # Compute entropy of cross-attention (measure of distribution)
                    cross_attn_entropy = -(cross_attn_sample * torch.log(cross_attn_sample + 1e-9)).sum(dim=-1)
                    avg_entropy = cross_attn_entropy.mean().item()

                    # Low entropy = concentrated attention (might miss diverse sentence info)
                    if avg_entropy < 1.0 and not is_correct:
                        failure_modes['cross_attention_misalignment'].append({
                            **sample_info,
                            'cross_attn_entropy': avg_entropy,
                            'reason': 'Cross-attention too concentrated, missing diverse sentence signals'
                        })

                    # Store metrics
                    attention_metrics['word_attn_mean'].append(attn_mean)
                    attention_metrics['word_attn_std'].append(attn_std)
                    attention_metrics['cross_attn_entropy'].append(avg_entropy)
                    attention_metrics['top_10_mass'].append(top_10_mass)
                    attention_metrics['is_correct'].append(is_correct.item())

        return failure_modes, attention_metrics

    def analyze_encoder_bias_propagation(self, data_loader, label_names, num_samples=100):
        """
        Analyze how encoder biases propagate through attention layers

        This examines:
        1. Contextual representation collapse (all words get similar representations)
        2. Position bias in LSTM encoding
        3. How poor word representations affect downstream attention
        """
        encoder_analysis = {
            'representation_collapse_cases': [],
            'position_bias_cases': [],
            'low_diversity_encoding': []
        }

        sample_count = 0

        with torch.no_grad():
            for batch in data_loader:
                if sample_count >= num_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)
                sentence_masks = batch['sentence_masks'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get encoder output
                batch_size, num_sent, sent_len = input_ids.shape
                input_ids_flat = input_ids.view(batch_size * num_sent, sent_len)
                masks_flat = sentence_masks.view(batch_size * num_sent, sent_len)

                # Get contextual embeddings from encoder
                contextual_embeddings = self.model.encoder(input_ids_flat, masks_flat)

                # Get predictions
                logits = self.model(input_ids, sentence_masks)
                predictions = torch.argmax(logits, dim=1)

                # Analyze each sample
                for i in range(min(batch_size, num_samples - sample_count)):
                    is_correct = predictions[i] == labels[i]

                    # Get embeddings for this sample
                    sample_embeddings = contextual_embeddings[i * num_sent:(i + 1) * num_sent]
                    sample_mask = masks_flat[i * num_sent:(i + 1) * num_sent]

                    # Flatten to get all word representations
                    all_word_embeds = []
                    for s in range(num_sent):
                        sent_mask = sample_mask[s]
                        sent_embeds = sample_embeddings[s][sent_mask == 1]
                        if len(sent_embeds) > 0:
                            all_word_embeds.append(sent_embeds)

                    if len(all_word_embeds) == 0:
                        continue

                    all_word_embeds = torch.cat(all_word_embeds, dim=0)

                    # 1. Representation collapse: Check if embeddings are too similar
                    # Compute pairwise cosine similarity
                    if len(all_word_embeds) > 1:
                        normalized = torch.nn.functional.normalize(all_word_embeds, dim=1)
                        similarity_matrix = torch.mm(normalized, normalized.t())

                        # Average similarity (excluding diagonal)
                        mask_matrix = 1 - torch.eye(similarity_matrix.size(0), device=self.device)
                        avg_similarity = (similarity_matrix * mask_matrix).sum() / mask_matrix.sum()

                        if avg_similarity > 0.9 and not is_correct:
                            encoder_analysis['representation_collapse_cases'].append({
                                'sample_idx': sample_count,
                                'true_label': label_names[labels[i].item()],
                                'pred_label': label_names[predictions[i].item()],
                                'avg_similarity': avg_similarity.item(),
                                'reason': 'Word representations too similar - encoder collapse'
                            })

                        # 2. Position bias: Check if first/last positions dominate
                        # This is simplified - in practice would track through actual attention
                        variance_across_positions = all_word_embeds.var(dim=0).mean().item()

                        if variance_across_positions < 0.01 and not is_correct:
                            encoder_analysis['low_diversity_encoding'].append({
                                'sample_idx': sample_count,
                                'true_label': label_names[labels[i].item()],
                                'pred_label': label_names[predictions[i].item()],
                                'position_variance': variance_across_positions,
                                'reason': 'Low variance in positional encoding - poor context modeling'
                            })

                    sample_count += 1

        return encoder_analysis

    def generate_failure_report(self, failure_modes, attention_metrics, encoder_analysis, save_path='failure_analysis.json'):
        """
        Generate comprehensive failure mode report
        """
        report = {
            'summary': {
                'total_low_quality_filtering': len(failure_modes['low_quality_filtering']),
                'total_sparsity_errors': len(failure_modes['attention_sparsity_errors']),
                'total_noisy_amplification': len(failure_modes['noisy_word_amplification']),
                'total_cross_attention_misalignment': len(failure_modes['cross_attention_misalignment']),
                'total_representation_collapse': len(encoder_analysis['representation_collapse_cases']),
                'total_low_diversity_encoding': len(encoder_analysis['low_diversity_encoding'])
            },
            'failure_modes': failure_modes,
            'encoder_analysis': encoder_analysis,
            'attention_statistics': {
                'correct_samples': {
                    'word_attn_mean': np.mean([m for m, c in zip(attention_metrics['word_attn_mean'], attention_metrics['is_correct']) if c]),
                    'word_attn_std': np.mean([m for m, c in zip(attention_metrics['word_attn_std'], attention_metrics['is_correct']) if c]),
                    'cross_attn_entropy': np.mean([m for m, c in zip(attention_metrics['cross_attn_entropy'], attention_metrics['is_correct']) if c]),
                },
                'incorrect_samples': {
                    'word_attn_mean': np.mean([m for m, c in zip(attention_metrics['word_attn_mean'], attention_metrics['is_correct']) if not c]),
                    'word_attn_std': np.mean([m for m, c in zip(attention_metrics['word_attn_std'], attention_metrics['is_correct']) if not c]),
                    'cross_attn_entropy': np.mean([m for m, c in zip(attention_metrics['cross_attn_entropy'], attention_metrics['is_correct']) if not c]),
                }
            }
        }

        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nFailure mode analysis saved to {save_path}")

        return report

    def print_failure_summary(self, report):
        """Print human-readable summary of failure modes"""

        print("\n" + "=" * 80)
        print("FAILURE MODE ANALYSIS SUMMARY")
        print("=" * 80)

        print("\n1. Word-Level Attention Failures:")
        print(f"   - Low-quality filtering (uniform attention): {report['summary']['total_low_quality_filtering']} cases")
        print(f"   - Attention sparsity errors (over-concentration): {report['summary']['total_sparsity_errors']} cases")
        print(f"   - Noisy word amplification: {report['summary']['total_noisy_amplification']} cases")

        print("\n2. Cross-Attention Failures:")
        print(f"   - Cross-attention misalignment: {report['summary']['total_cross_attention_misalignment']} cases")

        print("\n3. Encoder Representation Failures:")
        print(f"   - Representation collapse: {report['summary']['total_representation_collapse']} cases")
        print(f"   - Low diversity encoding: {report['summary']['total_low_diversity_encoding']} cases")

        print("\n4. Attention Pattern Differences (Correct vs Incorrect Predictions):")
        correct_stats = report['attention_statistics']['correct_samples']
        incorrect_stats = report['attention_statistics']['incorrect_samples']

        print(f"\n   Correct predictions:")
        print(f"     Word attention mean: {correct_stats['word_attn_mean']:.4f}")
        print(f"     Word attention std: {correct_stats['word_attn_std']:.4f}")
        print(f"     Cross-attention entropy: {correct_stats['cross_attn_entropy']:.4f}")

        print(f"\n   Incorrect predictions:")
        print(f"     Word attention mean: {incorrect_stats['word_attn_mean']:.4f}")
        print(f"     Word attention std: {incorrect_stats['word_attn_std']:.4f}")
        print(f"     Cross-attention entropy: {incorrect_stats['cross_attn_entropy']:.4f}")

        print("\n" + "=" * 80)


def propose_improvements(failure_report):
    """
    Propose architectural improvements based on detected failure modes
    """
    improvements = []

    # Based on failure modes, suggest improvements
    if failure_report['summary']['total_low_quality_filtering'] > 100:
        improvements.append({
            'issue': 'Low-quality word filtering (uniform attention)',
            'proposal': 'Implement learnable temperature parameter in word-level attention softmax to encourage sparsity, or use Gumbel-Softmax for more discriminative selection',
            'technical_detail': 'Replace softmax with temperature-scaled version: softmax(scores/tau) where tau is learned or annealed during training'
        })

    if failure_report['summary']['total_sparsity_errors'] > 100:
        improvements.append({
            'issue': 'Over-concentration in word attention (sparsity errors)',
            'proposal': 'Add diversity regularization loss to encourage attention over diverse word positions',
            'technical_detail': 'Add loss term: lambda * variance_loss where variance_loss = -log(var(attention_weights)) encourages spread'
        })

    if failure_report['summary']['total_cross_attention_misalignment'] > 100:
        improvements.append({
            'issue': 'Cross-attention misalignment between word and sentence signals',
            'proposal': 'Add residual connections between word-level and sentence-level representations, or implement gating mechanism to balance local vs global signals',
            'technical_detail': 'Use learned gate: g = sigmoid(W[word_repr; sent_repr]), output = g * word_repr + (1-g) * cross_attended'
        })

    if failure_report['summary']['total_representation_collapse'] > 50:
        improvements.append({
            'issue': 'Contextual representation collapse in encoder',
            'proposal': 'Replace Bi-LSTM with pre-trained BERT or add auxiliary reconstruction loss to preserve word-level information',
            'technical_detail': 'Use BERT embeddings or add loss: L_recon = MSE(decoded_words, original_embeddings) to prevent information loss'
        })

    if failure_report['summary']['total_noisy_amplification'] > 100:
        improvements.append({
            'issue': 'High attention to noisy or uninformative words',
            'proposal': 'Implement content-based filtering using TF-IDF weighting or learned importance scores based on word embeddings',
            'technical_detail': 'Pre-compute TF-IDF scores and multiply with attention: final_attention = attention_weights * tfidf_weights'
        })

    return improvements


def print_proposed_improvements(improvements):
    """Print proposed improvements"""

    print("\n" + "=" * 80)
    print("PROPOSED ARCHITECTURAL IMPROVEMENTS")
    print("=" * 80)

    for i, improvement in enumerate(improvements, 1):
        print(f"\n{i}. Issue: {improvement['issue']}")
        print(f"   Proposal: {improvement['proposal']}")
        print(f"   Technical Detail: {improvement['technical_detail']}")

    print("\n" + "=" * 80)
