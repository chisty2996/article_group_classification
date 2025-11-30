"""
Debug script to diagnose Hierarchical Attention model issues
"""

import torch
from data_loader import load_20newsgroups_data
from model import HierarchicalAttentionClassifier

def debug_hierarchical_model():
    """Debug the hierarchical attention model"""

    print("\n" + "=" * 80)
    print("DEBUGGING HIERARCHICAL ATTENTION MODEL")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader, vocab, label_names = load_20newsgroups_data(
        subset='all',
        max_vocab_size=20000,
        min_freq=2,
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15,
        batch_size=16
    )

    print(f"\nNumber of classes: {len(label_names)}")
    print(f"Vocab size: {len(vocab)}")

    # Get one batch
    batch = next(iter(train_loader))
    input_ids = batch['input_ids']
    sentence_masks = batch['sentence_masks']
    labels = batch['labels']

    print(f"\n2. Batch shapes:")
    print(f"   input_ids: {input_ids.shape}")
    print(f"   sentence_masks: {sentence_masks.shape}")
    print(f"   labels: {labels.shape}")

    # Check data
    print(f"\n3. Data ranges:")
    print(f"   Token ID range: [{input_ids.min()}, {input_ids.max()}]")
    print(f"   Vocab size: {len(vocab)}")
    print(f"   Valid IDs: {input_ids.max() < len(vocab)}")
    print(f"   Labels: {labels}")
    print(f"   Unique labels in batch: {torch.unique(labels).tolist()}")

    # Check for empty sentences
    non_zero_tokens = (input_ids != 0).sum()
    total_tokens = input_ids.numel()
    print(f"\n4. Token statistics:")
    print(f"   Non-zero tokens: {non_zero_tokens} / {total_tokens} ({100*non_zero_tokens/total_tokens:.1f}%)")
    print(f"   Avg tokens per sample: {non_zero_tokens / input_ids.shape[0]:.1f}")

    # Initialize model
    print(f"\n5. Initializing model...")
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
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total params: {total_params:,}")

    # Test forward pass with detailed inspection
    print(f"\n6. Testing forward pass with attention...")
    model.eval()
    with torch.no_grad():
        input_ids_gpu = input_ids.to(device)
        sentence_masks_gpu = sentence_masks.to(device)

        # Get detailed output
        logits, attn_dict = model(input_ids_gpu, sentence_masks=sentence_masks_gpu, return_attention=True)

        print(f"\n   Forward pass successful!")
        print(f"   Output shape: {logits.shape}")
        print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        print(f"   Logits std: {logits.std().item():.4f}")

        # Check attention outputs
        print(f"\n7. Attention mechanism outputs:")
        print(f"   Word attention scores shape: {attn_dict['word_attention_scores'].shape}")
        print(f"   Word attention range: [{attn_dict['word_attention_scores'].min().item():.4f}, {attn_dict['word_attention_scores'].max().item():.4f}]")
        print(f"   Sentence embeddings shape: {attn_dict['sentence_embeddings'].shape}")
        print(f"   Filtered words shape: {attn_dict['filtered_words'].shape}")

        # Check if embeddings are reasonable
        sent_emb_norm = attn_dict['sentence_embeddings'].norm(dim=-1).mean()
        print(f"   Sentence embedding avg norm: {sent_emb_norm:.4f}")

        # Check predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        print(f"\n8. Predictions:")
        print(f"   Predicted classes: {preds.cpu().tolist()}")
        print(f"   True labels: {labels.tolist()}")
        print(f"   Max probabilities: {probs.max(dim=1)[0].cpu().tolist()}")
        print(f"   Unique predictions: {torch.unique(preds).tolist()}")

        if len(torch.unique(preds)) == 1:
            print(f"   ⚠️  WARNING: All predictions are the same class!")

        # Check if logits are uniform
        logits_std = logits.std(dim=1).mean()
        print(f"   Logits std across classes: {logits_std:.4f}")
        if logits_std < 0.1:
            print(f"   ⚠️  WARNING: Logits are nearly uniform (model not discriminating)")

    # Test training step
    print(f"\n9. Testing training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Initial loss
    with torch.no_grad():
        logits_before = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)
        loss_before = criterion(logits_before, labels.to(device))

    print(f"   Loss before: {loss_before.item():.4f}")
    print(f"   Expected initial loss: ~{-torch.log(torch.tensor(1.0/len(label_names))):.4f} (random)")

    # Training step
    optimizer.zero_grad()
    logits = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)
    loss = criterion(logits, labels.to(device))
    loss.backward()

    # Check gradients
    grad_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_stats.append((name, grad_norm, param.numel()))

    grad_stats.sort(key=lambda x: x[1], reverse=True)

    print(f"\n10. Gradient analysis:")
    print(f"    Top 5 gradient norms:")
    for name, norm, size in grad_stats[:5]:
        print(f"      {name}: {norm:.6f} (size: {size})")

    print(f"\n    Bottom 5 gradient norms:")
    for name, norm, size in grad_stats[-5:]:
        print(f"      {name}: {norm:.6f} (size: {size})")

    # Check for gradient issues
    zero_grads = sum(1 for _, norm, _ in grad_stats if norm < 1e-7)
    exploding_grads = sum(1 for _, norm, _ in grad_stats if norm > 100)

    if zero_grads > 0:
        print(f"   ⚠️  WARNING: {zero_grads} parameters have near-zero gradients")
    if exploding_grads > 0:
        print(f"   ⚠️  WARNING: {exploding_grads} parameters have exploding gradients")

    # Clip and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    # Check weight update
    with torch.no_grad():
        logits_after = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)
        loss_after = criterion(logits_after, labels.to(device))

    print(f"\n11. After optimization step:")
    print(f"    Loss after: {loss_after.item():.4f}")
    print(f"    Loss change: {loss_before.item() - loss_after.item():.6f}")

    if abs(loss_before.item() - loss_after.item()) < 1e-5:
        print(f"    ⚠️  ERROR: Loss not changing (weights not updating!)")

    # Check model layers
    print(f"\n12. Model architecture check:")
    print(f"    Embedding dim: {model.encoder.embedding.embedding_dim}")
    print(f"    Hidden dim: {model.encoder.lstm.hidden_size * 2}")
    print(f"    Sentence repr output dim: {model.sentence_repr.output_dim}")
    print(f"    Cross attention word dim: {model.cross_attention.word_dim}")
    print(f"    Cross attention sentence dim: {model.cross_attention.sentence_dim}")

    # Verify dimensions match
    if model.encoder.lstm.hidden_size * 2 != model.cross_attention.word_dim:
        print(f"    ⚠️  ERROR: Hidden dim mismatch!")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    debug_hierarchical_model()
