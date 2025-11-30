"""
Debug script to diagnose BERT training issues
"""

import torch
from data_loader_bert import load_20newsgroups_data_bert
from model import HierarchicalAttentionBERT

def debug_bert_model():
    """Debug the BERT model to find issues"""

    print("\n" + "=" * 80)
    print("DEBUGGING BERT MODEL")
    print("=" * 80)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load one batch
    print("\n1. Loading data...")
    train_loader, val_loader, test_loader, tokenizer, label_names = load_20newsgroups_data_bert(
        subset='all',
        max_sent_len=100,
        max_num_sent=30,
        test_size=0.15
    )

    print(f"\nNumber of classes: {len(label_names)}")
    print(f"Classes: {label_names[:5]}...")

    # Get one batch
    batch = next(iter(train_loader))
    input_ids = batch['input_ids']
    sentence_masks = batch['sentence_masks']
    labels = batch['labels']

    print(f"\n2. Batch shapes:")
    print(f"   input_ids: {input_ids.shape}")
    print(f"   sentence_masks: {sentence_masks.shape}")
    print(f"   labels: {labels.shape}")
    print(f"   labels: {labels}")

    # Check token IDs
    print(f"\n3. Token ID ranges:")
    print(f"   Min token ID: {input_ids.min().item()}")
    print(f"   Max token ID: {input_ids.max().item()}")
    print(f"   BERT vocab size: {tokenizer.vocab_size}")
    print(f"   Valid: {input_ids.max().item() < tokenizer.vocab_size}")

    # Show first sentence
    first_sentence = input_ids[0, 0]  # First sample, first sentence
    print(f"\n4. First sentence tokens:")
    print(f"   Token IDs: {first_sentence[:20]}")
    decoded = tokenizer.decode(first_sentence, skip_special_tokens=False)
    print(f"   Decoded: {decoded[:100]}...")

    # Initialize model
    print(f"\n5. Initializing model...")
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
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # Test forward pass
    print(f"\n6. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        input_ids_gpu = input_ids.to(device)
        sentence_masks_gpu = sentence_masks.to(device)

        logits = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)

        print(f"   Output logits shape: {logits.shape}")
        print(f"   Expected: (batch_size={input_ids.shape[0]}, num_classes={len(label_names)})")
        print(f"   Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

        # Check predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        print(f"\n7. Predictions:")
        print(f"   Predicted classes: {preds.cpu()}")
        print(f"   True labels: {labels}")
        print(f"   Prediction probabilities (max): {probs.max(dim=1)[0].cpu()}")

        # Check if model is predicting same class always
        unique_preds = torch.unique(preds)
        print(f"\n8. Prediction diversity:")
        print(f"   Unique predictions in batch: {len(unique_preds)}")
        print(f"   Unique classes: {unique_preds.cpu()}")
        if len(unique_preds) == 1:
            print(f"   ⚠️  WARNING: Model always predicting class {unique_preds[0].item()}")

    # Test training step
    print(f"\n9. Testing training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Before training
    with torch.no_grad():
        logits_before = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)
        loss_before = criterion(logits_before, labels.to(device))

    print(f"   Loss before step: {loss_before.item():.4f}")

    # Training step
    optimizer.zero_grad()
    logits = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)
    loss = criterion(logits, labels.to(device))
    loss.backward()

    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad_norms.append((name, param.grad.norm().item()))

    grad_norms.sort(key=lambda x: x[1], reverse=True)
    print(f"\n10. Top 5 gradient norms:")
    for name, norm in grad_norms[:5]:
        print(f"    {name}: {norm:.6f}")

    if len(grad_norms) == 0:
        print(f"   ⚠️  ERROR: No gradients computed!")
    elif all(norm < 1e-6 for _, norm in grad_norms):
        print(f"   ⚠️  WARNING: All gradients very small (vanishing gradients)")

    optimizer.step()

    # After training
    with torch.no_grad():
        logits_after = model(input_ids_gpu, sentence_masks=sentence_masks_gpu)
        loss_after = criterion(logits_after, labels.to(device))

    print(f"   Loss after step: {loss_after.item():.4f}")
    print(f"   Loss change: {loss_before.item() - loss_after.item():.6f}")

    if abs(loss_before.item() - loss_after.item()) < 1e-6:
        print(f"   ⚠️  WARNING: Loss not changing (weights not updating)")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    debug_bert_model()
