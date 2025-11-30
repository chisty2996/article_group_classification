# Usage Guide: Hierarchical Attention Model for 20 Newsgroups

This guide provides step-by-step instructions for training, evaluating, and analyzing the hierarchical attention model.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Detailed Workflow](#detailed-workflow)
3. [Configuration Options](#configuration-options)
4. [Understanding Outputs](#understanding-outputs)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Option 1: Run Complete Pipeline (Recommended for First Time)

```bash
# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Run complete pipeline (trains all models, generates all analyses)
python run_complete_pipeline.py
```

This will:
- Load and preprocess the 20 Newsgroups dataset
- Train hierarchical attention model + 2 baselines
- Evaluate on test set
- Generate all visualizations
- Perform error and failure mode analysis
- Save all results

**Time estimate:** 1-2 hours on CPU, 20-30 minutes on GPU

### Option 2: Run Main Script with Options

```bash
# Train only the main model
python main.py --train_main --num_epochs 15

# Train all models with custom settings
python main.py --train_main --train_baselines \
    --hidden_dim 512 \
    --num_heads 8 \
    --num_epochs 20
```

### Option 3: Use Individual Scripts

```python
# In Python or Jupyter notebook
from data_loader import load_20newsgroups_data
from model import HierarchicalAttentionClassifier
from train import Trainer

# Load data
train_loader, val_loader, test_loader, vocab, label_names = \
    load_20newsgroups_data()

# Initialize model
model = HierarchicalAttentionClassifier(
    vocab_size=len(vocab),
    num_classes=len(label_names)
)

# Train
trainer = Trainer(model, train_loader, val_loader, device)
trainer.train(num_epochs=20)
```

---

## Detailed Workflow

### Step 1: Data Loading

```python
from data_loader import load_20newsgroups_data

train_loader, val_loader, test_loader, vocab, label_names = \
    load_20newsgroups_data(
        subset='all',           # 'train', 'test', or 'all'
        max_vocab_size=20000,   # Maximum vocabulary size
        min_freq=2,             # Minimum word frequency
        max_sent_len=100,       # Words per sentence
        max_num_sent=30,        # Sentences per document
        test_size=0.15          # Validation split ratio
    )
```

**What happens:**
- Downloads 20 Newsgroups dataset from scikit-learn
- Removes headers, footers, and quotes
- Builds vocabulary from most frequent words
- Splits text into sentences (NLTK)
- Creates PyTorch DataLoaders

**Output:**
- `train_loader`: Training data batches
- `val_loader`: Validation data batches
- `test_loader`: Test data batches
- `vocab`: Dictionary mapping words to indices
- `label_names`: List of 20 newsgroup categories

### Step 2: Model Initialization

```python
from model import HierarchicalAttentionClassifier

model = HierarchicalAttentionClassifier(
    vocab_size=len(vocab),
    num_classes=20,
    embedding_dim=200,        # Word embedding size
    hidden_dim=256,           # LSTM hidden dimension
    num_lstm_layers=2,        # LSTM depth
    num_attention_heads=4,    # Multi-head attention
    filter_ratio=0.5,         # Keep top 50% of words
    pooling_method='multi',   # 'max', 'mean', 'attention', 'multi'
    dropout=0.3,
    max_sent_len=100,
    max_num_sent=30
)

# Print model architecture
print(model)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")
```

### Step 3: Training

```python
from train import Trainer
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    learning_rate=0.001,
    weight_decay=1e-5,
    patience=5  # Early stopping patience
)

# Train model
train_losses, val_accuracies = trainer.train(
    num_epochs=20,
    save_path='models/my_model.pt'
)

# Plot training curves
from visualization import plot_training_curves
plot_training_curves(train_losses, val_accuracies, 'training_curves.png')
```

**What happens:**
- Trains for up to 20 epochs with early stopping
- Saves best model based on validation accuracy
- Uses Adam optimizer with learning rate scheduling
- Applies gradient clipping (max norm = 5.0)

### Step 4: Evaluation

```python
from train import evaluate_model

# Load best model
checkpoint = torch.load('models/my_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on test set
results, predictions, true_labels, logits = evaluate_model(
    model=model,
    test_loader=test_loader,
    device=device,
    label_names=label_names,
    save_path='results/evaluation.json'
)

print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 Score: {results['f1']:.4f}")
```

### Step 5: Visualization

```python
from visualization import (
    visualize_confusion_matrix,
    analyze_per_class_performance,
    visualize_word_attention,
    visualize_cross_attention
)

# Confusion matrix
visualize_confusion_matrix(
    true_labels, predictions, label_names,
    save_path='confusion_matrix.png'
)

# Per-class performance
analyze_per_class_performance(
    results, label_names,
    save_path='per_class_performance.png'
)

# Attention visualizations
model.eval()
visualize_word_attention(
    model, test_loader, device,
    num_samples=5,
    save_dir='word_attention/'
)

visualize_cross_attention(
    model, test_loader, device,
    num_samples=5,
    save_dir='cross_attention/'
)
```

### Step 6: Error Analysis

```python
from error_analysis import FailureModeAnalyzer

analyzer = FailureModeAnalyzer(model, device)

# Analyze failure modes
failure_modes, attention_metrics = analyzer.analyze_attention_cascade_failures(
    test_loader, label_names
)

# Analyze encoder biases
encoder_analysis = analyzer.analyze_encoder_bias_propagation(
    test_loader, label_names, num_samples=100
)

# Generate report
failure_report = analyzer.generate_failure_report(
    failure_modes, attention_metrics, encoder_analysis,
    save_path='failure_analysis.json'
)

analyzer.print_failure_summary(failure_report)
```

---

## Configuration Options

### Model Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `embedding_dim` | 200 | 100-300 | Word embedding dimension |
| `hidden_dim` | 256 | 128-512 | LSTM hidden dimension |
| `num_lstm_layers` | 2 | 1-3 | Number of LSTM layers |
| `num_attention_heads` | 4 | 2-8 | Multi-head attention heads |
| `filter_ratio` | 0.5 | 0.3-0.7 | Fraction of words to keep |
| `pooling_method` | 'multi' | max/mean/attention/multi | Sentence pooling strategy |
| `dropout` | 0.3 | 0.1-0.5 | Dropout rate |

### Training Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `learning_rate` | 0.001 | 0.0001-0.01 | Initial learning rate |
| `weight_decay` | 1e-5 | 0-1e-4 | L2 regularization |
| `num_epochs` | 20 | 10-50 | Maximum training epochs |
| `patience` | 5 | 3-10 | Early stopping patience |
| `batch_size` | 16 | 8-32 | Training batch size |

### Data Hyperparameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `vocab_size` | 20000 | 10000-50000 | Maximum vocabulary size |
| `min_freq` | 2 | 1-5 | Minimum word frequency |
| `max_sent_len` | 100 | 50-200 | Maximum words per sentence |
| `max_num_sent` | 30 | 20-50 | Maximum sentences per document |

---

## Understanding Outputs

### Directory Structure

```
article_group_classification/
├── models/
│   ├── hierarchical_attention_best.pt    # Best model checkpoint
│   ├── baseline_lstm_best.pt
│   └── baseline_cnn_best.pt
│
├── results/
│   ├── hierarchical_attention_results.json  # Evaluation metrics
│   ├── model_comparison.json                # Model comparison
│   ├── failure_analysis.json                # Failure mode analysis
│   └── proposed_improvements.json           # Suggested fixes
│
└── visualizations/
    ├── model_comparison.png                 # Bar chart
    ├── confusion_matrix.png                 # Confusion matrix
    ├── per_class_performance.png            # Per-class metrics
    ├── training_curves.png                  # Loss and accuracy
    ├── attention_distributions.png          # Attention stats
    ├── word_attention/
    │   ├── word_attention_sample_1.png
    │   ├── word_attention_sample_2.png
    │   └── word_attention_statistics.png
    └── cross_attention/
        ├── cross_attention_sample_1.png
        └── cross_attention_sample_2.png
```

### Results JSON Format

```json
{
  "accuracy": 0.7845,
  "precision": 0.7812,
  "recall": 0.7845,
  "f1": 0.7823,
  "per_class_metrics": {
    "alt.atheism": {
      "precision": 0.82,
      "recall": 0.79,
      "f1": 0.80,
      "support": 319
    },
    ...
  }
}
```

### Failure Analysis JSON Format

```json
{
  "summary": {
    "total_low_quality_filtering": 145,
    "total_sparsity_errors": 89,
    "total_noisy_amplification": 67,
    ...
  },
  "failure_modes": {
    "low_quality_filtering": [
      {
        "sample_idx": 42,
        "true_label": "sci.space",
        "pred_label": "sci.electronics",
        "attn_std": 0.08,
        "reason": "Uniform attention - model cannot distinguish important words"
      },
      ...
    ],
    ...
  }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory Error

**Problem:** `CUDA out of memory` or similar

**Solutions:**
```python
# Reduce batch size
train_loader = DataLoader(train_dataset, batch_size=8)  # Instead of 16

# Reduce model size
model = HierarchicalAttentionClassifier(
    hidden_dim=128,  # Instead of 256
    num_lstm_layers=1  # Instead of 2
)

# Use CPU instead of GPU
device = torch.device('cpu')
```

#### 2. NLTK Punkt Not Found

**Problem:** `Resource punkt not found`

**Solution:**
```python
import nltk
nltk.download('punkt')
```

#### 3. Slow Training

**Problem:** Training takes too long

**Solutions:**
- Use GPU if available
- Reduce dataset size for testing:
  ```python
  # Use subset of data
  train_loader, val_loader, test_loader, vocab, label_names = \
      load_20newsgroups_data(subset='train')  # Only training set
  ```
- Reduce epochs:
  ```python
  trainer.train(num_epochs=5)  # Instead of 20
  ```

#### 4. Poor Performance

**Problem:** Model accuracy is very low (<50%)

**Possible causes:**
- Learning rate too high/low
- Insufficient training (stopped too early)
- Poor hyperparameter choices

**Solutions:**
```python
# Try different learning rates
trainer = Trainer(model, train_loader, val_loader, device, learning_rate=0.0001)

# Increase patience
trainer = Trainer(..., patience=10)

# Adjust filter ratio
model = HierarchicalAttentionClassifier(..., filter_ratio=0.7)
```

---

## Advanced Usage

### Custom Dataset

To use your own dataset instead of 20 Newsgroups:

```python
from torch.utils.data import Dataset, DataLoader
from data_loader import NewsgroupsDataset

# Prepare your data
texts = ["Your document 1...", "Your document 2...", ...]
labels = [0, 1, 2, ...]  # Numeric labels

# Build vocabulary
from data_loader import build_vocabulary
vocab = build_vocabulary(texts, max_vocab_size=20000)

# Create dataset
dataset = NewsgroupsDataset(texts, labels, vocab)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train as usual
model = HierarchicalAttentionClassifier(vocab_size=len(vocab), num_classes=num_classes)
trainer = Trainer(model, loader, ...)
```

### Analyzing Specific Examples

```python
# Get attention for a specific document
import torch

model.eval()
with torch.no_grad():
    # Prepare input
    input_ids = ...  # Your tokenized document
    sentence_masks = ...

    # Get predictions with attention
    logits, attention_dict = model(
        input_ids.unsqueeze(0).to(device),
        sentence_masks.unsqueeze(0).to(device),
        return_attention=True
    )

    # Extract attention weights
    word_attention = attention_dict['word_attention_scores']
    cross_attention = attention_dict['cross_attention_weights']

    # Visualize or analyze
    print("Word attention scores:", word_attention)
    print("Most important words:", word_attention.topk(10))
```

### Hyperparameter Tuning

```python
# Grid search example
configs = [
    {'hidden_dim': 128, 'filter_ratio': 0.3},
    {'hidden_dim': 256, 'filter_ratio': 0.5},
    {'hidden_dim': 512, 'filter_ratio': 0.7},
]

best_acc = 0
best_config = None

for config in configs:
    model = HierarchicalAttentionClassifier(**config)
    trainer = Trainer(model, ...)
    trainer.train(num_epochs=10)

    # Evaluate
    results, _, _, _ = evaluate_model(model, test_loader, device, label_names)

    if results['accuracy'] > best_acc:
        best_acc = results['accuracy']
        best_config = config

print(f"Best config: {best_config}, Accuracy: {best_acc}")
```

### Transfer Learning

```python
# Load pre-trained model
checkpoint = torch.load('models/hierarchical_attention_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Fine-tune on new data
trainer = Trainer(model, new_train_loader, new_val_loader, device)
trainer.train(num_epochs=5)
```

---

## Performance Benchmarks

### Expected Results (20 Newsgroups)

| Model | Accuracy | F1 Score | Training Time (GPU) | Parameters |
|-------|----------|----------|---------------------|------------|
| Hierarchical Attention | 75-82% | 0.74-0.80 | 20-30 min | ~5.2M |
| Baseline LSTM | 70-76% | 0.68-0.74 | 15-20 min | ~4.1M |
| Baseline CNN | 68-74% | 0.66-0.72 | 10-15 min | ~3.8M |

### Hardware Requirements

| Hardware | Training Time | Inference Time | Memory |
|----------|---------------|----------------|--------|
| CPU (8 cores) | 1-2 hours | 10-15 sec/batch | 4-8 GB |
| GPU (GTX 1080) | 20-30 min | 1-2 sec/batch | 4-6 GB |
| GPU (RTX 3090) | 10-15 min | <1 sec/batch | 4-6 GB |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{hierarchical_attention_newsgroups,
  title={Hierarchical Attention Model for 20 Newsgroups Classification},
  author={Your Name},
  year={2025},
  howpublished={GitHub}
}
```

---

## Support

For issues, questions, or contributions:
- Check `REPORT.md` for detailed documentation
- Review `README.md` for quick reference
- Open an issue on GitHub
- Contact: your.email@example.com

---

**Good luck with your text classification project!** 🚀
