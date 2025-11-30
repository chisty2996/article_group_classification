# Hierarchical Attention Model for 20 Newsgroups Classification

A comprehensive deep learning project implementing a custom hierarchical attention architecture for text classification on the 20 Newsgroups dataset.

## Project Overview

This project implements a novel neural architecture combining:
- **Contextual Encoder** (Bi-LSTM)
- **Word-level Attention Filtering**
- **Sentence Representation** (multi-pooling)
- **Document-level Cross-Attention**
- **Classification Layer**

## Features

- Complete implementation of hierarchical attention model
- Two baseline models (LSTM, CNN) for comparison
- Comprehensive evaluation metrics and visualizations
- Detailed error analysis and failure mode detection
- Attention pattern visualization
- Bias propagation analysis

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Optional: Test GPU/MPS availability
python test_mps.py
```

### Training

```bash
# Train all models (hierarchical + baselines)
python main.py --train_main --train_baselines --num_epochs 20

# Train only the main model
python main.py --train_main --num_epochs 15

# Custom configuration
python main.py --hidden_dim 512 --num_heads 8 --filter_ratio 0.6
```

### Using Jupyter Notebook

For an interactive experience with detailed explanations:

```bash
jupyter notebook 20_newsgroups_classification.ipynb
```

## Project Structure

```
article_group_classification/
├── model.py                 # Model architectures
├── data_loader.py           # Data loading and preprocessing
├── train.py                 # Training and evaluation utilities
├── visualization.py         # Visualization tools
├── error_analysis.py        # Failure mode analysis
├── main.py                  # Main training script
├── 20_newsgroups_classification.ipynb  # Interactive notebook
├── REPORT.md                # Comprehensive project report
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Model Architecture

```
Input → Bi-LSTM Encoder → Word Attention Filter → Sentence Pooling
                                                          ↓
                                               Cross-Attention
                                                          ↓
                                                  Classification
```

See `REPORT.md` for detailed architecture description and design rationale.

## Results

Expected performance:
- **Hierarchical Attention**: 75-82% accuracy
- **Baseline LSTM**: 70-76% accuracy
- **Baseline CNN**: 68-74% accuracy

All results, visualizations, and trained models are saved to:
- `models/` - Model checkpoints
- `results/` - Evaluation metrics (JSON)
- `visualizations/` - Plots and attention heatmaps

## Deliverables

1. ✅ **Architecture Description** - See `REPORT.md` Section 1
2. ✅ **Baseline Comparison** - See `REPORT.md` Section 3
3. ✅ **Attention Visualizations** - See `visualizations/` directory
4. ✅ **Error Analysis** - See `REPORT.md` Section 5
5. ✅ **Attention Mechanism Impact** - See `REPORT.md` Section 4
6. ✅ **Architectural Limitations** - See `REPORT.md` Section 7
7. ✅ **Bias Propagation Analysis** - See `REPORT.md` Section 6

## Key Findings

### Failure Modes Identified:

1. **Low-Quality Word Filtering** - Uniform attention scores fail to distinguish important words
2. **Context Representation Collapse** - LSTM produces overly similar embeddings
3. **Attention Sparsity Errors** - Over-concentration on few words
4. **Noisy Word Amplification** - High attention to uninformative tokens
5. **Cross-Attention Misalignment** - Conflict between local and global signals

### Proposed Improvements:

1. **Adaptive Filtering** - Learn document-specific filter ratios
2. **Diversity Regularization** - Encourage distributed attention
3. **Gated Fusion** - Balance local and global information
4. **Pre-trained Encoders** - Use BERT instead of Bi-LSTM
5. **Efficient Attention** - Linear complexity mechanisms for scalability

See `REPORT.md` Section 6 for detailed technical solutions.

## Configuration Options

```bash
python main.py --help
```

Key parameters:
- `--embedding_dim` (default: 200) - Word embedding dimension
- `--hidden_dim` (default: 256) - LSTM hidden dimension
- `--num_heads` (default: 4) - Number of attention heads
- `--filter_ratio` (default: 0.5) - Percentage of words to keep
- `--pooling_method` (default: 'multi') - Sentence pooling strategy
- `--num_epochs` (default: 20) - Training epochs
- `--learning_rate` (default: 0.001) - Learning rate

## Visualizations

The project generates:
- Model comparison bar charts
- Confusion matrices
- Per-class performance analysis
- Training curves (loss and accuracy)
- Word-level attention heatmaps
- Document-level cross-attention patterns
- Attention distribution statistics

## Citation

If you use this code, please cite:

```
@misc{newsgroups_hierarchical_attention,
  title={Hierarchical Attention Model for 20 Newsgroups Classification},
  author={Your Name},
  year={2025},
  howpublished={GitHub repository}
}
```

## License

MIT License - feel free to use for research and educational purposes.

## Acknowledgments

- 20 Newsgroups dataset: http://qwone.com/~jason/20Newsgroups/
- Inspired by Yang et al., "Hierarchical Attention Networks for Document Classification" (2016)

## Contact

For questions or issues, please open a GitHub issue or contact the author.

---

**Happy classifying!** 🚀
