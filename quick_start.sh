#!/bin/bash

# Quick Start Script for Hierarchical Attention Model
# This script sets up the environment and runs the complete pipeline

set -e  # Exit on error

echo "=============================================================================="
echo "  Hierarchical Attention Model for 20 Newsgroups Classification"
echo "  Quick Start Script"
echo "=============================================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Download NLTK data
echo "Downloading NLTK data..."
python3 -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ NLTK data downloaded"
echo ""

# Test device availability
echo "Checking GPU/MPS availability..."
python3 test_mps.py
echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p models
mkdir -p results
mkdir -p visualizations
mkdir -p visualizations/word_attention
mkdir -p visualizations/cross_attention
echo "✓ Directories created"
echo ""

# Run the complete pipeline
echo "=============================================================================="
echo "  Starting Complete Pipeline"
echo "=============================================================================="
echo ""
echo "This will:"
echo "  1. Load 20 Newsgroups dataset"
echo "  2. Train hierarchical attention model + 2 baselines"
echo "  3. Evaluate all models on test set"
echo "  4. Generate visualizations and analyses"
echo "  5. Perform failure mode analysis"
echo ""
echo "Expected time: 1-2 hours on CPU, 20-30 minutes on GPU (MPS/CUDA)"
echo ""

read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Run pipeline
python3 run_complete_pipeline.py

echo ""
echo "=============================================================================="
echo "  Pipeline Complete!"
echo "=============================================================================="
echo ""
echo "Generated outputs:"
echo "  📁 models/           - Trained model checkpoints"
echo "  📁 results/          - Evaluation metrics (JSON)"
echo "  📁 visualizations/   - Charts and attention heatmaps"
echo ""
echo "Next steps:"
echo "  1. Check results/model_comparison.json for performance metrics"
echo "  2. View visualizations/ for charts and attention patterns"
echo "  3. Read REPORT.md for detailed analysis"
echo "  4. Review results/failure_analysis.json for error analysis"
echo ""
echo "✅ All tasks completed successfully!"
echo ""
