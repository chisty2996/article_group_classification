"""
Data loading and preprocessing for 20 Newsgroups dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from collections import Counter
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class NewsgroupsDataset(Dataset):
    """Dataset class for 20 Newsgroups with hierarchical structure"""

    def __init__(self, texts, labels, vocab, max_sent_len=100, max_num_sent=30):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Split into sentences
        sentences = nltk.sent_tokenize(text)

        # Limit number of sentences
        sentences = sentences[:self.max_num_sent]

        # Tokenize and convert to indices
        sentence_indices = []
        sentence_masks = []

        for sent in sentences:
            # Simple tokenization
            tokens = self._tokenize(sent)
            indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

            # Truncate or pad sentence
            if len(indices) > self.max_sent_len:
                indices = indices[:self.max_sent_len]
            else:
                indices = indices + [0] * (self.max_sent_len - len(indices))

            mask = [1 if idx != 0 else 0 for idx in indices]

            sentence_indices.append(indices)
            sentence_masks.append(mask)

        # Pad to max number of sentences
        while len(sentence_indices) < self.max_num_sent:
            sentence_indices.append([0] * self.max_sent_len)
            sentence_masks.append([0] * self.max_sent_len)

        return {
            'input_ids': torch.tensor(sentence_indices, dtype=torch.long),
            'sentence_masks': torch.tensor(sentence_masks, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def _tokenize(self, text):
        """Simple word tokenization"""
        # Convert to lowercase and split
        text = text.lower()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text)
        tokens = text.split()
        return tokens


def build_vocabulary(texts, max_vocab_size=20000, min_freq=2):
    """Build vocabulary from texts"""

    print("Building vocabulary...")
    word_counts = Counter()

    for text in texts:
        # Tokenize
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\.\,\!\?]', '', text)
        tokens = text.split()
        word_counts.update(tokens)

    # Create vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}

    # Add words that meet minimum frequency
    for word, count in word_counts.most_common(max_vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_20newsgroups_data(subset='all', max_vocab_size=20000, min_freq=2,
                          max_sent_len=100, max_num_sent=30, test_size=0.15,
                          use_bert_tokenizer=False):
    """
    Load and preprocess 20 newsgroups dataset

    Args:
        subset: 'train', 'test', or 'all'
        max_vocab_size: maximum vocabulary size
        min_freq: minimum word frequency
        max_sent_len: maximum sentence length
        max_num_sent: maximum number of sentences per document
        test_size: validation split ratio
        use_bert_tokenizer: if True, use BERT tokenizer instead of custom vocab

    Returns:
        train_loader, val_loader, test_loader, vocab, label_names
    """

    print("Loading 20 Newsgroups dataset...")

    # Load data
    if subset == 'all':
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        # Combine for vocab building
        all_texts = list(train_data.data) + list(test_data.data)
        train_texts = train_data.data
        train_labels = train_data.target
        test_texts = test_data.data
        test_labels = test_data.target

    else:
        data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        all_texts = data.data
        train_texts = data.data
        train_labels = data.target
        test_texts = []
        test_labels = []

    # Build vocabulary from all texts
    vocab = build_vocabulary(all_texts, max_vocab_size, min_freq)

    # Get label names
    label_names = fetch_20newsgroups(subset='train').target_names

    # Split training data into train and validation
    if len(train_texts) > 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels, test_size=test_size, random_state=42, stratify=train_labels
        )

        print(f"Train size: {len(train_texts)}")
        print(f"Validation size: {len(val_texts)}")
        print(f"Test size: {len(test_texts)}")

        # Create datasets
        train_dataset = NewsgroupsDataset(train_texts, train_labels, vocab, max_sent_len, max_num_sent)
        val_dataset = NewsgroupsDataset(val_texts, val_labels, vocab, max_sent_len, max_num_sent)
        test_dataset = NewsgroupsDataset(test_texts, test_labels, vocab, max_sent_len, max_num_sent)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

        return train_loader, val_loader, test_loader, vocab, label_names

    return None, None, None, vocab, label_names


def get_label_distribution(data_loader, label_names):
    """Get distribution of labels in dataset"""
    label_counts = Counter()

    for batch in data_loader:
        labels = batch['labels'].numpy()
        label_counts.update(labels)

    print("\nLabel distribution:")
    for label_id, count in sorted(label_counts.items()):
        print(f"{label_names[label_id]}: {count}")

    return label_counts
