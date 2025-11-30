"""
BERT-compatible data loader for 20 Newsgroups dataset
Uses BERT tokenizer instead of custom vocabulary
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class NewsgroupsDatasetBERT(Dataset):
    """Dataset class for 20 Newsgroups with BERT tokenization"""

    def __init__(self, texts, labels, max_sent_len=100, max_num_sent=30):
        self.texts = texts
        self.labels = labels
        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent

        # Use BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Split into sentences
        sentences = nltk.sent_tokenize(text)

        # Limit number of sentences
        sentences = sentences[:self.max_num_sent]

        # Tokenize with BERT tokenizer
        sentence_indices = []
        sentence_masks = []

        for sent in sentences:
            # BERT tokenization
            encoded = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=True,
                max_length=self.max_sent_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            sentence_indices.append(encoded['input_ids'].squeeze(0))
            sentence_masks.append(encoded['attention_mask'].squeeze(0))

        # Pad to max number of sentences
        while len(sentence_indices) < self.max_num_sent:
            sentence_indices.append(torch.zeros(self.max_sent_len, dtype=torch.long))
            sentence_masks.append(torch.zeros(self.max_sent_len, dtype=torch.long))

        # Stack into tensors
        input_ids = torch.stack(sentence_indices)
        attention_masks = torch.stack(sentence_masks)

        return {
            'input_ids': input_ids,
            'sentence_masks': attention_masks,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_20newsgroups_data_bert(subset='all', max_sent_len=100, max_num_sent=30, test_size=0.15):
    """
    Load and preprocess 20 newsgroups dataset for BERT

    Args:
        subset: 'train', 'test', or 'all'
        max_sent_len: maximum sentence length
        max_num_sent: maximum number of sentences per document
        test_size: validation split ratio

    Returns:
        train_loader, val_loader, test_loader, tokenizer, label_names
    """

    print("Loading 20 Newsgroups dataset for BERT...")

    # Load data
    if subset == 'all':
        train_data = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

        train_texts = train_data.data
        train_labels = train_data.target
        test_texts = test_data.data
        test_labels = test_data.target

    else:
        data = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))
        train_texts = data.data
        train_labels = data.target
        test_texts = []
        test_labels = []

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
        train_dataset = NewsgroupsDatasetBERT(train_texts, train_labels, max_sent_len, max_num_sent)
        val_dataset = NewsgroupsDatasetBERT(val_texts, val_labels, max_sent_len, max_num_sent)
        test_dataset = NewsgroupsDatasetBERT(test_texts, test_labels, max_sent_len, max_num_sent)

        # Get tokenizer for reference
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Create data loaders with reduced batch size for BERT
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

        print(f"\nUsing BERT tokenizer (vocab size: {tokenizer.vocab_size})")

        return train_loader, val_loader, test_loader, tokenizer, label_names

    return None, None, None, None, label_names
