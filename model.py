"""
Custom Deep Learning Model for 20 Newsgroups Classification

Architecture Components:
1. Contextual Encoder (Bi-LSTM)
2. Word-level Attention Filtering
3. Sentence Representation (pooling strategies)
4. Document-level Cross-Attention
5. Classification Layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WordLevelAttentionFilter(nn.Module):
    """
    Word-level attention filtering mechanism that scores and selects
    the most important words from the contextual representations.

    Uses self-attention to compute importance scores for each word,
    then filters based on a learned threshold or top-k selection.
    """
    def __init__(self, hidden_dim, dropout=0.3, filter_ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.filter_ratio = filter_ratio

        # Attention scoring network
        self.attention_query = nn.Linear(hidden_dim, hidden_dim)
        self.attention_key = nn.Linear(hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, word_embeddings, mask=None):
        """
        Args:
            word_embeddings: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - padding mask

        Returns:
            filtered_words: (batch, filtered_len, hidden_dim)
            attention_scores: (batch, seq_len)
            word_indices: (batch, filtered_len) - indices of selected words
        """
        batch_size, seq_len, hidden_dim = word_embeddings.shape

        # Ensure mask matches embeddings size
        if mask is not None and mask.size(1) != seq_len:
            # Pad or truncate mask to match sequence length
            if mask.size(1) < seq_len:
                padding = torch.zeros(batch_size, seq_len - mask.size(1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, padding], dim=1)
            else:
                mask = mask[:, :seq_len]

        # Compute self-attention scores
        query = self.attention_query(word_embeddings)  # (batch, seq_len, hidden_dim)
        key = self.attention_key(word_embeddings)      # (batch, seq_len, hidden_dim)

        # Scaled dot-product attention
        attention_logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_dim)

        if mask is not None:
            attention_logits = attention_logits.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attention_weights = F.softmax(attention_logits, dim=-1)
        attended_features = torch.matmul(attention_weights, word_embeddings)

        # Compute importance scores for each word
        importance_scores = self.attention_score(attended_features).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            importance_scores = importance_scores.masked_fill(mask == 0, -1e9)

        # Normalize scores
        attention_scores = torch.sigmoid(importance_scores)

        # Filter top-k words based on importance
        k = max(1, int(seq_len * self.filter_ratio))
        top_k_scores, top_k_indices = torch.topk(attention_scores, k, dim=-1)

        # Gather filtered word embeddings
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, k).to(word_embeddings.device)
        filtered_words = word_embeddings[batch_indices, top_k_indices]

        # Apply layer normalization and dropout
        filtered_words = self.layer_norm(filtered_words)
        filtered_words = self.dropout(filtered_words)

        return filtered_words, attention_scores, top_k_indices


class SentenceRepresentation(nn.Module):
    """
    Sentence representation module that aggregates word-level features
    into fixed-length sentence vectors using multiple pooling strategies.
    """
    def __init__(self, hidden_dim, pooling_method='multi', dropout=0.3):
        super().__init__()
        self.pooling_method = pooling_method
        self.hidden_dim = hidden_dim

        if pooling_method == 'multi':
            # Combine max, mean, and learned attention pooling
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            # Output will be concatenation of max, mean, and attention pooling
            self.output_dim = hidden_dim * 3
        elif pooling_method == 'attention':
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
            self.output_dim = hidden_dim
        else:  # 'max' or 'mean'
            self.output_dim = hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, word_embeddings, mask=None):
        """
        Args:
            word_embeddings: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len)

        Returns:
            sentence_repr: (batch, output_dim)
        """
        # Ensure mask matches embeddings size
        if mask is not None and mask.size(1) != word_embeddings.size(1):
            seq_len = word_embeddings.size(1)
            if mask.size(1) < seq_len:
                padding = torch.zeros(mask.size(0), seq_len - mask.size(1), device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, padding], dim=1)
            else:
                mask = mask[:, :seq_len]

        if self.pooling_method == 'max':
            if mask is not None:
                word_embeddings_masked = word_embeddings.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            else:
                word_embeddings_masked = word_embeddings
            sentence_repr = torch.max(word_embeddings_masked, dim=1)[0]

        elif self.pooling_method == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(word_embeddings)
                sum_embeddings = (word_embeddings * mask_expanded).sum(dim=1)
                sentence_repr = sum_embeddings / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                sentence_repr = word_embeddings.mean(dim=1)

        elif self.pooling_method == 'attention':
            # Attention-based pooling
            attention_scores = self.attention_pooling(word_embeddings).squeeze(-1)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
            sentence_repr = (word_embeddings * attention_weights).sum(dim=1)

        else:  # 'multi'
            # Max pooling
            if mask is not None:
                word_embeddings_masked = word_embeddings.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            else:
                word_embeddings_masked = word_embeddings
            max_pool = torch.max(word_embeddings_masked, dim=1)[0]

            # Mean pooling
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(word_embeddings)
                sum_embeddings = (word_embeddings * mask_expanded).sum(dim=1)
                mean_pool = sum_embeddings / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                mean_pool = word_embeddings.mean(dim=1)

            # Attention pooling
            attention_scores = self.attention_pooling(word_embeddings).squeeze(-1)
            if mask is not None:
                attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)
            attn_pool = (word_embeddings * attention_weights).sum(dim=1)

            sentence_repr = torch.cat([max_pool, mean_pool, attn_pool], dim=-1)

        sentence_repr = self.layer_norm(sentence_repr)
        sentence_repr = self.dropout(sentence_repr)

        return sentence_repr


class DocumentCrossAttention(nn.Module):
    """
    Document-level cross-attention module that uses filtered word representations
    as queries and sentence representations as keys/values to create a
    document-level representation.
    """
    def __init__(self, word_dim, sentence_dim, num_heads=4, dropout=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.word_dim = word_dim
        self.sentence_dim = sentence_dim

        # Multi-head attention components
        self.query_proj = nn.Linear(word_dim, word_dim)
        self.key_proj = nn.Linear(sentence_dim, word_dim)
        self.value_proj = nn.Linear(sentence_dim, word_dim)

        self.head_dim = word_dim // num_heads
        assert self.head_dim * num_heads == word_dim, "word_dim must be divisible by num_heads"

        self.output_proj = nn.Linear(word_dim, word_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(word_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(word_dim, word_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(word_dim * 4, word_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(word_dim)

    def forward(self, filtered_words, sentence_reprs):
        """
        Args:
            filtered_words: (batch, num_filtered_words, word_dim) - queries
            sentence_reprs: (batch, num_sentences, sentence_dim) - keys/values

        Returns:
            doc_repr: (batch, word_dim) - aggregated document representation
            cross_attention_weights: (batch, num_heads, num_filtered_words, num_sentences)
        """
        batch_size = filtered_words.size(0)

        # Project queries, keys, values
        queries = self.query_proj(filtered_words)  # (batch, num_words, word_dim)
        keys = self.key_proj(sentence_reprs)        # (batch, num_sentences, word_dim)
        values = self.value_proj(sentence_reprs)    # (batch, num_sentences, word_dim)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.word_dim)

        # Output projection with residual connection
        attended = self.output_proj(context)
        attended = self.layer_norm(attended + filtered_words)

        # Feed-forward network with residual
        output = self.ffn(attended)
        output = self.ffn_norm(output + attended)

        # Aggregate to document level (mean pooling over filtered words)
        doc_repr = output.mean(dim=1)

        return doc_repr, attention_weights


class ContextualEncoder(nn.Module):
    """
    Bi-LSTM based contextual encoder that processes word embeddings
    to produce contextual representations.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, input_ids, mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            mask: (batch, seq_len)

        Returns:
            contextual_embeddings: (batch, seq_len, hidden_dim)
        """
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        # Pack padded sequence for efficient LSTM processing
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            # Handle empty sequences (length 0) by clamping to at least 1
            lengths = torch.clamp(lengths, min=1).long()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeddings)

        lstm_out = self.layer_norm(lstm_out)
        return lstm_out


class HierarchicalAttentionClassifier(nn.Module):
    """
    Complete model combining:
    1. Contextual Encoder (Bi-LSTM)
    2. Word-level Attention Filtering
    3. Sentence Representation
    4. Document-level Cross-Attention
    5. Classification Layer
    """
    def __init__(
        self,
        vocab_size,
        num_classes,
        embedding_dim=200,
        hidden_dim=256,
        num_lstm_layers=2,
        num_attention_heads=4,
        filter_ratio=0.5,
        pooling_method='multi',
        dropout=0.3,
        max_sent_len=100,
        max_num_sent=30
    ):
        super().__init__()

        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent

        # Component 1: Contextual Encoder
        self.encoder = ContextualEncoder(
            vocab_size, embedding_dim, hidden_dim, num_lstm_layers, dropout
        )

        # Component 2: Word-level Attention Filter
        self.word_filter = WordLevelAttentionFilter(hidden_dim, dropout, filter_ratio)

        # Component 3: Sentence Representation
        self.sentence_repr = SentenceRepresentation(hidden_dim, pooling_method, dropout)

        # Component 4: Document-level Cross-Attention
        self.cross_attention = DocumentCrossAttention(
            hidden_dim, self.sentence_repr.output_dim, num_attention_heads, dropout
        )

        # Component 5: Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids, sentence_masks=None, return_attention=False):
        """
        Args:
            input_ids: (batch, num_sentences, max_sent_len)
            sentence_masks: (batch, num_sentences, max_sent_len)
            return_attention: bool - whether to return attention weights

        Returns:
            logits: (batch, num_classes)
            (optional) attention_dict: dictionary of attention weights and scores
        """
        batch_size, num_sentences, sent_len = input_ids.shape

        # Reshape for processing
        input_ids_flat = input_ids.view(batch_size * num_sentences, sent_len)
        if sentence_masks is not None:
            masks_flat = sentence_masks.view(batch_size * num_sentences, sent_len)
        else:
            masks_flat = None

        # 1. Contextual Encoding
        contextual_embeddings = self.encoder(input_ids_flat, masks_flat)

        # 2. Word-level Attention Filtering
        filtered_words, word_scores, word_indices = self.word_filter(
            contextual_embeddings, masks_flat
        )

        # 3. Sentence Representation
        sentence_embeddings = self.sentence_repr(contextual_embeddings, masks_flat)
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)

        # Reshape filtered words for document-level processing
        num_filtered = filtered_words.size(1)
        filtered_words = filtered_words.view(batch_size, num_sentences * num_filtered, -1)

        # 4. Document-level Cross-Attention
        doc_representation, cross_attn_weights = self.cross_attention(
            filtered_words, sentence_embeddings
        )

        # 5. Classification
        logits = self.classifier(doc_representation)

        if return_attention:
            attention_dict = {
                'word_attention_scores': word_scores.view(batch_size, num_sentences, sent_len),
                'word_indices': word_indices.view(batch_size, num_sentences, num_filtered),
                'cross_attention_weights': cross_attn_weights,
                'sentence_embeddings': sentence_embeddings,
                'filtered_words': filtered_words
            }
            return logits, attention_dict

        return logits


class BaselineLSTM(nn.Module):
    """Baseline: Simple Bi-LSTM with max pooling"""
    def __init__(self, vocab_size, num_classes, embedding_dim=200, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids, **kwargs):
        batch_size, num_sent, sent_len = input_ids.shape
        input_ids = input_ids.view(batch_size, -1)

        embeddings = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embeddings)
        pooled = torch.max(lstm_out, dim=1)[0]
        logits = self.fc(self.dropout(pooled))
        return logits


class BaselineCNN(nn.Module):
    """Baseline: CNN-based text classifier"""
    def __init__(self, vocab_size, num_classes, embedding_dim=200, num_filters=128, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        kernel_sizes = [3, 4, 5]
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, input_ids, **kwargs):
        batch_size, num_sent, sent_len = input_ids.shape
        input_ids = input_ids.view(batch_size, -1)

        embeddings = self.dropout(self.embedding(input_ids))
        embeddings = embeddings.transpose(1, 2)

        conv_outputs = [F.relu(conv(embeddings)) for conv in self.convs]
        pooled = [F.max_pool1d(out, out.size(2)).squeeze(2) for out in conv_outputs]
        concat = torch.cat(pooled, dim=1)

        logits = self.fc(self.dropout(concat))
        return logits


class BERTContextualEncoder(nn.Module):
    """
    BERT-based contextual encoder
    Uses pre-trained BERT for better contextual representations
    """
    def __init__(self, hidden_dim=768, dropout=0.3):
        super().__init__()
        from transformers import BertModel

        # Use smaller BERT model for efficiency
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_dim = hidden_dim

        # Freeze BERT parameters initially (can be fine-tuned)
        for param in self.bert.parameters():
            param.requires_grad = True  # Allow fine-tuning

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(768)  # BERT outputs 768-dim

        # Project BERT output to desired hidden_dim if different
        if hidden_dim != 768:
            self.projection = nn.Linear(768, hidden_dim)
        else:
            self.projection = None

    def forward(self, input_ids, mask=None):
        """
        Args:
            input_ids: (batch, seq_len)
            mask: (batch, seq_len) - attention mask

        Returns:
            contextual_embeddings: (batch, seq_len, hidden_dim)
        """
        # Create attention mask for BERT (1 for real tokens, 0 for padding)
        attention_mask = mask if mask is not None else torch.ones_like(input_ids)

        # Get BERT outputs
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (batch, seq_len, 768)

        # Apply layer norm and dropout
        sequence_output = self.layer_norm(sequence_output)
        sequence_output = self.dropout(sequence_output)

        # Project if needed
        if self.projection is not None:
            sequence_output = self.projection(sequence_output)

        return sequence_output


class HierarchicalAttentionBERT(nn.Module):
    """
    Hierarchical Attention model using BERT as encoder instead of Bi-LSTM
    Combines BERT contextual encoding with hierarchical attention mechanism
    """
    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        num_attention_heads=4,
        filter_ratio=0.5,
        pooling_method='multi',
        dropout=0.3,
        max_sent_len=100,
        max_num_sent=30
    ):
        super().__init__()

        self.max_sent_len = max_sent_len
        self.max_num_sent = max_num_sent

        # Component 1: BERT Contextual Encoder
        self.encoder = BERTContextualEncoder(hidden_dim=hidden_dim, dropout=dropout)

        # Component 2: Word-level Attention Filter
        self.word_filter = WordLevelAttentionFilter(hidden_dim, dropout, filter_ratio)

        # Component 3: Sentence Representation
        self.sentence_repr = SentenceRepresentation(hidden_dim, pooling_method, dropout)

        # Component 4: Document-level Cross-Attention
        self.cross_attention = DocumentCrossAttention(
            hidden_dim, self.sentence_repr.output_dim, num_attention_heads, dropout
        )

        # Component 5: Classification Layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, input_ids, sentence_masks=None, return_attention=False):
        """
        Args:
            input_ids: (batch, num_sentences, max_sent_len)
            sentence_masks: (batch, num_sentences, max_sent_len)
            return_attention: bool - whether to return attention weights

        Returns:
            logits: (batch, num_classes)
            (optional) attention_dict: dictionary of attention weights
        """
        batch_size, num_sentences, sent_len = input_ids.shape

        # Reshape for BERT processing
        input_ids_flat = input_ids.view(batch_size * num_sentences, sent_len)
        if sentence_masks is not None:
            masks_flat = sentence_masks.view(batch_size * num_sentences, sent_len)
        else:
            masks_flat = None

        # 1. BERT Contextual Encoding
        contextual_embeddings = self.encoder(input_ids_flat, masks_flat)

        # 2. Word-level Attention Filtering
        filtered_words, word_scores, word_indices = self.word_filter(
            contextual_embeddings, masks_flat
        )

        # 3. Sentence Representation
        sentence_embeddings = self.sentence_repr(contextual_embeddings, masks_flat)
        sentence_embeddings = sentence_embeddings.view(batch_size, num_sentences, -1)

        # Reshape filtered words for document-level processing
        num_filtered = filtered_words.size(1)
        filtered_words = filtered_words.view(batch_size, num_sentences * num_filtered, -1)

        # 4. Document-level Cross-Attention
        doc_representation, cross_attn_weights = self.cross_attention(
            filtered_words, sentence_embeddings
        )

        # 5. Classification
        logits = self.classifier(doc_representation)

        if return_attention:
            attention_dict = {
                'word_attention_scores': word_scores.view(batch_size, num_sentences, sent_len),
                'word_indices': word_indices.view(batch_size, num_sentences, num_filtered),
                'cross_attention_weights': cross_attn_weights,
                'sentence_embeddings': sentence_embeddings,
                'filtered_words': filtered_words
            }
            return logits, attention_dict

        return logits
