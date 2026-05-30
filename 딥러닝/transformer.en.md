# Transformer Deep Dive

[🇰🇷 Korean](./transformer.md) | 🇺🇸 English

A code-first walkthrough of the Transformer architecture introduced in  
"Attention Is All You Need" (Vaswani et al., 2017), and how it underpins BERT and GPT.

---

## 1. Self-Attention Mechanism

### Core Idea

Every word in a sentence computes an **attention weight** with every other word,  
producing a context-aware representation that goes far beyond static embeddings.

```
"The animal was too tired to cross the river. It was a lion."
                                               ↑
                    "It" attends strongly to "animal" and "lion"
```

### Q, K, V Matrices

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q (Query): What the current token is looking for
    K (Key):   What each token has to offer
    V (Value): The actual information to retrieve
    """
    d_k = Q.size(-1)

    # 1. Dot product between Q and K → similarity scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. (Optional) Causal mask: prevent attending to future tokens (used in GPT)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax → attention weights (sum to 1)
    attn_weights = F.softmax(scores, dim=-1)

    # 4. Weighted sum of V → context-aware representation
    return torch.matmul(attn_weights, V), attn_weights
```

**The Formula:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$: Query-Key similarity — higher score means more attention
- $\sqrt{d_k}$: Scaling prevents softmax from saturating in high dimensions
- Softmax: Normalizes scores into probabilities
- Output: Weighted sum of V — **a representation enriched with context**

---

## 2. Multi-Head Attention

A single attention head only captures one kind of relationship.  
**Multiple heads run in parallel**, each learning a different pattern (syntax, semantics, coreference, etc.).

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        # (B, seq, d_model) → (B, num_heads, seq, d_k)
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.split_heads(self.W_q(Q), B)
        K = self.split_heads(self.W_k(K), B)
        V = self.split_heads(self.W_v(V), B)

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate all heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)
```

**Example:** BERT-base has 12 heads, d_model=768 → each head has d_k=64.  
Different heads learn to track grammar, meaning, pronoun references, and more.

---

## 3. Positional Encoding

Attention has no inherent sense of order → **inject position information using sinusoidal functions**.

```python
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # Even dims: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dims: cos

        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])
```

---

## 4. Transformer Encoder Block

```python
class TransformerEncoderBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.GELU(),                 # BERT uses GELU activation
            torch.nn.Linear(d_ff, d_model)
        )
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + Residual Connection
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        # Feed-Forward + Residual Connection
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x
```

**Architecture Diagram:**

```
Token Embeddings
    + Positional Encoding
         ↓
  [Encoder Block] × N
  ┌────────────────────────────┐
  │  Multi-Head Self-Attention │
  │  + Residual + LayerNorm    │
  │  Feed-Forward Network      │
  │  + Residual + LayerNorm    │
  └────────────────────────────┘
         ↓
  Contextual Representations
```

---

## 5. BERT vs GPT Architecture

| | BERT | GPT |
|-|------|-----|
| Architecture | Encoder Only | Decoder Only |
| Attention | Bidirectional (sees all tokens) | Causal (left-to-right only) |
| Pretraining Goal | MLM (predict masked tokens) + NSP | Next token prediction (LM) |
| Strengths | Understanding (classification, NER, QA) | Generation (summarization, translation) |
| Representative Models | BERT, RoBERTa, KLUE-BERT | GPT-2/3/4, LLaMA, Mistral |

### BERT Pretraining: Masked Language Model (MLM)

```
Input:  "I [MASK] machine learning"    → Predict: "love"
Input:  "I love [MASK] learning"       → Predict: "machine"

→ Model must look in both directions to fill in the mask
```

### GPT Pretraining: Causal Language Modeling

```
Input:  "I love machine"               → Predict: "learning"
        Attention mask: only left context allowed

→ Naturally optimized for text generation
```

---

## 6. Fine-tuning Strategies

### BERT Fine-tuning for Classification

```python
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)

# Key: very small learning rate to preserve pretrained representations
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Warmup scheduler: ramp up slowly, then decay
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1),
    num_training_steps=total_steps
)

for batch in train_loader:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

**Fine-tuning Tips**
- Learning rate: `2e-5` ~ `5e-5` (too large destroys pretrained knowledge)
- Epochs: 3–5 (watch for overfitting)
- Warmup: ~10% of total steps

### GPT Prompt Engineering

```python
# Few-shot prompting
prompt = """
Classify the sentiment of each review as Positive or Negative.

Review: "The screen is crisp and the battery lasts all day."
Sentiment: Positive

Review: "Shipping took forever and the packaging was damaged."
Sentiment: Negative

Review: "Excellent value for the price."
Sentiment:"""

# Chain-of-Thought prompting
cot_prompt = """
Think step by step before answering.
Question: How would you address overfitting in a machine learning model?
Step-by-step reasoning:"""
```

---

## References

- [Attention Is All You Need (Original Paper)](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers (Original Paper)](https://arxiv.org/abs/1810.04805)
- [The Illustrated Transformer — Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
