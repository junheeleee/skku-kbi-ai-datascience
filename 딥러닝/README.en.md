# Deep Learning

[🇰🇷 Korean](./README.md) | 🇺🇸 English

Covers neural network theory and practice using PyTorch.  
Progresses through MLP → CNN → RNN/LSTM → Transformer (BERT/GPT) → Generative Models.

---

## Results Summary

| Task | Dataset | Model | Result |
|------|---------|-------|--------|
| Binary Classification | MNIST | MLP | Accuracy ~99% |
| Image Classification | MNIST | CNN | Accuracy ~99% |
| Stock Price Forecasting | Stock market data | LSTM | MAE minimized |
| Traffic Volume Forecasting | Traffic data | LSTM | Trend captured |
| Korean Text Classification | KorNLI etc. | BERT (klue/bert-base) | Accuracy ~90%+ |
| Image Generation | — | Diffusion Model | Noise→Image successful |

---

## Day 1 — PyTorch Basics · Neural Networks

### Tensors and Autograd

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype)       # torch.Size([2, 2]) torch.float32

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)

# Autograd: automatic gradient computation for backpropagation
w = torch.tensor(2.0, requires_grad=True)
y = w ** 2 + 3 * w
y.backward()
print(w.grad)                  # dy/dw = 2w + 3 = 7.0
```

### Neural Network Pattern

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(784, 256, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

### Training Loop

```python
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        acc = (model(X_val).argmax(1) == y_val).float().mean()
        print(f"Epoch {epoch+1}: val_acc={acc:.4f}")
```

---

## Day 2 — Classification & Regression

### Loss Function Selection Guide

| Task | Output Activation | Loss Function | Output Example |
|------|------------------|---------------|----------------|
| Binary Classification | Sigmoid | BCELoss | 0.87 (probability) |
| Multi-class Classification | None (built-in) | CrossEntropyLoss | [0.1, 0.7, 0.2] |
| Regression | None | MSELoss / MAELoss | 3.14 (continuous) |

```python
# Binary classification
binary_model = nn.Sequential(nn.Linear(n, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
loss = nn.BCELoss()(output, target.float())

# Multi-class classification
multi_model = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, num_classes))
loss = nn.CrossEntropyLoss()(output, target)   # Softmax is applied internally

# Regression
reg_model = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, 1))
loss = nn.MSELoss()(output.squeeze(), target.float())
```

---

## Day 3 — CNN · RNN / LSTM

### CNN (Convolutional Neural Network)

Dataset: MNIST (28×28) → **Accuracy ~99%**

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),  # → (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 14, 14)
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)   # → (B, 64, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**Layer Roles**

| Layer | Role |
|-------|------|
| `Conv2d` | Sliding filters detect edges and patterns |
| `BatchNorm2d` | Stabilizes training, speeds up convergence |
| `MaxPool2d` | Halves spatial size → location invariance, fewer params |
| `Dropout` | Regularization — randomly deactivates neurons during training |

**CNN Architecture Evolution:**

```
LeNet-5 (1998) → AlexNet (2012) → VGGNet (2014) → ResNet (2015) → EfficientNet (2019)
```

ResNet's key innovation — **Skip Connections:**

```python
class ResBlock(nn.Module):
    def forward(self, x):
        return self.conv_block(x) + x   # Identity shortcut → solves vanishing gradient
```

### RNN / LSTM / GRU — Time Series Forecasting

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers, batch_first=True, dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])   # Use last time step's hidden state
```

**LSTM Gate Equations:**

```
Forget Gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # How much of past to forget
Input Gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # How much new info to absorb
Output Gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # How much to output
Cell State:  C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(...)  # Long-term memory store
```

| Model | Params | Long-range Dependencies | Speed |
|-------|--------|------------------------|-------|
| RNN | Few | Weak (vanishing gradient) | Fast |
| LSTM | Many | Strong | Moderate |
| GRU | Medium | Strong | Faster than LSTM |

---

## Day 4 — BERT · GPT · LVLM

> For a deep dive into the Transformer architecture: [transformer.en.md](./transformer.en.md)

### BERT — Contextual Understanding

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)

inputs = tokenizer(
    "오늘 날씨가 정말 좋네요",
    return_tensors="pt", padding=True, truncation=True, max_length=128
)

# Fine-tune with a very small learning rate to preserve pretrained knowledge
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
outputs = model(**inputs, labels=torch.tensor([1]))
outputs.loss.backward()
optimizer.step()
```

### GPT — Text Generation

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("The future of AI", return_tensors="pt")
output = model.generate(
    input_ids, max_length=100,
    temperature=0.7,   # Lower = more consistent, Higher = more creative
    do_sample=True, top_p=0.9,
    no_repeat_ngram_size=2
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

| | BERT | GPT |
|-|------|-----|
| Architecture | Encoder Only | Decoder Only |
| Attention | Bidirectional | Causal (left-to-right) |
| Pretraining | MLM + NSP | Next token prediction |
| Strengths | Understanding (classification, QA) | Generation (summarization, writing) |

---

## Day 5 — Diffusion Models · ChatGPT Integration

### Diffusion Model — Image Generation

**Core Idea:** Train a UNet to reverse the process of adding Gaussian noise to images.  
At inference: start from pure noise and iteratively denoise to generate an image.

```python
# Forward process: image → noise (generates training targets)
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alpha * x_0 + sqrt_one_minus * noise

# Reverse process training: UNet predicts the noise
def p_losses(denoise_model, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)
    return F.mse_loss(noise, predicted_noise)   # actual noise vs predicted noise
```

> The foundation behind Stable Diffusion, DALL-E 2, and Midjourney.

### ChatGPT API Integration

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

conversation = [{"role": "system", "content": "You are a data science expert."}]

def chat(user_input):
    conversation.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4", messages=conversation,
        temperature=0.7, max_tokens=500
    )
    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    return reply
```

**Prompt Engineering Techniques**

| Technique | Description | Example |
|-----------|-------------|---------|
| System Prompt | Define the model's role and behavior | "You are a financial analyst." |
| Few-shot | Provide input-output examples | 2–3 examples before the actual query |
| Chain-of-Thought | Encourage step-by-step reasoning | "Think step by step before answering." |
| Temperature | Controls randomness | Summarization: 0.3 / Creative writing: 0.9 |
