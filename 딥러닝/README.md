# 딥러닝

PyTorch 기반의 신경망 이론과 실습을 다룹니다.  
MLP → CNN → RNN/LSTM → Transformer(BERT/GPT) → 생성 모델 순으로 학습했습니다.

---

## Day 1 — PyTorch 기초 · 신경망 입문

### Tensor 기본 연산

```python
import torch

# Tensor 생성
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype)    # torch.Size([2, 2]) torch.float32

# GPU 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)

# Autograd: 기울기 추적
w = torch.tensor(2.0, requires_grad=True)
y = w ** 2 + 3 * w
y.backward()
print(w.grad)    # dy/dw = 2w + 3 = 7.0
```

### 신경망 구성 패턴

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
```

### 학습 루프

```python
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val), y_val)
```

---

## Day 2 — 분류 · 회귀

### 이진 분류 (Binary Classification)

```python
# 출력층: 시그모이드 + BCELoss
model = nn.Sequential(
    nn.Linear(in_features, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
criterion = nn.BCELoss()

# 예측
with torch.no_grad():
    prob = model(X_test)
    pred = (prob > 0.5).float()
```

### 회귀 (Regression) — 주가 예측

```python
# 출력층: 활성화 함수 없음 + MSELoss
model = nn.Sequential(
    nn.Linear(in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 1)   # 연속값 출력
)
criterion = nn.MSELoss()
```

| 문제 유형 | 출력 활성화 | 손실 함수 |
|-----------|------------|----------|
| 이진 분류 | Sigmoid | BCELoss |
| 다중 분류 | Softmax (내장) | CrossEntropyLoss |
| 회귀 | 없음 | MSELoss / MAELoss |

---

## Day 3 — CNN · RNN / LSTM

### CNN (Convolutional Neural Network)

데이터셋: MNIST (28×28 흑백 이미지)

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2)                               # (B, 64, 7, 7)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.fc(self.conv_block(x))
```

**핵심 개념**
- `Conv2d`: 필터가 이미지를 슬라이딩하며 특징 추출
- `MaxPool2d`: 공간 크기 절반으로 줄여 계산량 감소, 위치 불변성 확보
- 필터 수를 늘릴수록 더 다양한 패턴 학습

### RNN / LSTM / GRU

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])   # 마지막 시점 출력
```

| 모델 | 장점 | 단점 |
|------|------|------|
| RNN | 단순, 빠름 | 장기 의존성 학습 어려움 (기울기 소실) |
| LSTM | 장기 의존성 해결 (Cell State) | 파라미터 많음 |
| GRU | LSTM보다 경량, 성능 유사 | |

**실습:** LSTM으로 주가 예측 · 교통량 예측

---

## Day 4 — BERT · GPT · LVLM

### BERT (Bidirectional Encoder Representations from Transformers)

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("klue/bert-base")  # 한국어
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)

# 토크나이징
inputs = tokenizer("오늘 날씨가 정말 좋네요", return_tensors="pt",
                   padding=True, truncation=True, max_length=128)

# 파인튜닝
outputs = model(**inputs, labels=torch.tensor([1]))
loss = outputs.loss
loss.backward()
```

- **핵심 특징:** 양방향(Bidirectional) 문맥 이해 → 문장 분류, 질의응답에 강점
- **한국어 모델:** `klue/bert-base` 활용

### GPT (Generative Pre-trained Transformer)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 텍스트 생성
input_ids = tokenizer.encode("The future of AI", return_tensors="pt")
output = model.generate(input_ids, max_length=100,
                        temperature=0.7, do_sample=True)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

- **핵심 특징:** 단방향(Autoregressive) 텍스트 생성 → 문장 완성, 창작에 강점

| | BERT | GPT |
|-|------|-----|
| 방향 | 양방향 | 단방향 (Left→Right) |
| 구조 | Encoder | Decoder |
| 강점 | 이해 (분류, NER, QA) | 생성 (요약, 번역, 작문) |

### LVLM — Qwen2-VL (멀티모달)

```python
# 이미지 + 텍스트를 함께 처리하는 Vision-Language Model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# 이미지와 질문을 함께 입력
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text": "이 이미지를 설명해줘"}
]}]
```

---

## Day 5 — Diffusion Model · ChatGPT 실습

### Diffusion Model

**핵심 아이디어:** 이미지에 점진적으로 노이즈를 추가(Forward)한 뒤, 역방향으로 노이즈를 제거(Reverse)하는 것을 학습

```python
# Forward process: 이미지 → 노이즈
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

# Reverse process: 노이즈 → 이미지 (UNet 기반 denoising)
def p_losses(denoise_model, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)
    return F.mse_loss(noise, predicted_noise)
```

**Stable Diffusion, DALL-E, Midjourney 등의 기반 기술**

### ChatGPT API 실습

```python
from openai import OpenAI

client = OpenAI(api_key="...")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "머신러닝과 딥러닝의 차이점을 설명해줘"}
    ],
    temperature=0.7
)
print(response.choices[0].message.content)
```

**프롬프트 엔지니어링 핵심**
- `system` 역할로 모델 행동 정의
- `temperature`: 낮을수록 일관성, 높을수록 창의성
- Few-shot 예시 포함 시 성능 향상
