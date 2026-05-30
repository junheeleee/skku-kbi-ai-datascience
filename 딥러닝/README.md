# 딥러닝

🇰🇷 한국어 | [🇺🇸 English](./README.en.md)

PyTorch 기반의 신경망 이론과 실습을 다룹니다.  
MLP → CNN → RNN/LSTM → Transformer(BERT/GPT) → 생성 모델 순으로 학습했습니다.

---

## 실험 결과 요약

| 과제 | 데이터셋 | 모델 | 결과 |
|------|---------|------|------|
| 이진 분류 | MNIST | MLP | Accuracy ~99% |
| 이미지 분류 | MNIST | CNN | Accuracy ~99% |
| 주가 예측 | Stock data | LSTM | MAE 최소화 |
| 교통량 예측 | Traffic data | LSTM | 추세 예측 성공 |
| 한국어 분류 | KorNLI 등 | BERT (klue/bert-base) | Accuracy ~90%+ |
| 이미지 생성 | — | Diffusion Model | 노이즈→이미지 생성 성공 |

---

## Day 1 — PyTorch 기초 · 신경망 입문

### Tensor와 Autograd

```python
import torch

# Tensor 생성 및 연산
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(x.shape, x.dtype)       # torch.Size([2, 2]) torch.float32

# GPU 이동
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)

# Autograd: 역전파를 위한 기울기 자동 계산
w = torch.tensor(2.0, requires_grad=True)
y = w ** 2 + 3 * w             # y = w² + 3w
y.backward()
print(w.grad)                  # dy/dw = 2w + 3 = 7.0
```

### 신경망 구성 패턴

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

### 학습 루프

```python
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = (model(X_val).argmax(1) == y_val).float().mean()
        print(f"Epoch {epoch+1}: val_acc={correct:.4f}")
```

---

## Day 2 — 분류 · 회귀

### 이진 분류 vs 다중 분류 vs 회귀

| 문제 유형 | 출력 활성화 | 손실 함수 | 출력 예시 |
|-----------|------------|----------|---------|
| 이진 분류 | Sigmoid | BCELoss | 0.87 (양성 확률) |
| 다중 분류 | 없음 (내장) | CrossEntropyLoss | [0.1, 0.7, 0.2] |
| 회귀 | 없음 | MSELoss / MAELoss | 3.14 (연속값) |

```python
# 이진 분류
binary_model = nn.Sequential(
    nn.Linear(in_features, 64), nn.ReLU(),
    nn.Linear(64, 1), nn.Sigmoid()
)
loss = nn.BCELoss()(output, target.float())

# 다중 분류
multi_model = nn.Sequential(
    nn.Linear(in_features, 128), nn.ReLU(),
    nn.Linear(128, num_classes)   # Softmax는 CrossEntropyLoss 내부에 포함
)
loss = nn.CrossEntropyLoss()(output, target)

# 회귀
reg_model = nn.Sequential(
    nn.Linear(in_features, 128), nn.ReLU(),
    nn.Linear(128, 1)             # 활성화 함수 없음
)
loss = nn.MSELoss()(output.squeeze(), target.float())
```

---

## Day 3 — CNN · RNN / LSTM

### CNN (Convolutional Neural Network)

데이터셋: MNIST (28×28) → **Accuracy ~99%**

```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B, 32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                              # (B, 32, 14, 14)
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # (B, 64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)                               # (B, 64, 7, 7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**핵심 레이어 설명**

| 레이어 | 역할 |
|--------|------|
| `Conv2d` | 필터가 이미지를 슬라이딩하며 엣지·패턴 추출 |
| `BatchNorm2d` | 학습 안정화, 수렴 속도 향상 |
| `MaxPool2d` | 공간 크기 절반으로 줄임 → 위치 불변성, 연산량 감소 |
| `Dropout` | 과적합 방지 (학습 시 랜덤하게 뉴런 비활성화) |

**대표 CNN 아키텍처 발전사**

```
LeNet-5 (1998) → AlexNet (2012) → VGGNet (2014) → ResNet (2015) → EfficientNet (2019)
```

ResNet의 핵심: **Residual Connection (Skip Connection)**

```python
# ResNet Block
class ResBlock(nn.Module):
    def forward(self, x):
        return self.conv_block(x) + x   # 입력을 그대로 더함 → 기울기 소실 해결
```

### RNN / LSTM / GRU — 시계열 예측

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        return self.fc(out[:, -1, :])   # 마지막 시점 hidden state 사용
```

**LSTM 게이트 구조**

```
Forget Gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # 이전 기억 얼마나 잊을지
Input Gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # 새 정보 얼마나 받아들일지
Output Gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # 출력 얼마나 내보낼지
Cell State:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ tanh(...)  # 장기 기억 저장소
```

| 모델 | 파라미터 수 | 장기 의존성 | 속도 |
|------|------------|------------|------|
| RNN | 적음 | 약함 (기울기 소실) | 빠름 |
| LSTM | 많음 | 강함 | 보통 |
| GRU | 중간 | 강함 | LSTM보다 빠름 |

---

## Day 4 — BERT · GPT · LVLM

> 자세한 Transformer 구조 설명: [transformer.md](./transformer.md)

### BERT — 문맥 이해 모델

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 한국어 BERT
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
model = BertForSequenceClassification.from_pretrained("klue/bert-base", num_labels=2)

# 토크나이징
inputs = tokenizer(
    "오늘 날씨가 정말 좋네요",
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)
# inputs: {'input_ids', 'attention_mask', 'token_type_ids'}

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
outputs = model(**inputs, labels=torch.tensor([1]))
outputs.loss.backward()
optimizer.step()
```

### GPT — 텍스트 생성 모델

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_ids = tokenizer.encode("The future of AI", return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=100,
    temperature=0.7,     # 낮을수록 일관성, 높을수록 창의성
    do_sample=True,
    top_p=0.9,           # Nucleus sampling
    no_repeat_ngram_size=2
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### LVLM — Qwen2-VL (멀티모달)

```python
# 이미지 + 텍스트를 함께 처리하는 Vision-Language Model
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text",  "text": "이 이미지를 설명해줘"}
]}]
```

---

## Day 5 — Diffusion Model · ChatGPT 실습

### Diffusion Model — 이미지 생성 원리

**핵심 아이디어:** 이미지에 점진적으로 가우시안 노이즈를 추가(Forward)한 뒤,  
역방향으로 노이즈를 제거(Reverse)하는 과정을 UNet으로 학습

```python
# Forward process: 이미지 → 노이즈 (학습 데이터 생성)
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    # α̅_t를 이용해 t 스텝의 노이즈 이미지 직접 계산
    sqrt_alpha = extract(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alpha = extract(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

# Reverse process 학습: UNet이 노이즈를 예측
def p_losses(denoise_model, x_start, t):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start, t, noise)
    predicted_noise = denoise_model(x_noisy, t)
    return F.mse_loss(noise, predicted_noise)   # 실제 노이즈 vs 예측 노이즈
```

**생성 시:** 순수 노이즈에서 시작 → T번 반복하며 노이즈 제거 → 이미지 완성

> Stable Diffusion, DALL-E 2, Midjourney의 핵심 기반 기술

### ChatGPT API 실습

```python
from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

# 대화 히스토리 관리
conversation = [
    {"role": "system", "content": "당신은 데이터사이언스 전문가입니다."}
]

def chat(user_input):
    conversation.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="gpt-4",
        messages=conversation,
        temperature=0.7,
        max_tokens=500
    )
    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})
    return reply

print(chat("머신러닝과 딥러닝의 차이점을 설명해줘"))
```

**프롬프트 엔지니어링 핵심**

| 기법 | 설명 | 예시 |
|------|------|------|
| System Prompt | 모델의 역할·행동 정의 | "당신은 금융 분석 전문가입니다" |
| Few-shot | 예시를 포함해 패턴 학습 | 입출력 예시 2~3개 포함 |
| Chain-of-Thought | 단계적 추론 유도 | "단계별로 생각해서 답해줘" |
| Temperature | 낮을수록 일관성, 높을수록 창의성 | 요약: 0.3 / 창작: 0.9 |
