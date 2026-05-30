# Transformer 심화

🇰🇷 한국어 | [🇺🇸 English](./transformer.en.md)

"Attention Is All You Need" (Vaswani et al., 2017) 논문에서 제안된 Transformer 아키텍처의 원리와  
이를 기반으로 한 BERT/GPT의 구조를 코드와 함께 정리합니다.

---

## 1. Self-Attention 메커니즘

### 핵심 아이디어

문장 내 각 단어가 **다른 모든 단어와의 관련성(가중치)**을 계산해  
문맥을 반영한 표현(representation)을 만든다.

```
"그 동물은 너무 지쳐서 강을 건너지 못했다. 그것은 사자였다."
                                              ↑
                          "그것"이 "동물", "사자"와 높은 attention 가중치를 가짐
```

### Q, K, V 행렬

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q (Query): 현재 단어가 "무엇을 찾고 있는가"
    K (Key):   각 단어가 "어떤 정보를 갖고 있는가"
    V (Value): 실제로 가져올 정보
    """
    d_k = Q.size(-1)

    # 1. Q와 K의 내적 → 유사도 점수
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. (선택) 미래 정보 마스킹 (GPT에서 사용)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. Softmax → 가중치 (합이 1)
    attn_weights = F.softmax(scores, dim=-1)

    # 4. V의 가중합 → 문맥 반영 표현
    return torch.matmul(attn_weights, V), attn_weights
```

**직관적 이해:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$: 쿼리-키 유사도 → 높을수록 많이 참조
- $\sqrt{d_k}$: 차원이 커질수록 내적값이 커지므로 스케일링
- Softmax: 가중치를 확률로 변환
- 최종 출력: V의 가중 평균 → **문맥이 담긴 단어 표현**

---

## 2. Multi-Head Attention

단일 Attention은 하나의 관점만 봄 → **여러 head가 서로 다른 관점**으로 병렬 처리

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

        # 모든 head 결합
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)
```

**예시:** BERT-base는 12개 head, d_model=768 → 각 head: d_k=64  
→ head마다 문법, 의미, 대명사 참조 등 서로 다른 패턴을 학습

---

## 3. Positional Encoding

Attention은 순서 정보가 없음 → **위치 정보를 사인/코사인 함수로 주입**

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

        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 차원: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 차원: cos

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
            torch.nn.GELU(),                 # BERT는 GELU 사용
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

**구성 요소 요약:**

```
입력 임베딩
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
      출력 표현
```

---

## 5. BERT vs GPT 구조 비교

| | BERT | GPT |
|-|------|-----|
| 구조 | Encoder Only | Decoder Only |
| Attention | 양방향 (Bidirectional) | 단방향 (Causal / Left-to-right) |
| 사전학습 목표 | MLM (마스크 토큰 예측) + NSP | 다음 토큰 예측 (LM) |
| 강점 | 이해 (분류, NER, QA) | 생성 (요약, 번역, 창작) |
| 대표 모델 | BERT, RoBERTa, KLUE-BERT | GPT-2/3/4, LLaMA, Mistral |

### BERT 사전학습: MLM (Masked Language Model)

```
입력: "나는 [MASK] 좋아한다"   → 예측: "사과"
     "나는 사과 [MASK]한다"   → 예측: "좋아"

→ 양방향으로 문맥을 봐야 맞출 수 있음
```

### GPT 사전학습: Causal LM

```
입력: "나는 사과를"            → 예측: "좋아한다"
      Attention 마스크: 현재 위치 이전 토큰만 참조 가능

→ 자연스럽게 텍스트 생성에 최적화
```

---

## 6. Fine-tuning 전략

### BERT Fine-tuning (분류)

```python
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

model = BertForSequenceClassification.from_pretrained(
    "klue/bert-base",
    num_labels=2
)

# Fine-tuning 핵심: 작은 학습률 사용 (사전학습 파라미터 보존)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Warmup 스케줄러: 처음에 천천히, 이후 감소
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps * 0.1,
    num_training_steps=total_steps
)

# 학습 루프
for batch in train_loader:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    outputs.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 기울기 클리핑
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

**Fine-tuning 팁**
- 학습률: `2e-5` ~ `5e-5` (너무 크면 사전학습 지식 손실)
- Epoch: 3~5회 (과적합 주의)
- Warmup: 전체 스텝의 10% 정도

### GPT 프롬프트 엔지니어링

```python
# Few-shot 프롬프트
prompt = """
다음 리뷰의 감성을 분석해 긍정/부정으로 답하세요.

리뷰: "화면이 선명하고 배터리가 오래 간다"
감성: 긍정

리뷰: "배송이 너무 느리고 포장이 엉망이었다"
감성: 부정

리뷰: "가격 대비 성능이 훌륭하다"
감성:"""

# Chain-of-Thought 프롬프트
cot_prompt = """
문제를 단계별로 생각해서 풀어주세요.
질문: 머신러닝 모델의 과적합을 어떻게 해결할 수 있나요?
단계별 분석:"""
```

---

## 참고 자료

- [Attention Is All You Need (원논문)](https://arxiv.org/abs/1706.03762)
- [BERT (원논문)](https://arxiv.org/abs/1810.04805)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
