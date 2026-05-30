# 머신러닝

🇰🇷 한국어 | [🇺🇸 English](./README.en.md)

scikit-learn 기반의 지도학습·비지도학습 전반을 다룹니다.

---

## 실험 결과 요약

| 과제 | 데이터셋 | 모델 | 결과 |
|------|---------|------|------|
| 스팸 분류 | Enron (~5,400 이메일) | Multinomial Naive Bayes | Accuracy ~97% |
| Iris 분류 | Iris (150샘플, 3클래스) | SVM (RBF kernel) | Accuracy ~97% |
| Iris 분류 | Iris | Decision Tree (depth=3) | Accuracy ~96% |
| 키-몸무게 예측 | Height-Weight dataset | Linear Regression (통합) | MSE 151.34 |
| 키-몸무게 예측 | Height-Weight dataset | Linear Regression (성별 분리) | MSE 95.5 / 102.6 |
| Titanic 생존 예측 | Titanic (891명) | Logistic Regression | Accuracy 81% |

---

## Day 1 — Python 기초 · NumPy · Pandas

### NumPy — 수치 연산

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a.shape, a.dtype)        # (2, 2) int64
print(a.mean(axis=0))          # [2. 3.]  (열 방향 평균)
print(a @ a)                   # 행렬 곱
```

### Pandas — 데이터프레임 조작

```python
import pandas as pd

df = pd.read_csv("data.csv")

# 결측치 처리
df.dropna(subset=["Age"], inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# 불리언 인덱싱
high_fare = df[df["Fare"] > 100]

# 그룹 집계
df.groupby("Pclass")["Survived"].mean()
```

**핵심 메서드:** `.loc[]` (레이블 기반) vs `.iloc[]` (위치 기반), `groupby`, `describe`, `value_counts`

---

## Day 2 — 텍스트 분류 · 모델 평가 · SVM · 의사결정나무

### 스팸 메일 분류 (Naive Bayes)

데이터셋: Enron 이메일 (~1,700 스팸 + ~3,700 정상)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text(doc):
    tokens = [lemmatizer.lemmatize(w.lower()) for w in doc.split()
              if w.isalpha() and len(w) > 2]
    return " ".join(tokens)

# 텍스트 → 수치 변환
cv = CountVectorizer(stop_words="english", max_features=500)
X_train_cv = cv.fit_transform(X_train)

# 모델 학습
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train_cv, y_train)
```

**Naive Bayes 작동 원리**

$$P(\text{spam} | \text{단어들}) \propto P(\text{spam}) \times \prod_i P(\text{단어}_i | \text{spam})$$

각 단어의 등장 확률을 독립으로 가정해 계산 → 빠르고 텍스트에 효과적

### 모델 평가 지표

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
```

| 지표 | 설명 | 언제 중요한가 |
|------|------|--------------|
| Accuracy | 전체 정확도 | 클래스 균형일 때 |
| Precision | 양성 예측 중 실제 양성 비율 | 오탐(False Positive) 비용이 클 때 |
| Recall | 실제 양성 중 맞게 예측한 비율 | 미탐(False Negative) 비용이 클 때 |
| F1-Score | Precision과 Recall의 조화평균 | 불균형 데이터셋 |
| AUC-ROC | 임계값 무관한 분류 성능 종합 지표 | 전반적 모델 비교 |

### SVM (Support Vector Machine)

데이터셋: Iris (4 features, 3 classes) → **Accuracy ~97%**

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))   # ~0.97
```

**핵심 개념**
- **마진 최대화:** 두 클래스를 가장 넓은 간격으로 분리하는 초평면 탐색
- **서포트 벡터:** 결정 경계에 가장 가까운 데이터 포인트들
- **커널 트릭:** 저차원 데이터를 고차원으로 변환 없이 비선형 분류 가능

| 커널 | 적합한 상황 |
|------|------------|
| linear | 선형 분리 가능한 데이터 |
| rbf | 비선형, 범용 (기본값) |
| poly | 다항식 관계 |

### 의사결정나무

데이터셋: Iris → **Accuracy ~96%**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# 트리 시각화
plt.figure(figsize=(12, 6))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# 특성 중요도 확인
for name, importance in zip(feature_names, dt.feature_importances_):
    print(f"{name}: {importance:.3f}")
```

**max_depth 조절이 핵심:** 너무 깊으면 과적합, 너무 얕으면 과소적합

---

## Day 3 — 선형 회귀 · 로지스틱 회귀

### 선형 회귀: 키-몸무게 예측

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
```

**실험 결과**

| 모델 | MSE |
|------|-----|
| 통합 모델 (성별 무시) | 151.34 |
| 성별 분리 — 남성 | 95.48 |
| 성별 분리 — 여성 | 102.56 |

> **인사이트:** 성별을 분리해 각각 모델을 만들면 MSE가 약 37% 감소  
> → 데이터 세분화 / 특성 엔지니어링의 효과를 직접 확인

### 로지스틱 회귀: Titanic 생존 예측

데이터셋: Titanic (891명, 생존율 38.4%) → **Accuracy 81%**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 범주형 인코딩
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 정규화 (로지스틱 회귀는 스케일에 민감)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
```

**시그모이드 함수:** 선형 출력을 0~1 확률로 변환

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

---

## Day 4 — 클러스터링 · 차원 축소 · 이상치 탐지

### K-Means 클러스터링

```python
from sklearn.cluster import KMeans

# 엘보우 방법으로 최적 k 탐색
inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_
            for k in range(1, 11)]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
```

### 차원 축소: PCA vs t-SNE

데이터셋: Digits dataset (64차원 → 2차원)

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA: 선형, 빠름, transform 재사용 가능
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"설명된 분산 비율: {pca.explained_variance_ratio_.sum():.1%}")

# t-SNE: 비선형, 시각화에 특화 (fit_transform만 가능)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)
```

| | PCA | t-SNE |
|-|-----|-------|
| 방식 | 선형 | 비선형 |
| 속도 | 빠름 | 느림 |
| 용도 | 전처리, 노이즈 제거 | 군집 시각화 |
| 신규 데이터 변환 | 가능 | 불가 |
| 하이퍼파라미터 | n_components | perplexity, n_iter |

### K-Means 기반 이상치 탐지

```python
import numpy as np

# 각 포인트 → 클러스터 중심까지 거리 계산
centroids = kmeans.cluster_centers_
distances = np.linalg.norm(X - centroids[labels], axis=1)

# 평균 + 2.5σ 초과 → 이상치
threshold = distances.mean() + 2.5 * distances.std()
outlier_mask = distances > threshold
outliers = X[outlier_mask]

print(f"이상치 비율: {outlier_mask.mean():.1%}")
```

> **활용 예시:** 금융 사기 탐지, 네트워크 침입 탐지, 제조 불량품 검출
