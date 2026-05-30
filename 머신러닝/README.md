# 머신러닝

scikit-learn 기반의 지도학습·비지도학습 전반을 다룹니다.

---

## Day 1 — Python 기초 · NumPy · Pandas

### Python 핵심 데이터 타입

```python
# 타입 변환 및 문자열 처리
x = "3.14"
print(float(x), int(float(x)))   # 3.14, 3

text = "Hello, World"
print(text.lower().split(", "))   # ['hello', 'world']
```

### NumPy — 수치 연산

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a.shape, a.dtype)          # (2, 2) int64
print(a.mean(axis=0))            # [2. 3.]  (열 방향 평균)
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

### 모델 평가 지표

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(classification_report(y_test, y_pred))
# precision / recall / f1-score 출력

# ROC-AUC
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
```

| 지표 | 설명 |
|------|------|
| Precision | 양성 예측 중 실제 양성 비율 |
| Recall | 실제 양성 중 맞게 예측한 비율 |
| F1-Score | Precision과 Recall의 조화평균 |
| AUC-ROC | 임계값 무관한 분류 성능 종합 지표 |

### SVM (Support Vector Machine)

데이터셋: Iris (4 features, 3 classes)

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))
```

- **핵심 개념:** 마진 최대화, 서포트 벡터, 커널 트릭 (linear / RBF / polynomial)

### 의사결정나무

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# 트리 시각화
plt.figure(figsize=(12, 6))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()

# 특성 중요도
importances = dt.feature_importances_
```

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
print(f"MSE: {mse:.2f}")   # 전체: 151.34 → 성별 분리 시 95.48 (남), 102.56 (여)
```

> **인사이트:** 성별을 분리해 각각 모델을 만들면 MSE가 약 37% 감소 → 데이터 세분화의 효과

### 로지스틱 회귀: Titanic 생존 예측

데이터셋: Titanic (891명, 생존율 38.4%)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 전처리
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)

# 학습
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
# 검증 정확도: ~81%
```

---

## Day 4 — 클러스터링 · 차원 축소 · 이상치 탐지

### K-Means 클러스터링

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

# 엘보우 방법으로 최적 k 탐색
inertias = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(1, 11)]
```

### 차원 축소: PCA vs t-SNE

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA: 선형 차원 축소, transform 가능
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)          # 64차원 → 2차원
print(pca.explained_variance_ratio_)  # 각 주성분의 분산 설명 비율

# t-SNE: 비선형, 클러스터 시각화에 강점 (fit_transform만 가능)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

| | PCA | t-SNE |
|-|-----|-------|
| 방식 | 선형 | 비선형 |
| 속도 | 빠름 | 느림 |
| 용도 | 전처리, 노이즈 제거 | 시각화 |
| transform | 가능 | 불가 |

### K-Means 기반 이상치 탐지

```python
import numpy as np

# 각 포인트 → 클러스터 중심까지의 거리 계산
distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)

# 평균 + 2.5σ 초과 → 이상치
threshold = distances.mean() + 2.5 * distances.std()
outliers = X[distances > threshold]
```
