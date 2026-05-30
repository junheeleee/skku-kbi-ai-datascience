# Machine Learning

[🇰🇷 Korean](./README.md) | 🇺🇸 English

A comprehensive study of supervised and unsupervised learning using scikit-learn.

---

## Results Summary

| Task | Dataset | Model | Result |
|------|---------|-------|--------|
| Spam Classification | Enron (~5,400 emails) | Multinomial Naive Bayes | Accuracy ~97% |
| Iris Classification | Iris (150 samples, 3 classes) | SVM (RBF kernel) | Accuracy ~97% |
| Iris Classification | Iris | Decision Tree (depth=3) | Accuracy ~96% |
| Height-Weight Regression | Height-Weight dataset | Linear Regression (combined) | MSE 151.34 |
| Height-Weight Regression | Height-Weight dataset | Linear Regression (gender-split) | MSE 95.5 / 102.6 |
| Titanic Survival | Titanic (891 samples) | Logistic Regression | Accuracy 81% |

---

## Day 1 — Python · NumPy · Pandas

### NumPy — Numerical Computing

```python
import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a.shape, a.dtype)        # (2, 2) int64
print(a.mean(axis=0))          # [2. 3.]  (column-wise mean)
print(a @ a)                   # matrix multiplication
```

### Pandas — DataFrame Operations

```python
import pandas as pd

df = pd.read_csv("data.csv")

# Handle missing values
df.dropna(subset=["Age"], inplace=True)
df["Fare"].fillna(df["Fare"].median(), inplace=True)

# Boolean indexing
high_fare = df[df["Fare"] > 100]

# Group aggregation
df.groupby("Pclass")["Survived"].mean()
```

**Key Methods:** `.loc[]` (label-based) vs `.iloc[]` (position-based), `groupby`, `describe`, `value_counts`

---

## Day 2 — Text Classification · Evaluation · SVM · Decision Tree

### Spam Email Classification (Naive Bayes)

Dataset: Enron emails (~1,700 spam + ~3,700 ham)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text(doc):
    tokens = [lemmatizer.lemmatize(w.lower()) for w in doc.split()
              if w.isalpha() and len(w) > 2]
    return " ".join(tokens)

# Text → numeric features
cv = CountVectorizer(stop_words="english", max_features=500)
X_train_cv = cv.fit_transform(X_train)

# Train classifier
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train_cv, y_train)
```

**How Naive Bayes Works:**

$$P(\text{spam} | \text{words}) \propto P(\text{spam}) \times \prod_i P(\text{word}_i | \text{spam})$$

Assumes word independence → fast and effective for text data.

### Evaluation Metrics

```python
from sklearn.metrics import classification_report, roc_auc_score

print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
```

| Metric | Description | When It Matters |
|--------|-------------|-----------------|
| Accuracy | Overall correctness | Balanced classes |
| Precision | True positives / predicted positives | High cost of false alarms |
| Recall | True positives / actual positives | High cost of missed detections |
| F1-Score | Harmonic mean of Precision & Recall | Imbalanced datasets |
| AUC-ROC | Threshold-independent performance | Model comparison |

### SVM (Support Vector Machine)

Dataset: Iris (4 features, 3 classes) → **Accuracy ~97%**

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel="rbf", C=1.0, gamma="scale")
svm.fit(X_train, y_train)
print(svm.score(X_test, y_test))   # ~0.97
```

**Core Concepts**
- **Margin maximization:** Find the hyperplane with the largest gap between classes
- **Support vectors:** Data points closest to the decision boundary
- **Kernel trick:** Enables non-linear classification without explicit feature transformation

| Kernel | Best For |
|--------|----------|
| linear | Linearly separable data |
| rbf | Non-linear, general purpose (default) |
| poly | Polynomial relationships |

### Decision Tree

Dataset: Iris → **Accuracy ~96%**

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(12, 6))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)

# Feature importance
for name, importance in zip(feature_names, dt.feature_importances_):
    print(f"{name}: {importance:.3f}")
```

**Key insight:** `max_depth` controls the bias-variance tradeoff — too deep = overfitting, too shallow = underfitting.

---

## Day 3 — Linear Regression · Logistic Regression

### Linear Regression: Height-Weight Prediction

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(X_train, y_train)
mse = mean_squared_error(y_test, lr.predict(X_test))
```

**Experimental Results**

| Model | MSE |
|-------|-----|
| Combined (ignoring gender) | 151.34 |
| Gender-split — Male | 95.48 |
| Gender-split — Female | 102.56 |

> **Insight:** Splitting by gender reduces MSE by ~37% — a clear demonstration of the value of feature segmentation and domain knowledge in model engineering.

### Logistic Regression: Titanic Survival Prediction

Dataset: Titanic (891 passengers, 38.4% survival rate) → **Accuracy 81%**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Encode categorical features
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Normalize (logistic regression is scale-sensitive)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
```

**Sigmoid function:** Converts linear output to a 0–1 probability.

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

---

## Day 4 — Clustering · Dimensionality Reduction · Outlier Detection

### K-Means Clustering

```python
from sklearn.cluster import KMeans

# Find optimal k with the Elbow Method
inertias = [KMeans(n_clusters=k, random_state=42).fit(X).inertia_
            for k in range(1, 11)]

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
```

### Dimensionality Reduction: PCA vs t-SNE

Dataset: Digits (64 dimensions → 2 dimensions)

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA: linear, fast, supports transform on new data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.1%}")

# t-SNE: non-linear, best for visualization (no transform on new data)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)
```

| | PCA | t-SNE |
|-|-----|-------|
| Approach | Linear | Non-linear |
| Speed | Fast | Slow |
| Use Case | Preprocessing, noise removal | Cluster visualization |
| New Data Transform | Yes | No |
| Key Hyperparameter | n_components | perplexity, n_iter |

### K-Means Based Outlier Detection

```python
import numpy as np

# Compute distance from each point to its cluster centroid
distances = np.linalg.norm(X - kmeans.cluster_centers_[labels], axis=1)

# Points beyond mean + 2.5σ are flagged as outliers
threshold = distances.mean() + 2.5 * distances.std()
outlier_mask = distances > threshold

print(f"Outlier rate: {outlier_mask.mean():.1%}")
```

> **Real-world applications:** Financial fraud detection, network intrusion detection, manufacturing defect identification
