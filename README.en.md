# SKKU KBI — AI-Based Data Science Portfolio

[🇰🇷 Korean](./README.md) | 🇺🇸 English

> A portfolio summarizing the **Machine Learning** and **Deep Learning** curriculum completed in the SKKU KBI program.  
> Covers the full data science stack from foundational theory to hands-on PyTorch implementation.

---

## Program Overview

| Course | Duration | Topics |
|--------|----------|--------|
| Machine Learning | Day 1–4 | Python/NumPy/Pandas, Classification, Regression, Unsupervised Learning |
| Deep Learning | Day 1–5 | PyTorch, CNN, RNN/LSTM, Transformer, LLMs |

---

## Key Experimental Results

| Task | Model | Result |
|------|-------|--------|
| Spam Email Classification | Multinomial Naive Bayes | Accuracy ~97% |
| Titanic Survival Prediction | Logistic Regression | Accuracy 81% |
| Height-Weight Prediction | Linear Regression (gender-split) | MSE 95.5 (M) / 102.6 (F) |
| MNIST Image Classification | CNN | Accuracy ~99% |
| Stock Price Forecasting | LSTM | MAE minimized |
| Korean Text Classification | BERT Fine-tuning | Accuracy ~90%+ |

---

## Tech Stack

**Data**
`Python` `NumPy` `Pandas`

**Machine Learning**
`scikit-learn` `NLTK`

**Deep Learning**
`PyTorch` `Hugging Face Transformers`

**Visualization**
`Matplotlib` `Seaborn`

---

## Detailed Contents

### [Machine Learning](./머신러닝/README.en.md)
- Data preprocessing with Python, NumPy, and Pandas
- Spam classification with Naive Bayes (Enron dataset)
- Iris classification with SVM and Decision Trees
- Linear Regression for height-weight prediction; Logistic Regression for Titanic survival
- K-Means clustering, PCA & t-SNE dimensionality reduction, outlier detection

### [Deep Learning](./딥러닝/README.en.md)
- PyTorch Tensor operations and Autograd
- MNIST classification with MLP (binary & multi-class)
- Image feature extraction and classification with CNN
- Time series forecasting (stock prices, traffic) with RNN, LSTM, GRU
- BERT fine-tuning for Korean NLP; GPT text generation
- Image generation with Diffusion Models; ChatGPT API integration

### [Transformer Deep Dive](./딥러닝/transformer.en.md)
- Self-Attention mechanism from scratch
- Multi-Head Attention and Positional Encoding
- BERT vs GPT architecture comparison and fine-tuning strategies
