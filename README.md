# IMDB Sentiment Classification with Logistic Regression & Linear SVC

A complete **Binary Sentiment Classification** project using the **IMDB 50K Movie Reviews Dataset**.

The project builds an end-to-end NLP classification pipeline, including text preprocessing, negation handling, Bag-of-Words feature extraction, Logistic Regression and Linear SVC models, hyperparameter tuning, convergence analysis, threshold experiments, error analysis, and feature interpretation.

---

## 📌 Project Overview

The goal of this project is to classify movie reviews as:

* **Negative → 0**
* **Positive → 1**

The project investigates how classical machine learning models perform on text classification using manually designed NLP preprocessing and **Bag-of-Words (BoW)** features.

Two linear classifiers are evaluated:

1. **Logistic Regression**
2. **Linear Support Vector Classifier (Linear SVC)**

The models are evaluated using multiple classification metrics and additional experiments are performed to understand model behavior.

---

## 📂 Dataset

The project uses the **IMDB Dataset**, containing:

* **50,000 movie reviews**
* `review` — Original movie review
* `sentiment` — Positive or Negative

The dataset is divided into:

* **80% Training**
* **20% Testing**

Stratified splitting is used to preserve the class distribution.

---

## 🛠️ Technologies & Libraries

* Python
* NumPy
* Pandas
* Matplotlib
* BeautifulSoup
* NLTK
* Contractions
* Scikit-learn
* Jupyter Notebook / Google Colab

### Machine Learning

* Logistic Regression
* Linear SVC
* CalibratedClassifierCV
* Cross-Validation
* Hyperparameter Tuning

### NLP

* Text Cleaning
* HTML Removal
* URL & Email Removal
* Contraction Expansion
* Negation Handling
* Stopword Removal
* Lemmatization
* Bag-of-Words
* Unigrams & Bigrams

---

## 🔄 NLP Preprocessing Pipeline

The reviews go through the following preprocessing steps:

```text
Raw Review
    ↓
Lowercasing
    ↓
HTML Removal
    ↓
URL Removal
    ↓
Email Removal
    ↓
Contraction Expansion
    ↓
HTML Entity Cleaning
    ↓
Elongated Word Reduction
    ↓
Negation Marking
    ↓
Special Character Removal
    ↓
Stopword Removal
    ↓
Lemmatization
    ↓
Final Clean Review
```

### 🔹 Negation Handling

A custom negation strategy is implemented to preserve the meaning of expressions such as:

```text
not good
not great
not excellent
```

Instead of simply removing the negation word, the pipeline transforms sentiment words into negation-aware tokens:

```text
not good → NOT_good
not great → NOT_great
```

Negation is reset when punctuation or clause-breaking words are encountered.

This helps preserve important sentiment information that could otherwise be lost during preprocessing.

---

## 🔢 Feature Extraction

The cleaned reviews are converted into numerical features using:

### Bag-of-Words

```python
CountVectorizer(
    ngram_range=(1, 2),
    lowercase=True,
    max_features=10000,
    min_df=10,
    max_df=0.9,
    binary=False
)
```

The model uses:

* **Unigrams** — single words
* **Bigrams** — two-word combinations
* Maximum of **10,000 features**
* Minimum document frequency = 10
* Maximum document frequency = 90%

The vectorizer is fitted **only on the training data** and then applied to the test data to avoid data leakage.

---

# 🤖 Models

## 1. Logistic Regression

A Logistic Regression classifier is trained using:

```python
LogisticRegression(
    solver="lbfgs",
    penalty="l2",
    max_iter=1000,
    random_state=42
)
```

The default Scikit-learn value of:

```text
C = 1.0
```

is used.

The model produces probabilities using the sigmoid function, allowing additional experiments with different classification thresholds.

---

## 2. Linear SVC

A Linear SVC classifier is evaluated using multiple values of the regularization parameter `C`:

```python
C_GRID = [0.01, 0.1, 1, 10, 100]
```

Five-fold cross-validation is used to evaluate each value of `C`.

Because LinearSVC does not directly provide probabilities, **CalibratedClassifierCV** with sigmoid calibration is used to obtain probability estimates.

The best `C` is selected based on mean cross-validation accuracy and the final model is retrained on the complete training set.

---

# 📊 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Log Loss
* Confusion Matrix
* Classification Report

### Why multiple metrics?

Accuracy alone does not provide a complete picture of classification performance.

Using Precision, Recall, F1-Score, and ROC-AUC provides a broader view of how the models handle positive and negative reviews.

---

# 📈 Model Comparison

The project generates a comparison table containing:

| Metric    | Logistic Regression | Linear SVC |
| --------- | ------------------: | ---------: |
| Accuracy  |                   — |          — |
| Precision |                   — |          — |
| Recall    |                   — |          — |
| F1-Score  |                   — |          — |
| ROC-AUC   |                   — |          — |
| Log Loss  |                   — |          — |

The actual values are generated automatically when the notebook is executed.

The project also visualizes the metrics using a bar chart for easier comparison.

---

# 📉 Convergence Analysis

The project investigates how the optimization process changes as the maximum number of iterations increases.

For Logistic Regression, training **Binary Cross-Entropy / Log Loss** is measured at different LBFGS iteration checkpoints:

```text
1, 2, 5, 10, 20, 30, 50, 75, 100, 200, 400, 500
```

A similar experiment is performed for LinearSVC using **Hinge Loss**.

### Logistic Regression

```text
LBFGS Iterations → Training BCE / Log Loss
```

### Linear SVC

```text
Iterations → Training Hinge Loss
```

The resulting curves help visualize how the optimization process approaches a stable solution.

---

# 🧮 Mathematical BCE Analysis

The project also includes a manual demonstration of the Logistic Regression forward pass.

Given:

```text
w_masterpiece = +1.8
w_boring      = -2.1
w_acting      = +0.5
bias          = -0.2
```

The logit is:

```text
z = wᵀx + b
```

which gives:

```text
z = 0
```

The sigmoid output is therefore:

```text
ŷ = 0.5
```

The Binary Cross-Entropy loss is calculated for both:

```text
y = 1
```

and

```text
y = 0
```

resulting in approximately:

```text
BCE = 0.6931
```

This demonstrates that a prediction of `0.5` represents maximum uncertainty between the two classes.

---

# 🎚️ Threshold Experiment

Instead of using only the default threshold:

```text
τ = 0.5
```

the project evaluates:

```text
τ = 0.3
τ = 0.5
τ = 0.7
```

For each threshold, the following are measured:

* Precision
* Recall
* F1-Score
* Number of Positive Predictions

This demonstrates how changing the decision threshold affects classification behavior.

---

# 🔍 Error Analysis

The project extracts:

* **2 False Positives**
* **2 False Negatives**

### False Positive

A negative review incorrectly classified as positive.

```text
True Label      → Negative
Predicted Label → Positive
```

### False Negative

A positive review incorrectly classified as negative.

```text
True Label      → Positive
Predicted Label → Negative
```

The selected examples can be manually analyzed for:

* Negation
* Sarcasm
* Mixed sentiment
* Strong sentiment words
* Context
* Word-order effects
* Ambiguous language

This provides qualitative insight into why the classifier makes mistakes.

---

# 🔎 Feature Interpretation

The Logistic Regression coefficients are analyzed to identify the most influential features.

### Positive Features

Features with large positive coefficients contribute more strongly toward:

```text
Positive → 1
```

### Negative Features

Features with large negative coefficients contribute more strongly toward:

```text
Negative → 0
```

The project extracts the:

```text
Top 10 Positive Features
Top 10 Negative Features
```

and visualizes them using horizontal bar charts.

This makes the model more interpretable by showing which words or bigrams have the strongest influence on its predictions.

---

# 🧪 Sentiment Feature Sanity Check

Selected sentiment words are inspected directly through their learned Logistic Regression coefficients.

Example sentiment vocabulary:

```text
excellent
amazing
wonderful
great
fantastic
perfect
terrible
awful
horrible
worst
boring
waste
```

This provides an additional sanity check for whether the learned coefficients align with expected sentiment associations.

---

# 🔮 Prediction Function

A reusable function is implemented:

```python
predict_sentiment(review_text, threshold=0.5)
```

It returns:

* Predicted class
* Positive probability
* Negative probability
* Logit
* Classification threshold

Example:

```python
result = predict_sentiment(
    "This movie was absolutely amazing!",
    threshold=0.5
)
```

The function can be used to test new reviews without retraining the model.

---

# 🧪 Benchmark Reviews

The final section evaluates the trained model on manually written reviews covering different linguistic patterns:

### Strong Positive

```text
An absolute triumph of cinema with breathtaking visuals and stellar acting.
```

### Sarcastic / Contradictory Language

```text
I thoroughly enjoyed wasting two hours of my life staring at a completely incoherent plot.
```

### Mixed / Negated Sentiment

```text
The cinematography was not bad, but the pacing was not good either.
```

These examples provide an additional qualitative test of the model's ability to handle sentiment, negation, and mixed opinions.

---

# 📁 Project Structure

```text
IMDB-Sentiment-Analysis/
│
├── IMDB Dataset.csv
├── sentiment_classification.ipynb
└── README.md
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd IMDB-Sentiment-Analysis
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib beautifulsoup4 nltk scikit-learn contractions
```

### 3. Open the notebook

Run the notebook using:

* Google Colab
* Jupyter Notebook
* JupyterLab

### 4. Upload the dataset

Place:

```text
IMDB Dataset.csv
```

in the expected path or update:

```python
DATA_PATH
```

### 5. Run all cells

The notebook will perform preprocessing, training, evaluation, experiments, and prediction automatically.

---

# 🎯 Key Learning Outcomes

This project demonstrates practical understanding of:

* End-to-end NLP classification
* Text preprocessing
* Negation-aware preprocessing
* Feature engineering with n-grams
* Bag-of-Words representation
* Logistic Regression for text classification
* Linear SVM classification
* Probability calibration
* Cross-validation
* Hyperparameter tuning
* Classification metrics
* Binary Cross-Entropy
* Hinge Loss
* Optimization convergence
* Decision threshold selection
* Error analysis
* Model interpretability

---

## 👩‍💻 Author

**Sherry Adel Riad Tawfik**

AI / Machine Learning Student

---

## ⭐ Project Highlights

> An end-to-end classical NLP project that goes beyond simply training a classifier by combining custom preprocessing, negation handling, model comparison, hyperparameter tuning, convergence analysis, threshold experiments, error analysis, and model interpretability.
