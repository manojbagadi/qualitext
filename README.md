# Sentence Quality Classifier (QualiText)
**Comparative Evaluation of Classical Machine Learning vs. Deep Neural Networks**

An automated text quality assessment and document health auditing system built with **Python**, **Scikit-Learn**, and **Streamlit**.

---

## 🌟 Key Features

1. **Comparative Algorithmic Hierarchy (Neural Network as Top Model)**:
   - 🧠 **Multi-Layer Perceptron (MLP)**: **Top Performing Model (100.0% Accuracy)** — Deep Feedforward Neural Network ($64 \rightarrow 32 \rightarrow 16$ neurons, ReLU, Adam) capturing complex non-linear language semantics.
   - 🌲🌲 **Random Forest Classifier**: **Ensemble Model (97.9% Accuracy)** — Ensemble of 100 decision trees averaging votes to reduce prediction variance.
   - 🌲 **Decision Tree Classifier**: **Baseline Model (97.9% Accuracy)** — Rapid, explainable single rule-based baseline.

2. **Per-Model Decision Explanations**:
   - Every model card dynamically explains **why it gave that specific confidence percentage** on the user's typed sentence (e.g. leaf split thresholds for Decision Tree, tree voting ratios for Random Forest, and layer activation levels for Neural Network).

3. **Domain-Aware Feature Engineering (9 Structural Dimensions + 350 TF-IDF)**:
   - Evaluates **Repeated Symbol Abuse** (`!!`, `???`, `$$$`) instead of penalizing natural single marks (`"Good morning!"`).
   - Understands **Academic Section Headings** in all caps (`ABSTRACT`, `SYSTEM ARCHITECTURE`) as clean titles rather than spam shouting.
   - Accurately generalizes to unseen technical phrases like `"Comparing Classical Machine Learning"`.

4. **Interactive Streamlit Web Dashboard**:
   - **Page 1 (Single Sentence Test)**: Clean interactive tester with real-time confidence scores, explicit explanations, and **Smart Next-Word Suggestion** autocomplete.
   - **Page 2 (Document & PDF Checker)**: Native multi-page PDF (`pypdf`) and Word document (`python-docx`) parsing with an Executive Document Quality Verdict.
   - **Page 3 (Model Comparison)**: Empirical accuracy charts, feature importance metrics, and architecture hierarchy tables.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application
```bash
python -m streamlit run app.py
```
Open your browser and navigate to: **`http://localhost:8501`**

### 3. (Optional) Retrain All Models
```bash
python train_all_models.py
```

---

## 📊 Dataset & Model Performance

- **Dataset**: 476 balanced samples (High Quality vs. Spam/Junk/Noise).
- **Split**: Stratified 80% Training (380 samples) and 20% Unseen Testing (96 samples).

| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🧠 **Neural Network (MLP)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** | **🥇 Best Model** |
| 🌲🌲 **Random Forest** | **97.9%** | 96.0% | 100.0% | 97.9% | **🥈 Strong Ensemble** |
| 🌲 **Decision Tree** | **97.9%** | 96.0% | 100.0% | 97.9% | **🥉 Baseline** |

---

## 📂 Project Structure

```
text_quality_classifier/
│
├── app.py                      # Main Streamlit web application
├── train_all_models.py         # Unified training pipeline for all 3 models
├── requirements.txt            # Python dependencies (Streamlit, scikit-learn, pypdf, docx)
├── README.md                   # Project documentation
├── PROJECT_PRESENTATION.md     # Detailed academic presentation guide for reviewers
│
├── data/
│   ├── dataset.csv             # Balanced corpus (476 samples)
│   └── features.npz            # Extracted feature matrix (359 dimensions)
│
└── models/
    ├── neural_mlp.pkl          # Trained Multi-Layer Perceptron (Top Model: 100%)
    ├── baseline_rf.pkl         # Trained Random Forest classifier (97.9%)
    ├── decision_tree.pkl       # Trained Decision Tree baseline (97.9%)
    ├── feature_scaler.pkl      # StandardScaler fitted on 9 structural features
    └── tfidf_vectorizer.pkl    # TfidfVectorizer fitted on 350 n-grams
```
