# LightGBM Fraud Detection Model: Reducing False Alarms for Frequent Customers

## 🎯 Project Overview

This project develops an optimized fraud detection model using **LightGBM** with a specific focus on **minimizing false positives for frequent, legitimate customers**. The approach combines custom evaluation functions, asymmetric sample weighting, and Bayesian hyperparameter optimization to achieve superior business-aligned performance.

### Key Achievements

- **Eliminated** false positives for frequent customers (0 incidents)
- **Reduced** overall false positive rate by **70%** (0.057% → 0.017%)
- **Increased** fraud recall from **78.7%** to **86.4%**
- **Improved** alert efficiency: (TP+FP)/TP ratio from 1.384 → 1.103

## 📁 Project Structure

```
lgbm_fraud_detection_model/
├── data/
│   ├── raw/
│   │   └── transactions.csv              # Original transaction data (1.85M records)
│   └── processed/
│       └── transactions_processed.csv    # Feature-enriched dataset (56 features)
├── notebooks/
│   ├── 01_exploration.ipynb             # EDA and data quality analysis
│   ├── 02_feature_engineering.ipynb     # Feature creation and transformation
│   └── 03_base_model.ipynb              # Model training and optimization
├── figures/
│   ├── base.png                         # Baseline model confusion matrix
│   └── final.png                        # Optimized model confusion matrix
├── frequent_customer.ipynb              # Frequent customer analysis
├── report.tex                           # Technical report (LaTeX)
├── report.pdf                           # Compiled technical report
├── .gitignore                           # Git ignore patterns
└── README.md                            # This file
```

## 🧮 Custom Evaluation Functions

The project implements **5 custom evaluation functions** designed to align model optimization with business objectives:

### 1. `fp_tp_ratio`

```python
def fp_tp_ratio(preds, data):
    """
    Custom metric: (TP + FP) / TP
    Lower is better. Measures overall alert burden relative to true positives
    """
```

**Purpose**: Simple ratio measuring total alerts per true positive.
**Limitation**: Can be "gamed" by models that make very few predictions.

### 2. `business_fp_tp_ratio` ⭐ **[Primary Optimization Metric]**

```python
def business_fp_tp_ratio(preds, data, min_recall=0.70, max_ratio=5.0, recall_penalty=50.0):
    """
    Business-aligned version that enforces minimum recall while optimizing alert burden.
    Lower is better.
    """
```

**Purpose**: Prevents metric gaming by enforcing business constraints:

- Minimum recall threshold (70%)
- Maximum acceptable alert ratio (5.0)
- Heavy penalties for violations

> **⚠️ Why use `business_fp_tp_ratio` instead of `fp_tp_ratio`?**
>
> The simple `fp_tp_ratio` can be "cheated" by models that make almost no positive predictions, resulting in very low alert burden but also extremely low recall (missing most fraud). The `business_fp_tp_ratio` adds constraints that force the model to maintain minimum business-viable performance while optimizing efficiency.

### 3. `balanced_cost`

```python
def balanced_cost(preds, data, w_fp=3.0, w_fn=10.0):
    """
    Custom cost: w_fp * FP_on_frequent + w_fn * FN
    Lower is better. Balances penalties for missing fraud vs. false alerts on frequent customers
    """
```

**Purpose**: Asymmetric cost function prioritizing frequent customer experience.

### 4. `f05_score`

```python
def f05_score(preds, data):
    """
    Custom F-beta score (beta=0.5), negated for maximization.
    Higher is better. Emphasizes precision (FP control) twice as much as recall.
    """
```

**Purpose**: F-beta score emphasizing precision over recall.

### 5. `freq_fpr`

```python
def freq_fpr(preds, data):
    """
    Frequent-customer false-positive rate: FP_freq / Legit_freq
    Lower is better. Measures share of frequent customers erroneously flagged
    """
```

**Purpose**: Direct monitoring of false positive rate among frequent customers.

## 🚀 Key Methodology

### 1. Frequent Customer Definition

- **Threshold**: ≥4 purchases per month at the same merchant
- **Rationale**: Inverse relationship between purchase frequency and fraud probability
- **Impact**: 1.4% of transactions, significantly lower fraud rates

### 2. Feature Engineering (56 Features)

- **Behavioral**: Transaction recency, regularity, amount anomalies
- **Temporal**: Hour cycles, day patterns, time since previous transaction
- **Geographical**: Distance from customer median location
- **Merchant**: Customer share, first-time interactions

### 3. Model Optimization

- **Base Model**: LightGBM with balanced class weights
- **Custom Weighting**: 10x weight for legitimate frequent customer transactions
- **Hyperparameter Tuning**: Optuna (50 trials) minimizing `business_fp_tp_ratio`

## 📊 Results Summary

| Metric              | Base Model | Optimized Model | Improvement |
| ------------------- | ---------- | --------------- | ----------- |
| Recall              | 78.7%      | 86.4%           | +7.7pp      |
| FP Rate (Overall)   | 0.057%     | 0.017%          | -70%        |
| FP Count (Frequent) | 3          | 0               | -100%       |
| Alert Efficiency    | 1.384      | 1.103           | -20%        |

## 🛠️ How to Run

### Prerequisites

```bash
pip install pandas numpy lightgbm optuna matplotlib seaborn jupyter scikit-learn
```

### Execution Steps

1. **Data Exploration**

   ```bash
   jupyter notebook notebooks/01_exploration.ipynb
   ```

2. **Feature Engineering**

   ```bash
   jupyter notebook notebooks/02_feature_engineering.ipynb
   ```

3. **Model Training & Optimization**

   ```bash
   jupyter notebook notebooks/03_base_model.ipynb
   ```

4. **Generate Report** (Optional)
   ```bash
   pdflatex report.tex
   ```

### Data Requirements

- Place `transactions.csv` in `data/raw/` directory
- Minimum columns: transaction amounts, timestamps, customer IDs, merchant IDs, fraud labels

## 🔬 Technical Highlights

- **Temporal Validation**: Train ≤ Sep 2020, Valid Oct-Nov 2020, Test Dec 2020
- **Imbalanced Classes**: 0.52% fraud rate, handled with balanced weights + custom metrics
- **Feature Importance**: Amount, frequent customer flag, merchant patterns, recency
- **Optimization**: TPE sampler with 42 seed for reproducibility

## 📈 Business Impact

The optimized model delivers:

- **Enhanced Customer Experience**: Zero false alarms for frequent customers
- **Operational Efficiency**: 70% reduction in overall false alerts
- **Improved Security**: Higher fraud detection rate (86.4% vs 78.7%)
- **Cost Reduction**: Better alert-to-fraud ratio reduces investigation costs

## 📝 License

See [LICENSE](LICENSE) for details.

## 🏗️ Future Enhancements

- Post-processing rules for residual false positives
- Deep learning models (TabNet, FT-Transformer) with same custom metrics
- Dynamic weight adjustment based on seasonality
- Real-time threshold calibration in production
