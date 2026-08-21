# 🏨 Lab_03 Hotel Booking Cancellation Prediction

> **Machine Learning Classification | Data Preprocessing | Model Evaluation**

---

## 📌 Assignment Overview

This lab focuses on predicting whether a hotel booking will be **canceled or not canceled** using machine learning classification models.

The workflow includes:

**Data Understanding → Data Cleaning → Outlier Handling → Preprocessing → Model Training → Performance Evaluation**

The main objective is to compare how two preprocessing approaches—**StandardScaler** and **MinMaxScaler**—affect the performance of **Logistic Regression** and **Decision Tree** classifiers.

---

## 👤 Student Information

| Details | Information |
|---|---|
| **Name** | Kaushal Trada |
| **Student ID** | 202618001 |
| **Assignment** | Lab_03 |
| **Title** | Hotel Booking Cancellation Prediction |

---

## 📂 Dataset

**Dataset:** Hotel Booking Demand Dataset

**Dataset Link:**  
https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand

The dataset contains hotel reservation information such as booking details, stay duration, customer information, room type, market segment, and other booking-related features.

### 🎯 Target Variable

`is_canceled`

- `0` → Booking was **Not Canceled**
- `1` → Booking was **Canceled**

---

# 🔍 Part A — Data Loading and Preprocessing

## 1️⃣ Data Understanding

The dataset was initially explored using:

- `head()`
- `shape`
- `info()`
- `describe()`
- `dtypes`

The class distribution of `is_canceled` was also examined before defining:

```text
X → Input features
y → is_canceled
```

Numerical and categorical columns were then identified separately for appropriate preprocessing.

---

## 2️⃣ Missing Values and Data Cleaning

Missing-value counts and percentages were checked for all columns.

### High Missingness

The `company` column was dropped because it contained a very high amount of missing data.

```python
X = X.drop(columns=["company"])
```

---

## 3️⃣ Data Leakage Prevention

The following columns were removed:

```text
reservation_status
reservation_status_date
```

These features directly reveal information about the final booking outcome. Keeping them could give the model access to information that would not normally be available when predicting cancellation.

Therefore, they were removed to prevent **data leakage**.

---

## 4️⃣ Outlier Detection and Removal

All numerical columns were analyzed for extreme outliers using the **IQR method**.

To remove only clear or extreme outliers, a wider threshold of:

```text
Q1 - 3 × IQR
Q3 + 3 × IQR
```

was used.

### Outlier Removal Result

| Measure | Result |
|---|---:|
| Original Rows | 119,390 |
| Rows After Removal | 60,249 |
| Rows Removed | 59,141 |
| Percentage Removed | 49.54% |

---

# ⚙️ Part B — Preprocessing and Model Training

## 5️⃣ Train-Test Split

The dataset was divided using:

```python
train_test_split(
    test_size=0.2,
    stratify=y,
    random_state=42
)
```

The **same train-test split** was used for all four experiments to ensure a fair comparison.

---

## 6️⃣ Preprocessing Pipelines

### 🔢 Numerical Features

Missing numerical values were handled using:

```text
KNNImputer(n_neighbors=5)
```

Two scaling approaches were tested.

### Pipeline A

```text
Numerical Features
        ↓
KNNImputer(n_neighbors=5)
        ↓
StandardScaler
```

### Pipeline B

```text
Numerical Features
        ↓
KNNImputer(n_neighbors=5)
        ↓
MinMaxScaler
```

### 🔤 Categorical Features

Categorical preprocessing:

```text
Categorical Features
        ↓
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

`ColumnTransformer` was used to combine numerical and categorical transformations.

All preprocessing was placed inside Scikit-learn `Pipeline` objects so that fitting occurred **only on the training data**, helping prevent data leakage.

---

# 🤖 Classification Models

The following models were trained with the same settings throughout the comparison.

## Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

## Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

This produced four model-pipeline combinations:

1. Logistic Regression + Pipeline A
2. Logistic Regression + Pipeline B
3. Decision Tree + Pipeline A
4. Decision Tree + Pipeline B

---

# 📊 Final Model Comparison

| Experiment | Train Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression + Pipeline A | 78.62% | 78.99% | 78.99% | 65.32% | 71.51% |
| Logistic Regression + Pipeline B | 78.70% | 79.05% | 79.17% | 65.25% | 71.54% |
| **Decision Tree + Pipeline A** | **99.59%** | **84.10%** | **80.33%** | **80.26%** | **80.30%** |
| Decision Tree + Pipeline B | 99.59% | 84.05% | 80.24% | 80.24% | 80.24% |

---

# 🏆 Best Model

## **Decision Tree + Pipeline A**

### Preprocessing

```text
Numerical Features
        ↓
KNNImputer(n_neighbors=5)
        ↓
StandardScaler

Categorical Features
        ↓
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
        ↓
DecisionTreeClassifier(random_state=42)
```

### Best Performance

| Metric | Score |
|---|---:|
| Training Accuracy | 99.59% |
| Testing Accuracy | **84.10%** |
| Precision | 80.33% |
| Recall | 80.26% |
| F1-Score | **80.30%** |

---

# 📈 Final Observations

1. **Decision Tree + Pipeline A achieved the best overall performance**, with the highest testing accuracy (**84.10%**) and F1-score (**80.30%**).

2. **Logistic Regression + Pipeline B performed slightly better than Pipeline A**, but the difference between the two scaling methods was very small.

3. The Decision Tree achieved much better recall for canceled bookings (**80.26%**) compared with Logistic Regression (approximately **65%**).

4. The Decision Tree shows **possible overfitting**, as its training accuracy (**99.59%**) is substantially higher than its testing accuracy (approximately **84%**).

5. **StandardScaler and MinMaxScaler made very little difference for the Decision Tree**, while the Logistic Regression results showed a small improvement with MinMaxScaler.

---

# 📊 Confusion Matrix Summary

The best models selected from the comparison were:

- **Best Logistic Regression:** Logistic Regression + Pipeline B
- **Best Decision Tree:** Decision Tree + Pipeline A

The confusion matrices were used to examine how correctly each model classified **canceled** and **not canceled** bookings.

The Decision Tree showed stronger overall cancellation detection performance, which is consistent with its higher **recall** and **F1-score**.

---

# ✅ Conclusion

This lab demonstrated a complete machine learning classification workflow, including:

- Data exploration
- Missing-value analysis
- Data leakage prevention
- Extreme outlier removal
- Feature preprocessing
- Pipeline construction
- Logistic Regression training
- Decision Tree training
- Performance comparison
- Confusion matrix analysis
- Overfitting evaluation

Among all four experiments, **Decision Tree + Pipeline A** achieved the best overall result with an **84.10% testing accuracy** and an **80.30% F1-score**.

However, the large gap between training and testing accuracy suggests **possible overfitting**. Logistic Regression produced more stable training and testing performance but had lower recall and F1-score than the Decision Tree.

---

**Author:** Kaushal Trada  
**Student ID:** 202618001  
**Assignment:** Lab_03
