# Lab-05: Machine Learning with Scikit-learn and From Scratch

**Student ID:** 202618001\
**Student Name:** Kaushal Trada\
**Course:** DS605 -- Fundamentals of Machine Learning\
**Lab:** 05

------------------------------------------------------------------------

## 1. Objective

This lab implements and compares machine-learning workflows using:

1.  **Scikit-learn**
2.  **From-scratch NumPy/Pandas implementations**
3.  **Optimized from-scratch Logistic Regression**
4.  **L2-regularized from-scratch Logistic Regression**

The lab compares predictive performance, training time, and prediction
time while using the same train-test split and feature representation
for fair comparison.

------------------------------------------------------------------------

## 2. Dataset

**Dataset:** UCI Productivity Prediction of Garment Employees

**File used:**

``` text
garments_worker_productivity.csv
```

The dataset contains information related to garment-production
productivity, including variables such as:

-   Department
-   Team
-   Targeted productivity
-   Over time
-   Incentives
-   WIP
-   Workers
-   Actual productivity

------------------------------------------------------------------------

## 3. Tasks

### Regression

The objective is to predict:

``` text
actual_productivity
```

using **Linear Regression**.

Evaluation metrics:

-   MAE
-   RMSE
-   R²
-   Training time
-   Prediction time

### Classification

A binary target was created:

``` text
MeetsTarget = 1
if actual_productivity >= targeted_productivity
else 0
```

`actual_productivity` was excluded from the classification features to
avoid target leakage.

The classification model is **Logistic Regression**.

Evaluation metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-Score
-   Training time
-   Prediction time

------------------------------------------------------------------------

## 4. Exploratory Data Analysis

The notebook performs:

-   Dataset shape and information inspection
-   Missing-value analysis
-   Duplicate-row checking
-   Descriptive statistics
-   Unique-value analysis
-   Categorical-value inspection
-   Numeric-feature analysis
-   Actual-productivity distribution analysis
-   Skewness calculation
-   Correlation analysis
-   Target-class distribution analysis

The `department` column was also cleaned using whitespace stripping.

------------------------------------------------------------------------

## 5. Train-Test Split

A single fixed 80/20 train-test split was used.

``` text
Test size   : 20%
Random state: 42
```

Because the regression target has a continuous distribution and the
classification target is binary, joint stratification was used.

The regression target was divided into five quantile bins and combined
with `MeetsTarget` to create the stratification variable.

This same train-test split was reused throughout the experiment to
ensure a fair comparison.

------------------------------------------------------------------------

## 6. Part A --- Scikit-learn Implementation

### Preprocessing

The Scikit-learn workflow uses:

-   Median imputation for numeric features
-   Standard scaling for numeric features
-   Most-frequent imputation for categorical features
-   One-hot encoding for categorical features
-   `handle_unknown="ignore"` for unseen categorical values

The preprocessing pipeline is fitted on the training data and then
applied to the test data.

### Models

``` text
LinearRegression()
LogisticRegression(max_iter=1000)
```

Training and prediction times are measured using:

``` python
time.perf_counter()
```

------------------------------------------------------------------------

## 7. Part B --- From-Scratch Implementation

Part B recreates the machine-learning workflow without Scikit-learn
model, preprocessing, or metric utilities.

### Manual preprocessing

The workflow includes:

1.  Missing-value handling
2.  Pandas one-hot encoding
3.  Train-data-based feature scaling

### Manual Linear Regression

The model uses the closed-form solution:

\[ `\beta `{=tex}= (X^TX)^{-1}X\^Ty \]

implemented using NumPy's numerically safer linear-system solver.

An intercept column is added manually.

### Manual Logistic Regression

The Logistic Regression implementation includes:

1.  Parameter initialization
2.  Linear score calculation
3.  Sigmoid function
4.  Predicted probability calculation
5.  Log-loss calculation
6.  Gradient calculation
7.  Gradient-descent parameter updates
8.  Probability-to-class conversion using a 0.5 threshold
9.  Manual confusion-matrix counts
10. Accuracy
11. Precision
12. Recall
13. F1-score

The training loop is vectorized using NumPy.

------------------------------------------------------------------------

## 8. Part C --- Comparison and Optimization

The main optimization target was the manual Logistic Regression
implementation because its original training time was substantially
higher than the Scikit-learn implementation.

### Optimization Step 1 --- Iteration and Convergence

The initial Logistic Regression used:

``` text
Learning rate = 0.01
Iterations    = 1000
Tolerance     = 1e-7
```

Increasing the maximum number of iterations to 5000 reduced the loss
further, but the strict convergence tolerance was still not reached.

Therefore, simply increasing the iteration count was not considered
sufficient optimization.

### Optimization Step 2 --- Learning-Rate Tuning

Learning rates were tested to investigate faster convergence.

The experiments showed that a learning rate of:

``` text
0.10
```

gave improved classification performance compared with the original
from-scratch implementation.

### Convergence Tolerance

The tolerance was changed from:

``` text
1e-7
```

to:

``` text
1e-5
```

The stricter criterion was not reached within 5000 iterations even
though the loss was decreasing. The less strict tolerance was adopted as
a practical stopping criterion, allowing the optimizer to stop once
further loss improvement became sufficiently small.

### Optimization Step 3 --- L2 Regularization

L2 regularization was added to the Logistic Regression objective:

\[ L\_{total} = L\_{logistic} + `\frac{\lambda}{2n}`{=tex}
`\sum`{=tex}\_{j=1}\^{p}`\beta`{=tex}\_j\^2 \]

The intercept was not regularized.

The following regularization strengths were tested:

``` text
0.001
0.01
0.1
1.0
10.0
```

This experiment investigated the effect of regularization on predictive
performance and convergence.

------------------------------------------------------------------------

## 9. Final Classification Comparison

Results obtained in the final comparison are:

  -------------------------------------------------------------------------------------
  Implementation     Accuracy   Precision     Recall   F1-Score   Training   Prediction
                                                                  Time (s)     Time (s)
  ---------------- ---------- ----------- ---------- ---------- ---------- ------------
  Scikit-learn       0.741667    0.765258   0.931429   0.840206   0.012849     0.000602

  From Scratch       0.733333    0.732218   1.000000   0.845411   0.238856     0.000401

  Optimized From     0.745833    0.761468   0.948571   0.844784   0.396844     0.000456
  Scratch                                                                  

  From Scratch +     0.745833    0.761468   0.948571   0.844784   0.302596     0.000090
  L2                                                                       
  -------------------------------------------------------------------------------------

Runtime values are measured during notebook execution and may vary
between runs because of system conditions and the very short execution
times involved.

------------------------------------------------------------------------

## 10. Regression Comparison

  ---------------------------------------------------------------------------------
  Implementation            MAE         RMSE           R²     Training   Prediction
                                                              Time (s)     Time (s)
  ---------------- ------------ ------------ ------------ ------------ ------------
  Scikit-learn         0.103598     0.143751     0.259663     0.006413     0.000567

  From Scratch         0.103598     0.143751     0.259662     0.001214     0.000364
  ---------------------------------------------------------------------------------

The from-scratch Linear Regression reproduces the Scikit-learn
predictive results almost exactly. The MAE and RMSE are identical at the
displayed precision, and the R² difference is approximately 0.000001.

------------------------------------------------------------------------

## 11. Baseline vs Optimized Logistic Regression

The original from-scratch Logistic Regression was compared with the
optimized version.

  Metric                  Baseline   Optimized
  --------------------- ---------- -----------
  Accuracy                0.733333    0.745833
  Precision               0.732218    0.761468
  Recall                  1.000000    0.948571
  F1-Score                0.845411    0.844784
  Training Time (s)       0.238856    0.302596
  Prediction Time (s)     0.000401    0.000090

The optimization increased accuracy and precision while recall decreased
slightly. The F1-score remained nearly unchanged.

------------------------------------------------------------------------

## 12. Discussion of Performance Differences

The predictive and runtime differences arise from differences in
implementation and optimization procedures.

### Predictive performance

The Scikit-learn and from-scratch implementations use different
optimization procedures, convergence criteria, and implementation
details. Therefore, their learned coefficients and predictions do not
have to be identical.

L2 regularization also changes the optimization objective by penalizing
large coefficients. This can change the decision boundary and the
resulting precision-recall trade-off.

### Training time

Scikit-learn uses a highly optimized Logistic Regression implementation.
The from-scratch version performs repeated vectorized NumPy
gradient-descent calculations, so training generally requires more
computation.

The manual Linear Regression implementation is different because it uses
a closed-form matrix solution rather than thousands of iterative
gradient-descent updates.

### Prediction time

Prediction is much simpler than training. Once the parameters are
learned, Logistic Regression mainly performs:

\[ X`\beta`{=tex} \]

followed by the sigmoid function and thresholding.

Therefore, prediction times are very small for all implementations.

Because the measured prediction times are extremely short, small
differences should be interpreted cautiously.

------------------------------------------------------------------------

## 13. Key Conclusions

1.  The Scikit-learn and manual Linear Regression implementations
    produced almost identical predictive results.
2.  The original manual Logistic Regression achieved comparable
    classification performance but required more training computation.
3.  Learning-rate tuning improved the manual Logistic Regression
    accuracy and precision.
4.  Convergence-based stopping provided a practical way to avoid
    unnecessarily continuing training after the loss had stabilized
    sufficiently.
5.  L2 regularization was tested as an additional optimization
    technique.
6.  The final manual implementations demonstrate how preprocessing,
    model training, prediction, metrics, optimization, and
    regularization can be implemented using NumPy/Pandas.
7.  Runtime differences depend on the implementation, number of
    optimization iterations, convergence behavior, and execution
    environment.

------------------------------------------------------------------------

## 14. Technologies Used

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Matplotlib
-   Jupyter Notebook

------------------------------------------------------------------------

## 15. How to Run

### 1. Install the required libraries

``` bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 2. Place the dataset in the notebook working directory

``` text
garments_worker_productivity.csv
```

### 3. Open the notebook

``` bash
jupyter notebook 202618001_Lab_05.ipynb
```

### 4. Run the notebook from top to bottom

The notebook is organized into:

``` text
Part A
  ├── Data loading and EDA
  ├── Train-test split
  ├── Preprocessing
  ├── Scikit-learn regression
  ├── Scikit-learn classification
  └── Evaluation

Part B
  ├── Manual preprocessing
  ├── Manual Linear Regression
  ├── Manual Logistic Regression
  └── Manual evaluation

Part C
  ├── Baseline comparison
  ├── Learning-rate optimization
  ├── Convergence tuning
  ├── L2 regularization
  └── Final comparison
```

------------------------------------------------------------------------

## 16. Files

``` text
202618001_Lab_05.ipynb
README.md
garments_worker_productivity.csv
```

------------------------------------------------------------------------

**Student:** Kaushal Trada\
**Student ID:** 202618001\
**Lab:** Lab-05 -- Machine Learning with Scikit-learn and From Scratch
