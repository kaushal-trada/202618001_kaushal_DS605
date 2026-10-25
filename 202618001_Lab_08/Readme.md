# DS605 — Lab Assignment 8
## Decision Trees and Support Vector Machines (SVM)

**Student Name:** Trada Kaushal  
**Student ID:** 202618001  
**Course:** DS605 — Fundamentals of Machine Learning

## Project Overview

This lab explores two supervised classification methods—**Decision Trees** and **Support Vector Machines (SVMs)**—through implementation, visualization, and evaluation. A Decision Tree and a linear soft-margin SVM are implemented **from scratch using Python and NumPy**. Their results are compared with equivalent models from Scikit-learn.

The notebook also investigates **linear, polynomial, and radial basis function (RBF) kernels** on nonlinear synthetic datasets. Finally, five classifiers are evaluated on a common test split of the Banknote Authentication dataset. The purpose is not only to compare accuracy, but also to understand model complexity, overfitting, feature scaling, interpretability, and computational cost.

## Dataset

**Primary dataset:** [UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

The dataset contains **1,372 observations**, four numerical input features extracted from wavelet-transformed banknote images, and a binary class label (`0` or `1`).

| Column | Description |
| --- | --- |
| `variance` | Variance of the wavelet-transformed image features |
| `skewness` | Skewness of the transformed image features |
| `curtosis` | Kurtosis of the transformed image features (column name retained from the data) |
| `entropy` | Entropy feature from the transformed image |
| `class` | Binary target label |

The notebook downloads the CSV-like text file from UCI. Initial access requires an internet connection.

**Feature usage:** `variance` and `skewness` are selected for **2D decision-boundary illustrations**. All **four** inputs are used for the final comparison so the models can benefit from the full feature information.

## Methodology

1. **Explore the data:** inspect records, data types, missing values, duplicate rows, class balance, and feature distributions.
2. **Split consistently:** use an **80% training / 20% testing** stratified split with `random_state=42`, giving approximately 1,097 training and 275 test observations.
3. **Prepare features appropriately:** Decision Trees use raw feature values; SVMs use standardization fitted **only on training data** to avoid leakage.
4. **Train and compare models:** use the same banknote training/test partitions, with the correct label format for each implementation.
5. **Evaluate:** compare test metrics, confusion matrices, model complexity, and training/prediction time. Do not infer final rankings until the notebook has been executed.

## Part A — Decision Tree From Scratch

A custom classification tree is implemented without calling Scikit-learn's tree-building algorithm. The implementation:

- Calculates **Gini impurity** and **entropy** to measure class mixing within a node.
- Tests candidate feature thresholds and selects the split with the **greatest reduction in weighted impurity**.
- Recursively builds left/right child nodes and applies stopping rules such as maximum depth, minimum leaf size, and pure nodes.
- Predicts by following learned split rules from the root to a leaf.

**Experiments:** Compare the manual tree with `DecisionTreeClassifier` across **Gini vs entropy** and maximum depths `1, 2, 3, 4, 5, 6, 8, 10`. The notebook reports training accuracy, testing accuracy, and agreement between the two implementations. Plots show accuracy versus tree depth and 2D decision boundaries at selected depths.

**Key idea:** Shallower trees may underfit, while deeper trees can model more detail but risk overfitting. Trees are relatively easy to interpret because predictions follow explicit rules.

## Part B — Linear SVM From Scratch

A **soft-margin linear SVM** is implemented with NumPy. It learns a weight vector `w` and bias `b`, giving the decision function:

`f(x) = wᵀx + b`

The implementation uses **hinge loss** and **full-batch subgradient descent**. It stores the optimization loss at each iteration so convergence can be inspected.

**Experiments include:**

- Visualizing the separating hyperplane (`f(x) = 0`) and functional-margin contours (`f(x) = ±1`).
- Separating **classification errors** from correctly classified points **inside the margin**. Every classification error is a margin violation, but not every margin violation is a classification error.
- Comparing manual and Scikit-learn linear SVM **weights, bias, accuracy, and margin width** (`2 / ||w||`).
- Studying the effects of regularization `C`, learning rate, and iteration count on loss, margin, and accuracy.

**Key idea:** Larger `C` places a greater penalty on margin violations, while the weight norm controls geometric margin width. Manual subgradient optimization may not match Scikit-learn exactly if it has not fully converged.

## Part C — Nonlinear Kernel SVM

Two synthetic binary datasets are generated:

- **`make_moons`:** two interlocking crescent-shaped groups.
- **`make_circles`:** an inner circle and an outer ring.

For **each dataset**, Scikit-learn's `SVC` is trained using **linear**, **polynomial**, and **RBF** kernels—six models in total. The experiments compare training/test accuracy, support-vector counts, decision regions, margin contours, correctly classified margin violations, and misclassified points.

**Key idea:** A linear SVM can only create a straight separating boundary in the input space. Polynomial and RBF kernels can produce nonlinear boundaries, which are often more suitable for curved or enclosed class patterns. Kernel flexibility must still be balanced against overfitting.

## Part D — Final Comparison on Banknote Data

Five models are trained and evaluated using **all four features** and the same banknote train/test split:

| Model | Implementation | Input preparation |
| --- | --- | --- |
| Decision Tree | From scratch | Original features |
| Decision Tree | Scikit-learn | Original features |
| Linear SVM | From scratch | Standardized features |
| Linear SVM | Scikit-learn | Standardized features |
| RBF SVM | Scikit-learn | Standardized features |

The final results table includes:

- **Accuracy, macro precision, macro recall, and macro F1-score** for predictive performance.
- **Training and testing accuracy** to assess potential overfitting.
- **Confusion matrices** to identify the two types of classification mistakes.
- **Training time (seconds)** and **prediction time (milliseconds)** to compare computational efficiency.
- **Tree node/leaf counts, linear-SVM coefficients, and RBF support-vector counts** as indicators of model structure and complexity.

Time plots are displayed separately from the score table. Reported times depend on the machine and the current workload; single-run timing differences should be treated cautiously.

## Main Concepts and Conclusions

| Aspect | Decision Tree | Linear SVM | RBF SVM |
| --- | --- | --- | --- |
| Interpretability | High: understandable feature rules | Moderate: coefficients and bias | Lower: nonlinear kernel-based predictions |
| Feature scaling | Generally unnecessary | Important | Important |
| Main complexity controls | Depth and minimum leaf size | `C` and weight regularization | `C`, `gamma`, and support vectors |
| Typical boundary | Axis-aligned regions | Straight hyperplane | Flexible nonlinear boundary |

Decision Trees emphasize explainable rules but can overfit as tree depth increases. Linear SVMs provide a relatively compact margin-based model, while RBF SVMs provide greater flexibility for nonlinear relationships. **The preferred classifier should be chosen from actual test-set performance, generalization behavior, interpretability needs, and execution cost—not from a presumed winner.**

## Requirements

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

## Running the Notebook

1. Clone or download this repository.
2. Install the dependencies listed above.
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the lab assignment notebook (`.ipynb`).
5. Run the cells **in order**, starting from dataset loading. The later experiments depend on models and variables created in earlier cells.
6. Review the generated comparison tables, loss curves, decision boundaries, confusion matrices, and timing plots.

> **Reproducibility note:** The dataset split and synthetic-data generation use fixed random seeds. Some solver behavior and measured execution times may vary across systems. Scikit-learn is used for splitting, scaling, evaluation, baseline models, and kernel experiments—not for the internal learning logic of the manual Decision Tree or manual linear SVM.
