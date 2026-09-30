# Lab 06 — Machine Learning Classification

| Student-ID | Name |
|---|---|
| 202618001 | Kaushal Trada |

## Overview

This notebook contains Lab 06 experiments covering image classification, text classification, and feature-selection-based dimensionality reduction.

The work is divided into three parts:

- **Part A:** Crack vs Non-Crack image classification using manually extracted image features.
- **Part B:** Spam email classification using precomputed text features.
- **Part C:** Feature reduction and comparison of original vs reduced text representations.

## Files

- `202618001_Lab_06.ipynb` — Main Jupyter Notebook
- `emails.csv` — Dataset used for spam email classification
- Image dataset folder containing:
  - `Cracks`
  - `Non-Cracks`

## Libraries Used

The notebook uses the following Python libraries:

- `os`
- `time`
- `numpy`
- `pandas`
- `matplotlib`
- `opencv-python (cv2)`
- `scikit-learn`

## Part A — Crack Image Classification

### Image Preprocessing

The image-processing pipeline includes:

1. Loading images from the dataset folders.
2. Resizing all images to **256 × 256**.
3. Converting images to grayscale.
4. Applying Gaussian blur using a **7 × 7** kernel.
5. Performing Canny edge detection.

The notebook supports both normal and adaptive Canny threshold selection. The adaptive method calculates thresholds from the median intensity of the blurred image.

### Extracted Features

Features extracted from each image include intensity- and edge-based information such as:

- Mean brightness
- Contrast
- Minimum intensity
- Maximum intensity
- Median intensity
- 25th percentile
- 75th percentile
- Interquartile range
- Dark-pixel ratio
- Bright-pixel ratio
- Intensity range
- Edge count
- Edge density

### Models

Three classifiers are trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest

### Image Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | Training Time | Prediction Time |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9375 | 0.9730 | 0.9000 | 0.9351 | 0.0051 s | 0.0007 s |
| Decision Tree | 0.9250 | 0.9474 | 0.9000 | 0.9231 | 0.0031 s | 0.0014 s |
| Random Forest | 0.9375 | 0.9487 | 0.9250 | 0.9367 | 0.1427 s | 0.0081 s |

Logistic Regression and Random Forest both achieved **93.75% accuracy**, while Random Forest produced the highest F1 score of **0.9367**.

## Part B — Spam Email Classification

The email dataset is loaded from `emails.csv`.

The target column used in the notebook is:

- `Prediction`

The remaining numeric columns are used as input features.

### Models

Two classifiers are compared:

- Multinomial Naive Bayes
- Logistic Regression

### Text Classification Results

| Model | Accuracy | Precision | Recall | F1 Score | Training Time | Prediction Time |
|---|---:|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 0.9420 | 0.8681 | 0.9433 | 0.9042 | 0.1091 s | 0.0726 s |
| Logistic Regression | 0.9826 | 0.9578 | 0.9833 | 0.9704 | 5.8784 s | 0.0648 s |

Logistic Regression achieved the strongest result for spam classification with **98.26% accuracy** and an **F1 score of 0.9704**.

## Part C — Feature Selection

Feature dimensionality is reduced using:

- `SelectKBest`
- Chi-square (`chi2`) feature scoring

The original representation contains **3000 features**, while the reduced representation keeps the best **1000 features**.

### Naive Bayes: Original vs Reduced Features

| Representation | Features | Accuracy | Precision | Recall | F1 Score | Training Time | Prediction Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 3000 | 0.9420 | 0.8681 | 0.9433 | 0.9042 | 0.1091 s | 0.0726 s |
| Reduced | 1000 | 0.9362 | 0.8589 | 0.9333 | 0.8946 | 0.0223 s | 0.0082 s |

### Overall Representation Comparison

| Model | Representation | Features | Accuracy | F1 Score | Training Time | Prediction Time |
|---|---|---:|---:|---:|---:|---:|
| Naive Bayes | Original | 3000 | 0.9420 | 0.9042 | 0.1091 s | 0.0726 s |
| Naive Bayes | Reduced | 1000 | 0.9362 | 0.8946 | 0.0223 s | 0.0082 s |
| Logistic Regression | Original | 3000 | 0.9826 | 0.9704 | 5.8784 s | 0.0648 s |
| Logistic Regression | Reduced | 1000 | 0.9729 | 0.9536 | 3.5636 s | 0.0066 s |

Reducing the number of text features from **3000 to 1000** substantially decreases training and prediction time while causing only a relatively small decrease in classification performance.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training Time
- Prediction Time

## Conclusion

For image classification, Logistic Regression and Random Forest both achieved **93.75% accuracy**, with Random Forest obtaining the highest F1 score.

For spam classification, Logistic Regression performed best in the experiments with **98.26% accuracy** and an **F1 score of 0.9704**.

Feature reduction from 3000 to 1000 text features improved computational efficiency considerably, with only a small reduction in predictive performance. This demonstrates the trade-off between model efficiency and classification accuracy.
