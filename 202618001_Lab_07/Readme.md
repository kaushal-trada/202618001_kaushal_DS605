# Face Recognition using PCA and Eigenfaces

**Image Representation, Principal Component Analysis, Eigenfaces, and Face Recognition**

## Id - 202618001 
## Name - Kaushal Trada 
 

## 1. Project Overview

This project explores dimensionality reduction and face recognition using Principal Component Analysis (PCA) on the AT&T Database of Faces (ORL).

PCA is implemented from scratch using NumPy linear algebra. Two mathematical approaches, covariance/eigendecomposition and Singular Value Decomposition (SVD), are used to obtain principal directions and compare their results.

Face recognition is performed using Euclidean distance in both the original pixel space and the reduced PCA space. Traditional machine learning classifiers, K-Nearest Neighbors (KNN) and Support Vector Machines (SVM), are also evaluated.

## 2. Dataset

**Dataset:** AT&T Database of Faces (ORL)

| Property | Description 
|---|---|
| Total images | 400 |
| Total subjects | 40 |
| Images per subject | 10 |
| Image dimensions | 112 × 92 |
| Image format | PGM (grayscale) |
| Features per image | 10,304 |
| Training images | 320 |
| Testing images | 80 |

The dataset contains facial images with variations in lighting, expressions, and appearance.

An 80:20 stratified train-test split ensures that each subject contributes 8 training and 2 testing images.

## 3. Methodology

### A. Image Representation

1. Load grayscale face images and their subject labels.
2. Inspect image dimensions and intensity distributions.
3. Flatten each image into a 10,304-dimensional feature vector.
4. Construct the complete feature matrix.
5. Perform a reproducible stratified train-test split.

The complete dataset is represented by:

$$
X \in \mathbb{R}^{400 \times 10304}$$

### B. PCA and Eigenfaces

**Mean Centering**

The mean face is calculated using training images only:

$$
\mu = \frac{1}{N}\sum_{i=1}^{N}x_i
$$

Each training image is centered by subtracting the mean face:

$$
X_c = X-\mu
$$

**Covariance/Eigendecomposition**

PCA identifies directions of maximum variance using eigenvalues and eigenvectors.

The covariance matrix is:

$$
C=\frac{X_c^TX_c}{N-1}
$$

To reduce computational requirements, a smaller Gram matrix is used:

$$
G=\frac{X_cX_c^T}{N-1}
$$

Its eigenvectors are used to recover principal directions in the original pixel space.

**Singular Value Decomposition**

SVD decomposes the centered matrix as:

$$
X_c=U\Sigma V^T
$$

The principal directions are obtained from the right singular vectors.

The eigenvalues are related to the singular values by:

$$
\lambda_i=\frac{\sigma_i^2}{N-1}
$$

Both PCA implementations are compared to verify their mathematical equivalence.

**Explained Variance**

The minimum number of principal components retaining at least 90% of total training variance is selected.

**Eigenfaces and Reconstruction**

Principal components are reshaped into images to visualize eigenfaces.

Images are projected into PCA space:

$$
Z=X_cV_k
$$

Reconstruction is performed using:

$$
\hat X=ZV_k^T+\mu
$$

Reconstruction quality is evaluated using Mean Squared Error (MSE). Storage requirements are compared with the original pixel representation, including the PCA coefficients, basis, and mean face.

### C. Face Recognition

Face recognition is implemented using a nearest-neighbor similarity approach.

For each test image, Euclidean distances are calculated against stored training images.

$$
d(a,b)=\sqrt{\sum_{i=1}^{k}(a_i-b_i)^2}
$$

The identity of the nearest training image is assigned to the test image.

Two approaches are compared:

- Raw-pixel nearest-neighbor recognition.
- PCA-feature nearest-neighbor recognition.

Recognition accuracy and examples of correct and incorrect predictions are examined.

### D. Traditional Machine Learning

Two classifiers are trained using PCA-transformed features:

- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)** with an RBF kernel

Hyperparameter optimization is performed using 3-fold stratified cross-validation.

Parameters evaluated include:

| Parameter | Candidate Values |
|---|---|
| PCA components | 20, 50, 100, 150 |
| KNN neighbors | 1, 3, 5 |
| SVM C | 0.1, 1, 10 |
| SVM kernel | RBF |

PCA is recalculated within each cross-validation training fold to prevent validation data leakage.

The selected configurations are then trained on the complete training set and evaluated on the held-out test set.

## 4. Experimental Results

*Replace the placeholders below with actual notebook outputs.*

### PCA Results

| Metric | Result |
|---|---|
| Original dimensions | 10,304 |
| Components for 90% variance | 98 |
| Reconstruction MSE | 334.9898829101203 |
| Original image storage | 3.14 MB |
| Total PCA storage | 4.01 MB |

### Face Recognition Results

| Method | Features | Accuracy (%) | Macro F1 |
|---|---|---|---|
| Raw Pixel 1-NN | 10,304 | 2.5 | 0.002 |
| PCA 1-NN | 98 | 96.25 | 0.954 |
| Tuned KNN | 50 | 98.75 | 0.987 |
| Tuned SVM | 50 | 98.75 | 0.987 |

The notebook also generates cumulative explained variance plots, eigenface visualizations, face reconstruction comparisons, and examples of correctly and incorrectly recognized faces.

## 5. Technologies Used

- **Python 3**
- NumPy — matrix operations, eigendecomposition, and SVD
- Pandas — result comparison tables
- Matplotlib — image visualization and plots
- Pillow — grayscale image loading
- Scikit-learn — train-test splitting, cross-validation, KNN, SVM, and evaluation metrics
- Jupyter Notebook — implementation and experiments

PCA and eigenface calculations are implemented using NumPy without Scikit-learn PCA.

## 6. How to Run

**Step 1:** Clone the repository.

**Step 2:** Install the required Python libraries:

```bash
pip install numpy pandas matplotlib pillow scikit-learn notebook
```

**Step 3:** Download the AT&T Database of Faces (ORL) and extract it into an `archive` folder containing subfolders `s1` through `s40`.

**Step 4:** Update the dataset path in the notebook if necessary.

**Step 5:** Open Jupyter Notebook:

```bash
jupyter notebook
```

Open the project notebook and execute the cells sequentially.

## 7. Key Observations

The experiments investigate the following:

- Dimensionality reduction using PCA while preserving at least 90% variance.
- Equivalence between covariance-based PCA and SVD.
- Visual reconstruction quality with reduced-dimensional representations.
- Storage trade-offs associated with retaining the PCA basis.
- Differences between raw-pixel and PCA-based face recognition.
- Effect of PCA component count and classifier hyperparameters on recognition accuracy.

## 8. Conclusion

This project demonstrates how PCA can represent high-dimensional facial images using a smaller set of principal components called eigenfaces.

By comparing covariance/eigendecomposition with SVD, the project establishes the mathematical relationship between the two methods.

Face recognition experiments using nearest-neighbor matching, KNN, and SVM evaluate how dimensionality reduction affects classification performance.

The experiments also illustrate the trade-offs between representation size, reconstruction quality, and recognition accuracy.

## 9. Limitations

- The dataset contains only 40 subjects and 400 images.
- Each subject has only two held-out test images.
- Recognition results may be sensitive to lighting, pose, and facial variations.
- Results on this dataset do not establish real-world face recognition reliability.
