# Airbnb Price Prediction

Deployed link: https://202618001kaushalds605-jdo4fn2zndms8vrekdlt6p.streamlit.app/

## Student Information

- **Student ID:** 202618001
- **Name:** Kaushal Trada

---

## 1. Project Overview

This project builds a machine learning model to predict the **nightly price of Airbnb listings in New York City** using the Airbnb NYC 2019 dataset.

The project covers:

- Data loading and exploration
- Data cleaning
- Missing-value handling
- Feature engineering
- Target transformation
- Train-test splitting
- Feature preprocessing
- Regression model training
- Model evaluation
- Random Forest hyperparameter tuning
- XGBoost comparison
- Saving the final model using Joblib

---

## 2. Dataset

The project uses the **AB_NYC_2019.csv** dataset.

### Dataset size

- **Rows:** 48,895
- **Columns:** 16
- **Target variable:** `price`

### Original columns

```text
id
name
host_id
host_name
neighbourhood_group
neighbourhood
latitude
longitude
room_type
price
minimum_nights
number_of_reviews
last_review
reviews_per_month
calculated_host_listings_count
availability_365
```

---

## 3. Data Cleaning

A copy of the original dataset was created for cleaning.

### Duplicate records

Duplicate rows were removed using:

```python
df_clean = df_clean.drop_duplicates()
```

The dataset remained:

```text
48,895 rows × 16 columns
```

after duplicate removal, indicating that no duplicate rows were removed.

### Unnecessary columns

The following columns were removed:

```python
[
    "id",
    "name",
    "host_id",
    "host_name"
]
```

These columns were not used as predictive inputs because IDs are identifiers and the name/host-name fields were not used as structured predictive variables.

### Missing values

Missing values in:

```text
reviews_per_month
```

were replaced with `0`, representing no recorded monthly reviews.

The `last_review` column was later transformed into a more useful numerical feature.

---

## 4. Exploratory Data Analysis

The analysis examined the distribution of Airbnb prices, numerical variables, categorical variables, and price differences across neighbourhood groups.

### Price distribution

The `price` variable is strongly right-skewed, with a relatively small number of high-priced listings.

Because some high prices may represent genuine Airbnb listings, the project does **not automatically remove all price outliers**.

---

## 5. Feature Engineering

Five additional features were created.

### 5.1 Distance to Midtown

An approximate distance from Midtown/Times Square was calculated using latitude and longitude.

```python
midtown_lat = 40.7580
midtown_lon = -73.9855

df_clean["distance_to_midtown"] = np.sqrt(
    (df_clean["latitude"] - midtown_lat) ** 2 +
    (df_clean["longitude"] - midtown_lon) ** 2
)
```

This feature captures the effect of location relative to a central NYC area.

### 5.2 Has Reviews

A binary feature indicates whether a listing has received at least one review.

```python
df_clean["has_reviews"] = (
    df_clean["number_of_reviews"] > 0
).astype(int)
```

### 5.3 Log Number of Reviews

The number of reviews was log-transformed:

```python
df_clean["log_number_of_reviews"] = np.log1p(
    df_clean["number_of_reviews"]
)
```

This reduces the influence of very large review counts.

### 5.4 Commercial Host

Hosts with more than one listing were identified as commercial hosts:

```python
df_clean["is_commercial_host"] = (
    df_clean["calculated_host_listings_count"] > 1
).astype(int)
```

### 5.5 Days Since Last Review

`last_review` was converted to a date and transformed into:

```text
days_since_last_review
```

using:

```text
Reference date = 2019-12-31
```

Listings without review history were assigned:

```text
-1
```

The original `last_review` column was then removed.

---

## 6. Target Transformation

The original `price` column was retained because it represents the actual Airbnb price in dollars.

Because price is highly right-skewed, a logarithmic target was created:

```python
df_clean["log_price"] = np.log1p(df_clean["price"])
```

The model uses `log_price` as the modeling target.

Predictions are converted back to the original dollar scale using:

```python
np.expm1(prediction)
```

This makes the final evaluation metrics interpretable in dollars.

---

## 7. Features Used for Modeling

The final model uses **15 input features**.

```python
features = [
    "neighbourhood_group",
    "neighbourhood",
    "latitude",
    "longitude",
    "distance_to_midtown",
    "room_type",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "has_reviews",
    "log_number_of_reviews",
    "days_since_last_review",
    "is_commercial_host"
]
```

### Categorical features

```text
neighbourhood_group
neighbourhood
room_type
```

### Numerical features

```text
latitude
longitude
distance_to_midtown
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
availability_365
has_reviews
log_number_of_reviews
days_since_last_review
is_commercial_host
```

---

## 8. Train-Test Split

Because the price distribution is skewed, price quantile bins were created for stratification.

```python
price_bins = pd.qcut(
    df_clean["price"],
    q=5,
    labels=False,
    duplicates="drop"
)
```

The data was split into:

- **80% training data**
- **20% testing data**

Using:

```python
random_state=42
```

### Split sizes

```text
Training samples: 39,116
Testing samples:   9,779
```

---

## 9. Preprocessing

A `ColumnTransformer` was used to apply different preprocessing steps to numerical and categorical variables.

### Numerical features

Numerical features were standardized using:

```python
StandardScaler()
```

### Categorical features

Categorical variables were converted using:

```python
OneHotEncoder(
    handle_unknown="ignore"
)
```

Using `handle_unknown="ignore"` prevents errors when an unseen category occurs during prediction.

---

## 10. Machine Learning Models

Three regression approaches were evaluated:

1. Linear Regression
2. Random Forest Regression
3. XGBoost Regression

The models were evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

### Metrics

**MAE:** Average absolute difference between actual and predicted price.

**RMSE:** Measures prediction error while giving more weight to large errors.

**R²:** Measures the proportion of variation in the target explained by the model.

---

## 11. Model Performance

The notebook produced the following test-set results:

| Model | MAE ($) | RMSE ($) | R² |
|---|---:|---:|---:|
| Linear Regression | 57.49 | 192.48 | 0.1406 |
| Random Forest | 54.36 | 185.01 | 0.2060 |
| XGBoost | 53.86 | 188.21 | 0.1784 |

### Best result

Based on the reported **R²**, Random Forest achieved the highest score:

```text
R² = 0.2060
```

It also achieved:

```text
MAE  = $54.36
RMSE = $185.01
```

XGBoost achieved a slightly lower MAE:

```text
MAE = $53.86
```

but Random Forest had the highest R² and lowest RMSE among the three models.

---

## 12. Random Forest Hyperparameter Tuning

A Random Forest pipeline was created with preprocessing and `RandomForestRegressor`.

The tuning section uses `GridSearchCV`.

### Parameters tuned

```python
param_grid = {
    "model__n_estimators": [100, 150],
    "model__max_depth": [10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}
```

This produces:

```text
16 parameter combinations
```

with:

```text
2-fold cross-validation
```

The scoring metric used for the grid search is:

```python
neg_root_mean_squared_error
```

### Important implementation note

The notebook defines and runs the Random Forest hyperparameter search, but the final model saved for deployment is the previously trained `random_forest_model` rather than `rf_grid_search.best_estimator_`.

Therefore, the reported final deployment model is the trained Random Forest with:

```python
n_estimators=100
random_state=42
n_jobs=-1
```

If the tuned Random Forest is intended to be the final model, `rf_grid_search.best_estimator_` should be evaluated first and then saved.

---

## 13. XGBoost Comparison

XGBoost was also trained as a stronger nonlinear ensemble model.

The model used:

```python
XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
```

Its test performance was:

```text
MAE  = $53.86
RMSE = $188.21
R²   = 0.1784
```

---

## 14. Final Model

Random Forest was selected as the final model in the notebook because it achieved the highest R² score among the three evaluated models.

The complete preprocessing and Random Forest pipeline was saved using Joblib:

```python
import joblib

final_model = random_forest_model

joblib.dump(
    final_model,
    "airbnb_price_model.pkl"
)
```

The saved pipeline includes:

- Feature preprocessing
- One-hot encoding
- Standardization
- Random Forest regression

The saved model can therefore be loaded directly for prediction.

---

## 15. Model Loading

The saved model can be loaded with:

```python
loaded_model = joblib.load(
    "airbnb_price_model.pkl"
)
```

The loaded object is a Scikit-learn:

```text
Pipeline
```

---

## 16. Project Structure

Recommended project structure:

```text
Airbnb_Price_Prediction/
│
├── AB_NYC_2019.csv
├── 202618001_lab_04imp.ipynb
├── airbnb_price_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 17. Deployment

The trained model can be used in a web application using frameworks such as:

- Streamlit
- Gradio

The application can accept Airbnb listing characteristics and return a predicted nightly price.

A typical Streamlit application can be started with:

```bash
streamlit run app.py
```

A Gradio application can be started with:

```bash
python app.py
```

---

## 18. Requirements

Example `requirements.txt`:

```text
pandas
numpy
scikit-learn==1.9.1
xgboost
joblib
gradio
streamlit
```

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## 19. Limitations

### Extreme prices

Very expensive Airbnb listings are difficult to predict because they represent a small portion of the dataset.

### Missing property-level information

The selected dataset does not provide several potentially important property characteristics such as:

- Number of bedrooms
- Number of bathrooms
- Property size
- Amenities

These missing variables can limit prediction accuracy.

### Moderate R²

The best reported R² is:

```text
0.2060
```

This indicates that a substantial amount of price variation remains unexplained by the available features.

### Historical data

The dataset represents Airbnb listings from 2019. Therefore, predictions should not be interpreted as current NYC market prices without retraining the model using newer data.

---

## 20. Conclusion

This project presents a complete machine learning workflow for Airbnb price prediction.

The data was cleaned, important features were engineered, and the skewed price target was log-transformed. Numerical features were standardized and categorical features were one-hot encoded.

Among the evaluated models, **Random Forest achieved the highest R² score of 0.2060** and an RMSE of **$185.01**. XGBoost produced a slightly lower MAE of **$53.86**, while Random Forest achieved an MAE of **$54.36**.

The final Random Forest pipeline was saved using Joblib and can be used for deployment in a web application.
