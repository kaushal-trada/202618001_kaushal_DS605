# DS605: Fundamentals of Machine Learning
## Lab Assignment 2: Vectorized Programming with NumPy and Data Wrangling with Pandas

**Name:** TRADA KAUSHAL MANSUKHBHAI  
**Student ID:** 202618001
**Course:** DS605 - Fundamentals of Machine Learning  

---

## Project Overview
This repository contains the deliverables for Lab Assignment 2. The objective of this project is to practice and demonstrate proficiency in handling arrays, vectorized mathematical operations, and linear algebra using **NumPy**, as well as performing comprehensive data wrangling, exploratory data analysis (EDA), and feature engineering using **Pandas**.

The project is divided into two main parts:
* **Part A (NumPy):** Array generation, statistical calculations, indexing/slicing, matrix operations (addition, multiplication, transpose, determinant, inverse), and simulating normal distributions.
* **Part B (Pandas):** Loading, inspecting, querying, grouping, missing value imputation, outlier detection, and data visualization using the famous Titanic dataset.

## Dataset
* **Source:** Kaggle Titanic Dataset
* **File:** `train.csv` (Original dataset containing passenger demographics and survival status).
* **Description:** Includes features such as Passenger Class (Pclass), Sex, Age, Siblings/Spouses Aboard (SibSp), Parents/Children Aboard (Parch), Fare, and Embarked location.

## Repository Structure
* `Lab2_NumPy_Pandas.ipynb`: Main Jupyter Notebook containing all the Python code for Tasks 1 through 9.
* `train.csv`: The original raw Titanic dataset.
* `cleaned_titanic_data.csv`: The processed dataset after missing value imputation and feature engineering.
* `README.md`: Project documentation and key observations.

## Technologies Used
* **Python 3.x**
* **NumPy** (Vectorized operations, Linear Algebra, Statistics)
* **Pandas** (Data wrangling, filtering, grouping, pivot tables)
* **Matplotlib & Seaborn** (Data visualization, heatmaps, scatter/bar plots)
* **SciPy** (Statistical distributions)

---

## Key Observations (Task 9)
Based on the numerical groupings, pivot tables, and visualizations generated in this assignment, here are the core findings regarding passenger survival:

1. **Sex is the Strongest Predictor:** Females had a significantly higher survival rate (~74%) compared to males (~19%), directly reflecting the "women and children first" maritime protocol.
2. **Passenger Class Greatly Impacts Survival:** There is a strong negative correlation between `Pclass` and `Survived`. First-class passengers had a significantly higher survival rate than those in second or third class.
3. **Fare Correlates with Survival:** `Fare` has a strong negative correlation with `Pclass` (lower class = higher number) and a positive correlation with `Survived`. Those who paid higher fares naturally resided in better classes and had better survival odds.
4. **Volume of Lower-Class Passengers:** The Age vs. Fare scatter plot shows a dense cluster of passengers paying very low fares (3rd class). Unfortunately, the vast majority of points in this dense cluster represent passengers who did not survive.
5. **High-Fare Outliers Survived:** Almost universally, the extreme fare outliers (passengers paying fares > 100) survived the disaster, regardless of their age.
6. **Family Size Matters:** Being completely alone (`IsAlone=1`) has a slight negative correlation with survival. Small families (FamilySize 2-4) appeared to have slightly better survival odds than those traveling completely solo or in massive families.

---

## How to Run
1. Clone this repository: `git clone [Your GitHub Repo Link]`
2. Ensure you have the required libraries installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy
