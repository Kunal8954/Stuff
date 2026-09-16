# 🤖 Scikit-Learn — Complete Machine Learning Fundamentals

A practical, step-by-step guide to **classical Machine Learning with Scikit-Learn**, starting from the basics and progressing toward complete ML pipelines.

The goal of this section is to understand the complete machine learning workflow:

```text
Data
 ↓
Understand Data
 ↓
Preprocess
 ↓
Split Data
 ↓
Train Model
 ↓
Predict
 ↓
Evaluate
 ↓
Improve
 ↓
Save Model
 ↓
Deploy
```

---

# 🎯 Goals

By completing this section, you should be able to:

* Load datasets
* Understand features and targets
* Perform EDA
* Clean data
* Handle missing values
* Encode categorical variables
* Scale numerical features
* Split datasets
* Train ML models
* Make predictions
* Evaluate models
* Compare models
* Tune hyperparameters
* Build pipelines
* Handle imbalanced datasets
* Perform cross-validation
* Save trained models
* Build complete ML projects

---

# 📂 Folder Structure

```text
03_sklearn/
│
├── 01_basics/
│   ├── sklearn_intro.py
│   ├── dataset.py
│   ├── train_test_split.py
│   ├── fit.py
│   └── predict.py
│
├── 02_data_preprocessing/
│   ├── missing_values.py
│   ├── encoding.py
│   ├── scaling.py
│   ├── normalization.py
│   └── preprocessing.py
│
├── 03_regression/
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   ├── ridge.py
│   ├── lasso.py
│   └── elasticnet.py
│
├── 04_classification/
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── svm.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   ├── naive_bayes.py
│   └── gradient_boosting.py
│
├── 05_clustering/
│   ├── kmeans.py
│   ├── hierarchical.py
│   └── dbscan.py
│
├── 06_dimensionality_reduction/
│   ├── pca.py
│   └── feature_selection.py
│
├── 07_evaluation/
│   ├── regression_metrics.py
│   ├── classification_metrics.py
│   ├── confusion_matrix.py
│   └── cross_validation.py
│
├── 08_model_selection/
│   ├── grid_search.py
│   ├── random_search.py
│   └── cross_validation.py
│
├── 09_pipelines/
│   ├── preprocessing_pipeline.py
│   └── complete_pipeline.py
│
├── 10_ensemble/
│   ├── voting.py
│   ├── bagging.py
│   ├── boosting.py
│   └── stacking.py
│
├── 11_imbalanced_data/
│
├── 12_model_persistence/
│   ├── joblib.py
│   └── pickle.py
│
├── projects/
│
└── README.md
```

---

# 1️⃣ What is Scikit-Learn?

Scikit-Learn is a Python library for classical Machine Learning.

It provides implementations for:

```text
Regression
Classification
Clustering
Dimensionality Reduction
Preprocessing
Model Selection
Evaluation
Pipelines
```

Typical workflow:

```python
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# 2️⃣ The Core ML Concept

Every supervised ML problem generally has:

```text
X = Features
y = Target
```

Example:

```text
House Dataset

X:
├── Area
├── Bedrooms
├── Location
└── Age

y:
└── Price
```

So:

```python
X = df.drop("price", axis=1)
y = df["price"]
```

---

# 3️⃣ Train/Test Split

Never evaluate a model only on the data it trained on.

Use:

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Conceptually:

```text
Dataset
   │
   ├──────────────→ Training Data
   │
   └──────────────→ Testing Data
```

Common split:

```text
80% Train
20% Test
```

---

# 4️⃣ The `fit()` Method

`fit()` means:

> Train the model using the training data.

```python
model.fit(X_train, y_train)
```

Conceptually:

```text
X_train + y_train
       ↓
     Model
       ↓
Learned Parameters
```

---

# 5️⃣ The `predict()` Method

After training:

```python
predictions = model.predict(X_test)
```

Conceptually:

```text
X_test
   ↓
Trained Model
   ↓
Predictions
```

---

# 6️⃣ Basic Regression

Regression predicts a continuous value.

Examples:

```text
House Price
Salary
Temperature
Sales
Revenue
```

---

## Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Concept:

```text
Features
   ↓
Linear Regression
   ↓
Continuous Value
```

---

# 7️⃣ Regression Models

Study:

### Linear Models

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet
* Polynomial Regression

### Tree-Based Models

* Decision Tree Regressor
* Random Forest Regressor
* Gradient Boosting Regressor
* HistGradientBoostingRegressor

### Other Models

* SVR
* KNN Regressor

---

# 8️⃣ Regression Evaluation

Important metrics:

### MAE

Mean Absolute Error.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
```

### MSE

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
```

### RMSE

```python
import numpy as np

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)
```

### R²

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, predictions)
```

---

# 9️⃣ Classification

Classification predicts categories.

Examples:

```text
Spam / Not Spam
Fraud / Not Fraud
Cat / Dog
Pass / Fail
Disease Classes
```

Concept:

```text
Features
   ↓
Classifier
   ↓
Class
```

---

# 🔟 Classification Models

Study:

* Logistic Regression
* KNN
* Decision Tree
* Random Forest
* SVM
* Naive Bayes
* Gradient Boosting
* HistGradientBoosting
* AdaBoost

---

# 1️⃣1️⃣ Logistic Regression

Despite its name, Logistic Regression is commonly used for classification.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# 1️⃣2️⃣ Classification Evaluation

### Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)
```

### Precision

Measures how many predicted positives were actually positive.

### Recall

Measures how many actual positives were detected.

### F1 Score

Balances precision and recall.

```python
from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        predictions
    )
)
```

---

# 1️⃣3️⃣ Confusion Matrix

A confusion matrix helps understand classification errors.

```text
                 Predicted
              Positive Negative

Actual Positive    TP       FN
Actual Negative    FP       TN
```

Use:

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    y_test,
    predictions
)
```

---

# 1️⃣4️⃣ Data Preprocessing

Real-world data is rarely ready for a model.

Typical workflow:

```text
Raw Data
   ↓
Missing Values
   ↓
Categorical Data
   ↓
Numerical Data
   ↓
Scaling
   ↓
Model
```

---

# 1️⃣5️⃣ Missing Values

Use:

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="mean"
)
```

Common strategies:

```text
mean
median
most_frequent
constant
```

---

# 1️⃣6️⃣ Encoding

Machine learning models generally need numerical representations.

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)
```

Example:

```text
City

Mumbai
Delhi
Pune
```

becomes:

```text
Mumbai  Delhi  Pune

1       0      0
0       1      0
0       0      1
```

---

# 1️⃣7️⃣ Feature Scaling

Common scalers:

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

### MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

Scaling is especially important for algorithms affected by feature magnitude, such as:

* KNN
* SVM
* Logistic Regression
* Linear models
* Neural networks

Tree-based models generally do not require feature scaling.

---

# 1️⃣8️⃣ Feature Selection

Not every feature is useful.

Study:

* SelectKBest
* SelectPercentile
* RFE
* RFECV
* Feature importance
* Mutual information

Goal:

```text
100 Features
     ↓
Useful Features
     ↓
20 Features
     ↓
Model
```

---

# 1️⃣9️⃣ Pipelines

Pipelines connect preprocessing and modeling.

Instead of:

```text
Imputer
 ↓
Encoder
 ↓
Scaler
 ↓
Model
```

manually, create:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
```

Then:

```python
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

---

# 2️⃣0️⃣ ColumnTransformer

Real datasets often contain both numerical and categorical columns.

Example:

```text
Age        → Numerical
Salary     → Numerical
City       → Categorical
Education  → Categorical
```

Use:

```python
from sklearn.compose import ColumnTransformer
```

Concept:

```text
                 Dataset
                    ↓
          ColumnTransformer
             /          \
            ↓            ↓
      Numerical       Categorical
          ↓                ↓
       Scaler            Encoder
             \          /
              ↓        ↓
                Model
```

This is an extremely important production ML concept.

---

# 2️⃣1️⃣ Cross Validation

Instead of relying on one train/test split:

```text
Dataset
 ↓
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Use:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
```

This gives a better estimate of model performance.

---

# 2️⃣2️⃣ Hyperparameter Tuning

Models have hyperparameters.

Example:

```text
Random Forest

n_estimators
max_depth
min_samples_split
```

We can search for good combinations.

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    model,
    param_grid,
    cv=5
)

grid.fit(X_train, y_train)
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
```

---

# 2️⃣3️⃣ Ensemble Learning

Combine multiple models.

Study:

```text
Bagging
Boosting
Voting
Stacking
```

Important models:

* Random Forest
* Gradient Boosting
* AdaBoost
* HistGradientBoosting
* VotingClassifier
* VotingRegressor
* StackingClassifier
* StackingRegressor

Concept:

```text
Model 1 ──┐
Model 2 ──┼──→ Ensemble → Final Prediction
Model 3 ──┘
```

---

# 2️⃣4️⃣ Clustering

Unsupervised learning does not require a target variable.

```text
X
 ↓
Clustering
 ↓
Groups
```

Study:

* K-Means
* DBSCAN
* Agglomerative Clustering

Example:

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=3,
    random_state=42
)

model.fit(X)

labels = model.labels_
```

---

# 2️⃣5️⃣ Dimensionality Reduction

Reduce the number of features.

Study:

* PCA
* TruncatedSVD

Example:

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=2
)

X_reduced = pca.fit_transform(X)
```

Concept:

```text
100 Features
     ↓
     PCA
     ↓
2 Features
```

Useful for:

* Visualization
* Noise reduction
* Feature compression
* High-dimensional datasets

---

# 2️⃣6️⃣ Imbalanced Data

Example:

```text
Fraud Detection

Normal → 99%
Fraud  → 1%
```

Accuracy can become misleading.

Study:

* Precision
* Recall
* F1
* ROC-AUC
* PR-AUC
* Class weights
* Stratified splitting

---

# 2️⃣7️⃣ Probability Predictions

Many classifiers provide:

```python
model.predict_proba(X_test)
```

Example:

```text
Class 0 → 0.12
Class 1 → 0.88
```

This is useful for:

* Threshold tuning
* Risk scoring
* Ranking
* Decision systems

---

# 2️⃣8️⃣ ROC-AUC

ROC-AUC evaluates ranking ability across classification thresholds.

```python
from sklearn.metrics import roc_auc_score

score = roc_auc_score(
    y_test,
    probabilities
)
```

For highly imbalanced problems, also study:

```text
Precision-Recall Curve
Average Precision
```

---

# 2️⃣9️⃣ Model Persistence

After training, save the model.

Using Joblib:

```python
import joblib

joblib.dump(
    model,
    "model.pkl"
)
```

Load it later:

```python
model = joblib.load(
    "model.pkl"
)
```

Concept:

```text
Training
   ↓
Model
   ↓
Save
   ↓
model.pkl
   ↓
Load
   ↓
Prediction
```

---

# 3️⃣0️⃣ Complete ML Pipeline

Eventually you should be comfortable building:

```text
                     DATASET
                        ↓
                    EDA / Analysis
                        ↓
                    Data Cleaning
                        ↓
                  Train/Test Split
                        ↓
                Feature Engineering
                        ↓
                ColumnTransformer
                 /              \
                ↓                ↓
          Numerical          Categorical
             ↓                    ↓
          Scaling              Encoding
                 \              /
                  ↓            ↓
                     Model
                       ↓
                    Training
                       ↓
                   Prediction
                       ↓
                   Evaluation
                       ↓
              Hyperparameter Tuning
                       ↓
                 Cross Validation
                       ↓
                  Model Selection
                       ↓
                  Save Model
                       ↓
                     API
                       ↓
                   Deployment
```

---

# 🧪 Practice Projects

## Project 1 — House Price Prediction

Learn:

```text
Regression
EDA
Preprocessing
Linear Regression
Random Forest
Evaluation
```

---

## Project 2 — Student Performance Prediction

Predict:

```text
Pass / Fail
```

Learn:

```text
Classification
Encoding
Scaling
Logistic Regression
Random Forest
Metrics
```

---

## Project 3 — Customer Churn

Predict whether a customer will leave.

Learn:

```text
Classification
Imbalanced Data
Precision
Recall
F1
ROC-AUC
Feature Engineering
```

---

## Project 4 — Customer Segmentation

Use:

```text
K-Means
PCA
Visualization
```

---

## Project 5 — End-to-End ML System

Build:

```text
CSV
 ↓
EDA
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
Multiple Models
 ↓
Cross Validation
 ↓
Hyperparameter Tuning
 ↓
Best Model
 ↓
Save Model
 ↓
FastAPI
 ↓
Docker
```

---

# 📊 Model Cheat Sheet

| Problem                  | Models              |
| ------------------------ | ------------------- |
| Regression               | Linear Regression   |
| Regression               | Ridge               |
| Regression               | Lasso               |
| Regression               | Random Forest       |
| Regression               | Gradient Boosting   |
| Classification           | Logistic Regression |
| Classification           | KNN                 |
| Classification           | SVM                 |
| Classification           | Decision Tree       |
| Classification           | Random Forest       |
| Classification           | Naive Bayes         |
| Classification           | Gradient Boosting   |
| Clustering               | K-Means             |
| Clustering               | DBSCAN              |
| Clustering               | Agglomerative       |
| Dimensionality Reduction | PCA                 |

---

# 🧠 Essential Scikit-Learn API

These methods/classes should become familiar:

```python
.fit()
.predict()
.predict_proba()
.transform()
.fit_transform()
.fit_predict()
```

Important modules:

```python
sklearn.model_selection
sklearn.preprocessing
sklearn.impute
sklearn.compose
sklearn.pipeline
sklearn.metrics
sklearn.linear_model
sklearn.tree
sklearn.ensemble
sklearn.neighbors
sklearn.svm
sklearn.naive_bayes
sklearn.cluster
sklearn.decomposition
```

---

# 🏁 Completion Checklist

## Fundamentals

* [ ] Understand X and y
* [ ] Train/test split
* [ ] `fit()`
* [ ] `predict()`
* [ ] `predict_proba()`

## Preprocessing

* [ ] Missing values
* [ ] Encoding
* [ ] Scaling
* [ ] Normalization
* [ ] Feature selection
* [ ] Feature engineering

## Regression

* [ ] Linear Regression
* [ ] Ridge
* [ ] Lasso
* [ ] ElasticNet
* [ ] Polynomial Regression
* [ ] Random Forest
* [ ] Gradient Boosting

## Classification

* [ ] Logistic Regression
* [ ] KNN
* [ ] SVM
* [ ] Decision Tree
* [ ] Random Forest
* [ ] Naive Bayes
* [ ] Gradient Boosting

## Unsupervised Learning

* [ ] K-Means
* [ ] DBSCAN
* [ ] Agglomerative Clustering
* [ ] PCA

## Evaluation

* [ ] MAE
* [ ] MSE
* [ ] RMSE
* [ ] R²
* [ ] Accuracy
* [ ] Precision
* [ ] Recall
* [ ] F1
* [ ] Confusion Matrix
* [ ] ROC-AUC
* [ ] PR-AUC

## Model Improvement

* [ ] Cross Validation
* [ ] GridSearchCV
* [ ] RandomizedSearchCV
* [ ] Ensemble Learning
* [ ] Class weights
* [ ] Threshold tuning

## Production

* [ ] Pipeline
* [ ] ColumnTransformer
* [ ] Model persistence
* [ ] FastAPI
* [ ] Docker
* [ ] Model monitoring

---

# 🚀 Final Mental Model

Don't learn Scikit-Learn as a collection of models.

Learn the **ML workflow**:

```text
                MACHINE LEARNING

                    DATA
                      ↓
                     EDA
                      ↓
                PREPROCESSING
                      ↓
                 FEATURES
                      ↓
                TRAIN / TEST
                      ↓
                   MODEL
                      ↓
                    FIT
                      ↓
                  PREDICT
                      ↓
                 EVALUATE
                      ↓
                  IMPROVE
                      ↓
                   TUNE
                      ↓
                  VALIDATE
                      ↓
                    SAVE
                      ↓
                  DEPLOY
```

Once this workflow becomes natural, learning individual Scikit-Learn algorithms becomes much easier.

---

# 🎯 Final Goal

By the end of this section, you should be able to receive an unfamiliar dataset and independently answer:

> **What is the problem?**

> **What are my features and target?**

> **How should I preprocess the data?**

> **Which models should I try?**

> **How should I evaluate them?**

> **How can I improve the model?**

> **How do I save and deploy it?**

That is the foundation required before moving deeper into **Deep Learning, PyTorch, Transformers, LLMs and MLOps**.
