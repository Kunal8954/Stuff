# 📊 Train-Test Split in Machine Learning

Train-Test Split is one of the most fundamental concepts in Machine Learning.

Before training a model, we need to answer an important question:

> **How do we know whether our model can make correct predictions on data it has never seen before?**

The answer is:

```text
                    DATASET
                       │
              Train-Test Split
                       │
              ┌────────┴────────┐
              ↓                 ↓
            TRAIN              TEST
              ↓                 ↓
        Model learns       Model is tested
              ↓                 ↓
        Learned pattern    Unseen data
              │                 │
              └────────┬────────┘
                       ↓
                  Evaluation
```

---

# 🧠 Why Do We Need Train-Test Split?

Suppose we have a dataset containing student study hours and exam marks.

```text
Hours Studied    Marks
----------------------
2                40
3                50
4                60
5                70
6                80
```

We want a model to learn:

```text
Hours Studied → Exam Marks
```

If we train the model using **all the data**:

```python
model.fit(X, y)
```

and then test it on the **same data**:

```python
model.predict(X)
```

the model has already seen those examples.

So a high score does not necessarily mean the model can handle new data.

---

# 🎓 Real-Life Analogy

Imagine you are preparing for an exam.

You practice:

```text
Question 1
Question 2
Question 3
Question 4
```

Then the teacher gives you:

```text
Question 1
Question 2
Question 3
Question 4
```

again.

You score:

```text
100%
```

Does that prove you understand the subject?

Not necessarily.

Now imagine the teacher gives:

```text
Question 5
Question 6
Question 7
Question 8
```

which you have never seen.

Your performance here tells us much more about your actual understanding.

Machine Learning works similarly.

```text
Training Data → Practice / Learning

Testing Data → New Questions / Evaluation
```

---

# 🔥 Core Idea

The dataset is divided into two parts:

```text
                    COMPLETE DATA
                         │
                Train-Test Split
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
          TRAIN                     TEST
             │                       │
             ↓                       ↓
       Model learns            Model evaluated
             │                       │
             ↓                       ↓
      Seen during training     Unseen during training
```

The model learns from the **training set**.

The model is evaluated using the **test set**.

---

# 📦 X and y

Before splitting, we usually separate:

```text
X = Features
y = Target
```

Example:

```text
Hours   Attendance   Marks
------  -----------  -----
2       70           40
4       80           60
6       90           80
```

Here:

```python
X = [
    [2, 70],
    [4, 80],
    [6, 90]
]

y = [
    40,
    60,
    80
]
```

So:

```text
X → Information used to make prediction

y → Answer we want the model to learn
```

---

# ✂️ Using `train_test_split`

Scikit-Learn provides:

```python
from sklearn.model_selection import train_test_split
```

Basic usage:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 🔍 What Does This Return?

It creates four outputs:

```text
X_train
X_test
y_train
y_test
```

Their meaning:

| Variable  | Meaning           |
| --------- | ----------------- |
| `X_train` | Training features |
| `y_train` | Training targets  |
| `X_test`  | Testing features  |
| `y_test`  | Testing targets   |

---

# 📊 Example

Suppose:

```text
1000 rows
```

and:

```python
test_size=0.2
```

Then approximately:

```text
800 rows → Training
200 rows → Testing
```

```text
             1000 ROWS
                 │
          ┌──────┴──────┐
          ↓             ↓
       800 rows       200 rows
        TRAIN           TEST
```

---

# ⚙️ `test_size`

This controls how much data goes into the test set.

### 20% Test

```python
test_size=0.2
```

```text
80% → Train
20% → Test
```

### 30% Test

```python
test_size=0.3
```

```text
70% → Train
30% → Test
```

### 10% Test

```python
test_size=0.1
```

```text
90% → Train
10% → Test
```

---

# 🎲 What is `random_state`?

The split normally contains randomness.

For example:

```python
train_test_split(X, y)
```

may produce a different split each time.

To make the result reproducible:

```python
random_state=42
```

Example:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Running the same code again with the same `random_state` gives the same split.

---

# ❓ Why 42?

There is nothing special about `42`.

You could use:

```python
random_state=10
```

or:

```python
random_state=100
```

or:

```python
random_state=7
```

The important thing is reproducibility.

```text
Same random_state
       ↓
Same split
       ↓
Reproducible experiment
```

---

# 🧠 Training the Model

Once the data is split:

```python
model.fit(X_train, y_train)
```

This means:

> Train the model using only the training data.

Conceptually:

```text
X_train + y_train
        ↓
      MODEL
        ↓
Learns patterns
```

---

# 🔮 Making Predictions

After training:

```python
predictions = model.predict(X_test)
```

Now:

```text
X_test
   ↓
Trained Model
   ↓
Predictions
```

The model has not used `X_test` during training.

---

# 📏 Evaluating the Model

We compare:

```text
Predictions
     ↓
     vs
Actual y_test
```

Example:

```python
from sklearn.metrics import mean_absolute_error

error = mean_absolute_error(
    y_test,
    predictions
)
```

For classification:

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    predictions
)
```

---

# 🔥 Complete Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

error = mean_absolute_error(
    y_test,
    predictions
)

print(error)
```

The complete flow:

```text
Dataset
   ↓
X / y
   ↓
Train-Test Split
   ↓
┌───────────────┐
│               │
↓               ↓
Train           Test
↓               ↓
fit()        predict()
↓               ↓
Model        Predictions
                 ↓
              y_test
                 ↓
             Evaluation
```

---

# ⚠️ What Happens If We Don't Split?

Suppose:

```python
model.fit(X, y)

predictions = model.predict(X)
```

You are testing on the same data used for training.

You might get:

```text
Training Accuracy = 99%
```

But that doesn't tell you how well the model performs on unseen data.

This can lead to a false sense of performance.

---

# 🧠 Generalization

One of the biggest goals of Machine Learning is:

> **Generalization**

Generalization means:

> A model should perform well not only on training examples but also on new, unseen examples from the same problem setting.

```text
Training Data
     ↓
   MODEL
     ↓
Unseen Data
     ↓
Good Predictions
```

A model that generalizes well has learned useful patterns rather than simply memorizing the training examples.

---

# 🚨 Overfitting

Suppose:

```text
Training Accuracy = 99%
Test Accuracy     = 60%
```

This large gap can indicate **overfitting**.

Conceptually:

```text
Training Data
      ↓
Model learns too specifically
      ↓
Training → Excellent
Testing  → Poor
```

The model may have learned noise or overly specific patterns in the training data.

---

# 🐌 Underfitting

Suppose:

```text
Training Accuracy = 60%
Test Accuracy     = 58%
```

The model is performing poorly on both.

This can indicate **underfitting**.

```text
Model
 ↓
Fails to learn enough useful structure
 ↓
Training → Poor
Testing  → Poor
```

---

# 📊 Three Possible Situations

### Good Generalization

```text
Train = 92%
Test  = 90%
```

The model performs similarly on both datasets.

---

### Overfitting

```text
Train = 99%
Test  = 65%
```

Large gap.

---

### Underfitting

```text
Train = 60%
Test  = 58%
```

Both are poor.

These numbers are illustrative; what counts as "good" depends on the dataset and metric.

---

# 🧪 Train / Validation / Test

For more serious ML development, we often distinguish three roles:

```text
                     DATASET
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
          TRAIN                  TEST
             ↓
        Model Training
             ↓
        Model Selection
             ↓
         Validation
             ↓
          Final Test
```

### Training Set

Used to learn model parameters.

```text
TRAIN → LEARN
```

### Validation Set

Used during development to compare models and tune hyperparameters.

```text
VALIDATION → CHOOSE / TUNE
```

### Test Set

Reserved for final evaluation.

```text
TEST → FINAL CHECK
```

---

# 🔄 Cross-Validation

Instead of depending on a single validation split, we can use cross-validation.

Example:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)
```

Conceptually:

```text
Dataset
   ↓
Fold 1
Fold 2
Fold 3
Fold 4
Fold 5
```

Each fold gets a turn as the validation portion.

This gives a more robust estimate of model performance during development.

---

# ⚠️ Data Leakage

A very important concept is **data leakage**.

Data leakage happens when information that should not be available during training or evaluation leaks into the model-building process.

Example:

```text
Test information
       ↓
Preprocessing / Training
       ↓
Model
```

This can produce overly optimistic evaluation results.

---

# 🧹 Correct Preprocessing Order

Suppose you need scaling.

Incorrect idea:

```text
Complete Dataset
      ↓
Scaler
      ↓
Train-Test Split
```

The scaler has seen information from the test set.

Better:

```text
Dataset
   ↓
Train-Test Split
   ↓
Train ──→ Fit Scaler
             ↓
          Transform
             
Test ─────────→ Transform
```

The preprocessing parameters should be learned from the training data only.

Scikit-Learn `Pipeline` is very useful for enforcing this pattern.

---

# 🏗️ Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(
    X_train,
    y_train
)

predictions = pipeline.predict(
    X_test
)
```

Conceptually:

```text
X_train
   ↓
Scaler learns from training data
   ↓
Model training
```

Then:

```text
X_test
   ↓
Already-fitted Scaler
   ↓
Model
   ↓
Prediction
```

---

# 🎯 Stratified Splitting

For classification problems, class proportions can matter.

Suppose:

```text
1000 samples

Class 0 → 900
Class 1 → 100
```

We often want train and test sets to maintain approximately similar class proportions.

Use:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Conceptually:

```text
Original Distribution
       ↓
    Split
       ↓
Train ≈ same class distribution
Test  ≈ same class distribution
```

---

# 📦 When Should You Use Train-Test Split?

Almost every supervised ML project needs some form of held-out evaluation.

Examples:

```text
House Price Prediction
Spam Detection
Fraud Detection
Customer Churn
Disease Classification
Student Performance
Sales Prediction
```

---

# 🧪 Practice Example

## Student Marks Prediction

Dataset:

```text
Hours   Marks
2       40
3       50
4       60
5       70
6       80
7       90
```

Features:

```python
X = df[["Hours"]]
```

Target:

```python
y = df["Marks"]
```

Split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Train:

```python
model.fit(
    X_train,
    y_train
)
```

Predict:

```python
predictions = model.predict(
    X_test
)
```

Evaluate:

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(
    y_test,
    predictions
)
```

---

# 🧠 Mental Model

Whenever you see:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y
)
```

think:

```text
X = Questions / Features
y = Answers / Target

X_train + y_train
        ↓
      LEARN

X_test
        ↓
    PREDICT

Predictions
        ↓
Compare with
        ↓
y_test
        ↓
   EVALUATE
```

---

# 🔥 The Most Important Rule

Remember these three lines:

```python
model.fit(X_train, y_train)
```

**Learn from training data.**

```python
predictions = model.predict(X_test)
```

**Predict on unseen test features.**

```python
metric(y_test, predictions)
```

**Compare predictions with actual test targets.**

---

# 🚀 Final Mental Model

```text
                       DATA
                        │
                        ↓
                  Split Dataset
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
          TRAIN                    TEST
             │                     │
             ↓                     │
        model.fit()                │
             │                     │
             ↓                     │
        Trained Model              │
             │                     │
             └──────────┐          │
                        ↓          ↓
                     predict(X_test)
                           │
                           ↓
                      Predictions
                           │
                           ↓
                     Compare with
                        y_test
                           │
                           ↓
                       METRIC
                           │
                           ↓
                    Model Evaluation
```

## 🏁 What You Should Remember

```text
Train  = Learn
Test   = Evaluate
X      = Features
y      = Target
fit()  = Learn parameters
predict() = Make predictions
```

And the fundamental reason for the split:

> **We separate training and testing data so that we can evaluate how well the trained model performs on data that was not used to fit it.**
