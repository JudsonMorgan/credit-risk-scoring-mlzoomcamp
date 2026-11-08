#!/usr/bin/env python
# coding: utf-8

import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import pickle

# Display settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:.3f}".format)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

RANDOM_STATE = 42

df = pd.read_excel('data/default of credit card clients.xls',
                  header=1)

df.head().T

df.shape

df.info()

df.describe()

# standardize column names
df.columns = (
    df.columns.str.lower().str.replace(' ', '_')
)
df.columns

df = df.rename(
    columns={'default_payment_next_month': 'default'}
)

df.columns.to_list()

# Checking for missing values
df.isnull().sum()

#duplicated records
df.duplicated().sum()

#check for unique column IDs
df.id.unique()

#Examine the target variables
df.default.value_counts(normalize=True)

sns.countplot(data=df, x='default')
plt.title('Distribution of Credit Card Defaults')
plt.xlabel('Default (0 = No, 1 = Yes)')
plt.ylabel('Number of Customers')
plt.show()


# Approximately 22% of customers defaulted on their payments. Since the majority did not default, accuracy alone would be misleading. We'll prioritize ROC AUC for model selection and also examine precision, recall, and F1-score.

df.describe().T

numerical_features = [
    'limit_bal',
    'age',
    'bill_amt1',
    'pay_amt1'
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for col, ax in zip(numerical_features, axes.flatten()):
    sns.histplot(df[col], bins=40, ax=ax)
    ax.set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

df.groupby('default')['limit_bal'].agg(
    ['mean', 'median', 'min', 'max']
)

df.groupby('pay_0')['default'].agg(
    ['count', 'mean']
)

# Select numerical features
numerical = [
    col for col in df.columns
    if col not in ['id', 'default', 'sex', 'education', 'marriage']
]

# Calculate correlations with target
correlations = (
    df[numerical + ['default']]
    .corr()['default']
    .drop('default')
    .sort_values(ascending=False)
)

correlations

plt.figure(figsize=(10, 8))

correlations.sort_values().plot(kind='barh')

plt.title('Feature Correlation with Credit Default')
plt.xlabel('Pearson Correlation')
plt.ylabel('Features')

plt.tight_layout()
plt.show()

#split dataset into train, test, and validation
df_full_train, df_test = train_test_split(
    df,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df['default']
)

df_train, df_val = train_test_split(
    df_full_train,
    test_size=0.25,
    random_state=RANDOM_STATE,
    stratify=df_full_train['default']
)

print("Training:", len(df_train))
print("Validation:", len(df_val))
print("Testing:", len(df_test))

y_train = df_train['default'].values
y_val = df_val['default'].values
y_test = df_test['default'].values

# Remove target and customer ID from model features
df_train = df_train.drop(columns=['default', 'id']).copy()
df_val = df_val.drop(columns=['default', 'id']).copy()
df_test = df_test.drop(columns=['default', 'id']).copy()

#verify class imbalance

print("Training default rate:", y_train.mean())
print("Validation default rate:", y_val.mean())
print("Testing default rate:", y_test.mean())

#prepare categorical and numerical features

categorical = ['sex', 'education', 'marriage']

numerical = [
    col for col in df_train.columns
    if col not in categorical
]

#Convert categorical to strings
for dataset in [df_train, df_val, df_test]:
    dataset[categorical] = dataset[categorical].astype(str)

print("Categorical:", categorical)
print("Numerical:", numerical)

train_dicts = df_train.to_dict(orient='records')
val_dicts = df_val.to_dict(orient='records')

dv = DictVectorizer(sparse=False)

X_train = dv.fit_transform(train_dicts)
X_val = dv.transform(val_dicts)

print("X_train:", X_train.shape)
print("X_val:", X_val.shape)

#Scale the numerical features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train Logistic Regression model
model = LogisticRegression(
    solver='liblinear',
    max_iter=1000,
    random_state=RANDOM_STATE
)

model.fit(X_train_scaled, y_train)


# ## Evaluate the base model

y_pred = model.predict_proba(X_val_scaled)[:, 1]

auc = roc_auc_score(y_val, y_pred)

print(f"Validation ROC AUC: {auc:.3f}")


# ## Train the baseline Decision Tree

dt = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)

dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)

print(f"Decision Tree ROC AUC: {auc:.3f}")

scores = []

for depth in [3, 4, 5, 6, 10, 15, 20]:
    for leaf in [1, 5, 10, 15, 20, 50, 100]:

        dt = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=RANDOM_STATE
        )

        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((depth, leaf, auc))

df_scores = pd.DataFrame(
    scores,
    columns=['max_depth', 'min_samples_leaf', 'auc']
)

df_scores.sort_values(
    by='auc',
    ascending=False
).head(10)


# Awesome! Our Tuned decision tree model on `max_depth` 10 and `min_samples_leaf` 100 has outperformed the Logistic Regression model.

# ### Train the baseline Random Forest Classifier model

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_val)[:, 1]

auc = roc_auc_score(y_val, y_pred)

print(f"Random Forest ROC AUC: {auc:.3f}")

#Tune Random forest
scores = []

for n in [50, 100, 150]:
    for depth in [5, 10, 15]:
        for leaf in [1, 5, 10, 20]:

            rf = RandomForestClassifier(
                n_estimators=n,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)

            y_pred = rf.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred)

            scores.append((n, depth, leaf, auc))

df_rf_scores = pd.DataFrame(
    scores,
    columns=[
        'n_estimators',
        'max_depth',
        'min_samples_leaf',
        'auc'
    ]
)

df_rf_scores.sort_values(
    by='auc',
    ascending=False
).head(10)


# **The results suggest that several Random Forest configurations perform similarly. The top three use `max_depth`=15 and `min_samples_leaf`=10, with 50, 100, or 150 trees. Using 100 trees is a reasonable compromise between computational cost and predictive performance.**

best_params = {
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_leaf': 10,
    'random_state': 42,
    'n_jobs': -1
}

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):

    df_train_fold = df_full_train.iloc[train_idx].copy()
    df_val_fold = df_full_train.iloc[val_idx].copy()

    y_train_fold = df_train_fold['default'].values
    y_val_fold = df_val_fold['default'].values

    df_train_fold = df_train_fold.drop(columns=['default', 'id'])
    df_val_fold = df_val_fold.drop(columns=['default', 'id'])

    categorical = ['sex', 'education', 'marriage']

    df_train_fold[categorical] = df_train_fold[categorical].astype(str)
    df_val_fold[categorical] = df_val_fold[categorical].astype(str)

    train_dicts = df_train_fold.to_dict(orient='records')
    val_dicts = df_val_fold.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)

    X_train_fold = dv.fit_transform(train_dicts)
    X_val_fold = dv.transform(val_dicts)

    model = RandomForestClassifier(**best_params)

    model.fit(X_train_fold, y_train_fold)

    y_pred = model.predict_proba(X_val_fold)[:, 1]

    auc = roc_auc_score(y_val_fold, y_pred)

    scores.append(auc)

print("Fold AUC scores:", np.round(scores, 3))
print(f"Mean ROC AUC: {np.mean(scores):.3f}")
print(f"Standard deviation: {np.std(scores):.3f}")


# The result above indicates that our Random Forest model is performing consistently across all five folds and the low standard deviation suggest a stable performance across these folds.

# ## Train the final model

# Prepare full training data
df_full_train_final = df_full_train.copy()

y_full_train = df_full_train_final['default'].values

df_full_train_final = df_full_train_final.drop(
    columns=['default', 'id']
)

categorical = ['sex', 'education', 'marriage']

df_full_train_final[categorical] = (
    df_full_train_final[categorical].astype(str)
)

# Prepare test features
df_test_final = df_test.copy()

df_test_final[categorical] = (
    df_test_final[categorical].astype(str)
)

# DictVectorizer
dv = DictVectorizer(sparse=False)

train_dicts = df_full_train_final.to_dict(orient='records')
test_dicts = df_test_final.to_dict(orient='records')

X_full_train = dv.fit_transform(train_dicts)
X_test = dv.transform(test_dicts)

# Train final Random Forest
final_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_full_train, y_full_train)


# ## Evaluate on the test data

# Predict probabilities
y_test_pred = final_model.predict_proba(X_test)[:, 1]

# Evaluate ROC AUC
test_auc = roc_auc_score(y_test, y_test_pred)

print(f"Final Test ROC AUC: {test_auc:.3f}")

#evaluate classification performance at the default threshold of 0.5
y_test_binary = (y_test_pred >= 0.5).astype(int)

print(classification_report(y_test, y_test_binary))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_test_binary))


# ## Save the model using `pickle`

output_file = 'models/credit_risk_model.bin'

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, final_model), f_out)

print(f'Model saved to {output_file}')


# ## Verify the save

with open(output_file, 'rb') as f_in:
    dv_loaded, model_loaded = pickle.load(f_in)

# Verify predictions
X_test_loaded = dv_loaded.transform(test_dicts)

y_pred_loaded = model_loaded.predict_proba(X_test_loaded)[:, 1]

print('Loaded model ROC AUC:', roc_auc_score(y_test, y_pred_loaded))
