"""Train the final credit-card default risk model from the ML Zoomcamp notebook."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent / 'data' / 'default of credit card clients.xls'
MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'credit_risk_model.bin'
CATEGORICAL = ['sex', 'education', 'marriage']


def prepare_features(data):
    features = data.drop(columns=['default', 'id']).copy()
    features[CATEGORICAL] = features[CATEGORICAL].astype(str)
    return features.to_dict(orient='records')


def main():
    # Dataset loading and cleaning copied from the notebook.
    df = pd.read_excel(DATA_PATH, header=1, engine='xlrd')
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.rename(columns={'default_payment_next_month': 'default'})

    # Same held-out test split as the notebook; the validation split is
    # unnecessary here because model selection has already been completed.
    df_full_train, df_test = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['default']
    )
    y_full_train = df_full_train['default'].values
    y_test = df_test['default'].values

    train_dicts = prepare_features(df_full_train)
    test_dicts = prepare_features(df_test)

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(train_dicts)
    X_test = dv.transform(test_dicts)

    # Best Random Forest configuration selected in the notebook.
    final_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X_full_train, y_full_train)

    y_test_pred = final_model.predict_proba(X_test)[:, 1]
    print(f'Final Test ROC AUC: {roc_auc_score(y_test, y_test_pred):.3f}')
    y_test_binary = (y_test_pred >= 0.5).astype(int)
    print(classification_report(y_test, y_test_binary))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_test_binary))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open('wb') as f_out:
        pickle.dump((dv, final_model), f_out)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
