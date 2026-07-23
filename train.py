# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

def train_on_features(feat_csv='data/features_table.csv'):
    df = pd.read_csv(feat_csv)
    X = df.drop(columns=['label','subj','trial'])
    y = df['label']
    groups = df['subj']

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=0))
    ])

    gkf = GroupKFold(n_splits=min(5, df['subj'].nunique()))
    scores = cross_val_score(pipe, X, y, cv=gkf.split(X,y,groups), scoring='roc_auc')
    print("Group (subject) CV AUROC: mean {:.3f} ± {:.3f}".format(scores.mean(), scores.std()))
    # fit final model on all data
    pipe.fit(X,y)
    dump(pipe, 'models/rf_load_detector.joblib')
    print("Model saved to models/rf_load_detector.joblib")

if __name__ == '__main__':
    train_on_features()
