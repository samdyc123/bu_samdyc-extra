from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


def feature_engineering(df):
    df['trans_hour'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S').dt.hour
    df['trans_day'] = pd.to_datetime(df['trans_date']).dt.day
    df['trans_month'] = pd.to_datetime(df['trans_date']).dt.month
    df['trans_year'] = pd.to_datetime(df['trans_date']).dt.year
    df['age'] = pd.to_datetime('today').year - pd.to_datetime(df['dob']).dt.year

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    df['distance'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    return df

train_data = feature_engineering(train_data)
test_data = feature_engineering(test_data)

features = ['amt', 'category', 'trans_hour', 'trans_day', 'trans_month', 'distance', 'age', 'city_pop', 'gender']
target = 'is_fraud'

X = train_data[features]
y = train_data[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

categorical_features = ['category', 'gender']
numerical_features = ['amt', 'trans_hour', 'trans_day', 'trans_month', 'distance', 'age', 'city_pop']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

pipeline.fit(X_train, y_train)

val_preds = pipeline.predict(X_val)
val_proba = pipeline.predict_proba(X_val)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score
print("Validation ROC AUC:", roc_auc_score(y_val, val_proba))
print("Validation Classification Report:\n", classification_report(y_val, val_preds))

test_preds = pipeline.predict_proba(test_data[features])[:, 1]

submission = test_data[['id']].copy()
submission['is_fraud'] = (test_preds > 0.5).astype(int)
submission.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully!")
