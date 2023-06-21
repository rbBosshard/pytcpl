import pandas as pd
import os

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
from imblearn.over_sampling import SMOTE


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

aeid = 117

# Define the file paths
assay_file_path = os.path.join(ROOT_DIR, 'export', 'out', f'{aeid}.csv')
fps_file_path = os.path.join(ROOT_DIR, 'input', 'df_sirius_fps.csv')

# Read the CSV files
assay_df = pd.read_csv(assay_file_path)
fps_df = pd.read_csv(fps_file_path)

dtxsid_assay = assay_df["dtxsid"].values.tolist()
dtxsid_fps = fps_df["dtxsid"].values.tolist()


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


print(len(dtxsid_assay), len(dtxsid_fps))
print(len(intersection(dtxsid_assay, dtxsid_fps)))

df = pd.merge(assay_df, fps_df, on="dtxsid").reset_index(drop=True)

activ = df[df["hitc"]]
not_activ = df[df["hitc"] == False]

print(activ.shape, not_activ.shape)

#####################################################################

# Split the data into features (X) and labels (y)
X = df.iloc[:, 2:]  # Select all columns starting from the third column as features
y = df['hitc']

# Perform SMOTE oversampling
oversampler = SMOTE(random_state=42)
X_oversampled, y_oversampled = oversampler.fit_resample(X, y)


# Split the oversampled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_oversampled, y_oversampled, test_size=0.2, random_state=42)


# Create an XGBoost classifier
model = xgb.XGBClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

# Print cross-validation scores
print("Cross-validation scores:", cv_scores)
print("Mean F1 score:", cv_scores.mean())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Classification Report:")
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
f1_score = f1_score(y_test, y_pred)
print("F1 Score:", f1_score)
precision = precision_score(y_test, y_pred)
print("Precision:", precision)
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

