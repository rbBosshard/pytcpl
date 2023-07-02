
import os
import yaml
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, VarianceThreshold
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

# Define the root directory
# Define the root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config', 'config_ml.yaml')


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        if config["ignore_warnings"]:
            import warnings
            warnings.filterwarnings("ignore")

    return config  

 
config = load_config(CONFIG_PATH)

print(f"ML pipeline for assay ID: {config['aeid']}\n")

# Prepare assay df
assay_file = f"{config['aeid']}.csv"
assay_file_path = os.path.join(ROOT_DIR, "export", "out", assay_file)
assay_df = pd.read_csv(assay_file_path)
print(f"Assay dataframe: {assay_df.shape[0]} chemical/hitcall datapoints")
# print(f"{assay_df['hitc'].value_counts()}\n")

# Prepare fingerprint df
fingerprint_file= "ToxCast_CSIfps.csv"
fps_file_path = os.path.join(ROOT_DIR, 'input', fingerprint_file)
# Skip the first 3 columns (relativeIndex, absoluteIndex, index) and transpose the dataframe
fps_df = pd.read_csv(fps_file_path)
fps_df = fps_df.T
# Transpose the DataFrame using numpy transpose
data = np.transpose(fps_df[1:, 3:].values.astype(int))
index = fps_df.iloc[0, 3:]
columns = fps_df.iloc[1:, 3]
transposed_df = pd.DataFrame(data=data, index=index, columns=columns)
fps_df = fps_df.iloc[:, 3:].T

data = fps_df.iloc[1:].values.astype(int)
index = fps_df.index[1:]
columns = fps_df.iloc[0]

fps_df = pd.DataFrame(data=data, index=index, columns=columns).reset_index()
fps_df = fps_df.rename(columns={"index": "dtxsid"})
assert fps_df.shape[0] == fps_df['dtxsid'].nunique()
print(f"Fingerprint dataframe ({fingerprint_file}): {fps_df.shape[0]} chemicals, {fps_df.iloc[:, 1:].shape[1]} binary features")

# Get intersection and merge the assay and fingerprint dataframes
df = pd.merge(assay_df, fps_df, on="dtxsid").reset_index(drop=True)
assert df.shape[0] == df['dtxsid'].nunique()
print(f"Merged dataframe for this ML pipeline: {df.shape[0]} datapoints (chemical fingerprint/hitcall)")
