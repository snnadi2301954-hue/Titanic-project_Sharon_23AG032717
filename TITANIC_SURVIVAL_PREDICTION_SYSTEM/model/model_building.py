import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# --------------------------------------------------
# 1. SET SAFE PATHS
# --------------------------------------------------

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "train.csv")

# Model save path
MODEL_PATH = os.path.join(BASE_DIR, "model", "titanic_survival_model.pkl")

# --------------------------------------------------
# 2. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 3. FEATURE SELECTION
# --------------------------------------------------

features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
target = "Survived"

df = df[features + [target]]

# --------------------------------------------------
# 4. HANDLE MISSING VALUES
# --------------------------------------------------

df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# --------------------------------------------------
# 5. ENCODE CATEGORICAL VARIABLES
# --------------------------------------------------

le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Embarked"] = le_embarked.fit_transform(df["Embarked"])

# --------------------------------------------------
# 6. SPLIT DATA
# --------------------------------------------------

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 7. FEATURE SCALING
# --------------------------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 8. TRAIN MODEL
# --------------------------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------------------------
# 9. EVALUATE MODEL
# --------------------------------------------------

y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 10. SAVE MODEL
# --------------------------------------------------

with open(MODEL_PATH, "wb") as file:
    pickle.dump((model, scaler, le_sex, le_embarked), file)

print("\n✅ Model trained and saved successfully!")
print(f"📁 Model location: {MODEL_PATH}")
