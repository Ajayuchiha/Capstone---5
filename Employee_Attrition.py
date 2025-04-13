import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Load data
emp_att = pd.read_csv(r"C:\Users\AJAY\Downloads\PROJECT\Capstone_5\Employee-Attrition.csv")
emp_att.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"], inplace=True)

# Encode categorical variables
cat_cols = emp_att.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    emp_att[col] = le.fit_transform(emp_att[col])

# Fill missing values
emp_att.fillna(emp_att.median(numeric_only=True), inplace=True)

# Split features and target
X = emp_att.drop("Attrition", axis=1)
y = emp_att["Attrition"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Logistic Regression (Balanced)
lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000)
lr_balanced.fit(X_train, y_train)
y_pred_lr = lr_balanced.predict(X_test)
y_proba_lr = lr_balanced.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))
print("AUC-ROC:", roc_auc_score(y_test, y_proba_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

import pickle

filename = 'Attrition_Model.pkl'
pickle.dump(lr_balanced, open(filename, 'wb'))

