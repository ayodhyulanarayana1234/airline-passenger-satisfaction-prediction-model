# ============================================================
# AIRLINE PASSENGER SATISFACTION PROJECT
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# ------------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ------------------------------------------------------------

dataset_path = Path(__file__).with_name("Airline Passenger Satisfaction.csv")
df = pd.read_csv(dataset_path)

print("First five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all"))

# ------------------------------------------------------------
# STEP 2: DROP UNNAMED COLUMN AND ID
# ------------------------------------------------------------

df.drop(columns=["Unnamed: 0", "id"], inplace=True, errors="ignore")
print("\nShape after dropping Unnamed column and id:", df.shape)

# ------------------------------------------------------------
# STEP 3: CLEAN TEXT AND REMOVE DUPLICATES
# ------------------------------------------------------------

for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

print("\nDuplicate rows:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Shape after removing duplicates:", df.shape)

# ------------------------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# Numerical columns: mean
# Categorical columns: mode
# ------------------------------------------------------------

print("\nMissing values before handling:")
print(df.isnull().sum())

numerical_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(include="object").columns

for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mean())

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing values after handling:")
print(df.isnull().sum())

# ------------------------------------------------------------
# STEP 5: CORRELATION, COVARIANCE AND OUTLIERS
# ------------------------------------------------------------

print("\nCorrelation matrix:")
print(df.corr(numeric_only=True).round(2))

print("\nCovariance matrix:")
print(df.cov(numeric_only=True).round(2))

print("\nSkewness:")
print(df.select_dtypes(include="number").skew().round(2))

print("\nKurtosis:")
print(df.select_dtypes(include="number").kurt().round(2))

for column in ["Age", "Flight Distance",
               "Departure Delay in Minutes", "Arrival Delay in Minutes"]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[
        (df[column] < Q1 - 1.5 * IQR) |
        (df[column] > Q3 + 1.5 * IQR)
    ]
    print(f"Outliers in {column}: {len(outliers)}")

# ------------------------------------------------------------
# STEP 6: EXPLORATORY DATA ANALYSIS AND VISUALIZATION
# ------------------------------------------------------------

# Plot 1: Overall passenger satisfaction

plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="satisfaction", palette=["skyblue", "orange"])
plt.title("Distribution of Passenger Satisfaction")
plt.xlabel("Satisfaction")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

# Plot 2: Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", bins=30, kde=True)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Plot 3: Satisfaction by class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Class", hue="satisfaction")
plt.title("Satisfaction by Travel Class")
plt.xlabel("Class")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Plot 4: Satisfaction by type of travel
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Type of Travel", hue="satisfaction")
plt.title("Satisfaction by Type of Travel")
plt.xlabel("Type of Travel")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Plot 5: Satisfaction by customer type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Customer Type", hue="satisfaction")
plt.title("Satisfaction by Customer Type")
plt.xlabel("Customer Type")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.show()

# Plot 6: Service ratings by satisfaction
service_columns = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness"
]

service_means = df.groupby("satisfaction")[service_columns].mean().T
service_means.plot(kind="bar", figsize=(14, 6))
plt.title("Average Service Ratings by Satisfaction")
plt.xlabel("Service Feature")
plt.ylabel("Average Rating")
plt.xticks(rotation=50, ha="right")
plt.tight_layout()
plt.show()

# Correlation Heatmap
# Correlation heatmap for selected features
selected_columns = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Food and drink",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes"
]

correlation_matrix = df[selected_columns].corr()

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 8}
)

plt.title("Correlation Between Service Ratings and Flight Delays")
plt.tight_layout()
plt.show()

# Plot 8: Arrival delay by satisfaction
# Box Plot: Arrival Delay by Satisfaction
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="satisfaction",
    y="Arrival Delay in Minutes",
    hue="satisfaction",
    palette=["#FF9999", "#66B3FF"],
    legend=False
)

plt.yscale("symlog", linthresh=1)
plt.title("Arrival Delay by Satisfaction")
plt.xlabel("Satisfaction")
plt.ylabel("Arrival Delay in Minutes (Symmetric Log Scale)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 7: FIVE PROJECT OBJECTIVES
# ------------------------------------------------------------

# Objective 1: Overall satisfaction
print("\nOBJECTIVE 1: Overall Passenger Satisfaction")
print(df["satisfaction"].value_counts())
print("\nSatisfaction percentage:")
print((df["satisfaction"].value_counts(normalize=True) * 100).round(2))

# Objective 2: Satisfaction by class
print("\nOBJECTIVE 2: Satisfaction by Class")
print(pd.crosstab(df["Class"], df["satisfaction"]))
print("\nPercentage within each class:")
print((pd.crosstab(df["Class"], df["satisfaction"], normalize="index") * 100).round(2))

# Objective 3: Satisfaction by type of travel
print("\nOBJECTIVE 3: Satisfaction by Type of Travel")
print(pd.crosstab(df["Type of Travel"], df["satisfaction"]))
print("\nPercentage within each travel type:")
print((pd.crosstab(df["Type of Travel"], df["satisfaction"], normalize="index") * 100).round(2))

# Objective 4: Service ratings by satisfaction
print("\nOBJECTIVE 4: Average Service Ratings by Satisfaction")
print(df.groupby("satisfaction")[service_columns].mean().T.round(2))

# Objective 5: Flight delays by satisfaction
print("\nOBJECTIVE 5: Flight Delays by Satisfaction")
print(df.groupby("satisfaction")[
    ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
].mean().round(2))

# ------------------------------------------------------------
# STEP 8: FEATURE ENGINEERING
# ------------------------------------------------------------

df["Age Group"] = pd.cut(
    df["Age"],
    bins=[0, 24, 39, 59, 100],
    labels=["18-24", "25-39", "40-59", "60+"]
)

# ------------------------------------------------------------
# STEP 9: PREPARE DATA FOR CLASSIFICATION
# ------------------------------------------------------------

# satisfaction is the target we want to predict.
X = df.drop("satisfaction", axis=1)
y = df["satisfaction"]

# Convert text columns into 0/1 columns using pandas.
# This is a simple alternative to OneHotEncoder.
X = pd.get_dummies(X, drop_first=True)

# Convert the target labels into numbers.
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nTarget label mapping:")
print(dict(zip(label_encoder.classes_,
               label_encoder.transform(label_encoder.classes_))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler is used for Logistic Regression and SVM.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# STEP 10: MODEL 1 - LOGISTIC REGRESSION
# ------------------------------------------------------------

model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train_scaled, y_train)
y_pred1 = model1.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, y_pred1))
print("Precision:", precision_score(y_test, y_pred1, average="weighted"))
print("Recall:", recall_score(y_test, y_pred1, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred1, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("Classification Report:\n",
      classification_report(y_test, y_pred1,
                            target_names=label_encoder.classes_))

# Logistic Regression - Confusion Matrix Plot
plt.figure(figsize=(6, 4))
sns.heatmap(
    confusion_matrix(y_test, y_pred1),
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted Satisfaction")
plt.ylabel("Actual Satisfaction")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 11: MODEL 2 - DECISION TREE
# ------------------------------------------------------------

model2 = DecisionTreeClassifier(max_depth=15, random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

print("\nDECISION TREE")
print("Accuracy:", accuracy_score(y_test, y_pred2))
print("Precision:", precision_score(y_test, y_pred2, average="weighted"))
print("Recall:", recall_score(y_test, y_pred2, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred2, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("Classification Report:\n",
      classification_report(y_test, y_pred2,
                            target_names=label_encoder.classes_))

plt.figure(figsize=(18, 9))
plot_tree(
    model2,
    feature_names=X.columns,
    class_names=label_encoder.classes_,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=7
)
plt.title("Decision Tree (First Three Levels)")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 12: MODEL 3 - RANDOM FOREST
# ------------------------------------------------------------

model3 = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, y_pred3))
print("Precision:", precision_score(y_test, y_pred3, average="weighted"))
print("Recall:", recall_score(y_test, y_pred3, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred3, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("Classification Report:\n",
      classification_report(y_test, y_pred3,
                            target_names=label_encoder.classes_))

# ------------------------------------------------------------
# STEP 13: MODEL 4 - SUPPORT VECTOR MACHINE (SVM)
# ------------------------------------------------------------

# SVM can take a long time on a very large dataset.
# Train it on a random sample of the training rows to keep runtime practical.
svm_sample_size = min(15000, len(X_train))
X_svm_train = X_train_scaled[:svm_sample_size]
y_svm_train = y_train[:svm_sample_size]

model4 = SVC(kernel="linear")
model4.fit(X_svm_train, y_svm_train)
y_pred4 = model4.predict(X_test_scaled)

print("\nSUPPORT VECTOR MACHINE")
print("SVM training sample size:", svm_sample_size)
print("Accuracy:", accuracy_score(y_test, y_pred4))
print("Precision:", precision_score(y_test, y_pred4, average="weighted"))
print("Recall:", recall_score(y_test, y_pred4, average="weighted"))
print("F1 Score:", f1_score(y_test, y_pred4, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred4))
print("Classification Report:\n",
      classification_report(y_test, y_pred4,
                            target_names=label_encoder.classes_))

# ------------------------------------------------------------
# STEP 14: COMPARE MODELS AND SELECT THE BEST MODEL
# ------------------------------------------------------------

models = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine"
]

accuracies = [
    accuracy_score(y_test, y_pred1),
    accuracy_score(y_test, y_pred2),
    accuracy_score(y_test, y_pred3),
    accuracy_score(y_test, y_pred4)
]

f1_scores = [
    f1_score(y_test, y_pred1, average="weighted"),
    f1_score(y_test, y_pred2, average="weighted"),
    f1_score(y_test, y_pred3, average="weighted"),
    f1_score(y_test, y_pred4, average="weighted")
]

model_comparison = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies,
    "Weighted F1 Score": f1_scores
})

model_comparison = model_comparison.sort_values(
    by="Weighted F1 Score", ascending=False
)

print("\nMODEL COMPARISON")
print(model_comparison.to_string(index=False))

best_model_name = model_comparison.iloc[0]["Model"]
best_accuracy = model_comparison.iloc[0]["Accuracy"]

print("\nBEST-FIT MODEL:", best_model_name)
print("Best model accuracy:", round(best_accuracy, 4))

plt.figure(figsize=(9, 5))

sns.barplot(
    data=model_comparison,
    x="Model",
    y="Accuracy",
    hue="Model",
    palette=["skyblue", "orange", "lightgreen", "salmon"],
    legend=False
)

plt.title("Accuracy Comparison of Classification Models")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# STEP 15: FINAL CONCLUSIONS AND RECOMMENDATIONS
# ------------------------------------------------------------

print("\nFINAL CONCLUSIONS AND RECOMMENDATIONS")

print("\n1. Overall satisfaction percentage:")
print((df["satisfaction"].value_counts(normalize=True) * 100).round(2))

print("\n2. Satisfaction percentage by class:")
print((pd.crosstab(df["Class"], df["satisfaction"], normalize="index") * 100).round(2))

print("\n3. Satisfaction percentage by travel type:")
print((pd.crosstab(df["Type of Travel"], df["satisfaction"], normalize="index") * 100).round(2))

print("\n4. Average service ratings by satisfaction:")
print(df.groupby("satisfaction")[service_columns].mean().T.round(2))

print("\n5. Average delays by satisfaction:")
print(df.groupby("satisfaction")[
    ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
].mean().round(2))

print("\n6. Best model:", best_model_name)
print("   Accuracy:", round(best_accuracy, 4))

print("\nRecommendations:")
print("- Improve service areas where dissatisfied passengers give lower ratings.")
print("- Compare satisfaction between travel classes and types of travel.")
print("- Monitor departure and arrival delays.")
print("- Validate the selected model on new data before using it in practice.")

