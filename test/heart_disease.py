import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
url = "CSC14003_Decision_Tree/data/processed.cleveland.data"
columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Missing values are marked as '?'
df = pd.read_csv(url, names=columns, na_values="?")

# Drop rows with missing values
df = df.dropna()

# Convert target to binary: 0 - no disease, 1 - disease
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Prediction
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test,
    y_test,
    display_labels=["No Disease", "Disease"],
    cmap=plt.cm.Blues,
    normalize=None,
)
plt.title("Confusion Matrix")
plt.show()

# Visualization
plt.figure(figsize=(20, 10))
plot_tree(
    clf, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True
)
plt.title("Decision Tree of heart disease (Dataset 1)")
plt.show()

# Plot feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind="barh")
plt.title("Feature Importance")
plt.show()

# Analyze max depth
train_accuracies = []
test_accuracies = []
depth_range = range(1, 15)

for depth in depth_range:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_accuracies.append(model.score(X_train, y_train))
    test_accuracies.append(model.score(X_test, y_test))

plt.figure()
plt.plot(depth_range, train_accuracies, label="Train Accuracy", marker="o")
plt.plot(depth_range, test_accuracies, label="Test Accuracy", marker="o")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Effect of max_depth on Accuracy")
plt.show()
