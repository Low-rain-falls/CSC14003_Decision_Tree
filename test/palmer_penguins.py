import matplotlib.pyplot as plt
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
url = "CSC14003_Decision_Tree/data/penguins.csv"
df = pd.read_csv(url)
# df = sns.load_dataset("penguins")

# Drop rows with missing values
df = df.dropna()

# Convert species to binary: 0 - Adelie, 1 - Gentoo, 2 - Chinstrap
df["species"] = df["species"].apply(
    lambda x: 0 if x == "Adelie" else (1 if x == "Gentoo" else 2)
)

# Feature selection
X = df.drop("species", axis=1)
y = df["species"]

# Convert categorical features
X = pd.get_dummies(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
disp = ConfusionMatrixDisplay.from_estimator(
    clf,
    X_test,
    y_test,
    display_labels=["Adelie", "Gentoo", "Chinstrap"],
    cmap=plt.cm.Blues,
    normalize=None,
)
plt.title("Confusion Matrix")
plt.show()

# Plot Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["Adelie", "Gentoo", "Chinstrap"],
    filled=True,
)
plt.title("Decision Tree of Palmer Penguins (Dataset 2)")
plt.show()

# Plot feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 6))
feat_importances.nlargest(10).plot(kind="barh")
plt.title("Feature Importance")
plt.show()

# Analyze max_depth
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
