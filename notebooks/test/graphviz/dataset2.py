import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import graphviz

# Load dataset
df = sns.load_dataset("penguins")

# Drop rows with missing values
df = df.dropna()

# Features & Labels
X = df.drop(columns=["species"])
y = df["species"]

# Encode categorical variables
X = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision tree with max 7 nodes
clf = DecisionTreeClassifier(random_state=42, max_leaf_nodes=7)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Số lượng node trong cây:", clf.tree_.node_count)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Export tree as DOT file
dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=clf.classes_,
    filled=True,
    rounded=True,
    special_characters=True,
)

# Render tree using Graphviz
graph = graphviz.Source(dot_data)
graph.render("penguins_tree", format="pdf", cleanup=False)  # sẽ tạo penguins_tree.pdf
graph.render("penguins_tree", format="png", cleanup=False)  # sẽ tạo penguins_tree.png
graph.view("penguins_tree")  # mở ảnh cây

# Nếu muốn xem trực tiếp trong notebook: graph.view()
