from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, random_state=0)

models = {
    'Logistic Regression': LogisticRegression(max_iter=200),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

N_FOLDS = 5 # Número de folds para la validación cruzada

for name, model in models.items():
    y_pred = cross_val_predict(model, X_train, y_train, cv=N_FOLDS)
    
    print(f"{name} Metrics with Cross-Validation:")
    print(f"CV Accuracy: {cross_val_score(model, X_train, y_train, cv=N_FOLDS, scoring='accuracy').mean():.2f}")
    print(f"Precision: {precision_score(y_train, y_pred, average='weighted'):.2f}")
    print(f"Recall: {recall_score(y_train, y_pred, average='weighted'):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_train, y_pred))
    print("\n")
