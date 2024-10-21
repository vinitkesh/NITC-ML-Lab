import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC # Main inport
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
columns = [f"Feature_{i+1}" for i in range(57)] + ['Label']
data = pd.read_csv(url, names=columns)

X = data.iloc[:, :-1]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
accuracy_linear = accuracy_score(y_test, y_pred_linear)
print(f"Accuracy (Linear SVM): {accuracy_linear * 100:.2f}%")

rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy (RBF SVM): {accuracy_rbf * 100:.2f}%")

poly_svm = SVC(kernel='poly', degree=3)
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)
accuracy_poly = accuracy_score(y_test, y_pred_poly)
print(f"Accuracy (Polynomial SVM, degree 3): {accuracy_poly * 100:.2f}%")

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

plot_confusion_matrix(y_test, y_pred_linear, 'Linear SVM')
plot_confusion_matrix(y_test, y_pred_rbf, 'RBF SVM')
plot_confusion_matrix(y_test, y_pred_poly, 'Polynomial SVM (degree 3)')

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

linear_svm.fit(X_train_pca, y_train)
rbf_svm.fit(X_train_pca, y_train)
poly_svm.fit(X_train_pca, y_train)

def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    plt.title(f"Decision Boundary: {title}")
    plt.show()

plot_decision_boundary(X_train_pca, y_train, linear_svm, "Linear SVM")
plot_decision_boundary(X_train_pca, y_train, rbf_svm, "RBF SVM")
plot_decision_boundary(X_train_pca, y_train, poly_svm, "Polynomial SVM (degree 3)")

print(f"Accuracy (Linear SVM): {accuracy_linear * 100:.2f}%")
print(f"Accuracy (RBF SVM): {accuracy_rbf * 100:.2f}%")
print(f"Accuracy (Polynomial SVM, degree 3): {accuracy_poly * 100:.2f}%")
