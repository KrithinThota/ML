import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

data = pd.read_csv("Data\parkinsson_data.csv")

X = data.drop(columns=['name', 'status'])
y = data['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)

conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

precision_train = precision_score(y_train, y_train_pred)
precision_test = precision_score(y_test, y_test_pred)

recall_train = recall_score(y_train, y_train_pred)
recall_test = recall_score(y_test, y_test_pred)

f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)

print("Confusion Matrix (Training Data):\n", conf_matrix_train)
print("Precision (Training Data):", precision_train)
print("Recall (Training Data):", recall_train)
print("F1 Score (Training Data):", f1_train)

print("\nConfusion Matrix (Test Data):\n", conf_matrix_test)
print("Precision (Test Data):", precision_test)
print("Recall (Test Data):", recall_test)
print("F1 Score (Test Data):", f1_test)

# Observations and Inferences
print("\nObservations and Inferences:")
print("- Precision measures the accuracy of positive predictions, while recall measures the ability to find all positive samples.")
print("- F1-score is the harmonic mean of precision and recall, providing a balanced measure of model performance.")
print("- If the model's performance metrics are similar between the training and test datasets, it suggests the model is fitting well.")
print("- Significant differences in performance metrics between training and test datasets may indicate overfitting or underfitting.")
