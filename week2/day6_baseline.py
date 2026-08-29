import pandas as pd

df = pd.read_csv("incidents.csv")

# Baseline: always predict no breach
df["prediction"] = False

accuracy = (df["prediction"] == df["sla_breached"]).mean()

print("Accuracy:", accuracy)
print("Accuracy %:", accuracy * 100)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

actual = df["sla_breached"]
predicted = df["prediction"]

print("Confusion Matrix:")
print(confusion_matrix(actual, predicted))

print("Precision:", precision_score(actual, predicted))
print("Recall:", recall_score(actual, predicted))
print("F1 Score:", f1_score(actual, predicted))