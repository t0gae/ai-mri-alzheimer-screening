import numpy as np
from sklearn.metrics import classification_report, roc_auc_score

y_true = np.load("results/y_test_true.npy")
y_proba = np.load("results/y_test_proba.npy")

y_pred = (y_proba > 0.5).astype(int)

print("AUC:", roc_auc_score(y_true, y_proba))
print(classification_report(y_true, y_pred, target_names=["Normal", "Dementia"]))
