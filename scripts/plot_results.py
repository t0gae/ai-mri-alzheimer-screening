import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

y_true = np.load(RESULTS_DIR / "y_test_true.npy")
y_proba = np.load(RESULTS_DIR / "y_test_proba.npy")
y_pred = (y_proba > 0.5).astype(int)

fpr, tpr, _ = roc_curve(y_true, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve on Test Set")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
plt.close()

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Dementia"],
    yticklabels=["Normal", "Dementia"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Test Set")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
plt.close()

print("Figures saved to figures/")
