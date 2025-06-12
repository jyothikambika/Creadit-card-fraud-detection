# -------------------- IMPORTS --------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# -------------------- LOAD & EXPLORE DATA --------------------
df = pd.read_csv("creditcard.csv")
print("Dataset shape:", df.shape)
print(df["Class"].value_counts())

# -------------------- PREPROCESSING --------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Selection with RFE
log_reg = LogisticRegression(max_iter=1000)
rfe = RFE(log_reg, n_features_to_select=20)
X_selected = rfe.fit_transform(X_scaled, y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# -------------------- MODEL DEFINITIONS --------------------
models = {
    "Logistic Regression": LogisticRegression(penalty="l2", solver="liblinear"),
    "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# -------------------- TRAIN & EVALUATE --------------------
def evaluate_model(model, X_test, y_test, title="Model"):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    print(f"\n🧪 Evaluation for: {title}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # ROC-AUC Curve
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {title}")
        plt.legend()
        plt.grid()
        plt.show()

# -------------------- TRAINING --------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(model, X_test, y_test, title=name)
