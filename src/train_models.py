from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.5, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=60,
            class_weight='balanced', n_jobs=-1
        ),
        "SVM": SVC(kernel='linear', C=1.2, class_weight='balanced'),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5
        )
    }
