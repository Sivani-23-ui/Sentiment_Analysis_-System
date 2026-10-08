"""Algorithm 4 (extra): Random Forest.   Run: python train_random_forest.py [--data ...]"""
from sklearn.ensemble import RandomForestClassifier
from src.common import get_args, run_experiment

if __name__ == "__main__":
    run_experiment("random_forest", "Random Forest",
                   RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=42),
                   {"clf__n_estimators": [200, 300], "clf__max_depth": [None, 40]}, get_args())
