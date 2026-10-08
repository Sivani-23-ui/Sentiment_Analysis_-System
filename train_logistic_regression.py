"""Algorithm 2: Logistic Regression.   Run: python train_logistic_regression.py [--data ...]"""
from sklearn.linear_model import LogisticRegression
from src.common import get_args, run_experiment

if __name__ == "__main__":
    run_experiment("logistic_regression", "Logistic Regression",
                   LogisticRegression(max_iter=2000, class_weight="balanced"),
                   {"clf__C": [0.5, 1, 5, 10]}, get_args())
