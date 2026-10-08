"""Algorithm 3: Linear SVM (calibrated so it can output probabilities).   Run: python train_svm.py [--data ...]"""
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from src.common import get_args, run_experiment

if __name__ == "__main__":
    run_experiment("linear_svm", "Linear SVM",
                   CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
                   {"clf__estimator__C": [0.1, 0.5, 1, 2]}, get_args())
