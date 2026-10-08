"""Algorithm 1: Multinomial Naive Bayes.   Run: python train_naive_bayes.py [--data data/yourfile.csv]"""
from sklearn.naive_bayes import MultinomialNB
from src.common import get_args, run_experiment

if __name__ == "__main__":
    run_experiment("naive_bayes", "Naive Bayes", MultinomialNB(),
                   {"clf__alpha": [0.05, 0.1, 0.5, 1.0]}, get_args())
