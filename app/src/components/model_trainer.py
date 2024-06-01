"""
Functions to train and evaluate models.
"""
from sklearn.model_selection import cross_val_score, train_test_split

from src.utils import Kernel


def cross_validate(model, train_features, train_target, kernel: Kernel):
    cross_val_settings = kernel.config['cross_val_settings']

    cv_scores = cross_val_score(
        model, train_features, train_target, **cross_val_settings)

    print("Cross-Validation Scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())
    print("Standard Deviation of CV Scores:", cv_scores.std())


def test_model(
        model,
        train_features,
        train_target,
        test_features,
        test_target,
        kernel: Kernel):
    model.fit(train_features, train_target)

    # Assess the model on the holdout set
    test_score = model.score(test_features, test_target)

    print("Test Score:", test_score)

    return test_score


def train_and_evaluate(model, data, kernel: Kernel):
    config = kernel.config
    features = data.drop(config['target'], axis=1)
    target = data[config['target']]
    train_features, test_features, train_target, test_target = (
        train_test_split(features, target, **config['split_settings']))

    cross_validate(model, train_features, train_target, kernel)
    result = test_model(model, train_features, train_target,
                        test_features, test_target, kernel)

    return result



