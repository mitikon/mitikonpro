import numpy as np
import pandas as pd
import pytest

from leading_signal_lambda.model import LeadingLambdaClassifier


def _training_data(rows=90):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {"f1": rng.normal(0, 1, rows), "f2": rng.normal(0, 1, rows)},
    )
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, 1, -1), name="target_class")
    return X, y


def test_confidence_temperature_defaults_to_one_and_is_backward_compatible():
    X, y = _training_data()
    model = LeadingLambdaClassifier().fit(X, y)
    assert model.confidence_temperature == 1.0


def test_confidence_temperature_rejects_non_positive_values():
    with pytest.raises(ValueError, match="confidence_temperature"):
        LeadingLambdaClassifier(confidence_temperature=0.0)
    with pytest.raises(ValueError, match="confidence_temperature"):
        LeadingLambdaClassifier(confidence_temperature=-1.0)


def test_higher_temperature_softens_confidence_without_changing_the_predicted_class():
    X, y = _training_data()
    row = X.iloc[-1]
    baseline = LeadingLambdaClassifier(confidence_temperature=1.0).fit(X, y).predict_one(row)
    softened = LeadingLambdaClassifier(confidence_temperature=3.0).fit(X, y).predict_one(row)
    assert softened.predicted_class == baseline.predicted_class
    assert softened.confidence <= baseline.confidence


def test_temperature_of_one_is_numerically_identical_to_the_untouched_softmax():
    X, y = _training_data()
    row = X.iloc[-1]
    explicit_default = LeadingLambdaClassifier(confidence_temperature=1.0).fit(X, y).predict_one(row)
    implicit_default = LeadingLambdaClassifier().fit(X, y).predict_one(row)
    assert explicit_default.probabilities == implicit_default.probabilities
