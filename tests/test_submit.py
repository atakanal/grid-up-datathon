import pandas as pd

from src.submit import validate_submission


def test_valid_submission():
    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    submission = pd.DataFrame({"id": [1, 2], "target": [0.1, 0.2]})
    assert validate_submission(submission, sample) == []


def test_wrong_columns_are_rejected():
    sample = pd.DataFrame({"id": [1, 2], "target": [0.0, 0.0]})
    submission = pd.DataFrame({"id": [1, 2], "prediction": [0.1, 0.2]})
    assert validate_submission(submission, sample)
