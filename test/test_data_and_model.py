import pytest
from sklearn.metrics import classification_report
from data.datamanager import data_loader
import joblib
import numpy as np


@pytest.fixture
def adult_test_dataset():
    path = './data/adult_test.csv'
    x, y = data_loader(path)
    return x, y, path


def test_dataloader(adult_test_dataset):
    # Test whether there are columns containing unique values within the cleaned dataset or whether there are
    x, y, _ = adult_test_dataset

    # perform unique count for each column of the dataframe
    n_unique = x.nunique(axis=0).values

    # perform unique count for each CATEGORICAL column of the dataframe
    n_unique_categorical = np.array([x[i].nunique() for i in x.columns[x.dtypes == 'object']])

    # Perform tests on unique counts
    assert n_unique.min() > 1
    assert all(n_unique_categorical/x.shape[0] < 0.9)


def test_model_metrics(adult_test_dataset):
    x, y, data_path = adult_test_dataset
    clf = joblib.load('./model.pkl')
    predictions = clf.predict(x)
    metrics = classification_report(y, predictions, output_dict=True)
    # just adding a comment
    assert len(np.unique(predictions)) > 1
    assert metrics['>50K']['precision'] > 0.7  # fill here
    assert metrics['>50K']['recall'] > 0.1  # fill here

