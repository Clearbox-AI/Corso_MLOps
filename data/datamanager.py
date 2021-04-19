import pandas as pd


def data_loader(path: str):
    data_csv_path = path
    dataset = pd.read_csv(data_csv_path)

    target_column = 'income'
    y = dataset[target_column]
    x = dataset.drop(target_column, axis=1)
    return x, y

