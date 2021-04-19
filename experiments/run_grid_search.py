import mlflow
from itertools import product
import warnings
from model.model_training import train_random_forest_model
from model.utils import model_metrics
import argparse
import json


def grid_search_random_forest(name_experiment):

    # this function runs a grid search over the hyper-parameters specified below
    max_depth = [3, 6]
    criterion = ['gini', 'entropy']
    min_samples_leaf = [5, 10]
    n_estimators = [50, 100]

    parameters = product(max_depth, criterion, min_samples_leaf, n_estimators)
    parameters_list = list(parameters)

    print('Number of experiments:', len(parameters_list))

    # Hyperparameter search
    results = []
    best_param = None
    best_f1 = 0.0
    warnings.filterwarnings('ignore')

    for i, param in enumerate(parameters_list):
        print('Running experiment number ', i)
        with mlflow.start_run(run_name=name_experiment):
            # Tell mlflow to log the following parameters for the experiments dashboard
            mlflow.log_param('depth', param[0])
            mlflow.log_param('criterion', param[1])
            mlflow.log_param('minsamplesleaf', param[2])
            mlflow.log_param('nestimators', param[3])

            try:
                parameters = dict(n_estimators=param[3],
                                  max_depth=param[0],
                                  criterion=param[1],
                                  min_sample_leaf=param[2])

                clf = train_random_forest_model(data_path='./data/adult_training.csv',
                                                parameters=parameters)

                metrics = model_metrics(clf, data_path='./data/adult_validation.csv')

                # Tell mlflow to log the following metrics
                mlflow.log_metric("precision", metrics['>50K']['precision'])
                mlflow.log_metric("F1", metrics['>50K']['f1-score'])

                # Store this artifact for each run
                json.dump(metrics, open("metrics.json", "w"))
                mlflow.log_artifact('./metrics.json')

                # save the best experiment yet (in terms of precision)
                if metrics['>50K']['f1-score'] > best_f1:
                    best_param = parameters
                    best_f1 = metrics['>50K']['f1-score']

                results.append([param, metrics['>50K']['f1-score']])

            except ValueError:
                print('bad parameter combination:', param)
                continue

    mlflow.end_run()
    print('Best F1 was:', best_f1)
    print('Using the following parameters')
    print(best_param)
    return results, best_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="experiment_name")
    args, leftovers = parser.parse_known_args()

    results, best_param = grid_search_random_forest(args.name)
    json.dump(best_param, open("best_params.json", "w"))
