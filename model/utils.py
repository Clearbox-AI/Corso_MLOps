from skl2onnx.common.data_types import FloatTensorType, Int64TensorType, StringTensorType
from sklearn.metrics import classification_report
from mlflow.models.signature import infer_signature
import mlflow
from data.datamanager import data_loader
from skl2onnx import convert_sklearn


def convert_dataframe_schema(df, drop=None):
    inputs = []
    for k, v in zip(df.columns, df.dtypes):
        if drop is not None and k in drop:
            continue
        if v == 'int64':
            t = Int64TensorType([None, 1])
        elif v == 'float64':
            t = FloatTensorType([None, 1])
        else:
            t = StringTensorType([None, 1])
        inputs.append((k, t))
    return inputs


def model_metrics(clf, data_path):

    x_test, y_test = data_loader(data_path)
    metrics = classification_report(y_test, clf.predict(x_test), output_dict=True)

    return metrics


def convert_sklearn_mlflow(clf, x_sample):

    signature = infer_signature(x_sample, clf.predict(x_sample))
    input_example = {}
    for i in x_sample.columns:
        input_example[i] = x_sample[i][0]

    mlflow.sklearn.save_model(clf, "best_model", signature=signature, input_example=input_example)

    return


def convert_sklearn_onnx(clf, x_sample):
    inputs = convert_dataframe_schema(x_sample)
    onnx_model = convert_sklearn(clf, 'model_pipeline', inputs, target_opset=12)

    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    return onnx_model


def onnx_input(x):
    inputs = {c: x[c].values for c in x.columns}
    for k in inputs:
        inputs[k] = inputs[k].reshape((inputs[k].shape[0], 1))
    return inputs
