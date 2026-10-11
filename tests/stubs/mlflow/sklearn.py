import types, mlflow
def log_model(pipe, name=None, artifact_path=None, signature=None, input_example=None):
    return types.SimpleNamespace(model_uri=f"runs:/stub/{name or artifact_path}")
def load_model(uri): raise RuntimeError("stub")
