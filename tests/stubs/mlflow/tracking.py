import mlflow
class MlflowClient:
    def set_model_version_tag(self, n, v, k, val): pass
    def set_registered_model_alias(self, n, a, v): mlflow.LOG.append(("alias", n, a, v))
    def update_registered_model(self, n, description=None): pass
