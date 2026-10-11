"""Minimal in-memory stand-in for `mlflow`, used ONLY to smoke-test control flow when the
real package is not installed (e.g. an offline sandbox). Never used if real mlflow exists."""
import contextlib, itertools, types

_runs = itertools.count(1)
_versions = {}
LOG = []
data = types.SimpleNamespace(from_pandas=lambda df, source=None, name=None: ("dataset", name))


def set_tracking_uri(u): LOG.append(("uri", u))
def get_experiment_by_name(n): return None
def create_experiment(n, artifact_location=None): LOG.append(("exp", n))
def set_experiment(n): pass
def set_tags(t): pass
def log_params(p): LOG.append(("params", dict(p)))
def log_metrics(m): LOG.append(("metrics", dict(m)))
def log_artifact(p): pass
def log_dict(d, f): pass
def log_input(ds, context=None): pass


@contextlib.contextmanager
def start_run(run_name=None):
    rid = f"run{next(_runs)}"
    LOG.append(("run", run_name, rid))
    yield types.SimpleNamespace(info=types.SimpleNamespace(run_id=rid))


def register_model(uri, name):
    v = _versions.setdefault(name, [])
    v.append(uri)
    LOG.append(("register", name, len(v)))
    return types.SimpleNamespace(version=str(len(v)))
