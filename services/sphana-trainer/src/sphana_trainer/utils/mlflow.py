"""Lightweight MLflow logging helper."""

from __future__ import annotations

from typing import Dict, Optional


class MlflowLogger:
    """Context manager that wraps optional MLflow logging."""

    def __init__(
        self,
        enabled: bool,
        *,
        tracking_uri: Optional[str],
        experiment: Optional[str],
        run_name: Optional[str],
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.enabled = enabled
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.tags = tags or {}
        self._mlflow = None
        self._active_run = None

    def __enter__(self) -> Optional["MlflowLogger"]:
        if not self.enabled:
            return None
        try:
            import mlflow  # type: ignore
        except ImportError as exc:  # pragma: no cover - executed only when dependency missing
            raise RuntimeError("MLflow logging requested but 'mlflow' is not installed.") from exc
        self._mlflow = mlflow
        # Always set tracking URI, default to target/mlruns if not specified
        tracking_uri = self.tracking_uri
        if not tracking_uri:
            from sphana_trainer.cli import DEFAULT_MLFLOW_URI
            tracking_uri = DEFAULT_MLFLOW_URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if self.experiment:
            mlflow.set_experiment(self.experiment)
        self._active_run = mlflow.start_run(run_name=self.run_name, tags=self.tags or None)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mlflow and self._active_run:
            self._mlflow.end_run()
        self._active_run = None

    def log_params(self, params: Dict[str, object]) -> None:
        if self._mlflow and self._active_run:
            self._mlflow.log_params(_stringify_dict(params))

    def log_metrics(self, metrics: Dict[str, float], *, step: Optional[int] = None) -> None:
        if self._mlflow and self._active_run:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if self._mlflow and self._active_run:
            self._mlflow.log_artifact(path)


def _stringify_dict(payload: Dict[str, object]) -> Dict[str, object]:
    formatted: Dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            formatted[key] = value
        else:
            formatted[key] = str(value)
    return formatted


