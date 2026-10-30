"""IO-Recon-TCN package."""

from .config import IOConfig

# TCNForecastModel/Detector(=torch)는 지연 로드 → 앱 추론 경로(onnx)에서 torch 미포함.
__all__ = ["IOConfig", "TCNForecastDetector", "TCNForecastModel"]


def __getattr__(name):
    if name in ("TCNForecastModel", "TCNForecastDetector"):
        from . import model_tcn_forecast as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
