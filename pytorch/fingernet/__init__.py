"""
Lightweight package exports with lazy-loading.

Importing this package should be cheap. Heavy submodules (API, model,
plot) are imported only when their symbols are actually used. This
prevents slow startup when the CLI entrypoint imports
``fingernet.cli`` (which imports the package) just to show help.
"""
from typing import Any


class _LazyImport:
    """Proxy that imports the real object on first access/call."""

    def __init__(self, module: str, name: str) -> None:
        self._module = module
        self._name = name
        self._obj: Any | None = None

    def _load(self) -> Any:
        if self._obj is None:
            mod = __import__(f"{__package__}.{self._module}", fromlist=[self._name])
            self._obj = getattr(mod, self._name)
        return self._obj

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __repr__(self) -> str:  # pragma: no cover - simple helper
        return f"<LazyImport {self._module}.{self._name}>"

# Expose lazy proxies for public API
run_inference = _LazyImport("api", "run_inference")
get_fingernet_core = _LazyImport("model", "get_fingernet_core")
get_fingernet = _LazyImport("wrapper", "get_fingernet")
FingerNetWrapper = _LazyImport("model", "FingerNetWrapper")
plot_output = _LazyImport("plot", "plot_output")
plot_raw_output = _LazyImport("plot", "plot_raw_output")
get_fingernet_logger = _LazyImport("fnet_utils", "get_fingernet_logger")
FnetTimer = _LazyImport("fnet_utils", "FnetTimer")
