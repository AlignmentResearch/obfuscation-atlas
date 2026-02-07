import contextlib
import time
from collections import defaultdict


class Profiler:
    def __init__(self, prefix: str = "profiling/Time taken"):
        self.metrics: dict[str, float] = defaultdict(float)
        self.prefix = prefix

    @contextlib.contextmanager
    def profile(self, name: str):
        start = time.perf_counter()
        yield
        self.metrics[f"{self.prefix}: {name}"] += time.perf_counter() - start

    def get_and_reset(self) -> dict[str, float]:
        metrics = dict(self.metrics)
        self.metrics.clear()
        return metrics
