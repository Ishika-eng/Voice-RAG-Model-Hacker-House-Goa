"""Per-stage timing (pipeline harness) + percentile helpers (benchmark script)."""
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings_ms[name] = (time.perf_counter() - start) * 1000

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


def percentiles(values: list[float], pcts: tuple[int, ...] = (50, 70, 100)) -> dict[str, float]:
    if not values:
        return {f"p{p}": 0.0 for p in pcts}
    ordered = sorted(values)
    result = {}
    for p in pcts:
        idx = min(max(int(round(p / 100 * len(ordered))) - 1, 0), len(ordered) - 1)
        result[f"p{p}"] = ordered[idx]
    return result