"""In-memory sliding-window rate limiter, per client IP.

Deliberately simple (a dict + deque, no Redis) -- correct for a single
FastAPI worker process. A multi-worker/multi-instance production deploy
would need a shared store (Redis) instead, since each worker would
otherwise track its own independent counters.
"""
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.max_requests:
            retry_after = max(1, int(self.window_seconds - (now - hits[0])))
            raise HTTPException(
                status_code=429,
                detail="You're asking faster than I can verify answers -- please wait a moment and try again.",
                headers={"Retry-After": str(retry_after)},
            )
        hits.append(now)


ask_limiter = RateLimiter(max_requests=20, window_seconds=300)


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    ask_limiter.check(client_ip)
