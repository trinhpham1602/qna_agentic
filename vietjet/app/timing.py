from __future__ import annotations

import inspect
import time
import typing
from datetime import datetime
from functools import wraps


def log_api_time(name: str):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            start = datetime.now()
            counter_start = time.perf_counter()
            print(f"[api {name}] start={start.isoformat()}")
            try:
                return await fn(*args, **kwargs)
            finally:
                end = datetime.now()
                duration = time.perf_counter() - counter_start
                print(
                    f"[api {name}] end={end.isoformat()} duration={duration:.3f}s"
                )

        try:
            hints = typing.get_type_hints(fn)
            sig = inspect.signature(fn)
            params = [
                p.replace(annotation=hints.get(p.name, p.annotation))
                for p in sig.parameters.values()
            ]
            wrapper.__signature__ = sig.replace(
                parameters=params,
                return_annotation=hints.get("return", sig.return_annotation),
            )
        except Exception:
            pass

        return wrapper

    return decorator
