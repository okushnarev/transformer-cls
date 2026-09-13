import functools


def get_callable_name(func) -> str:
    if isinstance(func, functools.partial):
        return get_callable_name(func.func)

    if hasattr(func, "__qualname__"):
        return func.__qualname__

    if hasattr(func, "__name__"):
        return func.__name__

    return func.__class__.__name__
