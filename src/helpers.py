import functools
import hashlib

from sympy.printing.pytorch import torch


def get_callable_name(func) -> str:
    if isinstance(func, functools.partial):
        return get_callable_name(func.func)

    if hasattr(func, "__qualname__"):
        return func.__qualname__

    if hasattr(func, "__name__"):
        return func.__name__

    return func.__class__.__name__


def get_model_hash(state_dict) -> str:
    """
    Generates a hash based on the model's state dictionary (architecture and weights)
    """
    hasher = hashlib.sha256()

    # Sort to maintain order
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]

        # Update hash with the key name
        hasher.update(key.encode('utf-8'))

        # Update hash with tensor values
        chunk = tensor.cpu().to(torch.float32).numpy().tobytes()
        hasher.update(chunk)

    return hasher.hexdigest()
