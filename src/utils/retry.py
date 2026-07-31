"""
Retry utilities.
"""
import asyncio
from typing import Callable, Any, Optional


async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    **kwargs,
) -> Any:
    """
    Retry async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Function arguments
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff base
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (exponential_base**attempt), max_delay)
                await asyncio.sleep(delay)

    if last_exception:
        raise last_exception


def retry_sync(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> Any:
    """
    Retry sync function with exponential backoff.

    Args:
        func: Function to retry
        *args: Function arguments
        max_retries: Maximum number of retries
        base_delay: Initial delay in seconds
        **kwargs: Function keyword arguments

    Returns:
        Function result

    Raises:
        Last exception if all retries fail
    """
    import time

    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                time.sleep(delay)

    if last_exception:
        raise last_exception
