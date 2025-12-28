"""
-*- coding:utf-8 -*-
@Time      :2025/8/20 上午10:15
@Author    :Chen Junpeng

"""
import logging
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_in_threads(max_workers: int = 10):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                raise ValueError(
                    f"function '{func.__name__}' needs at least one argument, which is the tasks iterable"
                )
            logger = None
            if 'logger' in kwargs:
                logger = kwargs['logger']
            elif args and hasattr(args[0], '__dict__') and 'logger' in args[0].__dict__:
                logger = args[0].logger
            else:
                for arg in args:
                    if isinstance(arg, logging.Logger):
                        logger = arg
                        break
            log = logger.info if logger else print

            tasks_iterable = args[0]
            common_args = args[1:]

            results = {}

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(func, task, *common_args, **kwargs): index
                    for index, task in enumerate(tasks_iterable)
                }

                log(f"submit {len(future_to_index)} tasks to thread pool executor")

                for future in as_completed(future_to_index):
                    original_index = future_to_index[future]
                    try:
                        result = future.result()
                        results[original_index] = result
                    except Exception as e:
                        log(f"task {original_index} raised exception: {e}")
                        raise e

            sorted_results = [results[i] for i in sorted(results.keys())]
            return sorted_results

        return wrapper

    return decorator