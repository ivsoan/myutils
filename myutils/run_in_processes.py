"""
-*- coding:utf-8 -*-
@Time      :2025/8/20 上午10:24
@Author    :Chen Junpeng

"""
import logging
import multiprocessing
import os
import traceback
import warnings
from functools import wraps, partial
from typing import Optional, Callable, Any
from .utils_waring import UtilsWarning


def _safe_worker(func: Callable, task_item: Any) -> tuple:
    try:
        result = func(task_item)
        return 'success', result
    except:
        error_str = traceback.format_exc()
        return 'error', (task_item, error_str)


def run_in_processes(num_processes: Optional[int] = None, remove_error_results: bool = True):
    """
    Decorator to run a function in multiple processes.
    :param num_processes: number of processes to run in parallel, defaults to the number of CPUs available.
    :param remove_error_result: whether to remove the error result when running the function. If False, return [success_task 1, success_task 2, None, ..., success_task n]
    :return:
    :usage:
        step1: Instantiate this decorator with the desired number of processes. e.g. parallel_decorator = run_in_processes()
        step2: decorate function with this decorator, if the func has a _worker version, use it instead. e.g. parallel_normalize_smiles = parallel_decorator(normalize_smiles)
        step3: call the decorated function with the tasks iterable as the first argument and any other arguments as needed. e.g. results = parallel_normalize_smiles(smiles_list, sanitize=True)
    """
    if num_processes is None and os.cpu_count() > 48:
        warnings.warn(UtilsWarning(f"If you are running on a server, make sure to set the `num_processes` argument to a lower value to avoid overloading the server."))
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                raise ValueError(
                    f"function '{func.__name__}' must have at least one argument, which is the tasks iterable"
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
            log_info = logger.info if logger else print
            log_error = logger.error if logger else print

            effective_processes = num_processes if num_processes is not None else os.cpu_count()

            tasks_iterable = args[0]
            common_args = args[1:]

            original_worker = partial(func, *common_args, **kwargs)
            safe_worker = partial(_safe_worker, original_worker)

            with multiprocessing.Pool(processes=effective_processes) as pool:
                log_info(f"submitting {len(tasks_iterable)} tasks to {effective_processes} processes")
                results = pool.map(safe_worker, tasks_iterable)

            successful_results = []
            failed_tasks_count = 0
            for item in results:
                status, payload = item
                if status == 'success':
                    successful_results.append(payload)
                else:
                    failed_tasks_count += 1
                    original_task, error_message = payload
                    if remove_error_results is False:
                        successful_results.append(None)
                    log_error(f"Task failed for input: {repr(original_task)}\nError: {error_message}")

            if failed_tasks_count > 0:
                log_info(f"Completed with {failed_tasks_count} failed tasks out of {len(tasks_iterable)} total.")

            return successful_results

        return wrapper

    return decorator

