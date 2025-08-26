"""
-*- coding:utf-8 -*-
@Time      :2025/8/18 下午5:50
@Author    :Chen Junpeng

"""
import multiprocessing
import functools


class TimeoutError(Exception):
    pass


def timeout():
    """
    This decorator function is used to set a timeout for a function call.
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            exception_container = multiprocessing.Queue()
            result_container = multiprocessing.Queue()

            timeout = kwargs.pop('timeout', None)
            if timeout is None:
                raise ValueError("parameter \"timeout\" must be specified in function call arguments.")

            def target():
                try:
                    result = func(*args, **kwargs)
                    result_container.put(result)
                except Exception as e:
                    exception_container.put(e)

            process = multiprocessing.Process(target=target)
            process.start()

            process.join(timeout)

            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout} seconds")
            else:
                if not exception_container.empty():
                    exception = exception_container.get()
                    raise exception
                else:
                  result = result_container.get()
                  return result

        return wrapper

    return decorator