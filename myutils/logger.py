"""
-*- coding:utf-8 -*-
@Time      :2025/8/18 下午5:16
@Author    :Chen Junpeng

"""
import os
import sys
import datetime
import logging
import functools
import inspect
import warnings
from .utils_waring import UtilsWarning
from .util import get_datetime_str


def log_call(level=logging.INFO):
    """
    Decorator to log function execution.
    :param level: the logging level to use (default is logging.INFO)
    :return:
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # try to get logger from args or kwargs
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

            if logger is None:
                warnings.warn(UtilsWarning("No logger found in arguments, using root logger."))
                logger = logging.getLogger()

            try:
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                param_strs = []
                for k, v in bound_args.arguments.items():
                    if k == 'self':
                        continue

                    v_str = str(v)
                    param_strs.append(f"{k}: {v_str}")

                if param_strs:
                    args_content = ", \n".join(param_strs)
                    all_args_str = f"Running function: {func.__name__} with:\n{args_content}"
                else:
                    all_args_str = f"Running function: {func.__name__} (No arguments)"

            except Exception:
                all_args_str = f"Running function: {func.__name__} with args: {args}, kwargs: {kwargs}"

            logger.log(level, all_args_str)

            result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def default_log_file_name():
    """
    Generate a default log file name based on the script name and the current time.
    :return: the default log file name
    """
    script_name_with_ext = os.path.basename(sys.argv[0])
    script_name = os.path.splitext(script_name_with_ext)[0]
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")
    log_file = f'{script_name}_{formatted_time}.log'
    return log_file


def logging_init(log_path: str = None, log_file_name: str = None, console_level=logging.DEBUG, file_level=logging.DEBUG, send_to_console=True, name=None):
    """
    Initialize the logging system.
    :param log_path: the path of the log file
    :param log_file_name: the name of the log file (default is generated based on the script name and the current time)
    :param console_level: the logging level in the console to use
    :param file_level: the logging level in the file to use
    :param send_to_console: if True, log messages will be sent to the console as well as to the file
    :param name: the name of the logger to use (default is the name of the script)
    :return: a logger instance
    """
    if name is None:
        name = os.path.basename(inspect.stack()[1].filename)

    log_path = log_path if log_path is not None else './logs'

    if not os.path.exists(log_path):
        os.makedirs(log_path)
        warnings.warn(UtilsWarning(f'Create logs directory at {log_path}.'))

    if log_file_name:
        log_file_name += get_datetime_str()
        if not log_file_name.endswith('.log'):
            log_file_name += '.log'
    else:
        log_file_name = default_log_file_name()

    log_file = os.path.join(log_path, log_file_name)

    print(f'logging_init: Logging to file {log_file}.')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)

    if send_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if send_to_console:
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.addHandler(file_handler)

    return logger