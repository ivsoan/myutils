"""
-*- coding:utf-8 -*-
@Time      :2025/8/23 下午4:15
@Author    :Chen Junpeng

"""
import logging
import os.path


def read_txt_to_list(file_path: str, logger: logging.Logger = None) -> list:
    """
    Read txt file to list.
    :param file_path: txt file path
    :param logger: the logger to use
    :return: a list of lines in the txt file
    """
    log = logger.info if logger else print
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    log(f"Read {len(lines)} lines from {file_path}.")
    return [line.strip() for line in lines if line.strip()]


def merge_txt_files(file_path_list: list, save_path: str, name: str, logger: logging.Logger = None) -> None:
    """
    Merge txt files to one file.
    :param file_path_list: txt file paths to be merged
    :param save_path: txt file path to save merged file
    :param name: the name of the merged file
    :param logger: the logger to use
    :return:
    """
    log = logger.info if logger else print
    line_set = set()
    original_line_length = 0

    for file_path in file_path_list:
        lines = read_txt_to_list(file_path, logger)
        original_line_length += len(lines)
        line_set.update(lines)

    log(f"Original line length: {original_line_length}.")
    log(f"After merging, line length: {len(line_set)}.")
    log(f"Delete {original_line_length - len(line_set)} duplicates.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, f"{name}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in line_set:
            if line.strip():
                f.write(line + '\n')
    log(f"Saved merged file to {output_file}.")


def get_unique_lines(txt_file_1: str, txt_file_2: str, save_path: str, name: str, logger: logging.Logger = None) -> None:
    """
    Get lines in txt file 1 but not in txt file 2.
    :param txt_file_1: path to txt file 1
    :param txt_file_2: path to txt file 2
    :param save_path: txt file path to save merged file
    :param name: the name of the merged file
    :param logger: the logger to use
    :return:
    """
    log = logger.info if logger else print
    line1 = set(read_txt_to_list(txt_file_1, logger))
    line2 = set(read_txt_to_list(txt_file_2, logger))
    unique_lines = line1.difference(line2)
    log(f"Unique lines: {len(unique_lines)}.")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_file = os.path.join(save_path, f"{name}.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            if line.strip():
                f.write(line + '\n')
    log(f"Saved unique lines to {output_file}.")