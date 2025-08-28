"""
-*- coding:utf-8 -*-
@Time      :2025/8/13 下午1:39
@Author    :Chen Junpeng

"""
import warnings

class UtilsWarning(UserWarning):
    def __init__(self, msg=None):
        self.msg = msg

    def __str__(self):
        return self.msg