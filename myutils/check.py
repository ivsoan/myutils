"""
-*- coding:utf-8 -*-
@Time      :2025/9/8 下午3:00
@Author    :Chen Junpeng

"""
import numpy as np
import pandas as pd
from plots import draw_violin_plot

np.random.seed(0)
data = pd.DataFrame({
    'Category': np.repeat(['A', 'B', 'C'], 200),
    'Value': np.concatenate([
        np.random.normal(0, 0.5, 200),
        np.random.normal(0, 1, 200),
        np.random.normal(0, 1.5, 200)
    ])
})


draw_violin_plot(data, 'test', './test', 'test')