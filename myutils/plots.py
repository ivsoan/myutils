"""
-*- coding:utf-8 -*-
@Time      :2025/8/28 上午9:53
@Author    :Chen Junpeng

"""
import logging
import os
import numpy as np


def draw_distribution_plot(data: list | dict,
                      save_path: str,
                      name: str,
                      xlabel: str,
                      ylabel: str,
                      title: str,
                      color: str = 'lightblue',
                      bins: int = None,
                      rotate_45: bool = False,
                      x_log_scale: bool = False,
                      y_log_scale: bool = False,
                      logger: logging.Logger = None):
    """
    Draw distribution plot using seaborn.
    :param data: a list or a dictionary of data to plot. eg. [120, 120, 235, 124] or {'A': 120, 'B': 235, 'C': 124}
    :param save_path: the path to save the figure
    :param name: the name of the figure
    :param xlabel: the name of the x axis
    :param ylabel: the name of the y axis
    :param title: the title of the plot
    :param color: the color of the bar
    :param bins: the number of bins
    :param rotate_45: rotate 45 degrees for x labels or not
    :param x_log_scale: set x axis to log scale or not
    :param y_log_scale: set y axis to log scale or not
    :param logger: the logger to use
    :return:
    """
    import matplotlib.pyplot as plt
    import seaborn as sns


    log = logger.info if logger else print
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")
    output_file = os.path.join(save_path, f"{name}.png")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12
    })

    if isinstance(data, list):
        plt.figure(figsize=(8, 8))
        sns.histplot(data, bins=bins, kde=True, color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if x_log_scale:
            plt.xscale('log')
        if y_log_scale:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_file, dpi=1200)
        plt.close()
        log(f"Save distribution plot to {output_file}")

    elif isinstance(data, dict):
        plt.figure(figsize=(8, 8))
        sns.barplot(x=list(data.keys()), y=list(data.values()), color=color, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if rotate_45:
            plt.xticks(rotation=45, ha='right')
        if x_log_scale:
            plt.xscale('log')
        if y_log_scale:
            plt.yscale('log')
        plt.tight_layout()
        plt.savefig(output_file, dpi=1200)
        plt.close()
        log(f"Save distribution plot to {output_file}")


def draw_parity_plot(pred: list | np.ndarray,
                     target: list | np.ndarray,
                     save_path: str,
                     name: str,
                     title: str,
                     save_result: bool = False,
                     msg: list = None,
                     logger: logging.Logger = None):
    """
    Draw parity plot for comparing predicted values and target values.
    :param pred: the predicted values
    :param target: the target values
    :param save_path: the path to save the figure
    :param name: the name of the figure
    :param title: the title of the plot
    :param save_result: save the result or not
    :param msg: the message to display, chosen from ['mae', 'mse', 'rmse', 'r2']
    :param logger: the logger to use
    :return:
    """
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from scipy.stats import gaussian_kde
    from matplotlib import pyplot as plt
    from matplotlib import ticker
    from .file_process import save_data_to_json_file

    log = logger.info if logger else print
    assert len(pred) == len(target), f"The length of pred and target should be the same, got {len(pred)}, {len(target)}."
    default_msg_list = ['mae', 'mse', 'rmse', 'r2']
    if msg is None:
        msg = default_msg_list
    else:
        default_msg_set = set(default_msg_list)
        _msg_set = set(msg)
        if not _msg_set.issubset(default_msg_set):
            raise ValueError(f"Valid message should be in {default_msg_set}, but got invalid message: {_msg_set - default_msg_set}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")
    output_file = os.path.join(save_path, f"{name}.png")

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 10,
        'ytick.labelsize': 12
    })

    _pred, _targ = [], []
    for p, t in zip(pred, target):
        if p is not None and t is not None:
            _pred.append(p)
            _targ.append(t)
    pred, targ = np.array(_pred), np.array(_targ)

    if len(pred) < 2:
        raise ValueError(f"The length of pred should be greater than 2.")

    r2 = r2_score(targ, pred)
    mae = mean_absolute_error(targ, pred)
    mse = mean_squared_error(targ, pred)
    rmse = np.sqrt(mse)

    result_dict = {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}
    if save_result:
        save_data_to_json_file(result_dict, save_path, 'result')

    lim_min = min(np.min(pred), np.min(targ))
    lim_max = max(np.max(pred), np.max(targ))
    margin = (lim_max - lim_min) * 0.1
    lim_min -= margin
    lim_max += margin
    axis_limit = [lim_min, lim_max]

    xy = np.vstack([targ, pred])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    targ, pred, z = targ[idx], pred[idx], z[idx]

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter_plot = ax.scatter(targ, pred, c=z, s=20, cmap='viridis', zorder=2)

    cbar = fig.colorbar(scatter_plot, ax=ax)
    cbar.set_label('Data Point Density')

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    cbar.ax.yaxis.set_major_formatter(formatter)

    ax.plot(axis_limit, axis_limit, 'r--', label='y=x', zorder=1)
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title(title)
    ax.set_xlim(axis_limit)
    ax.set_ylim(axis_limit)
    ax.set_aspect('equal', adjustable='box')

    m = ''

    for i, _m in enumerate(msg):
        value = result_dict[_m]
        if _m == 'r2':
            _m = 'R$^2$'
        elif _m == 'mae':
            _m = 'MAE'
        elif _m =='mse':
            _m = 'MSE'
        elif _m =='rmse':
            _m = 'RMSE'
        m += f'{_m}: {value:.4f}\n'
    m = m[:-2]

    ax.text(0.55, 0.05, m,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes,
            fontsize=20,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    ax.legend(loc='upper left')
    plt.savefig(output_file, dpi=1200)
    plt.close()
    log(f"Save parity plot to {output_file}")