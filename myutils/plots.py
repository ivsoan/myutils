"""
-*- coding:utf-8 -*-
@Time      :2025/8/28 上午9:53
@Author    :Chen Junpeng

"""
import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from .utils_waring import UtilsWarning

default_rc_params = matplotlib.rc_params()


def set_plot_config(font_size: int = 16,
                    title_size: int = 24,
                    label_size: int = 20,
                    xtick_size: int = 16,
                    ytick_size: int = 16):
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': title_size,
        'axes.labelsize': label_size,
        'xtick.labelsize': xtick_size,
        'ytick.labelsize': ytick_size
    })


def _check_plot_config():
    if default_rc_params == plt.rcParams:
        warnings.warn(UtilsWarning(f'Use default plot config, or call set_plot_config() to set custom config.'))


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
    log = logger.info if logger else print
    _check_plot_config()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")
    output_file = os.path.join(save_path, f"{name}.png")

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
    from matplotlib import ticker
    from .file_process import save_data_to_json_file


    log = logger.info if logger else print
    _check_plot_config()
    assert len(pred) == len(target), f"The length of pred and target should be the same, got {len(pred)}, {len(target)}."
    default_msg_list = ['mae', 'mse', 'rmse', 'r2']
    if msg is None:
        warnings.warn(UtilsWarning(f"No message is given, use default message: {default_msg_list}"))
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

    fig, ax = plt.subplots(figsize=(10, 8))
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


def draw_violin_plot(df: pd.DataFrame,
                     title: str,
                     save_path: str,
                     name: str,
                     figsize: tuple[float, float] = (12, 8),
                     asymmetrical_y: bool = False,
                     logger: logging.Logger = None):
    """
    draw violin plot
    :param df: input dataframe, must have exactly two columns for x and y axes
    :param title: title of plot
    :param save_path: path to save plot
    :param name: name of plot
    :param figsize: size of figure
    :param asymmetrical_y: whether to set y axis as asymmetrical or not.
    :param logger: logger object to use
    :return:

    :example:
    df = pd.DataFrame({
        'Category': np.repeat(['A', 'B', 'C'], 200),
        'Value': np.concatenate([
            np.random.normal(0, 0.5, 200),
            np.random.normal(0, 1, 200),
            np.random.normal(0, 1.5, 200)
        ])
    })
    """
    log = logger.info if logger else print
    _check_plot_config()
    if df.shape[1] != 2:
        raise ValueError(f"Input DataFrame must have exactly two columns for x and y axes, but got {df.shape[1]} columns. Columns: {list(df.columns)}.")

    x_label = df.columns[0]
    y_label = df.columns[1]

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")
    output_file = os.path.join(save_path, f"{name}.png")
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=x_label, y=y_label, palette='muted', hue=x_label, legend=False)
    if asymmetrical_y:
        current_ylim = plt.ylim()
        max_abs_val = max(abs(current_ylim[0]), abs(current_ylim[1]))
        plt.ylim((-max_abs_val * 1.05, max_abs_val * 1.05))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_file, dpi=1200)
    plt.close()

    log(f"Save violin plot to {output_file}")


def draw_radar_plot(df: pd.DataFrame,
                     title: str,
                     save_path: str,
                     name: str,
                     figsize: tuple[float, float] = (8, 8),
                     logger: logging.Logger = None):
    """
    draw radar plot
    :param df: input dataframe
    :param title: title of the plot
    :param save_path: save path of the plot
    :param name: the name of the plot
    :param figsize: the size of the plot
    :param logger: the logger to use
    :return:

    :example:
    df = pd.DataFrame({
        'Warrior': [85, 60, 75, 30, 70, 50],
        'Mage': [40, 70, 45, 95, 85, 65],
        'Ranger': [55, 90, 60, 65, 80, 75]
        }, index=['STR', 'AGI', 'CON', 'INT', 'WIS', 'CHA'])
    """
    log = logger.info if logger else print
    _check_plot_config()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")

    save_path = os.path.join(save_path, f"{name}.png")

    labels = df.index.tolist()
    categories = df.columns.tolist()
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = plt.cm.get_cmap("tab20c", len(categories))
    for i, category in enumerate(categories):
        values = df[category].tolist()
        values += values[:1]

        ax.plot(angles, values, color=colors(i), linewidth=2, linestyle='solid', label=category)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=14)
    ax.set_rlabel_position(180)
    ax.grid(color='grey', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 100)
    plt.title(title, color='black', y=1.1)
    plt.legend(loc='lower right', bbox_to_anchor=(1.05, -0.05))

    plt.savefig(save_path, dpi=1200)
    plt.close()

    log(f"Save radar plot to {save_path}")


def draw_line_plot(df: pd.DataFrame,
                   title: str,
                   save_path: str,
                   name: str,
                   y_label: str,
                   x_label: str,
                   figsize: tuple[float, float] = (8, 8),
                   T: bool = False,
                   y_log_scale: bool = False,
                   logger: logging.Logger = None):
    log = logger.info if logger else print
    _check_plot_config()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")

    output_file = os.path.join(save_path, f"{name}.png")

    if T:
        df = df.T

    ax = df.plot(
        kind='line',
        figsize=figsize,
        marker='o',
        linestyle='-',
        logy=y_log_scale,
    )

    ax.set_xticks(range(len(df.index)))
    ax.set_xticklabels(df.index, rotation=0)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.legend(loc='upper left')
    plt.savefig(output_file, dpi=1200)
    plt.close()
    log(f"Save line plot to {output_file}")


def draw_multi_kde_plot(data: dict[str, list],
                        title: str,
                        save_path: str,
                        name: str,
                        x_label: str,
                        legend_title: str,
                        show_legend: bool = True,
                        figsize: tuple[float, float] = (8, 8),
                        cmap: str | list = None,
                        label_peaks: bool = False,
                        logger: logging.Logger = None) -> None:
    from scipy.stats import gaussian_kde
    from matplotlib.patches import Patch


    log = logger.info if logger else print
    _check_plot_config()

    if cmap is None:
        cmap = 'tab10'
        warnings.warn(UtilsWarning(f"No cmap is given, use default cmap: {cmap}"))

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        log(f"Create directory: {save_path}")

    output_file = os.path.join(save_path, f"{name}.png")

    plt.figure(figsize=figsize)
    ax = plt.gca()

    valid_data_values = [v for v in data.values() if v and len(v) > 1 and np.var(v) > 0]

    all_values = np.concatenate(valid_data_values)
    global_min, global_max = all_values.min(), all_values.max()

    # 2. 增加边界余量 (padding)，让曲线有空间回到0
    range_width = global_max - global_min
    padding = range_width * 0.15  # 增加15%的余量
    plot_min = global_min - padding
    plot_max = global_max + padding

    # 3. 生成一个统一的、更宽的X坐标轴用于绘图
    global_x_coords = np.linspace(plot_min, plot_max, 500)

    if isinstance(cmap, str):
        colors = plt.cm.get_cmap(cmap, len(data.keys()))
        indices = np.linspace(0, 1, len(data.keys()))
        colors = [colors(i) for i in indices]
    else:
        colors = cmap
        if len(colors) != len(data.keys()):
            raise ValueError(f"The length of colors should be the same as the number of data, but got {len(colors)} and {len(data.keys())}.")

    legend_handles = []

    for i, (label, values) in enumerate(data.items()):
        if not values or len(values) < 2 or np.var(values) == 0:
            warnings.warn(f"Skipping '{label}' because its data is empty, has too few points, or has zero variance.")
            continue

        try:
            kde = gaussian_kde(values)
            # y_coords = kde(global_x_coords)
            x_coords = np.linspace(min(values) * 1.2, max(values), 500)
            y_coords = kde(x_coords)
        except np.linalg.LinAlgError:
            warnings.warn(f"Skipping '{label}' because KDE calculation failed (likely due to singular matrix).")
            continue

        ax.plot(x_coords, y_coords, color=colors[i], linewidth=1.5)
        ax.fill_between(x_coords, y_coords, color=colors[i], alpha=0.5)

        legend_handles.append(Patch(facecolor=colors[i], alpha=0.6, label=label))

        if label_peaks:
            peak_index = np.argmax(y_coords)
            peak_x = x_coords[peak_index]
            peak_y = y_coords[peak_index]

            ax.text(
                peak_x, peak_y + 0.02, label,
                ha='center', fontsize=14, fontweight='bold'
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Probability Density")
    ax.set_title(title, pad=20)
    if show_legend:
        ax.legend(handles=legend_handles, title=legend_title, loc='upper right')

    ax.set_ylim(-0.1, 10)
    plt.tight_layout()

    plt.savefig(output_file, dpi=1200)
    plt.close()
    log(f"Save multi-KDE plot to {output_file}")