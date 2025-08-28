"""
-*- coding:utf-8 -*-
@Time      :2025/8/28 上午9:53
@Author    :Chen Junpeng

"""
import logging
import os


def draw_distribution_plot(data: list | dict,
                      save_path: str,
                      name: str,
                      xlabel: str,
                      ylabel: str,
                      title: str,
                      color: str = 'lightblue',
                      rotate_45: bool = False,
                      x_log_scale: bool = False,
                      y_log_scale: bool = False,
                      logger: logging.Logger = None):
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
        sns.histplot(data, bins='auto', kde=True, color=color, edgecolor='black')
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