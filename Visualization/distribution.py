import numpy as np
import pandas as pd
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


def vis_pdf(data: pd.DataFrame or np.array or list,
            label: str or list = None,
            figsize: tuple=(8, 6),
            save_path: str=None,
            grid: bool=True,
            title: str=None,
            xlabel: str=None,
            ylabel: str=None) -> None:

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.kdeplot(d, label=label[i], fill=True, alpha=0.5)
            plt.legend(loc='upper left')

        else:
            for d in data:
                sns.kdeplot(d, fill=True, alpha=0.5)
    else:
        if label is not None:
            sns.kdeplot(data, label=label, fill=True, alpha=0.5)
            plt.legend(loc='upper left')
        else:
            sns.kdeplot(data, fill=True, alpha=0.5)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def vis_cdf(data: pd.DataFrame or np.array or List[pd.DataFrame] or List[np.array],
            label: str or List[str] = None,
            figsize: tuple=(8, 6),
            save_path: str=None,
            grid: bool=True,
            title: str=None,
            xlabel: str=None,
            ylabel: str=None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.grid(visible=grid)

    if isinstance(data, list):
        if label is not None:
            assert len(data) == len(label), 'Data and label must have the same length'

            for i, d in enumerate(data):
                sns.kdeplot(d, label=label[i], cumulative=True)
            plt.legend(loc='upper left')

        else:
            for d in data:
                sns.kdeplot(d, cumulative=True)
    else:
        if label is not None:
            sns.kdeplot(data, label=label, cumulative=True)
            plt.legend(loc='upper left')
        else:
            sns.kdeplot(data, cumulative=True)

    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
