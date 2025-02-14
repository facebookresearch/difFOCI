# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import pandas as pd

from ._utils import KNeighbors


def pd_sort(x, ascending):
    if ascending:
        return torch.argsort(torch.argsort(x)) + 1
    else:
        return torch.argsort(torch.argsort(x, descending=True)) + 1


def conditional_dependence(x, y, beta=0.2, device='cpu') -> float:
    n = len(y)
    r_y = pd_sort(y, ascending=True)

    l_y = pd_sort(y, ascending=False)
    s = (l_y * (n - l_y)).sum() / (n ** 3)

    z_neighbors = KNeighbors(k=2, device=device)
    m = z_neighbors.fit(x, beta)

    r_m = torch.vmap(lambda x: torch.dot(r_y.float(), x))(m)

    q = (torch.minimum(r_y, r_m) - (l_y * l_y) / n).sum() / (n * n)
    return q / s


def conditional_dependence_with_x(
    y: pd.Series, z: pd.DataFrame, x: pd.DataFrame, beta=0.2, device='cpu'
) -> float:

    r_y = pd_sort(y, ascending=True)
    x_z = torch.concat([x, z], axis=1)

    x_z_neighbors = KNeighbors(k=2, device=device)
    m = x_z_neighbors.fit(x_z, beta)
    r_m = torch.vmap(lambda x: torch.dot(r_y.float(), x))(m)

    x_neighbors = KNeighbors(k=2, device=device)
    n_i = x_neighbors.fit(x, beta)
    r_n = torch.vmap(lambda x: torch.dot(r_y.float(), x))(n_i)

    num = (torch.minimum(r_y, r_m) - torch.minimum(r_y, r_n)).sum()
    den = (r_y - torch.minimum(r_y, r_n)).sum()
    return num / den
