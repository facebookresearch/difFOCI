# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch

def pd_sort(x, ascending):
    if ascending:
        return torch.argsort(torch.argsort(x)) + 1
    else:
        return torch.argsort(torch.argsort(x, descending=True)) + 1

def softmaxb(x, beta=1e-5):
    """Compute softmax values for each sets of scores in x."""
    e_x = torch.exp((x - torch.max(x))/beta)
    return e_x / e_x.sum()


class KNeighbors:
    def __init__(self, k=2, use_faiss=False):
        self.index = None
        self.k = k
        self.faiss = use_faiss
        assert use_faiss is False

    def fit(self, X, beta=1e-5):
        output = torch.cdist(X, X)
        return torch.vmap(lambda x: softmaxb(x, beta))(-output-1e10*torch.eye(output.shape[0]))
        
        
def conditional_dependence(x, y, beta=1e-5) -> float:
    # if y.min() == y.max():
        # return 0

    n = len(y)
    r_y = pd_sort(y, ascending=True)

    l_y = pd_sort(y, ascending=False)
    s = (l_y * (n - l_y)).sum() / (n ** 3)

    z_neighbors = KNeighbors(k=2)
    m = z_neighbors.fit(x, beta)    
    
    r_m = torch.vmap(lambda x: torch.dot(r_y.float(), x))(m)
    
    q = (torch.minimum(r_y, r_m) - (l_y * l_y) / n).sum() / (n * n)
    return q / s
