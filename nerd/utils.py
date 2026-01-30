import torch


def pairwise_d2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distances.

    Parameters
    ----------
    a : torch.Tensor
        Input tensor of shape ``(B, D)`` containing query vectors.
    b : torch.Tensor
        Input tensor of shape ``(K, D)`` containing codebook vectors.

    Returns
    -------
    torch.Tensor
        Squared L2 distances with shape ``(B, K)`` where entry ``(i, j)`` is
        the distance between ``a[i]`` and ``b[j]``.
    """
    return (
        a.pow(2).sum(dim=1, keepdim=True)
        - 2 * a @ b.t()
        + b.pow(2).sum(dim=1, keepdim=True).t()
    )
