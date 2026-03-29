import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, B)


N = 512


def get_inputs():
    A = torch.randn((N, N), dtype=torch.float32)
    B = torch.randn((N, N), dtype=torch.float32)
    return [A, B]


def get_init_inputs():
    return []
