import torch
import torch.nn as nn
import numpy as np

class FourierFeatureEncoder:
    def __init__(self, input_dims: int, mapping_size: int, scale: float, device: torch.device):
        self.input_dims = input_dims
        self.mapping_size = mapping_size
        self.scale = float(scale)
        self.device = device
        self.B = torch.randn((mapping_size, input_dims), device=self.device) * self.scale
        self.B.requires_grad = False

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ConductivityNetwork(nn.Module):
    def __init__(
        self,
        ffe_dims: int,
        actv: nn.Module = nn.SiLU(),
        actv_last_layer: bool = True,
        layers: int = 4,
        neurons: int = 128,
        min_gamma: float = 0.5,
        max_gamma: float = 2.5,
    ):
        super().__init__()
        net_layers = []
        in_dim = ffe_dims
        for _ in range(layers):
            net_layers.append(nn.Linear(in_dim, neurons))
            net_layers.append(actv)
            in_dim = neurons
        net_layers.append(nn.Linear(neurons, 1))
        if actv_last_layer:
            net_layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*net_layers)

        self.min_gamma = float(min_gamma)
        self.max_gamma = float(max_gamma)

    def forward(self, x_ffe: torch.Tensor) -> torch.Tensor:
        raw = self.network(x_ffe)
        return self.min_gamma + (self.max_gamma - self.min_gamma) * raw
