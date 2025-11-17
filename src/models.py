# src/models.py
import torch
import torch.nn as nn
from typing import Sequence

class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (256, 128, 64, 32),
        act: type[nn.Module] = nn.ELU,
        out_bias_init: float = -1.0,
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(act())
            d = h
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(d, out_dim)
        self._init_weights(out_bias_init)

    def _init_weights(self, out_bias_init: float):
        # Use ReLU gain for ELU blocks (PyTorch calculate_gain doesn't support 'elu')
        for m in self.backbone:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        # Output head: small Xavier + bias near dataset mean log-power (~-1.1)
        nn.init.xavier_uniform_(self.head.weight, gain=1.0)
        nn.init.constant_(self.head.bias, float(out_bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self.backbone(x)
        return self.head(x)
