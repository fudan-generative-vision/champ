import copy
from typing import List, Optional

import torch


class AdaptiveLayerNorm1D(torch.nn.Module):
    def __init__(self, data_dim: int, norm_cond_dim: int):
        super().__init__()
        if data_dim <= 0:
            raise ValueError(f"data_dim must be positive, but got {data_dim}")
        if norm_cond_dim <= 0:
            raise ValueError(f"norm_cond_dim must be positive, but got {norm_cond_dim}")
        self.norm = torch.nn.LayerNorm(
            data_dim
        )  # TODO: Check if elementwise_affine=True is correct
        self.linear = torch.nn.Linear(norm_cond_dim, 2 * data_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (batch, ..., data_dim)
        # t: (batch, norm_cond_dim)
        # return: (batch, data_dim)
        x = self.norm(x)
        alpha, beta = self.linear(t).chunk(2, dim=-1)

        # Add singleton dimensions to alpha and beta
        if x.dim() > 2:
            alpha = alpha.view(alpha.shape[0], *([1] * (x.dim() - 2)), alpha.shape[1])
            beta = beta.view(beta.shape[0], *([1] * (x.dim() - 2)), beta.shape[1])

        return x * (1 + alpha) + beta


class SequentialCond(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        for module in self:
            if isinstance(module, (AdaptiveLayerNorm1D, SequentialCond, ResidualMLPBlock)):
                # print(f'Passing on args to {module}', [a.shape for a in args])
                input = module(input, *args, **kwargs)
            else:
                # print(f'Skipping passing args to {module}', [a.shape for a in args])
                input = module(input)
        return input


def normalization_layer(norm: Optional[str], dim: int, norm_cond_dim: int = -1):
    if norm == "batch":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer":
        return torch.nn.LayerNorm(dim)
    elif norm == "ada":
        assert norm_cond_dim > 0, f"norm_cond_dim must be positive, got {norm_cond_dim}"
        return AdaptiveLayerNorm1D(dim, norm_cond_dim)
    elif norm is None:
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown norm: {norm}")


def linear_norm_activ_dropout(
    input_dim: int,
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",  # Options: ada/batch/layer
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers = []
    layers.append(torch.nn.Linear(input_dim, output_dim, bias=bias))
    if norm is not None:
        layers.append(normalization_layer(norm, output_dim, norm_cond_dim))
    layers.append(copy.deepcopy(activation))
    if dropout > 0.0:
        layers.append(torch.nn.Dropout(dropout))
    return SequentialCond(*layers)


def create_simple_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: torch.nn.Module = torch.nn.ReLU(),
    bias: bool = True,
    norm: Optional[str] = "layer",  # Options: ada/batch/layer
    dropout: float = 0.0,
    norm_cond_dim: int = -1,
) -> SequentialCond:
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            linear_norm_activ_dropout(
                prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            )
        )
        prev_dim = hidden_dim
    layers.append(torch.nn.Linear(prev_dim, output_dim, bias=bias))
    return SequentialCond(*layers)


class ResidualMLPBlock(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",  # Options: ada/batch/layer
        dropout: float = 0.0,
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        if not (input_dim == output_dim == hidden_dim):
            raise NotImplementedError(
                f"input_dim {input_dim} != output_dim {output_dim} is not implemented"
            )

        layers = []
        prev_dim = input_dim
        for i in range(num_hidden_layers):
            layers.append(
                linear_norm_activ_dropout(
                    prev_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
                )
            )
            prev_dim = hidden_dim
        self.model = SequentialCond(*layers)
        self.skip = torch.nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.model(x, *args, **kwargs)


class ResidualMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        output_dim: int,
        activation: torch.nn.Module = torch.nn.ReLU(),
        bias: bool = True,
        norm: Optional[str] = "layer",  # Options: ada/batch/layer
        dropout: float = 0.0,
        num_blocks: int = 1,
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.model = SequentialCond(
            linear_norm_activ_dropout(
                input_dim, hidden_dim, activation, bias, norm, dropout, norm_cond_dim
            ),
            *[
                ResidualMLPBlock(
                    hidden_dim,
                    hidden_dim,
                    num_hidden_layers,
                    hidden_dim,
                    activation,
                    bias,
                    norm,
                    dropout,
                    norm_cond_dim,
                )
                for _ in range(num_blocks)
            ],
            torch.nn.Linear(hidden_dim, output_dim, bias=bias),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(x, *args, **kwargs)


class FrequencyEmbedder(torch.nn.Module):
    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer("frequencies", frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(
            N, -1
        )  # (N, D * 2 * num_frequencies + D)
        return embedded

