from inspect import isfunction
from typing import Callable, Optional

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from .t_cond_mlp import (
    AdaptiveLayerNorm1D,
    FrequencyEmbedder,
    normalization_layer,
)
# from .vit import Attention, FeedForward


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: Callable, norm: str = "layer", norm_cond_dim: int = -1):
        super().__init__()
        self.norm = normalization_layer(norm, dim, norm_cond_dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if isinstance(self.norm, AdaptiveLayerNorm1D):
            return self.fn(self.norm(x, *args), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = default(context_dim, dim)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context=None):
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args):
        for attn, ff in self.layers:
            x = attn(x, *args) + x
            x = ff(x, *args) + x
        return x


class TransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(
                dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout
            )
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ca, norm=norm, norm_cond_dim=norm_cond_dim),
                        PreNorm(dim, ff, norm=norm, norm_cond_dim=norm_cond_dim),
                    ]
                )
            )

    def forward(self, x: torch.Tensor, *args, context=None, context_list=None):
        if context_list is None:
            context_list = [context] * len(self.layers)
        if len(context_list) != len(self.layers):
            raise ValueError(f"len(context_list) != len(self.layers) ({len(context_list)} != {len(self.layers)})")

        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            x = self_attn(x, *args) + x
            x = cross_attn(x, *args, context=context_list[i]) + x
            x = ff(x, *args) + x
        return x


class DropTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[0, :, 0], self.p).bernoulli().bool()
            # TODO: permutation idx for each batch using torch.argsort
            if zero_mask.any():
                x = x[:, ~zero_mask, :]
        return x


class ZeroTokenDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "dropout probability has to be between 0 and 1, " "but got {}".format(p)
            )
        self.p = p

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_len, dim)
        if self.training and self.p > 0:
            zero_mask = torch.full_like(x[:, :, 0], self.p).bernoulli().bool()
            # Zero-out the masked tokens
            x[zero_mask, :] = 0
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = "drop",
        emb_dropout_loc: str = "token",
        norm: str = "layer",
        norm_cond_dim: int = -1,
        token_pe_numfreq: int = -1,
    ):
        super().__init__()
        if token_pe_numfreq > 0:
            token_dim_new = token_dim * (2 * token_pe_numfreq + 1)
            self.to_token_embedding = nn.Sequential(
                Rearrange("b n d -> (b n) d", n=num_tokens, d=token_dim),
                FrequencyEmbedder(token_pe_numfreq, token_pe_numfreq - 1),
                Rearrange("(b n) d -> b n d", n=num_tokens, d=token_dim_new),
                nn.Linear(token_dim_new, dim),
            )
        else:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        else:
            raise ValueError(f"Unknown emb_dropout_type: {emb_dropout_type}")
        self.emb_dropout_loc = emb_dropout_loc

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout, norm=norm, norm_cond_dim=norm_cond_dim
        )

    def forward(self, inp: torch.Tensor, *args, **kwargs):
        x = inp

        if self.emb_dropout_loc == "input":
            x = self.dropout(x)
        x = self.to_token_embedding(x)

        if self.emb_dropout_loc == "token":
            x = self.dropout(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]

        if self.emb_dropout_loc == "token_afterpos":
            x = self.dropout(x)
        x = self.transformer(x, *args)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        emb_dropout_type: str = 'drop',
        norm: str = "layer",
        norm_cond_dim: int = -1,
        context_dim: Optional[int] = None,
        skip_token_embedding: bool = False,
    ):
        super().__init__()
        if not skip_token_embedding:
            self.to_token_embedding = nn.Linear(token_dim, dim)
        else:
            self.to_token_embedding = nn.Identity()
            if token_dim != dim:
                raise ValueError(
                    f"token_dim ({token_dim}) != dim ({dim}) when skip_token_embedding is True"
                )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, dim))
        if emb_dropout_type == "drop":
            self.dropout = DropTokenDropout(emb_dropout)
        elif emb_dropout_type == "zero":
            self.dropout = ZeroTokenDropout(emb_dropout)
        elif emb_dropout_type == "normal":
            self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerCrossAttn(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            norm=norm,
            norm_cond_dim=norm_cond_dim,
            context_dim=context_dim,
        )

    def forward(self, inp: torch.Tensor, *args, context=None, context_list=None):
        x = self.to_token_embedding(inp)
        b, n, _ = x.shape

        x = self.dropout(x)
        x += self.pos_embedding[:, :n]

        x = self.transformer(x, *args, context=context, context_list=context_list)
        return x

