from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput

from models.attention import TemporalBasicTransformerBlock
from models.motion_module import zero_module
from models.resnet import InflatedConv3d, InflatedGroupNorm
from models.transformer_3d import Transformer3DModel


class GuidanceEncoder(ModelMixin):
    def __init__(
        self,
        guidance_embedding_channels: int,
        guidance_input_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        attention_num_heads: int = 8,
    ):
        super().__init__()
        self.guidance_input_channels = guidance_input_channels
        self.conv_in = InflatedConv3d(
            guidance_input_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]

            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.attentions.append(
                Transformer3DModel(
                    attention_num_heads,
                    channel_in // attention_num_heads,
                    channel_in,
                    norm_num_groups=1,
                    unet_use_cross_frame_attention=False,
                    unet_use_temporal_attention=False,
                )
            )

            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
            self.attentions.append(
                Transformer3DModel(
                    attention_num_heads,
                    channel_out // attention_num_heads,
                    channel_out,
                    norm_num_groups=32,
                    unet_use_cross_frame_attention=False,
                    unet_use_temporal_attention=False,
                )
            )

        attention_channel_out = block_out_channels[-1]
        self.guidance_attention = Transformer3DModel(
            attention_num_heads,
            attention_channel_out // attention_num_heads,
            attention_channel_out,
            norm_num_groups=32,
            unet_use_cross_frame_attention=False,
            unet_use_temporal_attention=False,
        )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                guidance_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, condition):
        embedding = self.conv_in(condition)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        # FIXME: Temporarily only use the last attention.
        embedding = self.attentions[-1](embedding).sample
        embedding = self.conv_out(embedding)

        return embedding
