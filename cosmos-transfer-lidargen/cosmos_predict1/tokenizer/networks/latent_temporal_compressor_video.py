# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen 2D tokenizer with latent-side temporal compression."""

from collections import namedtuple
import math
from typing import Any

from einops import rearrange
from loguru import logger as logging
import torch
import torch.nn.functional as F
from torch import nn

from cosmos_predict1.tokenizer.modules import ContinuousFormulation
from cosmos_predict1.tokenizer.networks.continuous_image import ContinuousImageTokenizer
from cosmos_predict1.tokenizer.modules.layers3d import (
    CausalAttnBlock,
    CausalConv3d,
    CausalHybridDownsample3d,
    CausalHybridUpsample3d,
    CausalNormalize,
    CausalResnetBlockFactorized3d,
    CausalTemporalAttnBlock,
    nonlinearity,
)

NetworkEval = namedtuple(
    "LatentTemporalCompressorNetworkEval",
    [
        "reconstructions",
        "posteriors",
        "latent",
        "compressed_latent",
        "reconstructed_latents",
        "target_latents",
    ],
)


def _group_count(num_groups: int, num_channels: int) -> int:
    if num_channels % num_groups == 0:
        return num_groups
    return math.gcd(num_groups, num_channels) or 1


def _front_pad_temporal(x: torch.Tensor, pad_frames: int) -> torch.Tensor:
    if pad_frames <= 0:
        return x
    return F.pad(x, (0, 0, 0, 0, pad_frames, 0), mode="constant", value=0.0)


class OpenSoraStyleResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        activation_fn=nn.SiLU,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activate = activation_fn()

        norm_groups_in = _group_count(num_groups, in_channels)
        norm_groups_out = _group_count(num_groups, out_channels)
        self.norm1 = nn.GroupNorm(norm_groups_in, in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(norm_groups_out, out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.shortcut is not None:
            residual = self.shortcut(residual)
        return x + residual


class OpenSoraStyleTemporalEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_embed_dim: int,
        filters: int,
        num_res_blocks: int,
        channel_multipliers: tuple[int, ...] | list[int],
        temporal_downsample: tuple[bool, ...] | list[bool],
        num_groups: int = 32,
        activation_fn=nn.SiLU,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = tuple(channel_multipliers)
        self.temporal_downsample = tuple(temporal_downsample)
        self.activate = activation_fn()
        self.num_groups = num_groups

        self.conv_in = CausalConv3d(in_channels, filters, kernel_size=3, padding=1, bias=False)
        self.block_res_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])

        prev_filters = filters
        for i in range(self.num_blocks):
            filters_i = self.filters * self.channel_multipliers[i]
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(
                    OpenSoraStyleResBlock(
                        prev_filters,
                        filters_i,
                        num_groups=self.num_groups,
                        activation_fn=activation_fn,
                    )
                )
                prev_filters = filters_i
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i]:
                    self.conv_blocks.append(
                        CausalConv3d(
                            prev_filters,
                            filters_i,
                            kernel_size=3,
                            stride=1,
                            time_stride=2,
                            padding=1,
                            bias=False,
                        )
                    )
                else:
                    self.conv_blocks.append(nn.Identity())

        self.res_blocks = nn.ModuleList(
            [
                OpenSoraStyleResBlock(
                    prev_filters,
                    prev_filters,
                    num_groups=self.num_groups,
                    activation_fn=activation_fn,
                )
                for _ in range(self.num_res_blocks)
            ]
        )
        self.norm_out = nn.GroupNorm(_group_count(self.num_groups, prev_filters), prev_filters)
        self.conv_out = CausalConv3d(prev_filters, latent_embed_dim, kernel_size=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for i in range(self.num_blocks):
            for block in self.block_res_blocks[i]:
                x = block(x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)

        for block in self.res_blocks:
            x = block(x)

        x = self.norm_out(x)
        x = self.activate(x)
        return self.conv_out(x)


class OpenSoraStyleTemporalDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_embed_dim: int,
        filters: int,
        num_res_blocks: int,
        channel_multipliers: tuple[int, ...] | list[int],
        temporal_downsample: tuple[bool, ...] | list[bool],
        num_groups: int = 32,
        activation_fn=nn.SiLU,
    ) -> None:
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.channel_multipliers = tuple(channel_multipliers)
        self.temporal_downsample = tuple(temporal_downsample)
        self.num_blocks = len(self.channel_multipliers)
        self.num_groups = num_groups
        self.activate = activation_fn()

        filters_last = self.filters * self.channel_multipliers[-1]
        prev_filters = filters_last
        self.conv_in = CausalConv3d(latent_embed_dim, filters_last, kernel_size=3, padding=1, bias=True)
        self.res_blocks = nn.ModuleList(
            [
                OpenSoraStyleResBlock(
                    filters_last,
                    filters_last,
                    num_groups=self.num_groups,
                    activation_fn=activation_fn,
                )
                for _ in range(self.num_res_blocks)
            ]
        )

        self.block_res_blocks = nn.ModuleList([])
        self.conv_blocks = nn.ModuleList([])
        for i in reversed(range(self.num_blocks)):
            filters_i = self.filters * self.channel_multipliers[i]
            block_items = nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(
                    OpenSoraStyleResBlock(
                        prev_filters,
                        filters_i,
                        num_groups=self.num_groups,
                        activation_fn=activation_fn,
                    )
                )
                prev_filters = filters_i
            self.block_res_blocks.insert(0, block_items)

            if i > 0:
                if self.temporal_downsample[i - 1]:
                    self.conv_blocks.insert(
                        0,
                        CausalConv3d(
                            prev_filters,
                            prev_filters * 2,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    )
                else:
                    self.conv_blocks.insert(0, nn.Identity())

        self.norm_out = nn.GroupNorm(_group_count(self.num_groups, prev_filters), prev_filters)
        self.conv_out = CausalConv3d(prev_filters, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.res_blocks:
            x = block(x)

        for i in reversed(range(self.num_blocks)):
            for block in self.block_res_blocks[i]:
                x = block(x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                if t_stride > 1:
                    x = rearrange(x, "b (c ts) t h w -> b c (t ts) h w", ts=t_stride)

        x = self.norm_out(x)
        x = self.activate(x)
        return self.conv_out(x)


class OpenSoraStyleTemporalVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        dropout: float,
        resolution: int,
        z_channels: int,
        latent_channels: int,
        temporal_compression: int = 4,
        formulation: str = ContinuousFormulation.VAE.name,
        temporal_downsample: tuple[bool, ...] | list[bool] | None = None,
        num_groups: int = 32,
        activation_fn: str = "swish",
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        del dropout, resolution, ignore_kwargs
        self.latent_channels = latent_channels
        self.formulation = ContinuousFormulation[formulation]
        self.distribution = self.formulation.value()
        self.activation_fn = nn.SiLU if activation_fn == "swish" else nn.ReLU

        if temporal_downsample is None:
            num_temporal_downs = int(math.log2(temporal_compression))
            temporal_downsample = tuple(False for _ in range(len(channels_mult) - 1 - num_temporal_downs)) + tuple(
                True for _ in range(num_temporal_downs)
            )
        self.temporal_downsample = tuple(temporal_downsample)
        self.time_downsample_factor = 2 ** sum(self.temporal_downsample)
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = out_channels

        encoder_out_channels = 2 * z_channels if self.formulation is ContinuousFormulation.VAE else z_channels
        quant_out_channels = 2 * latent_channels if self.formulation is ContinuousFormulation.VAE else latent_channels

        self.encoder = OpenSoraStyleTemporalEncoder(
            in_channels=in_channels,
            latent_embed_dim=encoder_out_channels,
            filters=channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channels_mult,
            temporal_downsample=self.temporal_downsample,
            num_groups=num_groups,
            activation_fn=self.activation_fn,
        )
        self.quant_conv = CausalConv3d(encoder_out_channels, quant_out_channels, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(latent_channels, z_channels, kernel_size=1, padding=0)
        self.decoder = OpenSoraStyleTemporalDecoder(
            out_channels=out_channels,
            latent_embed_dim=z_channels,
            filters=channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channels_mult,
            temporal_downsample=self.temporal_downsample,
            num_groups=num_groups,
            activation_fn=self.activation_fn,
        )

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                latent_size.append(None)
            elif i == 0:
                time_padding = (
                    0
                    if (input_size[i] % self.time_downsample_factor == 0)
                    else self.time_downsample_factor - input_size[i] % self.time_downsample_factor
                )
                latent_size.append((input_size[i] + time_padding) // self.patch_size[i])
            else:
                latent_size.append(input_size[i] // self.patch_size[i])
        return latent_size

    def encode(self, x: torch.Tensor):
        time_padding = (
            0
            if (x.shape[2] % self.time_downsample_factor == 0)
            else self.time_downsample_factor - x.shape[2] % self.time_downsample_factor
        )
        x = _front_pad_temporal(x, time_padding)
        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature).to(x.dtype)
        return self.distribution(moments)

    def decode(self, z: torch.Tensor, num_frames: int | None = None):
        if num_frames is None:
            num_frames = z.shape[2] * self.time_downsample_factor
        time_padding = (
            0
            if (num_frames % self.time_downsample_factor == 0)
            else self.time_downsample_factor - num_frames % self.time_downsample_factor
        )
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        if time_padding > 0:
            x = x[:, :, time_padding:]
        return x


class LatentTemporalAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        dropout: float,
        resolution: int,
        z_channels: int,
        latent_channels: int,
        temporal_compression: int = 4,
        attn_resolutions: list[int] | None = None,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        attn_resolutions = attn_resolutions or []
        self.latent_channels = latent_channels
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks
        self.num_temporal_downs = int(math.log2(temporal_compression))
        if self.num_temporal_downs > self.num_resolutions - 1:
            raise ValueError(
                f"temporal_compression={temporal_compression} requires {self.num_temporal_downs} downsamples, "
                f"but only {self.num_resolutions - 1} are available."
            )

        self.conv_in = nn.Sequential(
            CausalConv3d(in_channels, channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(channels_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CausalHybridDownsample3d(
                    block_in,
                    spatial_down=False,
                    temporal_down=temporal_down,
                )
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, z_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(z_channels, z_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.quant_conv = CausalConv3d(z_channels, latent_channels, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(latent_channels, z_channels, kernel_size=1, padding=0)

        self.decoder_conv_in = nn.Sequential(
            CausalConv3d(z_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0),
        )
        self.decoder_mid = nn.Module()
        self.decoder_mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.decoder_mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.decoder_mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                temporal_up = i_level >= self.num_resolutions - self.num_temporal_downs
                up.upsample = CausalHybridUpsample3d(block_in, spatial_up=False, temporal_up=temporal_up)
            self.up.insert(0, up)

        self.decoder_norm_out = CausalNormalize(block_in, num_groups=1)
        self.decoder_conv_out = nn.Sequential(
            CausalConv3d(block_in, out_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_channels, out_channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def encode(self, x: torch.Tensor):
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        latent = self.quant_conv(h)
        zero = torch.zeros(1, device=latent.device, dtype=latent.dtype)
        return latent, (zero, zero)

    def decode(self, z: torch.Tensor, num_frames: int | None = None):
        h = self.post_quant_conv(z)
        h = self.decoder_conv_in(h)
        h = self.decoder_mid.block_1(h)
        h = self.decoder_mid.attn_1(h)
        h = self.decoder_mid.block_2(h)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.decoder_norm_out(h)
        h = nonlinearity(h)
        return self.decoder_conv_out(h)


class LatentTemporalCompressorVideoTokenizer(nn.Module):
    def __init__(
        self,
        image_tokenizer: dict[str, Any],
        temporal_compressor: dict[str, Any],
        frozen_image_tokenizer_ckpt: str,
        expected_input_frames: int = 29,
        expected_compressed_frames: int = 8,
        exact_context_frames: int = 1,
        trainable_image_tokenizer_modules: tuple[str, ...] | list[str] | None = None,
        trainable_image_tokenizer_lr_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.name = kwargs.get("name", "LTCV")
        self.expected_input_frames = expected_input_frames
        self.expected_compressed_frames = expected_compressed_frames
        self.exact_context_frames = exact_context_frames
        self.fixed_input_frames = expected_input_frames
        self.fixed_compressed_frames = expected_compressed_frames
        self.streaming_enabled = False
        self.streaming_require_full_chunks = False
        self.trainable_image_tokenizer_modules = tuple(trainable_image_tokenizer_modules or ())
        self.trainable_image_tokenizer_lr_scale = float(trainable_image_tokenizer_lr_scale)

        if self.exact_context_frames != 1:
            raise ValueError(f"{self.name} currently supports exactly one uncompressed context frame.")
        if self.expected_input_frames < 2:
            raise ValueError(f"{self.name} expects at least 2 frames, but got {self.expected_input_frames}.")
        if self.expected_compressed_frames < 2:
            raise ValueError(
                f"{self.name} expects at least 2 compressed frames (context + compressed future), "
                f"but got {self.expected_compressed_frames}."
            )

        self.image_tokenizer = ContinuousImageTokenizer(**image_tokenizer)
        self.temporal_compressor = self._build_temporal_compressor(temporal_compressor)
        self.latent_channels = getattr(self.temporal_compressor, "latent_channels", temporal_compressor["latent_channels"])

        self._load_frozen_image_tokenizer_weights(frozen_image_tokenizer_ckpt)
        self._configure_image_tokenizer_trainability(train_mode=False)

        num_total_parameters = sum(param.numel() for param in self.parameters())
        num_trainable_parameters = sum(param.numel() for param in self.parameters() if param.requires_grad)
        logging.info(
            f"model={self.name}, total_parameters={num_total_parameters:,}, "
            f"trainable_parameters={num_trainable_parameters:,}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Re-apply trainability every time the module switches mode, so nothing can
        # silently re-enable gradients on frozen modules between construction and
        # the first training step.
        self._configure_image_tokenizer_trainability(train_mode=mode)
        return self

    def trainable_parameters(self):
        return self.optimizer_parameter_groups()

    def optimizer_parameter_groups(self, base_lr: float | None = None):
        temporal_params = [param for param in self.temporal_compressor.parameters() if param.requires_grad]
        groups = []
        if temporal_params:
            groups.append({"params": temporal_params})

        image_params = self._trainable_image_tokenizer_parameters()
        if image_params:
            image_group = {"params": image_params}
            if base_lr is not None:
                image_group["lr"] = base_lr * self.trainable_image_tokenizer_lr_scale
            groups.append(image_group)
        return groups

    def frozen_state_dict_prefixes(self) -> tuple[str, ...]:
        # The frozen pieces of the 2D image tokenizer are re-hydrated from
        # frozen_image_tokenizer_ckpt at construction time, so they do not need
        # to be serialized into every training checkpoint.
        top_level_modules = ("encoder", "quant_conv", "post_quant_conv", "decoder")
        trainable_top_level = {module_name.split(".", 1)[0] for module_name in self.trainable_image_tokenizer_modules}
        return tuple(
            f"image_tokenizer.{module_name}."
            for module_name in top_level_modules
            if module_name not in trainable_top_level
        )

    def encoder_jit(self):
        raise RuntimeError(f"{self.name} only supports full-model .pt inference; encoder JIT export is disabled.")

    def decoder_jit(self):
        raise RuntimeError(f"{self.name} only supports full-model .pt inference; decoder JIT export is disabled.")

    def last_decoder_layer(self):
        return self.image_tokenizer.last_decoder_layer()

    def _load_frozen_image_tokenizer_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Unsupported frozen image tokenizer checkpoint format: {type(checkpoint)}")

        if any(key.startswith("network.") for key in checkpoint):
            checkpoint = {key.removeprefix("network."): value for key, value in checkpoint.items() if key.startswith("network.")}

        missing_keys, unexpected_keys = self.image_tokenizer.load_state_dict(checkpoint, strict=False)
        allowed_missing_prefixes = ("encoder.patcher", "decoder.unpatcher")
        disallowed_missing = [key for key in missing_keys if not key.startswith(allowed_missing_prefixes)]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                f"Failed to load frozen 2D tokenizer strictly enough. "
                f"Missing keys: {disallowed_missing}. Unexpected keys: {list(unexpected_keys)}"
            )

    def _resolve_image_tokenizer_module(self, module_name: str) -> nn.Module:
        module = self.image_tokenizer
        for name_part in module_name.split("."):
            if not hasattr(module, name_part):
                raise ValueError(f"{self.name} cannot resolve image_tokenizer module '{module_name}'.")
            module = getattr(module, name_part)
        if not isinstance(module, nn.Module):
            raise ValueError(f"{self.name} target '{module_name}' is not an nn.Module.")
        return module

    def _build_temporal_compressor(self, temporal_compressor: dict[str, Any]) -> nn.Module:
        temporal_config = dict(temporal_compressor)
        architecture = temporal_config.pop("architecture", "cosmos_factorized_ae")
        if architecture == "opensora_temporal_vae":
            return OpenSoraStyleTemporalVAE(**temporal_config)
        if architecture == "cosmos_factorized_ae":
            return LatentTemporalAutoencoder(**temporal_config)
        raise ValueError(f"{self.name} received unsupported temporal_compressor architecture '{architecture}'.")

    def _trainable_image_tokenizer_parameters(self) -> list[nn.Parameter]:
        trainable_parameters = []
        seen_parameter_ids = set()
        for module_name in self.trainable_image_tokenizer_modules:
            module = self._resolve_image_tokenizer_module(module_name)
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    continue
                parameter_id = id(parameter)
                if parameter_id in seen_parameter_ids:
                    continue
                seen_parameter_ids.add(parameter_id)
                trainable_parameters.append(parameter)
        return trainable_parameters

    def _configure_image_tokenizer_trainability(self, train_mode: bool) -> None:
        for module_name in ("encoder", "quant_conv", "post_quant_conv", "decoder"):
            module = getattr(self.image_tokenizer, module_name)
            module.requires_grad_(False)
            module.eval()

        for module_name in self.trainable_image_tokenizer_modules:
            module = self._resolve_image_tokenizer_module(module_name)
            module.requires_grad_(True)
            module.train(train_mode)

    def _validate_input_frames(self, tensor: torch.Tensor, expected_frames: int, tensor_name: str) -> None:
        if tensor.ndim != 5:
            raise ValueError(f"{tensor_name} must be 5D [B,C,T,H,W], but got shape {tuple(tensor.shape)}.")
        if tensor.shape[2] != expected_frames:
            raise ValueError(
                f"{self.name} expects {tensor_name} to contain exactly {expected_frames} frames, "
                f"but got {tensor.shape[2]}."
            )

    @staticmethod
    def _video_to_frame_batch(video: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, channels, num_frames, height, width = video.shape
        frame_batch = video.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, height, width)
        return frame_batch, batch_size, num_frames

    @staticmethod
    def _frame_batch_to_video(frame_batch: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
        channels, height, width = frame_batch.shape[1:]
        return frame_batch.reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)

    def _encode_video_to_target_latents(self, input_video: torch.Tensor) -> torch.Tensor:
        self._validate_input_frames(input_video, self.expected_input_frames, "input_video")
        frame_batch, batch_size, num_frames = self._video_to_frame_batch(input_video)
        with torch.no_grad():
            frame_latents, _ = self.image_tokenizer.encode(frame_batch)
            frame_latents = frame_latents.detach()
        return self._frame_batch_to_video(frame_latents, batch_size, num_frames)

    def _prepare_compressed_latent(
        self, target_latents: torch.Tensor, compressed_latent_from_encoder: torch.Tensor
    ) -> torch.Tensor:
        self._validate_input_frames(target_latents, self.expected_input_frames, "target_latents")
        self._validate_input_frames(
            compressed_latent_from_encoder,
            self.expected_compressed_frames,
            "compressed_latent_from_encoder",
        )
        exact_context_latent = target_latents[:, :, : self.exact_context_frames, :, :]
        compressed_future = compressed_latent_from_encoder[:, :, self.exact_context_frames :, :, :]
        return torch.cat([exact_context_latent, compressed_future], dim=2)

    def _decode_temporal_latents(
        self,
        compressed_latent: torch.Tensor,
        exact_context_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._validate_input_frames(compressed_latent, self.expected_compressed_frames, "compressed_latent")
        reconstructed_latents = self.temporal_compressor.decode(
            compressed_latent,
            num_frames=self.expected_input_frames,
        )
        reconstructed_latents = reconstructed_latents[:, :, : self.expected_input_frames, :, :]
        if exact_context_latent is None:
            # Backward-compatible fallback for callers that still pass a
            # bypass-prepared compressed latent whose frame-0 slot already
            # carries the exact 2D latent. The training forward() path now keeps
            # the decoder input on the temporal-compressor manifold and applies
            # the exact-context bypass only after decode.
            exact_context_latent = compressed_latent[:, :, : self.exact_context_frames, :, :]
        else:
            self._validate_input_frames(
                exact_context_latent, self.exact_context_frames, "exact_context_latent"
            )
        reconstructed_future = reconstructed_latents[:, :, self.exact_context_frames :, :, :]
        return torch.cat([exact_context_latent, reconstructed_future], dim=2)

    def _decode_frame_latents_to_video(self, reconstructed_latents: torch.Tensor) -> torch.Tensor:
        self._validate_input_frames(reconstructed_latents, self.expected_input_frames, "reconstructed_latents")
        frame_batch, batch_size, num_frames = self._video_to_frame_batch(reconstructed_latents)
        decoded_frame_batch = self.image_tokenizer.decode(frame_batch)
        return self._frame_batch_to_video(decoded_frame_batch, batch_size, num_frames)

    def encode(self, input_video: torch.Tensor):
        target_latents = self._encode_video_to_target_latents(input_video)
        compressed_latent_from_encoder, posteriors = self.temporal_compressor.encode(target_latents)
        compressed_latent = self._prepare_compressed_latent(target_latents, compressed_latent_from_encoder)
        return compressed_latent, posteriors

    def decode(
        self,
        compressed_latent: torch.Tensor,
        exact_context_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reconstructed_latents = self._decode_temporal_latents(compressed_latent, exact_context_latent)
        return self._decode_frame_latents_to_video(reconstructed_latents)

    def forward(self, input_video: torch.Tensor):
        target_latents = self._encode_video_to_target_latents(input_video)
        compressed_latent_from_encoder, posteriors = self.temporal_compressor.encode(target_latents)
        # Keep the decoder input on the temporal-VAE manifold; the exact-context
        # bypass is only re-injected after temporal decode.
        reconstructed_latents = self._decode_temporal_latents(
            compressed_latent_from_encoder,
            exact_context_latent=target_latents[:, :, : self.exact_context_frames, :, :],
        )
        reconstructions = self._decode_frame_latents_to_video(reconstructed_latents)

        output_dict = dict(
            reconstructions=reconstructions,
            posteriors=posteriors,
            latent=compressed_latent_from_encoder,
            compressed_latent=compressed_latent_from_encoder,
            reconstructed_latents=reconstructed_latents,
            target_latents=target_latents,
        )
        if self.training:
            return output_dict
        return NetworkEval(**output_dict)
