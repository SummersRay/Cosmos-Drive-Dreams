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

"""The model definition for 3D layers

Adapted from: https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L889

[MIT License Copyright (c) 2023 Phil Wang]
https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/LICENSE
"""
import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger as logging

from cosmos_predict1.tokenizer.modules.patching import Patcher, Patcher3D, UnPatcher, UnPatcher3D
from cosmos_predict1.tokenizer.modules.utils import (
    CausalNormalize,
    batch2space,
    batch2time,
    cast_tuple,
    is_odd,
    nonlinearity,
    replication_pad,
    space2batch,
    time2batch,
)

_LEGACY_NUM_GROUPS = 32


def _cache_key(module: nn.Module, kind: str) -> tuple[str, int]:
    return (kind, id(module))


def _get_cache(cache_state, module: nn.Module, kind: str, x: torch.Tensor):
    if cache_state is None:
        return None
    cached = cache_state.get(_cache_key(module, kind))
    if cached is None:
        return None
    return cached.to(device=x.device, dtype=x.dtype)


def _set_cache(cache_state, module: nn.Module, kind: str, value: torch.Tensor) -> None:
    if cache_state is not None:
        if cache_state.get("_detach_cache", False):
            value = value.detach()
        cache_state[_cache_key(module, kind)] = value


def _forward_with_cache(module: nn.Module, x: torch.Tensor, cache_state=None) -> torch.Tensor:
    if cache_state is None:
        return module(x)
    if isinstance(module, nn.Sequential):
        for layer in module:
            x = _forward_with_cache(layer, x, cache_state)
        return x
    if hasattr(module, "forward") and "cache_state" in module.forward.__code__.co_varnames:
        return module(x, cache_state=cache_state)
    return module(x)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        chan_out: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        time_stride = kwargs.pop("time_stride", 1)
        time_dilation = kwargs.pop("time_dilation", 1)
        padding = kwargs.pop("padding", 1)

        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = time_pad
        self.time_kernel_size = time_kernel_size

        self.spatial_pad = (padding, padding, padding, padding)

        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def _ensure_min_temporal_length(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] >= self.time_kernel_size:
            return x
        extra = self.time_kernel_size - x.shape[2]
        x_prev = x[:, :, :1, ...].repeat(1, 1, extra, 1, 1)
        return torch.cat([x_prev, x], dim=2)

    def _replication_pad(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        x = torch.cat([x_prev, x], dim=2)
        x = self._ensure_min_temporal_length(x)
        padding = self.spatial_pad + (0, 0)
        return F.pad(x, padding, mode=self.pad_mode, value=0.0)

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        raw_x = x
        if cache_state is not None and self.time_pad > 0:
            cached = _get_cache(cache_state, self, "conv", x)
            if cached is not None:
                x = torch.cat([cached, x], dim=2)
            remaining_pad = max(self.time_pad - (0 if cached is None else cached.shape[2]), 0)
            if remaining_pad > 0:
                x_prev = x[:, :, :1, ...].repeat(1, 1, remaining_pad, 1, 1)
                x = torch.cat([x_prev, x], dim=2)
            x = self._ensure_min_temporal_length(x)
            padding = self.spatial_pad + (0, 0)
            x = F.pad(x, padding, mode=self.pad_mode, value=0.0)
            cache_source = raw_x if cached is None else torch.cat([cached, raw_x], dim=2)
            _set_cache(cache_state, self, "conv", cache_source[:, :, -self.time_pad :, ...])
        else:
            x = self._replication_pad(x)
        return self.conv3d(x)


class CausalUpsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()
        x = x.repeat_interleave(int(time_factor), dim=2)
        x = self.conv(x, cache_state=cache_state)
        return x[..., int(time_factor - 1) :, :, :]


class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            time_stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        pad = (0, 1, 0, 1, 0, 0)
        x = F.pad(x, pad, mode="constant", value=0)
        x = replication_pad(x)
        x = self.conv(x, cache_state=cache_state)
        return x


class CausalHybridUpsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_up: bool = True,
        temporal_up: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=1, padding=0)
            if temporal_up
            else nn.Identity()
        )
        self.conv2 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=1, time_stride=1, padding=1)
            if spatial_up
            else nn.Identity()
        )
        self.conv3 = (
            CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, time_stride=1, padding=0)
            if spatial_up or temporal_up
            else nn.Identity()
        )
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        if not self.spatial_up and not self.temporal_up:
            return x

        # hybrid upsample temporally.
        if self.temporal_up:
            if cache_state is not None:
                seen_key = _cache_key(self, "temporal_up")
                if seen_key in cache_state:
                    x = x.repeat_interleave(2, dim=2)
                else:
                    cache_state[seen_key] = True
                x = self.conv1(x, cache_state=cache_state) + x
            else:
                time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                x = x.repeat_interleave(int(time_factor), dim=2)
                x = x[..., int(time_factor - 1) :, :, :]
                x = self.conv1(x) + x

        # hybrid upsample spatially.
        if self.spatial_up:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.conv2(x, cache_state=cache_state) + x

        # final 1x1x1 conv.
        x = self.conv3(x, cache_state=cache_state)
        return x


class CausalHybridDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_down: bool = True,
        temporal_down: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=2, time_stride=1, padding=0)
            if spatial_down
            else nn.Identity()
        )
        self.conv2 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=2, padding=0)
            if temporal_down
            else nn.Identity()
        )
        self.conv3 = (
            CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, time_stride=1, padding=0)
            if spatial_down or temporal_down
            else nn.Identity()
        )

        self.spatial_down = spatial_down
        self.temporal_down = temporal_down

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        if not self.spatial_down and not self.temporal_down:
            return x

        # hybrid downsample spatially.
        if self.spatial_down:
            pad = (0, 1, 0, 1, 0, 0)
            x = F.pad(x, pad, mode="constant", value=0)
            x1 = self.conv1(x, cache_state=cache_state)
            x2 = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            x = x1 + x2

        # hybrid downsample temporally.
        if self.temporal_down:
            if cache_state is None:
                pooled_input = replication_pad(x)
                x1 = self.conv2(pooled_input)
                x2 = F.avg_pool3d(pooled_input, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            else:
                conv_cache = _get_cache(cache_state, self.conv2, "conv", x)
                if conv_cache is None and x.shape[2] == 1:
                    x1 = self.conv2(replication_pad(x))
                else:
                    x1 = self.conv2(x, cache_state=cache_state)
                pooled_input = replication_pad(x) if x.shape[2] == 1 else x
                x2 = F.avg_pool3d(pooled_input, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            x = x1 + x2

        # final 1x1x1 conv.
        x = self.conv3(x, cache_state=cache_state)
        return x


class CausalResnetBlock3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=num_groups)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h, cache_state=cache_state)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, cache_state=cache_state)
        x = _forward_with_cache(self.nin_shortcut, x, cache_state)

        return x + h


class CausalResnetBlockFactorized3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=1)
        self.conv1 = nn.Sequential(
            CausalConv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = _forward_with_cache(self.conv1, h, cache_state)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = _forward_with_cache(self.conv2, h, cache_state)
        x = _forward_with_cache(self.nin_shortcut, x, cache_state)

        return x + h


class CausalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size = time2batch(q)
        k, batch_size = time2batch(k)
        v, batch_size = time2batch(v)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = batch2time(h_, batch_size)
        h_ = self.proj_out(h_)
        return x + h_


class CausalTemporalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cache_state=None) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        use_history = cache_state is not None and not cache_state.get("_disable_temporal_attn_cache", False)
        history = _get_cache(cache_state, self, "temporal_attn", h_) if use_history else None
        if history is not None:
            h_context = torch.cat([history, h_], dim=2)
        else:
            h_context = h_
        if use_history:
            max_history = cache_state.get("_temporal_attn_max_history", -1)
            if isinstance(max_history, int) and max_history > 0 and h_context.shape[2] > max_history:
                h_context = h_context[:, :, -max_history:, ...]
            _set_cache(cache_state, self, "temporal_attn", h_context)
        q = self.q(h_)
        k = self.k(h_context)
        v = self.v(h_context)

        # compute attention
        q, batch_size, height = space2batch(q)
        k, _, _ = space2batch(k)
        v, _, _ = space2batch(v)

        bhw, c, t_query = q.shape
        t_total = k.shape[-1]
        q = q.permute(0, 2, 1)  # (bhw, t, c)
        k = k.permute(0, 2, 1)  # (bhw, t, c)
        v = v.permute(0, 2, 1)  # (bhw, t, c)

        w_ = torch.bmm(q, k.permute(0, 2, 1))  # (bhw, t_query, t_total)
        w_ = w_ * (int(c) ** (-0.5))

        # Apply causal mask
        if cache_state is None or history is None:
            mask = torch.tril(torch.ones_like(w_))
        else:
            past_len = t_total - t_query
            query_idx = torch.arange(t_query, device=w_.device).view(1, t_query, 1)
            key_idx = torch.arange(t_total, device=w_.device).view(1, 1, t_total)
            mask = (key_idx <= (past_len + query_idx)).to(w_.dtype)
        w_ = w_.masked_fill(mask == 0, float("-inf"))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        h_ = torch.bmm(w_, v)  # (bhw, t, c)
        h_ = h_.permute(0, 2, 1).reshape(bhw, c, t_query)  # (bhw, c, t)

        h_ = batch2space(h_, batch_size, height)
        h_ = self.proj_out(h_)
        return x + h_


class EncoderBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher = Patcher(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        in_channels = in_channels * patch_size * patch_size

        # downsampling
        self.conv_in = CausalConv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        # num of groups for GroupNorm, num_groups=1 for LayerNorm.
        num_groups = ignore_kwargs.get("num_groups", _LEGACY_NUM_GROUPS)
        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlock3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock(block_in, num_groups=num_groups))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = CausalDownsample3d(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.mid.attn_1 = CausalAttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=num_groups)
        self.conv_out = CausalConv3d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def patcher3d(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = time2batch(x)
        x = self.patcher(x)
        x = batch2time(x, batch_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher3d(x)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            else:
                # temporal downsample (last level)
                time_factor = 1 + 1 * (hs[-1].shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                hs[-1] = replication_pad(hs[-1])
                hs.append(
                    F.avg_pool3d(
                        hs[-1],
                        kernel_size=[time_factor, 1, 1],
                        stride=[2, 1, 1],
                    )
                )

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def forward_streaming_chunk(self, x: torch.Tensor, cache_state, is_first_chunk: bool) -> torch.Tensor:
        x = self.patcher3d.forward_streaming(x, is_first_chunk=is_first_chunk)

        hs = [_forward_with_cache(self.conv_in, x, cache_state)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], cache_state=cache_state)
                if len(self.down[i_level].attn) > 0:
                    h = _forward_with_cache(self.down[i_level].attn[i_block], h, cache_state)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], cache_state=cache_state))

        h = hs[-1]
        h = self.mid.block_1(h, cache_state=cache_state)
        h = _forward_with_cache(self.mid.attn_1, h, cache_state)
        h = self.mid.block_2(h, cache_state=cache_state)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = _forward_with_cache(self.conv_out, h, cache_state)
        return h


class DecoderBase(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher = UnPatcher(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        out_ch = out_channels * patch_size * patch_size

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.info("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # num of groups for GroupNorm, num_groups=1 for LayerNorm.
        num_groups = ignore_kwargs.get("num_groups", _LEGACY_NUM_GROUPS)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.mid.attn_1 = CausalAttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlock3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock(block_in, num_groups=num_groups))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = CausalUpsample3d(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=num_groups)
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def unpatcher3d(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = time2batch(x)
        x = self.unpatcher(x)
        x = batch2time(x, batch_size)

        return x

    def forward(self, z):
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            else:
                # temporal upsample (last level)
                time_factor = 1.0 + 1.0 * (h.shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                h = h.repeat_interleave(int(time_factor), dim=2)
                h = h[..., int(time_factor - 1) :, :, :]

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher3d(h)
        return h

    def forward_streaming_chunk(self, z: torch.Tensor, cache_state, is_first_chunk: bool) -> torch.Tensor:
        h = _forward_with_cache(self.conv_in, z, cache_state)

        h = self.mid.block_1(h, cache_state=cache_state)
        h = _forward_with_cache(self.mid.attn_1, h, cache_state)
        h = self.mid.block_2(h, cache_state=cache_state)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, cache_state=cache_state)
                if len(self.up[i_level].attn) > 0:
                    h = _forward_with_cache(self.up[i_level].attn[i_block], h, cache_state)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_state=cache_state)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = _forward_with_cache(self.conv_out, h, cache_state)
        return self.unpatcher3d.forward_streaming(h, is_first_chunk=is_first_chunk)


class EncoderFactorized(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher3d = Patcher3D(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        in_channels = in_channels * patch_size * patch_size * patch_size

        # calculate the number of downsample operations
        self.num_spatial_downs = int(math.log2(spatial_compression)) - int(math.log2(patch_size))
        assert (
            self.num_spatial_downs <= self.num_resolutions
        ), f"Spatially downsample {self.num_resolutions} times at most"

        self.num_temporal_downs = int(math.log2(temporal_compression)) - int(math.log2(patch_size))
        assert (
            self.num_temporal_downs <= self.num_resolutions
        ), f"Temporally downsample {self.num_resolutions} times at most"

        # downsampling
        self.conv_in = nn.Sequential(
            CausalConv3d(
                in_channels,
                channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
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
                spatial_down = i_level < self.num_spatial_downs
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CausalHybridDownsample3d(
                    block_in,
                    spatial_down=spatial_down,
                    temporal_down=temporal_down,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
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

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, z_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(
                z_channels,
                z_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher3d(x)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def forward_streaming_chunk(self, x: torch.Tensor, cache_state, is_first_chunk: bool) -> torch.Tensor:
        x = self.patcher3d.forward_streaming(x, is_first_chunk=is_first_chunk)

        hs = [_forward_with_cache(self.conv_in, x, cache_state)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], cache_state=cache_state)
                if len(self.down[i_level].attn) > 0:
                    h = _forward_with_cache(self.down[i_level].attn[i_block], h, cache_state)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1], cache_state=cache_state))

        h = hs[-1]
        h = self.mid.block_1(h, cache_state=cache_state)
        h = _forward_with_cache(self.mid.attn_1, h, cache_state)
        h = self.mid.block_2(h, cache_state=cache_state)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = _forward_with_cache(self.conv_out, h, cache_state)
        return h


class DecoderFactorized(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher3d = UnPatcher3D(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        out_ch = out_channels * patch_size * patch_size * patch_size

        # calculate the number of upsample operations
        self.num_spatial_ups = int(math.log2(spatial_compression)) - int(math.log2(patch_size))
        assert self.num_spatial_ups <= self.num_resolutions, f"Spatially upsample {self.num_resolutions} times at most"
        self.num_temporal_ups = int(math.log2(temporal_compression)) - int(math.log2(patch_size))
        assert (
            self.num_temporal_ups <= self.num_resolutions
        ), f"Temporally upsample {self.num_resolutions} times at most"

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.info("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Sequential(
            CausalConv3d(z_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        # middle
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

        legacy_mode = ignore_kwargs.get("legacy_mode", False)
        # upsampling
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
                # The layer index for temporal/spatial downsampling performed
                # in the encoder should correspond to the layer index in
                # reverse order where upsampling is performed in the decoder.
                # If you've a pre-trained model, you can simply finetune.
                i_level_reverse = self.num_resolutions - i_level - 1
                if legacy_mode:
                    temporal_up = i_level_reverse < self.num_temporal_ups
                else:
                    temporal_up = 0 < i_level_reverse < self.num_temporal_ups + 1
                spatial_up = temporal_up or (
                    i_level_reverse < self.num_spatial_ups and self.num_spatial_ups > self.num_temporal_ups
                )
                up.upsample = CausalHybridUpsample3d(block_in, spatial_up=spatial_up, temporal_up=temporal_up)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, out_ch, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def forward(self, z):
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher3d(h)
        return h

    def forward_streaming_chunk(self, z: torch.Tensor, cache_state, is_first_chunk: bool) -> torch.Tensor:
        h = _forward_with_cache(self.conv_in, z, cache_state)

        h = self.mid.block_1(h, cache_state=cache_state)
        h = _forward_with_cache(self.mid.attn_1, h, cache_state)
        h = self.mid.block_2(h, cache_state=cache_state)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, cache_state=cache_state)
                if len(self.up[i_level].attn) > 0:
                    h = _forward_with_cache(self.up[i_level].attn[i_block], h, cache_state)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_state=cache_state)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = _forward_with_cache(self.conv_out, h, cache_state)
        return self.unpatcher3d.forward_streaming(h, is_first_chunk=is_first_chunk)
