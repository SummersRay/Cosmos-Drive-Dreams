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

"""The causal continuous video tokenizer with VAE or AE formulation for 3D data.."""
from collections import OrderedDict, namedtuple

from loguru import logger as logging
import torch
from torch import nn

from cosmos_predict1.tokenizer.modules import ContinuousFormulation, Decoder3DType, Encoder3DType
from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "posteriors", "latent"])


def validate_streaming_chunk_alignment(total_frames: int, chunk_size: int, sequence_name: str = "frames") -> None:
    if total_frames < 1:
        raise ValueError(f"{sequence_name} must contain at least 1 frame, but got {total_frames}.")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, but got {chunk_size}.")
    if (total_frames - 1) % chunk_size != 0:
        raise ValueError(
            f"{sequence_name} must satisfy 1 + n*{chunk_size} for strict streaming chunking, but got {total_frames}."
        )


class CausalContinuousVideoTokenizer(nn.Module):
    def __init__(self, z_channels: int, z_factor: int, latent_channels: int, **kwargs) -> None:
        super().__init__()
        self.name = kwargs.get("name", "CausalContinuousVideoTokenizer")
        self.latent_channels = latent_channels
        self.streaming_enabled = kwargs.get("streaming_enabled", False)
        self.streaming_raw_chunk_size = kwargs.get("streaming_raw_chunk_size", kwargs.get("temporal_compression", 1))
        self.streaming_latent_chunk_size = kwargs.get("streaming_latent_chunk_size", 1)
        self.streaming_attention_max_history = kwargs.get("streaming_attention_max_history", -1)
        self.streaming_detach_cache = kwargs.get("streaming_detach_cache", True)
        self.streaming_disable_temporal_attn_cache = kwargs.get("streaming_disable_temporal_attn_cache", False)
        self.streaming_train_use_full_path = kwargs.get("streaming_train_use_full_path", False)
        self.streaming_require_full_chunks = kwargs.get("streaming_require_full_chunks", False)

        encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)
        decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)

        self.quant_conv = CausalConv3d(
            z_factor * z_channels,
            z_factor * latent_channels,
            kernel_size=1,
            padding=0,
        )
        self.post_quant_conv = CausalConv3d(latent_channels, z_channels, kernel_size=1, padding=0)

        formulation_name = kwargs.get("formulation", ContinuousFormulation.AE.name)
        self.distribution = ContinuousFormulation[formulation_name].value()
        logging.info(f"{self.name} based on {formulation_name} formulation, with {kwargs}.")

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, latent_channels={self.latent_channels}.")

    def encoder_jit(self):
        if self.streaming_enabled:
            raise RuntimeError("Streaming video tokenizer does not export fixed-shape encoder JIT. Disable JIT for this experiment.")
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("distribution", self.distribution),
                ]
            )
        )

    def decoder_jit(self):
        if self.streaming_enabled:
            raise RuntimeError("Streaming video tokenizer does not export fixed-shape decoder JIT. Disable JIT for this experiment.")
        return nn.Sequential(
            OrderedDict(
                [
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def last_decoder_layer(self):
        return self.decoder.conv_out

    def _iter_raw_chunks(self, x: torch.Tensor):
        total_frames = x.shape[2]
        if self.streaming_require_full_chunks:
            validate_streaming_chunk_alignment(total_frames, self.streaming_raw_chunk_size, sequence_name="input frames")
        yield True, x[:, :, :1, :, :]
        for start in range(1, total_frames, self.streaming_raw_chunk_size):
            end = min(start + self.streaming_raw_chunk_size, total_frames)
            yield False, x[:, :, start:end, :, :]

    def _iter_latent_chunks(self, z: torch.Tensor):
        total_frames = z.shape[2]
        if self.streaming_require_full_chunks:
            validate_streaming_chunk_alignment(
                total_frames, self.streaming_latent_chunk_size, sequence_name="latent frames"
            )
        yield True, z[:, :, :1, :, :]
        for start in range(1, total_frames, self.streaming_latent_chunk_size):
            end = min(start + self.streaming_latent_chunk_size, total_frames)
            yield False, z[:, :, start:end, :, :]

    def _use_streaming_path(self) -> bool:
        if not self.streaming_enabled:
            return False
        if self.training and self.streaming_train_use_full_path:
            return False
        return True

    def _encode_full(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return self.distribution(moments)

    def _decode_full(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def encode_streaming(self, x):
        if not hasattr(self.encoder, "forward_streaming_chunk"):
            raise RuntimeError("The selected encoder does not support streaming.")
        cache_state = {
            "_temporal_attn_max_history": self.streaming_attention_max_history,
            "_detach_cache": self.training and self.streaming_detach_cache,
            "_disable_temporal_attn_cache": self.streaming_disable_temporal_attn_cache,
        }
        hidden_chunks = []
        for is_first_chunk, input_chunk in self._iter_raw_chunks(x):
            hidden_chunks.append(self.encoder.forward_streaming_chunk(input_chunk, cache_state, is_first_chunk))
        h = torch.cat(hidden_chunks, dim=2)
        moments = self.quant_conv(h)
        return self.distribution(moments)

    def decode_streaming(self, z):
        if not hasattr(self.decoder, "forward_streaming_chunk"):
            raise RuntimeError("The selected decoder does not support streaming.")
        z = self.post_quant_conv(z)
        cache_state = {
            "_temporal_attn_max_history": self.streaming_attention_max_history,
            "_detach_cache": self.training and self.streaming_detach_cache,
            "_disable_temporal_attn_cache": self.streaming_disable_temporal_attn_cache,
        }
        decoded_chunks = []
        for is_first_chunk, latent_chunk in self._iter_latent_chunks(z):
            decoded_chunks.append(self.decoder.forward_streaming_chunk(latent_chunk, cache_state, is_first_chunk))
        return torch.cat(decoded_chunks, dim=2)

    def encode(self, x):
        if self._use_streaming_path():
            return self.encode_streaming(x)
        return self._encode_full(x)

    def decode(self, z):
        if self._use_streaming_path():
            return self.decode_streaming(z)
        return self._decode_full(z)

    def forward(self, input):
        latent, posteriors = self.encode(input)
        reconstructions = self.decode(latent)
        if self._use_streaming_path():
            reconstructions = reconstructions[:, :, : input.shape[2], :, :]
        if self.training:
            return dict(
                reconstructions=reconstructions,
                posteriors=posteriors,
                latent=latent,
            )
        return NetworkEval(
            reconstructions=reconstructions,
            posteriors=posteriors,
            latent=latent,
        )
