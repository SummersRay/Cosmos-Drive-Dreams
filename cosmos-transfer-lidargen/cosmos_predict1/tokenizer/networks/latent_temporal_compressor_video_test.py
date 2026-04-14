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

"""Tests for frozen 2D + latent temporal compressor tokenizer.

PYTHONPATH=$PWD pytest -v cosmos_predict1/tokenizer/networks/latent_temporal_compressor_video_test.py
"""

from pathlib import Path

import torch

from cosmos_predict1.tokenizer.networks.continuous_image import ContinuousImageTokenizer
from cosmos_predict1.tokenizer.networks.latent_temporal_compressor_video import (
    LatentTemporalAutoencoder,
    LatentTemporalCompressorVideoTokenizer,
)


def _tiny_image_tokenizer_config():
    return dict(
        attn_resolutions=[],
        channels=8,
        channels_mult=[1, 1],
        dropout=0.0,
        in_channels=3,
        spatial_compression=8,
        num_res_blocks=1,
        out_channels=3,
        resolution=32,
        patch_size=2,
        patch_method="haar",
        latent_channels=16,
        z_channels=16,
        z_factor=1,
        num_groups=1,
        name="ContinuousImageTokenizer",
        formulation="AE",
        encoder="Default",
        decoder="Default",
    )


def _tiny_temporal_compressor_config(resolution: int):
    return dict(
        attn_resolutions=[],
        channels=8,
        channels_mult=[1, 1, 1],
        dropout=0.0,
        in_channels=16,
        num_res_blocks=1,
        out_channels=16,
        resolution=resolution,
        patch_size=1,
        patch_method="haar",
        latent_channels=16,
        z_channels=16,
        z_factor=1,
        num_groups=1,
        legacy_mode=False,
        spatial_compression=1,
        temporal_compression=4,
        formulation="AE",
        encoder="FACTORIZED",
        decoder="FACTORIZED",
        name="LatentTemporalCompressorAE",
    )


def _write_image_tokenizer_checkpoint(tmp_path: Path, image_config: dict) -> tuple[Path, ContinuousImageTokenizer]:
    tokenizer = ContinuousImageTokenizer(**image_config)
    checkpoint_path = tmp_path / "ci8x8_waymo_mock.pt"
    torch.save({"model": {f"network.{key}": value for key, value in tokenizer.state_dict().items()}}, checkpoint_path)
    return checkpoint_path, tokenizer


def test_temporal_compressor_matches_29_to_8_to_29_latent_shapes():
    temporal_model = LatentTemporalAutoencoder(**_tiny_temporal_compressor_config(resolution=64))
    latent_video = torch.randn(1, 16, 29, 64, 112)

    with torch.no_grad():
        compressed_latent, _ = temporal_model.encode(latent_video)
        reconstructed_latents = temporal_model.decode(compressed_latent)

    assert compressed_latent.shape == (1, 16, 8, 64, 112)
    assert reconstructed_latents.shape == (1, 16, 29, 64, 112)


def test_latent_temporal_compressor_loads_frozen_2d_tokenizer_exactly(tmp_path: Path):
    image_config = _tiny_image_tokenizer_config()
    checkpoint_path, source_tokenizer = _write_image_tokenizer_checkpoint(tmp_path, image_config)

    model = LatentTemporalCompressorVideoTokenizer(
        image_tokenizer=image_config,
        temporal_compressor=_tiny_temporal_compressor_config(resolution=4),
        frozen_image_tokenizer_ckpt=str(checkpoint_path),
        expected_input_frames=29,
        expected_compressed_frames=8,
        exact_context_frames=1,
    )

    assert all(not parameter.requires_grad for parameter in model.image_tokenizer.parameters())
    for key, value in source_tokenizer.state_dict().items():
        torch.testing.assert_close(model.image_tokenizer.state_dict()[key], value)


def test_latent_temporal_compressor_only_backprops_through_temporal_model(tmp_path: Path):
    image_config = _tiny_image_tokenizer_config()
    checkpoint_path, _ = _write_image_tokenizer_checkpoint(tmp_path, image_config)

    model = LatentTemporalCompressorVideoTokenizer(
        image_tokenizer=image_config,
        temporal_compressor=_tiny_temporal_compressor_config(resolution=4),
        frozen_image_tokenizer_ckpt=str(checkpoint_path),
        expected_input_frames=29,
        expected_compressed_frames=8,
        exact_context_frames=1,
    )
    model.train()

    input_video = torch.randn(1, 3, 29, 32, 32)
    output_batch = model(input_video)
    loss = output_batch["reconstructions"].sum()
    loss.backward()

    assert output_batch["target_latents"].shape == (1, 16, 29, 4, 4)
    assert output_batch["compressed_latent"].shape == (1, 16, 8, 4, 4)
    assert output_batch["reconstructed_latents"].shape == (1, 16, 29, 4, 4)
    assert output_batch["reconstructions"].shape == (1, 3, 29, 32, 32)
    torch.testing.assert_close(
        output_batch["reconstructed_latents"][:, :, :1],
        output_batch["target_latents"][:, :, :1],
    )
    assert any(parameter.grad is not None for parameter in model.temporal_compressor.parameters())
    assert all(parameter.grad is None for parameter in model.image_tokenizer.parameters())


def test_latent_temporal_compressor_can_finetune_decoder_modules_only(tmp_path: Path):
    image_config = _tiny_image_tokenizer_config()
    checkpoint_path, _ = _write_image_tokenizer_checkpoint(tmp_path, image_config)

    model = LatentTemporalCompressorVideoTokenizer(
        image_tokenizer=image_config,
        temporal_compressor=_tiny_temporal_compressor_config(resolution=4),
        frozen_image_tokenizer_ckpt=str(checkpoint_path),
        expected_input_frames=29,
        expected_compressed_frames=8,
        exact_context_frames=1,
        trainable_image_tokenizer_modules=("post_quant_conv", "decoder"),
        trainable_image_tokenizer_lr_scale=0.2,
    )
    model.train()

    input_video = torch.randn(1, 3, 29, 32, 32)
    output_batch = model(input_video)
    loss = output_batch["reconstructions"].sum()
    loss.backward()

    assert all(parameter.grad is None for parameter in model.image_tokenizer.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.image_tokenizer.quant_conv.parameters())
    assert any(parameter.grad is not None for parameter in model.image_tokenizer.post_quant_conv.parameters())
    assert any(parameter.grad is not None for parameter in model.image_tokenizer.decoder.parameters())


def test_latent_temporal_compressor_optimizer_groups_apply_lr_scale(tmp_path: Path):
    image_config = _tiny_image_tokenizer_config()
    checkpoint_path, _ = _write_image_tokenizer_checkpoint(tmp_path, image_config)

    model = LatentTemporalCompressorVideoTokenizer(
        image_tokenizer=image_config,
        temporal_compressor=_tiny_temporal_compressor_config(resolution=4),
        frozen_image_tokenizer_ckpt=str(checkpoint_path),
        expected_input_frames=29,
        expected_compressed_frames=8,
        exact_context_frames=1,
        trainable_image_tokenizer_modules=("post_quant_conv", "decoder"),
        trainable_image_tokenizer_lr_scale=0.2,
    )

    param_groups = model.optimizer_parameter_groups(base_lr=5e-6)
    assert len(param_groups) == 2
    assert "lr" not in param_groups[0]
    assert abs(param_groups[1]["lr"] - 1e-6) < 1e-12
    assert len(param_groups[0]["params"]) > 0
    assert len(param_groups[1]["params"]) > 0
