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

"""Regression tests for continuous tokenizer losses.

PYTHONPATH=$PWD pytest -v cosmos_predict1/tokenizer/training/losses/continuous_test.py
"""

from types import SimpleNamespace

import torch

from cosmos_predict1.tokenizer.training.datasets.utils import INPUT_KEY, MASK_KEY
from cosmos_predict1.tokenizer.training.losses.continuous import (
    RECONSTRUCTED_LATENTS_KEY,
    TARGET_LATENTS_KEY,
    LatentReconstructionLoss,
    TemporalDeltaLoss,
    TokenizerLoss,
)


class ConstantLoss(torch.nn.Module):
    def __init__(self, name: str, value: float) -> None:
        super().__init__()
        self.name = name
        self.value = float(value)

    def forward(self, inputs, output_batch, iteration):
        return {self.name: torch.tensor(self.value)}


def test_tokenizer_loss_does_not_double_count_last_module():
    config = SimpleNamespace(
        reduce="mean",
        color={
            "_target_": "cosmos_predict1.tokenizer.training.losses.continuous_test.ConstantLoss",
            "name": "color",
            "value": 1.0,
        },
        video_consistency={
            "_target_": "cosmos_predict1.tokenizer.training.losses.continuous_test.ConstantLoss",
            "name": "video_consistency",
            "value": 5.0,
        },
    )
    loss_module = TokenizerLoss(config)

    loss_dict, total_loss = loss_module(
        inputs={INPUT_KEY: torch.zeros(1, 3, 1, 2, 2)},
        output_batch={},
        iteration=0,
    )

    assert set(loss_dict["loss"].keys()) == {"color", "video_consistency"}
    assert torch.isclose(loss_dict["loss"]["color"], torch.tensor(1.0))
    assert torch.isclose(loss_dict["loss"]["video_consistency"], torch.tensor(5.0))
    assert torch.isclose(total_loss, torch.tensor(6.0))


class MaskEchoLoss(torch.nn.Module):
    def forward(self, inputs, output_batch, iteration):
        return {"mask_sum": inputs[MASK_KEY].sum()}


def test_tokenizer_loss_preserves_input_loss_mask():
    config = SimpleNamespace(
        reduce="mean",
        color={
            "_target_": "cosmos_predict1.tokenizer.training.losses.continuous_test.MaskEchoLoss",
        },
    )
    loss_module = TokenizerLoss(config)
    mask = torch.tensor([[[[1.0, 0.0], [0.5, 0.0]]]])

    loss_dict, total_loss = loss_module(
        inputs={
            INPUT_KEY: torch.zeros(1, 3, 2, 2),
            MASK_KEY: mask,
        },
        output_batch={},
        iteration=0,
    )

    expected = mask.sum()
    assert torch.isclose(loss_dict["loss"]["mask_sum"], expected)
    assert torch.isclose(total_loss, expected)


def test_tokenizer_loss_includes_latent_reconstruction_term():
    config = SimpleNamespace(
        reduce="mean",
        latent_recon={
            "_target_": "cosmos_predict1.tokenizer.training.losses.continuous.LatentReconstructionLoss",
            "config": SimpleNamespace(
                boundaries=[0],
                values=[1.0],
            ),
        },
    )
    loss_module = TokenizerLoss(config)
    reconstructed_latents = torch.tensor([[[[[1.0, 2.0]]]]])
    target_latents = torch.tensor([[[[[0.0, 1.0]]]]])

    loss_dict, total_loss = loss_module(
        inputs={
            INPUT_KEY: torch.zeros(1, 3, 1, 2, 2),
            MASK_KEY: torch.ones(1, 3, 1, 1, 2),
        },
        output_batch={
            RECONSTRUCTED_LATENTS_KEY: reconstructed_latents,
            TARGET_LATENTS_KEY: target_latents,
        },
        iteration=0,
    )

    expected = torch.abs(reconstructed_latents - target_latents).mean()
    assert torch.isclose(loss_dict["loss"]["latent_recon"], expected)
    assert torch.isclose(total_loss, expected)


def test_latent_reconstruction_loss_respects_loss_mask_after_resize():
    loss_module = LatentReconstructionLoss(SimpleNamespace(boundaries=[0], values=[1.0]))
    reconstructed_latents = torch.tensor([[[[[2.0, 4.0], [6.0, 8.0]]]]])
    target_latents = torch.zeros_like(reconstructed_latents)
    mask = torch.tensor(
        [
            [
                [
                    [[1.0, 1.0], [0.0, 0.0]],
                ]
            ]
        ]
    )

    loss_dict = loss_module(
        {
            INPUT_KEY: torch.zeros(1, 3, 1, 2, 2),
            MASK_KEY: mask,
        },
        {
            RECONSTRUCTED_LATENTS_KEY: reconstructed_latents,
            TARGET_LATENTS_KEY: target_latents,
        },
        iteration=0,
    )

    expected = torch.tensor([[[[[2.0, 4.0], [0.0, 0.0]]]]])
    torch.testing.assert_close(loss_dict["latent_recon"], expected)


def test_latent_reconstruction_loss_uses_conservative_mask_pooling_when_downsampling():
    loss_module = LatentReconstructionLoss(SimpleNamespace(boundaries=[0], values=[1.0]))
    reconstructed_latents = torch.tensor([[[[[3.0, 0.0], [0.0, 0.0]]]]])
    target_latents = torch.zeros_like(reconstructed_latents)
    mask = torch.zeros(1, 3, 1, 4, 4)
    mask[:, :, :, 1, 1] = 1.0

    loss_dict = loss_module(
        {
            INPUT_KEY: torch.zeros(1, 3, 1, 4, 4),
            MASK_KEY: mask,
        },
        {
            RECONSTRUCTED_LATENTS_KEY: reconstructed_latents,
            TARGET_LATENTS_KEY: target_latents,
        },
        iteration=0,
    )

    expected = torch.tensor([[[[[3.0, 0.0], [0.0, 0.0]]]]])
    torch.testing.assert_close(loss_dict["latent_recon"], expected)


def test_temporal_delta_loss_tracks_frame_differences_with_mask():
    loss_module = TemporalDeltaLoss(SimpleNamespace(enabled=True, boundaries=[0], values=[1.0]))
    inputs = {
        INPUT_KEY: torch.tensor([[[[[0.0]], [[2.0]], [[4.0]]]]]),
        MASK_KEY: torch.tensor([[[[[1.0]], [[1.0]], [[0.0]]]]]),
    }
    reconstructions = torch.tensor([[[[[0.0]], [[1.0]], [[5.0]]]]])

    loss_dict = loss_module(inputs, {"reconstructions": reconstructions}, iteration=0)

    # Frame deltas are [2, 2] for target and [1, 4] for recon.
    # With min-adjacent masking, only the first temporal delta remains active.
    expected = torch.tensor([[[[[1.0]], [[0.0]]]]])
    torch.testing.assert_close(loss_dict["temporal_delta"], expected)
