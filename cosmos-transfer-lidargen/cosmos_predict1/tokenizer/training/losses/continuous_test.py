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
from cosmos_predict1.tokenizer.training.losses.continuous import TokenizerLoss


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
