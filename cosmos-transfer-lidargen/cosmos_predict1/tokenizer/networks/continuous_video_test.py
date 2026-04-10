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

"""Lightweight tests for streaming chunk alignment helpers.

PYTHONPATH=$PWD pytest -v cosmos_predict1/tokenizer/networks/continuous_video_test.py
"""

import pytest

from cosmos_predict1.tokenizer.networks.continuous_video import validate_streaming_chunk_alignment


@pytest.mark.parametrize("total_frames", [1, 5, 9, 17, 29])
def test_validate_streaming_chunk_alignment_accepts_one_plus_multiple_of_four(total_frames):
    validate_streaming_chunk_alignment(total_frames, 4, sequence_name="input frames")


@pytest.mark.parametrize("total_frames", [2, 6, 15, 18, 30])
def test_validate_streaming_chunk_alignment_rejects_ragged_lengths(total_frames):
    with pytest.raises(ValueError, match=r"1 \+ n\*4"):
        validate_streaming_chunk_alignment(total_frames, 4, sequence_name="input frames")
