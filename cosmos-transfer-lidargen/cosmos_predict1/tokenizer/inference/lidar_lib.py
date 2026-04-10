# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
from tqdm import tqdm

from cosmos_predict1.tokenizer.inference.image_lib import ImageTokenizer
from cosmos_predict1.tokenizer.inference.utils import pad_image_batch, pad_video_batch, unpad_image_batch, unpad_video_batch
from cosmos_predict1.tokenizer.inference.video_lib import CausalVideoTokenizer
from cosmos_predict1.utils.lidar_rangemap import normalize_range_map, unnormalize_range_map, RangeMapDownsampler, range_map_to_ray_directions, load_pandar128_elevations


class LidarTokenizer(ImageTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    @torch.no_grad()
    def autoencode(self, image: np.ndarray) -> np.ndarray:
        """Reconstrcuts a numpy image after embedding into a latent first.

        Args:
            image: The input image BxHxWx3 layout, range [0..255].
        Returns:
            The reconstructed image, layout BxHxWx3, range [0..255].
        """
        input_array = image.transpose(0, 3, 1, 2)
        input_tensor = torch.from_numpy(input_array).to(self._dtype).to(self._device)
        
        if self._full_model is not None:
            output_tensor = self._full_model(input_tensor)
            output_tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)

        padded_output_image = output_tensor.float().cpu().numpy()
        padded_output_image = padded_output_image.transpose(0, 2, 3, 1)

        return padded_output_image

    
    @torch.no_grad()
    def forward(self, image: np.ndarray) -> np.ndarray:
        """Reconstructs an image using a pre-trained tokenizer.

        Args:
            image: The input image BxHxWxC layout, range [0..255].
        Returns:
            The reconstructed image in range [0..255], layout BxHxWxC.
        """
        padded_image, crop_region = pad_image_batch(image)
        padded_output_image = self.autoencode(padded_image)
        return unpad_image_batch(padded_output_image, crop_region)

class LidarProcessor:
    def __init__(
        self,
        checkpoint=None,
        checkpoint_enc=None,
        checkpoint_dec=None,
        tokenizer_config=None,
        dtype="bfloat16",
        device="cuda",
        min_range=0.75,
        max_range=75,
        n_rows_repeat=4,
        n_cols_repeat=1,
        downsample_factor_row=1,
        downsample_factor_col=2,
        downsample_method="scatter_min",
        elevation_angles=None,
        tokenizer_type="image",
        temporal_window=9,
        use_standalone_context_frame=True,
    ):
        """Initialize the LidarProcessor.

        Args:
            checkpoint: Path to full autoencoder JIT model
            checkpoint_enc: Path to encoder JIT model
            checkpoint_dec: Path to decoder JIT model
            mode: Backend mode ('torch' or 'jit')
            short_size: Size to resample inputs
            dtype: Precision type
            device: Device for model execution
            max_range: Maximum range value
            n_rows_repeat: Number of times to repeat each row
            n_cols_repeat: Number of times to repeat each column
            inverse_depth: Whether to process depth as inverse depth
            log_space: Whether to normalize in log space
            dynamic_range: Whether to dynamically adjust the range
            stack_frames: Whether to stack frames
        """
        self.min_range = min_range
        self.max_range = max_range
        self.n_rows_repeat = n_rows_repeat
        self.n_cols_repeat = n_cols_repeat
        self.downsample_factor_row = downsample_factor_row
        self.downsample_factor_col = downsample_factor_col
        self.downsample_method = downsample_method
        self.max_value = 1
        self.min_value = -1
        self.tokenizer_type = tokenizer_type.lower()
        self.temporal_window = temporal_window
        self.use_standalone_context_frame = use_standalone_context_frame
        if self.temporal_window < 1:
            raise ValueError(f"temporal_window must be >= 1, but got {self.temporal_window}")
        if self.tokenizer_type == "video" and self.temporal_window < 2:
            raise ValueError(
                f"temporal_window must be >= 2 for video tokenizers because each window should contain "
                f"at least one context frame and one new frame, but got {self.temporal_window}"
            )

        if checkpoint is None and checkpoint_enc is None and checkpoint_dec is None:
            raise ValueError("checkpoint or checkpoint_enc/checkpoint_dec must be provided")
        
        self.elevation_angles = elevation_angles
        self.range_map_downsampler = RangeMapDownsampler(
            row_factor=downsample_factor_row,
            col_factor=downsample_factor_col,
            method=downsample_method,
        )

        # Initialize the autoencoder
        if self.tokenizer_type == "image":
            self.autoencoder = LidarTokenizer(
                checkpoint=checkpoint,
                checkpoint_enc=checkpoint_enc,
                checkpoint_dec=checkpoint_dec,
                tokenizer_config=tokenizer_config,
                device=device,
                dtype=dtype,
            )
        elif self.tokenizer_type == "video":
            self.autoencoder = CausalVideoTokenizer(
                checkpoint=checkpoint,
                checkpoint_enc=checkpoint_enc,
                checkpoint_dec=checkpoint_dec,
                tokenizer_config=tokenizer_config,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unsupported tokenizer_type: {self.tokenizer_type}")
        self._validate_video_frame_count(self.temporal_window, "temporal_window")

    def _uses_streaming_full_model(self) -> bool:
        full_model = getattr(self.autoencoder, "_full_model", None)
        return self.tokenizer_type == "video" and full_model is not None and getattr(full_model, "streaming_enabled", False)

    def _requires_strict_streaming_alignment(self) -> bool:
        full_model = getattr(self.autoencoder, "_full_model", None)
        return self._uses_streaming_full_model() and getattr(full_model, "streaming_require_full_chunks", False)

    def _streaming_raw_chunk_size(self) -> int:
        full_model = getattr(self.autoencoder, "_full_model", None)
        return getattr(full_model, "streaming_raw_chunk_size", 4)

    def _validate_video_frame_count(self, n_frames: int, value_name: str) -> None:
        if not self._requires_strict_streaming_alignment():
            return
        chunk_size = self._streaming_raw_chunk_size()
        if (n_frames - 1) % chunk_size != 0:
            raise ValueError(
                f"{value_name} must satisfy 1 + n*{chunk_size} for the strict streaming tokenizer, but got {n_frames}."
            )

    def _prepare_normalized_video(self, input_video: np.ndarray) -> np.ndarray:
        input_video = np.clip(input_video, self.min_range, self.max_range)
        normalised_video = normalize_range_map(
            input_video,
            self.max_range,
            self.min_range,
            self.min_value,
            False,
        )
        normalised_video = np.repeat(normalised_video[..., np.newaxis], 3, axis=-1)
        normalised_video = np.repeat(normalised_video, self.n_rows_repeat, axis=1)
        normalised_video = np.repeat(normalised_video, self.n_cols_repeat, axis=2)
        return normalised_video.astype(np.float32, copy=False)

    def _autoencode_image_video(self, normalised_video: np.ndarray) -> np.ndarray:
        output_images = []
        for frame_idx in tqdm(range(normalised_video.shape[0])):
            batch_image = np.expand_dims(normalised_video[frame_idx], axis=0)
            output_image = self.autoencoder(batch_image)[0]
            output_images.append(output_image)
        return np.stack(output_images, axis=0)

    def _autoencode_video_chunk(self, input_chunk: np.ndarray) -> np.ndarray:
        batch_video = input_chunk[np.newaxis, ...]
        padded_video, crop_region = pad_video_batch(batch_video, temporal_align=1)
        input_tensor = torch.from_numpy(padded_video.transpose(0, 4, 1, 2, 3)).to(self.autoencoder._dtype).to(
            self.autoencoder._device
        )
        output_tensor = self.autoencoder.autoencode(input_tensor)
        padded_output = output_tensor.float().cpu().numpy().transpose(0, 2, 3, 4, 1)
        return unpad_video_batch(padded_output, crop_region)[0]

    def _build_causal_input_chunk(self, input_chunk: np.ndarray) -> tuple[np.ndarray, int]:
        """Create the fixed-length model input for a causal video chunk."""
        valid_frames = input_chunk.shape[0]
        causal_input = input_chunk
        if not self.use_standalone_context_frame:
            causal_input = np.concatenate([input_chunk[:1], input_chunk], axis=0)
        if causal_input.shape[0] < self.temporal_window:
            causal_input = np.concatenate(
                [
                    causal_input,
                    np.repeat(causal_input[-1:], self.temporal_window - causal_input.shape[0], axis=0),
                ],
                axis=0,
            )
        return causal_input, valid_frames

    def _autoencode_temporal_video(self, normalised_video: np.ndarray) -> np.ndarray:
        n_frames = normalised_video.shape[0]
        if n_frames == 0:
            return normalised_video
        self._validate_video_frame_count(n_frames, "input frame count")

        if self.use_standalone_context_frame and self._uses_streaming_full_model():
            output_chunk = self._autoencode_video_chunk(normalised_video)
            return output_chunk[:n_frames]

        input_window = self.temporal_window if self.use_standalone_context_frame else self.temporal_window - 1
        if n_frames > input_window:
            raise ValueError(
                f"video tokenizer currently expects a single fixed window of at most {input_window} raw frames, "
                f"but got {n_frames}. Set --max_frames to match --temporal_window for aligned 1+T reconstruction."
            )

        causal_input, valid_frames = self._build_causal_input_chunk(normalised_video)
        output_chunk = self._autoencode_video_chunk(causal_input)
        if self.use_standalone_context_frame:
            return output_chunk[:valid_frames]
        return output_chunk[1 : valid_frames + 1]
        

    def process_video(self, input_video):
        """Process a range image video.

        Args:
            input_video: Input video array of shape [N, H, W], the values are the effective range values
            max_frames: Maximum number of frames to process

        Returns:
            Tuple of (processed_video, original_video, difference)
        """
        valid_mask = (input_video > self.min_range + 0.1) & (input_video < self.max_range - 0.1)

        if self.elevation_angles is not None:
            elevation_angles = self.elevation_angles
        else:
            elevation_angles = load_pandar128_elevations()
        ray_directions = range_map_to_ray_directions(input_video.shape[-1], elevation_angles)  # shape: (H, W, 3)
        ray_directions = np.repeat(ray_directions[np.newaxis, ...], input_video.shape[0], axis=0)
        
        # downsample the range map if needed 
        input_video, [ray_directions, valid_mask] = self.range_map_downsampler.downsample(
            input_video,
            extra_maps=[ray_directions, valid_mask],
        )
        
        input_video = np.clip(input_video, self.min_range, self.max_range)
        n_frames = input_video.shape[0]
        normalised_video = self._prepare_normalized_video(input_video)
        if self.tokenizer_type == "image":
            recon_video = self._autoencode_image_video(normalised_video)
        else:
            recon_video = self._autoencode_temporal_video(normalised_video)

        recon_video = recon_video[
            :,
            self.n_rows_repeat // 2 :: self.n_rows_repeat,
            self.n_cols_repeat // 2 :: self.n_cols_repeat,
            :,
        ]
        recon_video = recon_video.mean(axis=-1)
        recon_video = np.clip(recon_video, self.min_value, self.max_value)

        output_images_list = []
        for frame_idx in range(n_frames):
            output_frame, _ = unnormalize_range_map(
                recon_video[frame_idx],
                self.max_range,
                self.min_range,
                self.min_value,
                False,
                near_buffer=0.1,
                far_buffer=0.1,
                valid_mask=valid_mask[frame_idx],
            )

            output_images_list.append(output_frame)
        

        output_video = np.stack(output_images_list, axis=0)
        return input_video, output_video, valid_mask, ray_directions
