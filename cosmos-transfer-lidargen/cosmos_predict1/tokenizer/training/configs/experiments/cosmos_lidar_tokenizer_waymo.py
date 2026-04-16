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

from copy import deepcopy

from hydra.core.config_store import ConfigStore

from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict

Cosmos_LidarTokenizer_CI8x8_Waymo: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_image"},
            {"override /loss": "image"},
            {"override /data_train": "lidar_range_map_rRow4_waymo"},
            {"override /data_val": "lidar_range_map_rRow4_waymo"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["float32", "wandbLidar"]},
            "_self_",
        ],
        job=dict(
            group="tokenizer",
            name="Cosmos-LidarTokenizer-CI8x8-Waymo",
            wandb_mode="disabled",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/Cosmos-0.1-Tokenizer-CI8x8/autoencoder.pt",
            load_training_state=False,
            save_iter=1000,
            strict_resume=False,
            jit=dict(strict=False, dtype="float32"),
        ),
        trainer=dict(
            max_iter=20000,
            validation_iter=500,
            max_val_iter=5,
            logging_iter=100,
        ),
        dataloader_val=dict(
            dataset=dict(
                lidar_crop_size=[-1, 896],
                frame_sampling_method="sequential_from_zero",
            ),
        ),
        model=dict(
            config=dict(
                network=dict(
                    resolution=512,
                ),
                loss=dict(
                    config=dict(
                        perceptual=dict(
                            config=dict(
                                gram_enabled=False,
                                gram_boundaries=[0],
                                gram_values=[0.0],
                            )
                        )
                    )
                ),
                disc_optimizer=dict(
                    lr=0.00016,
                ),
                ema=dict(
                    enabled=False,
                ),
                precision="float32"
            )
        ),
        optimizer=dict(
            lr=0.00004,
        ),
    )
)

Cosmos_LidarTokenizer_CV4x8x8_Waymo: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_factorized_video"},
            {"override /loss": "video"},
            {"override /data_train": "lidar_range_map_video_rRow4_waymo"},
            {"override /data_val": "lidar_range_map_video_rRow4_waymo"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic", "wandbLidar"]},
            "_self_",
        ],
        job=dict(
            group="tokenizer",
            name="Cosmos-LidarTokenizer-CV4x8x8-Waymo",
            wandb_mode="disabled",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/Cosmos-0.1-Tokenizer-CI8x8/autoencoder.pt",
            load_training_state=False,
            save_iter=1000,
            strict_resume=False,
            jit=dict(strict=False, dtype="bfloat16", input_shape=[1, 3, 9, 512, 896]),
        ),
        trainer=dict(
            max_iter=20000,
            validation_iter=500,
            max_val_iter=1,
            logging_iter=100,
        ),
        model=dict(
            config=dict(
                network=dict(
                    resolution=512,
                    patch_size=2,
                    temporal_compression=4,
                    spatial_compression=8,
                ),
                loss=dict(
                    config=dict(
                        perceptual=dict(
                            config=dict(
                                lpips_boundaries=[1000],
                                lpips_values=[0.0, 0.1],
                                gram_enabled=False,
                                gram_boundaries=[0],
                            )
                        ),
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[1.0],
                                num_frames=9,
                                step=4,
                            )
                        ),
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[1_000_000],
                                values=[0.0, 0.01],
                                scale=2,
                                dtype="bfloat16",
                                checkpoint_activations=False,
                            )
                        ),
                    )
                ),
                disc_optimizer=dict(
                    lr=0.00016,
                ),
                ema=dict(
                    enabled=False,
                ),
                precision="bfloat16",
            )
        ),
        optimizer=dict(
            lr=0.00004,
        ),
    )
)

Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "continuous_factorized_video"},
            {"override /loss": "video"},
            {"override /data_train": "lidar_range_map_video_rRow4_waymo_t29"},
            {"override /data_val": "lidar_range_map_video_rRow4_waymo_t29"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic", "wandbLidar"]},
            "_self_",
        ],
        job=dict(
            group="tokenizer",
            name="Cosmos-LidarTokenizer-CV4x8x8-Waymo-T29",
            wandb_mode="disabled",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/Cosmos-0.1-Tokenizer-CI8x8/autoencoder.pt",
            load_training_state=False,
            save_iter=1000,
            strict_resume=False,
            jit=dict(strict=False, dtype="bfloat16", input_shape=[1, 3, 29, 512, 832]),
        ),
        trainer=dict(
            max_iter=20000,
            validation_iter=500,
            max_val_iter=1,
            logging_iter=100,
        ),
        model=dict(
            config=dict(
                network=dict(
                    resolution=512,
                    patch_size=2,
                    temporal_compression=4,
                    spatial_compression=8,
                ),
                loss=dict(
                    config=dict(
                        perceptual=dict(
                            config=dict(
                                enabled=False,
                                lpips_boundaries=[1500],
                                lpips_values=[0.0, 0.1],
                                gram_enabled=False,
                                gram_boundaries=[0],
                            )
                        ),
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[1.0],
                                num_frames=29,
                                step=4,
                            )
                        ),
                        flow=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                                scale=4,
                                dtype="bfloat16",
                                checkpoint_activations=True,
                            )
                        ),
                    )
                ),
                disc_optimizer=dict(
                    lr=0.00016,
                ),
                ema=dict(
                    enabled=False,
                ),
                precision="bfloat16",
            )
        ),
        optimizer=dict(
            lr=0.00004,
        ),
    )
)

Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Flow: LazyDict = deepcopy(Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Flow["job"]["name"] = "Cosmos-LidarTokenizer-CV4x8x8-Waymo-T29-Flow"
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Flow["model"]["config"]["loss"]["config"]["flow"]["config"].update(
    {
        "enabled": True,
        "boundaries": [6000],
        "values": [0.0, 0.002],
    }
)

Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming: LazyDict = deepcopy(Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming["job"]["name"] = "Cosmos-LidarTokenizer-CV4x8x8-Waymo-T29-Streaming"
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming["checkpoint"]["jit"]["enabled"] = False
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming["checkpoint"]["jit"]["input_shape"] = [1, 3, 29, 512, 768]
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming["dataloader_train"] = dict(
    dataset=dict(
        lidar_crop_size=[-1, 768],
    ),
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming["dataloader_val"] = dict(
    dataset=dict(
        lidar_crop_size=[-1, 768],
    ),
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming["model"]["config"]["network"].update(
    {
        "streaming_enabled": True,
        "streaming_raw_chunk_size": 4,
        "streaming_latent_chunk_size": 1,
        "streaming_detach_cache": True,
        "streaming_disable_temporal_attn_cache": True,
        "streaming_train_use_full_path": True,
        "streaming_require_full_chunks": True,
    }
)

Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming: LazyDict = deepcopy(Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["defaults"] = [
    {"override /network": "continuous_factorized_video"},
    {"override /loss": "video"},
    {"override /data_train": "lidar_range_map_video_rRow4_waymo_t15"},
    {"override /data_val": "lidar_range_map_video_rRow4_waymo_t15"},
    {"override /optimizer": "fused_adam"},
    {"override /callbacks": ["basic", "wandbLidar"]},
    "_self_",
]
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["job"]["name"] = "Cosmos-LidarTokenizer-CV4x8x8-Waymo-T15-Streaming"
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["checkpoint"]["jit"]["enabled"] = False
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["checkpoint"]["jit"]["input_shape"] = [1, 3, 15, 512, 896]
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["dataloader_train"] = dict(
    dataset=dict(
        lidar_crop_size=[-1, 896],
    ),
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["dataloader_val"] = dict(
    dataset=dict(
        lidar_crop_size=[-1, 896],
    ),
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["model"]["config"]["loss"]["config"]["video_consistency"][
    "config"
]["num_frames"] = 15
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming["model"]["config"]["network"].update(
    {
        "streaming_enabled": True,
        "streaming_raw_chunk_size": 4,
        "streaming_latent_chunk_size": 1,
        "streaming_detach_cache": True,
        "streaming_disable_temporal_attn_cache": False,
        "streaming_train_use_full_path": False,
    }
)

Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming: LazyDict = deepcopy(Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["defaults"] = [
    {"override /network": "continuous_factorized_video"},
    {"override /loss": "video"},
    {"override /data_train": "lidar_range_map_video_rRow4_waymo_t17"},
    {"override /data_val": "lidar_range_map_video_rRow4_waymo_t17"},
    {"override /optimizer": "fused_adam"},
    {"override /callbacks": ["basic", "wandbLidar"]},
    "_self_",
]
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["job"]["name"] = "Cosmos-LidarTokenizer-CV4x8x8-Waymo-T17-Streaming"
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["checkpoint"]["jit"]["enabled"] = False
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["checkpoint"]["jit"]["input_shape"] = [1, 3, 17, 512, 896]
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/"
    "checkpoints/iter_000020000.pt"
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["checkpoint"]["strict_resume"] = False
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["dataloader_train"] = dict(
    dataset=dict(
        lidar_crop_size=[-1, 896],
    ),
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["dataloader_val"] = dict(
    dataset=dict(
        lidar_crop_size=[-1, 896],
    ),
)
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["model"]["config"]["loss"]["config"]["video_consistency"][
    "config"
]["num_frames"] = 17
Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming["model"]["config"]["network"].update(
    {
        "streaming_enabled": True,
        "streaming_raw_chunk_size": 4,
        "streaming_latent_chunk_size": 1,
        "streaming_detach_cache": True,
        "streaming_disable_temporal_attn_cache": False,
        "streaming_train_use_full_path": False,
        "streaming_require_full_chunks": True,
    }
)

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /network": "latent_temporal_compressor_video"},
            {"override /loss": "video"},
            {"override /data_train": "lidar_range_map_video_rRow4_waymo_t29_latent"},
            {"override /data_val": "lidar_range_map_video_rRow4_waymo_t29_latent"},
            {"override /optimizer": "fused_adam"},
            {"override /callbacks": ["basic", "wandbLidar"]},
            "_self_",
        ],
        job=dict(
            group="tokenizer",
            name="Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor",
            wandb_mode="online",
            wandb_entity="leishuangming1103-zhejiang-university",
            wandb_project="Lidar Tokenizer",
        ),
        checkpoint=dict(
            load_path="",
            load_training_state=False,
            save_iter=1000,
            strict_resume=False,
            jit=dict(enabled=False, strict=False, dtype="bfloat16", input_shape=[1, 3, 29, 512, 896]),
        ),
        trainer=dict(
            max_iter=20000,
            validation_iter=500,
            max_val_iter=5,
            logging_iter=100,
        ),
        model=dict(
            config=dict(
                network=dict(
                    name="LTCV",
                    expected_input_frames=29,
                    expected_compressed_frames=8,
                    exact_context_frames=1,
                    frozen_image_tokenizer_ckpt=(
                        "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/"
                        "checkpoints/iter_000020000.pt"
                    ),
                    image_tokenizer=dict(
                        resolution=512,
                    ),
                    temporal_compressor=dict(
                        resolution=64,
                    ),
                ),
                loss=dict(
                    config=dict(
                        kl=dict(
                            config=dict(
                                boundaries=[0],
                                values=[1e-5],
                            )
                        ),
                        latent_recon=dict(
                            config=dict(
                                boundaries=[0],
                                values=[1.0],
                            )
                        ),
                        temporal_delta=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                            )
                        ),
                        perceptual=dict(
                            config=dict(
                                enabled=False,
                                gram_enabled=False,
                                gram_boundaries=[0],
                                gram_values=[0.0],
                            )
                        ),
                        flow=dict(
                            config=dict(
                                enabled=True,
                                boundaries=[5000],
                                values=[0.0, 0.002],
                                scale=4,
                                dtype="bfloat16",
                                checkpoint_activations=True,
                            )
                        ),
                        video_consistency=dict(
                            config=dict(
                                enabled=False,
                                boundaries=[0],
                                values=[0.0],
                                num_frames=29,
                                step=4,
                            )
                        ),
                    )
                ),
                disc_optimizer=dict(
                    lr=0.00016,
                ),
                ema=dict(
                    enabled=False,
                ),
                precision="bfloat16",
            )
        ),
        optimizer=dict(
            lr=0.00004,
        ),
    )
)

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["defaults"] = [
    {"override /network": "latent_temporal_compressor_video"},
    {"override /loss": "video"},
    {"override /data_train": "lidar_range_map_video_rRow4_waymo_t29_latent"},
    {"override /data_val": "lidar_range_map_video_rRow4_waymo_t29_latent"},
    {"override /optimizer": "fused_adam"},
    {"override /scheduler": "warmup_cosine"},
    {"override /callbacks": ["basic", "wandbLidar"]},
    "_self_",
]
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["job"]["wandb_mode"] = "online"
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["job"]["wandb_entity"] = (
    "leishuangming1103-zhejiang-university"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["job"]["wandb_project"] = "Lidar Tokenizer"
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/"
    "checkpoints/iter_000027000.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["checkpoint"]["load_training_state"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["trainer"].update(
    {
        "max_iter": 13000,
        "validation_iter": 500,
        "max_val_iter": 10,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["optimizer"]["lr"] = 1e-5
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["scheduler"] = dict(
    warmup_iters=500,
    lr_decay_iters=13000,
    min_lr=1e-6,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["model"]["config"]["loss"]["config"]["flow"][
    "config"
]["boundaries"] = [0]
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT["model"]["config"]["loss"]["config"]["flow"][
    "config"
]["values"] = [0.002]

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V2"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V2"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["job"]["wandb_log_validation_media"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["job"]["wandb_init_timeout"] = 300
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/"
    "checkpoints/iter_000010000.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["checkpoint"]["load_training_state"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["trainer"].update(
    {
        "max_iter": 10000,
        "validation_iter": 500,
        "max_val_iter": 10,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["optimizer"]["lr"] = 5e-6
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["scheduler"] = dict(
    warmup_iters=200,
    lr_decay_iters=10000,
    min_lr=5e-7,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2["model"]["config"]["loss"]["config"]["flow"][
    "config"
]["values"] = [0.001]

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V3"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V3"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/"
    "checkpoints/iter_000009000.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["trainer"].update(
    {
        "max_iter": 12000,
        "validation_iter": 500,
        "max_val_iter": 5,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["optimizer"]["lr"] = 5e-6
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["scheduler"] = dict(
    warmup_iters=200,
    lr_decay_iters=12000,
    min_lr=5e-7,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["model"]["config"]["loss"]["config"][
    "latent_recon"
]["config"] = dict(boundaries=[5000], values=[1.0, 0.25])
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["model"]["config"]["loss"]["config"][
    "temporal_delta"
]["config"] = dict(enabled=True, boundaries=[0], values=[0.25])
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["model"]["config"]["loss"]["config"]["flow"][
    "config"
]["values"] = [0.0005]
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3["job"]["wandb_log_validation_media"] = False

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["job"]["wandb_mode"] = "offline"
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["job"]["wandb_log_validation_media"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V3/"
    "checkpoints/iter_000008000.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["checkpoint"]["load_training_state"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["trainer"].update(
    {
        "max_iter": 10000,
        "validation_iter": 500,
        "max_val_iter": 5,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["dataloader_val"] = dict(
    prefetch_factor=2,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["optimizer"]["lr"] = 5e-6
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["scheduler"] = dict(
    warmup_iters=200,
    lr_decay_iters=10000,
    min_lr=5e-7,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["model"]["config"]["network"][
    "trainable_image_tokenizer_modules"
] = ("post_quant_conv", "decoder")
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4["model"]["config"]["network"][
    "trainable_image_tokenizer_lr_scale"
] = 0.2

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V5"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V5"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/"
    "checkpoints/iter_000008000.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["checkpoint"]["load_training_state"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["checkpoint"]["save_iter"] = 500
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["trainer"].update(
    {
        "max_iter": 10000,
        "validation_iter": 500,
        "max_val_iter": 5,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["dataloader_val"] = dict(
    prefetch_factor=2,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["optimizer"]["lr"] = 3e-6
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["scheduler"] = dict(
    warmup_iters=200,
    lr_decay_iters=10000,
    min_lr=5e-7,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["model"]["config"]["network"][
    "trainable_image_tokenizer_modules"
] = ("quant_conv", "post_quant_conv", "decoder")
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["model"]["config"]["network"][
    "trainable_image_tokenizer_lr_scale"
] = 0.1
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["model"]["config"]["loss"]["config"][
    "latent_recon"
]["config"] = dict(boundaries=[6500], values=[0.25, 0.1])
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["model"]["config"]["loss"]["config"][
    "temporal_delta"
]["config"] = dict(enabled=True, boundaries=[0], values=[0.35])
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5["model"]["config"]["loss"]["config"]["flow"][
    "config"
] = dict(enabled=False, boundaries=[0], values=[0.0], scale=4, dtype="bfloat16", checkpoint_activations=True)

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["defaults"] = [
    {"override /network": "latent_temporal_compressor_video"},
    {"override /loss": "video"},
    {"override /data_train": "lidar_range_map_video_rRow4_waymo_t29_latent"},
    {"override /data_val": "lidar_range_map_video_rRow4_waymo_t29_latent"},
    {"override /optimizer": "fused_adam"},
    {"override /scheduler": "warmup_cosine"},
    {"override /callbacks": ["basic", "wandbLidar"]},
    "_self_",
]
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["job"]["wandb_mode"] = "offline"
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["job"]["wandb_log_validation_media"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["checkpoint"]["load_path"] = ""
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["checkpoint"]["load_training_state"] = False
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["checkpoint"]["save_iter"] = 500
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["trainer"].update(
    {
        "max_iter": 30000,
        "validation_iter": 500,
        "max_val_iter": 5,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["dataloader_val"] = dict(
    prefetch_factor=2,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["optimizer"]["lr"] = 1e-5
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["scheduler"] = dict(
    warmup_iters=500,
    lr_decay_iters=30000,
    min_lr=1e-6,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["network"][
    "temporal_compressor"
] = dict(
    architecture="opensora_temporal_vae",
    channels=64,
    channels_mult=[1, 2, 2, 4],
    dropout=0.0,
    in_channels=16,
    num_res_blocks=2,
    out_channels=16,
    resolution=64,
    patch_size=1,
    patch_method="haar",
    latent_channels=16,
    z_channels=16,
    z_factor=1,
    num_groups=16,
    legacy_mode=False,
    spatial_compression=1,
    temporal_compression=4,
    temporal_downsample=(False, True, True),
    formulation="VAE",
    encoder="FACTORIZED",
    decoder="FACTORIZED",
    name="OpenSoraTemporalCompressorVAE",
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["network"][
    "trainable_image_tokenizer_modules"
] = ()
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["network"][
    "trainable_image_tokenizer_lr_scale"
] = 1.0
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["loss"]["config"][
    "color"
] = dict(config=dict(boundaries=[0], values=[0.1]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["loss"]["config"][
    "latent_recon"
] = dict(config=dict(boundaries=[0], values=[1.0]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["loss"]["config"][
    "temporal_delta"
] = dict(config=dict(enabled=False, boundaries=[0], values=[0.0]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1["model"]["config"]["loss"]["config"]["flow"][
    "config"
] = dict(enabled=False, boundaries=[0], values=[0.0], scale=4, dtype="bfloat16", checkpoint_activations=True)

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/"
    "checkpoints/iter_000029500.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["trainer"].update(
    {
        "max_iter": 40000,
        "validation_iter": 500,
        "max_val_iter": 5,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["optimizer"]["lr"] = 3e-6
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["scheduler"] = dict(
    warmup_iters=200,
    lr_decay_iters=40000,
    min_lr=5e-7,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["model"]["config"]["network"][
    "trainable_image_tokenizer_modules"
] = ("post_quant_conv", "decoder")
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["model"]["config"]["network"][
    "trainable_image_tokenizer_lr_scale"
] = 0.1
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["model"]["config"]["loss"]["config"][
    "color"
] = dict(config=dict(boundaries=[0], values=[0.25]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["model"]["config"]["loss"]["config"][
    "latent_recon"
] = dict(config=dict(boundaries=[0], values=[1.0]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2["model"]["config"]["loss"]["config"][
    "temporal_delta"
] = dict(config=dict(enabled=True, boundaries=[0], values=[0.1]))

Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3: LazyDict = deepcopy(
    Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["job"]["name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["job"]["wandb_name"] = (
    "Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S3"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["checkpoint"]["load_path"] = (
    "checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2/"
    "checkpoints/REPLACE_WITH_BEST_S2_CHECKPOINT.pt"
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["trainer"].update(
    {
        "max_iter": 6000,
        "validation_iter": 500,
        "max_val_iter": 5,
        "logging_iter": 100,
    }
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["optimizer"]["lr"] = 2e-6
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["scheduler"] = dict(
    warmup_iters=100,
    lr_decay_iters=6000,
    min_lr=2e-7,
)
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["model"]["config"]["loss"]["config"][
    "color"
] = dict(config=dict(boundaries=[0], values=[1.0]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["model"]["config"]["loss"]["config"][
    "latent_recon"
] = dict(config=dict(boundaries=[1000], values=[0.25, 0.05]))
Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3["model"]["config"]["loss"]["config"][
    "temporal_delta"
] = dict(config=dict(enabled=True, boundaries=[0], values=[0.25]))

cs = ConfigStore.instance()

for experiment_name, experiment_cfg in [
    ("cosmos_lidar_tokenizer_waymo", Cosmos_LidarTokenizer_CI8x8_Waymo),
    ("cosmos_lidar_tokenizer_cv4x8x8_waymo", Cosmos_LidarTokenizer_CV4x8x8_Waymo),
    ("cosmos_lidar_tokenizer_cv4x8x8_waymo_t29", Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29),
    ("cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_flow", Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Flow),
    ("cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_streaming", Cosmos_LidarTokenizer_CV4x8x8_Waymo_T29_Streaming),
    ("cosmos_lidar_tokenizer_cv4x8x8_waymo_t15_streaming", Cosmos_LidarTokenizer_CV4x8x8_Waymo_T15_Streaming),
    ("cosmos_lidar_tokenizer_cv4x8x8_waymo_t17_streaming", Cosmos_LidarTokenizer_CV4x8x8_Waymo_T17_Streaming),
    ("cosmos_lidar_tokenizer_waymo_t29_latent_compressor", Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v2",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V2,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v3",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V3,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v4",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V4,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v5",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_LRDecayFT_V5,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s1",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S1,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s2",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S2,
    ),
    (
        "cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s3",
        Cosmos_LidarTokenizer_Waymo_T29_LatentCompressor_OpenSora_S3,
    ),
]:
    log.info(f"Registering experiment: {experiment_name}")
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=experiment_cfg,
    )
