# Cosmos Drive Dreams - Waymo LiDAR Tokenizer 工作流

## 1. 数据转换：Waymo rds_hq → LiDAR Tokenizer 格式

将 Waymo rds_hq 的 `lidar_raw` 转换为 tokenizer 训练所需的 sparse range map 格式。
这里的 `xyz` 原始保存在 vehicle frame，`lidar_to_world` 实际上是 `vehicle_to_world`；
转换脚本会先把点云变换到 Waymo TOP LiDAR sensor frame，再写出真正的 `lidar_to_world` metadata。

**输出格式**：
- `metadata/{clip_id}.npz` — pose_list, timestamps_list, frame_indices
- `lidar/{clip_id}.tar` — sparse range maps (row/col/range per frame)

**Range map 参数**：128 rows × 3600 cols，使用 Waymo TOP LiDAR 的真实 64 条非均匀 beam inclination 插值到 128 行

```bash
cd /root/workspace/Cosmos-Drive-Dreams
conda activate cosmos-predict1

# 转换训练集 (798 clips)
python cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py \
    --input_root /data2/rds_hq_waymo/training \
    --output_root /data2/rds_hq_waymo/lidar_tokenizer/training \
    --num_workers 16

# 转换验证集 (202 clips)
python cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py \
    --input_root /data2/rds_hq_waymo/validation \
    --output_root /data2/rds_hq_waymo/lidar_tokenizer/validation \
    --num_workers 16
```

**可选参数**：
- `--split_file`：指定 clip 列表文件
- `--n_cols`：range map 宽度（默认 3600）
- `--n_rows`：当前固定要求为 128，和 tokenizer 训练配置保持一致
- `--num_workers`：按 clip 并行转换的进程数（默认 8，建议从 8 或 16 起试）

**注意事项**：
- 每个 clip 的最后一帧会被保留；其 `pose_list[-1]` 使用当前帧 pose 自复制，时间戳补 `+100000us`
- 如果原始 `timestamp/` 目录存在，会优先使用真实时间戳；否则按顺序生成 `idx * 100000`
- metadata 中保存的 `pose_list` 是真实的 `lidar_to_world`，与输出 range map 所在 LiDAR frame 对齐

## 2. 创建软链接和 split 文件

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

# 软链接数据到 datasets/ 目录
ln -s /data2/rds_hq_waymo/lidar_tokenizer/training datasets/waymo_lidar_training
ln -s /data2/rds_hq_waymo/lidar_tokenizer/validation datasets/waymo_lidar_validation

# 生成 split 文件
ls datasets/waymo_lidar_training/lidar/ | sed 's/.tar//' > assets/lidar/waymo_train_split.lst
ls datasets/waymo_lidar_validation/lidar/ | sed 's/.tar//' > assets/lidar/waymo_val_split.lst
```

## 3. 后训练 LiDAR Tokenizer (8x A100-80G)

当前支持两条后训练路线：
- `CI8x8`：保持现有逐帧 image tokenizer，不做时间压缩
- `CV4x8x8`：切换到 causal video tokenizer，对时间维做 `4x` 压缩

### 3.1 逐帧版本：CI8x8

基于预训练的 Cosmos-Tokenizer-CI8x8-Lidar 在 Waymo 数据上 fine-tune。

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints

torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py \
    -- \
    experiment=cosmos_lidar_tokenizer_waymo
```

**训练配置**（`cosmos_lidar_tokenizer_waymo.py`）：
- 预训练权重：`checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/Cosmos-0.1-Tokenizer-CI8x8/autoencoder.pt`
- max_iter=20000, validation_iter=500, save_iter=1000, lr=4e-5, precision=float32
- 输出目录：`checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/`
- 支持断点续训：自动从 `latest_checkpoint.txt` 恢复

### 3.2 时间压缩版本：CV4x8x8

如果希望 tokenizer 在 latent 里同时压缩时间维，可以训练 `CV4x8x8` 版本：

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py \
    -- \
    experiment=cosmos_lidar_tokenizer_cv4x8x8_waymo
```

**训练配置**（同文件内新增 `cosmos_lidar_tokenizer_cv4x8x8_waymo`）：
- 网络：`continuous_factorized_video`
- 压缩率：`temporal_compression=4`, `spatial_compression=8`, `patch_size=2`
- 时间维语义：输入定义为 `1 + T`，latent 时间长度为 `1 + T/4`
- 当前配置示例：`T=8`，顺序采样真实的 `1+T=9` 帧，其中第 1 帧是 standalone context，后 8 帧是后续真实帧；latent 长度为 `3`
- 训练输入 shape：`[B, 3, 9, 512, 896]`
- 训练精度：`precision=bfloat16`，并使用 `basic` callbacks 中的 `low_precision`
- validation：`validation_iter=500`, `max_val_iter=5`
- JIT 导出精度：`checkpoint.jit.dtype=bfloat16`
- 输出目录：`checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo/`

**初始化说明**：
- 默认仍从 `CI8x8` LiDAR checkpoint 读取可兼容参数
- 由于 2D image tokenizer 和 3D video tokenizer 的部分层 shape 不同，`strict_resume=False`
- 不匹配的 tensor 会被自动跳过；新增的时序层会随机初始化再 fine-tune
- 当前 Waymo `CV4x8x8` 已对齐 `cosmos-transfer2.5` 的 causal 语义：前导 `1` 是真实首帧，不再通过重复首帧伪造
- 这只会影响后续新训练/新推理；此前已经训练完成的旧 checkpoint 仍属于 legacy“复制首帧”语义，如需完全对齐需要按新配置重新训练

**当前基线策略（先跑通）**：
- 当前 `CV4x8x8` 先以可稳定收敛的 baseline 为主，`video_consistency.enabled=False`，`flow.enabled=False`
- `flow` 对应的是基于 `RAFT` 的光流一致性损失；当前训练只跑 `20000` iter，而它的权重调度是 `boundaries=[1_000_000], values=[0.0, 0.01]`
- 因此在这版配置里，即使把 `flow` 打开，前 `20000` iter 内它的权重也仍然是 `0.0`，不会真正参与优化
- 先关闭这两项的好处是避免额外的 `RAFT` 初始化、checkpoint 下载和显存占用，优先把 `1+T -> 1+T/4` 的重建基线训稳
- `LPIPS` 当前改为短 warmup：前 `1000` iter 权重为 `0.0`，之后切到 `0.1`，让随机初始化的时序层先用重建项稳定下来，再引入 perceptual loss
- 如果后续需要强化时间一致性，建议先完成这版 baseline，再从其 checkpoint 出发做第二阶段微调，并以较小权重逐步打开 `video_consistency` 或 `flow`
- 当前 8 卡实测中，`CV4x8x8` 的 OOM 主要出现在 `LPIPS/PerceptualLoss` 的 float32 路径；切到 `bfloat16` 后可稳定跑过首个 iter 并成功保存 checkpoint
- 如果遇到显存碎片问题，建议保留 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**T29 训练补充**：
- 如果目标改成 `1+28 -> 1+7 -> 1+28`，仓库里已准备好 `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29` 和 `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_flow`
- 当前可稳定跑通 8x80G 的 `T29` baseline 使用宽度 `832`，JIT 输入是 `[1, 3, 29, 512, 832]`
- `T29` baseline 当前显式关闭了 `LPIPS`，因为即使去掉 `RAFT`，`29 x 896` 的 decoder 峰值也仍会在 80G 卡上 OOM；把宽度收为 `832` 后，8 卡 smoke 和完整训练起步已验证可过
- `video_consistency` 目前仍保持关闭；原因不是权重大小，而是它当前实现会先把 full-window 输入切成更短的重叠子窗口再送进网络，这和我们想训练的固定 `29` 帧压缩目标不一致
- `flow` 代码仍保留为第二阶段选项，但不再作为默认长训入口；当前 `FlowLoss` 已改成按需初始化，不会在权重为 `0` 的阶段提前加载 `RAFT`
- 当前 `T29` 长训配置把 `max_val_iter` 设为 `1`，优先保证首轮 validation 和训练启动稳定

**T17 Streaming 主线（当前默认配置）**：
- 当前主线配置已切到 `cosmos_lidar_tokenizer_cv4x8x8_waymo_t17_streaming`
- 输入定义为真实 `1+16=17` 帧，空间宽度保持 `896`，时间压缩仍为 `4x`，因此语义上是 `17 -> 5 -> 17`
- 第二阶段训练现已默认从 [iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt) 初始化，也就是先做 `CI8x8-Waymo` 域适配，再进入 `T17 Streaming`
- 第二阶段继续保持 `strict_resume=False`，这样可以在加载 `CI8x8-Waymo` 的同时兼容新增的 3D/streaming 层 warm start
- 当前 streaming 配置为：`streaming_enabled=True`、`streaming_raw_chunk_size=4`、`streaming_latent_chunk_size=1`、`streaming_train_use_full_path=False`、`streaming_require_full_chunks=True`
- 当前主线额外约束为：`streaming_disable_temporal_attn_cache=False`，让训练和推理都使用同一套 temporal attention cache 语义，避免“训练禁用、推理启用”导致后半段帧质量明显漂移
- 这意味着训练阶段已经直接走 streaming path，而不是只在验证/推理时才启用 streaming
- 当前主线严格遵循 Wan 风格分块约束：输入总帧数只能是 `1 + 4k`，因此 `9 / 17 / 29` 这类长度有效，而 `15` 这种 `1+4+4+4+2` 的尾块输入不再作为主线配置
- 旧的 `T15` / `T29` / `T29 Streaming` 方案代码仍完整保留；其中 `T15 Streaming` 仅作为 legacy ragged-tail 实验保留，不再作为默认主线
- 本地 GPU smoke 已验证当前代码链路：
  - `1` 卡训练 `cosmos_lidar_tokenizer_cv4x8x8_waymo_t17_streaming` 跑通 `1 iter`，并成功保存 [iter_000000001.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo-T17-Streaming-Debug1GPU/checkpoints/iter_000000001.pt)
  - 使用这份 debug checkpoint 做 `29` 帧推理可以正常完成，输出写到了 [range_map_video](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t17_streaming_debug1gpu_29f/range_map_video/10203656353524179475_7625_000_7645_000.mp4) 和 [histogram](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t17_streaming_debug1gpu_29f/histogram/10203656353524179475_7625_000_7645_000.png)
  - 同时也验证了严格约束生效：`15` 帧输入会直接报错 `input frame count must satisfy 1 + n*4`

**当前已知问题（已记录，暂不阻塞当前主线）**：
- 当前 streaming 实现已经对齐了 Wan2.1 的核心思路：`1 + 4 + 4 + ...` 原始帧分块、`1 + 1 + 1 + ...` latent 分块、跨 chunk causal cache 和 temporal attention cache
- 早期 `T15 Streaming` 试验里曾把 `streaming_disable_temporal_attn_cache=True` 打开；由于旧实现只在 `training` 阶段应用这个开关，出现了“训练禁用 temporal attention cache、推理启用 temporal attention cache”的行为不一致。当前代码已修正为 train/infer 都遵循同一个开关，并且当前主线默认改为 `False`
- 当前主线显式开启了 `streaming_require_full_chunks=True`：输入总长度必须满足 `1 + 4k`，避免 `15` 这类 ragged tail 输入和 Wan2.1 的严格分块语义不一致
- 但它还不是“完全等价的 Wan2.1 VAE”：`quant_conv` 和 `post_quant_conv` 目前仍在整段 hidden / latent 上执行，而不是逐 chunk 流式执行
- 因此当前 streaming 路径已经能支持变长输入，但显存占用仍会随总时长增长，不是严格意义上的 constant-memory streaming
- 这一点对当前 `17` 帧训练和 `29` 帧推理不是阻塞项，所以先不改；如果以后目标扩展到更长序列，再继续往“fully streaming quant/post-quant”推进
- `BASE` 版 3D encoder/decoder 的 streaming path 目前没有和 full path 做等价性对齐；当前 Waymo 主线使用的是 `FACTORIZED` 结构，因此不受这个限制
- 目前“超过训练窗口长度的 streaming 推理”优先保证的是 full `.pt` checkpoint + `config.yaml` 路径；如果只给 encoder/decoder 分离权重，接口行为仍按固定窗口更稳妥
- `29` 帧推理“已通过”的含义是结构和加载链路已打通，不代表 smoke checkpoint 的重建质量已经可用于最终指标对比；质量仍应以完整训练后的 checkpoint 为准
- `CI8x8-Waymo` 的 stage-1 warm start 当前继续保持 `strict_resume=False`，优先兼容已有通用 `CI8x8` 初始化路径；更严格的 2D->2D checkpoint 校验如果后面需要，再单独切回来
- `CI8x8-Waymo` 的 validation 现已改为更稳定的配置：`max_val_iter=5`，并且使用 `512x896` 裁剪与 `sequential_from_zero` 取帧，便于把 stage-1 的 val loss 当作更可靠的趋势信号
- `CI8x8-Waymo` 中原本继承来的 Gram loss 调度在 `20000 iter` 内不会真正生效；现已显式关闭，避免出现“配置里写了 Gram、实际上整轮训练都没用到”的误解
- `TokenizerLoss` 里之前会无条件把 `loss_mask` 覆盖成全 1；这个 bug 现已修复，后续如果数据真的提供 `loss_mask`，loss 会按 mask 生效
- `TokenizerModel` 里原先基于 `video.size(2) == 3` 的视频维度判断也已收紧：现在优先把 `dim=1` 视为 channel 维，只有 `dim=1 != 3 且 dim=2 == 3` 时才会做 `BTCHW -> BCTHW` 的 permute，避免恰好 3 帧视频被误判

**权重存储**：权重实际保存在 `/data2/checkpoints/posttraining/`，通过软链接映射到 `checkpoints/posttraining`。

**训练结果**（Validation Loss，前 5000 iter）：

注：下面这组历史数值来自早期 `max_val_iter=1` 配置，只适合看整体趋势，不适合过度解读相邻 checkpoint 的细小波动。当前仓库配置已将 `max_val_iter` 提高到 `5`，后续验证信号会更稳。

| Iteration | Val Loss |
|-----------|----------|
| 0         | 0.0532   |
| 500       | 0.0248   |
| 1000      | 0.0197   |
| 2000      | 0.0179   |
| 3000      | 0.0163   |
| 3500      | 0.0124   |
| 5000      | 0.0135   |

## 4. 推理评估

### 4.1 逐帧版本：CI8x8

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

# 注意：必须使用 --tokenizer_dtype float32（JIT 模型用 float32 traced）
python -m cosmos_predict1.tokenizer.inference.lidar_cli \
    --sample_path="/data2/rds_hq_waymo/lidar_tokenizer/validation/lidar/<clip_id>.tar" \
    --enc_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000005000_enc.jit" \
    --dec_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000005000_dec.jit" \
    --output_folder="waymo_eval" \
    --tokenizer_dtype float32 \
    --max_frames 20 \
    --waymo_top \
    --display_frame vehicle
```

### 4.2 时间压缩版本：CV4x8x8

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

python -m cosmos_predict1.tokenizer.inference.lidar_cli \
    --sample_path="/data2/rds_hq_waymo/lidar_tokenizer/validation/lidar/<clip_id>.tar" \
    --enc_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo/checkpoints/iter_000005000_enc.jit" \
    --dec_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo/checkpoints/iter_000005000_dec.jit" \
    --output_folder="waymo_eval_cv4x8x8" \
    --tokenizer_type video \
    --temporal_window 9 \
    --tokenizer_dtype bfloat16 \
    --max_frames 20 \
    --waymo_top \
    --display_frame vehicle
```

对于当前 `T17 Streaming` / full `.pt` checkpoint，也可以直接走整模加载：

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

python -m cosmos_predict1.tokenizer.inference.lidar_cli \
    --sample_path="/data2/rds_hq_waymo/lidar_tokenizer/validation/lidar/<clip_id>.tar" \
    --model_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo-T17-Streaming/checkpoints/iter_0000xxxxx.pt" \
    --tokenizer_config_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo-T17-Streaming/config.yaml" \
    --output_folder="waymo_eval_t17_streaming" \
    --tokenizer_type video \
    --temporal_window 17 \
    --tokenizer_dtype bfloat16 \
    --max_frames 29 \
    --vis_pcd 0 \
    --waymo_top \
    --display_frame vehicle
```

**推理参数**：
- `--tokenizer_dtype`：`CI8x8` 用 `float32`；当前 `CV4x8x8` 导出的 JIT 是 `bfloat16`，所以这里应显式设成 `bfloat16`
- `--tokenizer_type`：`image` 或 `video`。现有 `CI8x8` 用 `image`，新的 `CV4x8x8` 用 `video`
- `--temporal_window`：仅 `video` 模式使用，表示模型总输入长度 `1+T`，不是“原始帧块长度”；对当前严格 streaming 主线，它必须满足 `1 + 4k`
- 当前 `CV4x8x8` 配置里 `T=8`，所以这里应设为 `9`；如果以后改成 `1+28 -> 1+7`，这里就应设为 `29`
- 当前 `video` 推理只做单窗口 `1+T -> 1+T/4 -> 1+T` autoencode，不再做长视频分块拼接；评估时建议令 `--max_frames` 与 `--temporal_window` 保持一致
- 对于当前 `T17 Streaming` 的 full `.pt` 模型，`--max_frames` 可以大于 `--temporal_window`；但 `--max_frames` 本身也必须满足 `1 + 4k`，例如使用 `17` 帧训练权重进行 `29` 帧推理
- 这条“`17` 训 `29` 推”的能力是当前严格主线的目标用法；如果只使用分离的 `enc/dec` 权重，仍建议按固定窗口方式评估
- 本地 GPU smoke 中，使用 `T17` debug checkpoint 进行 `29` 帧推理时已实际跑通，单条样本的测试指标为：`RMSE 39.00 / MAE 34.49 / Rel 2.51`
- 这组数值只用于证明训练和推理代码链路可运行；由于 checkpoint 只训练了 `1 iter`，不代表最终模型质量
- 如果需要复现旧版“复制首帧”的历史 checkpoint，可额外加 `--legacy_duplicate_context`
- `--max_frames`：限制评估帧数（-1 为全部，默认 20）
- `--vis_pcd`：是否渲染点云对比（默认 1）
- `--waymo_top`：使用 Waymo TOP LiDAR 的真实 beam inclination 反投影
- `--display_frame vehicle`：将点云可视化从 LiDAR frame 变换回 vehicle frame，避免看起来“斜着”
- `--downsample_factor_col`：列方向下采样因子（默认 2，3600→1800）
- `--max_range` / `--min_range`：range 裁剪范围（默认 100/5）

**评估指标**（验证集单 clip）：
- RMSE: 0.72 m
- MAE: 0.39 m
- Relative Error: 2%

**输出文件**：
- `dump_results/lidar_tokenizer/<output_folder>/range_map_video/` — 原始 vs 重建 range map 对比视频
- `dump_results/lidar_tokenizer/<output_folder>/histogram/` — 误差分布直方图
- `dump_results/lidar_tokenizer/<output_folder>/point_cloud/` — 原始 vs 重建点云对比视频

## 5. 可视化原始数据

```bash
cd /root/workspace/Cosmos-Drive-Dreams
conda activate cosmos-predict1

# Range map 视频（Spectral colormap）
python cosmos-drive-dreams-toolkits/visualize_waymo_rangemap.py \
    --tar_path="<lidar_tar_path>" \
    --output_dir="/data2/waymo_visualizations/raw"

# Range map + 点云渲染
python cosmos-drive-dreams-toolkits/visualize_waymo_rangemap.py \
    --tar_path="<lidar_tar_path>" \
    --output_dir="/data2/waymo_visualizations/raw" \
    --vis_pcd \
    --camera_view front_view \
    --display_frame vehicle \
    --max_frames 20
```

**目录约定**：
- 原始 tokenizer 数据的可视化统一放在 `/data2/waymo_visualizations/raw/`
- 推理评估产物继续保留在 `dump_results/lidar_tokenizer/<output_folder>/`，不额外归档到 `/data2`

**可视化可选参数**：
- `--camera_view`：`front_view` 或 `top_down_view`
- `--display_frame`：`vehicle` 或 `lidar`。Waymo tokenizer 数据建议用 `vehicle`
- `--elevation_mode`：默认 `waymo_top`，与当前转换脚本输出匹配
- `--max_frames`：限制帧数（-1 为全部）
- `--save_frames`：同时保存单帧图片
- `--colormap`：colormap 名称（默认 Spectral）
- `--max_workers`：点云渲染并行数（默认 8）

## 6. 关键文件

| 文件 | 说明 |
|------|------|
| `cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py` | Waymo → tokenizer 格式转换 |
| `cosmos-drive-dreams-toolkits/visualize_waymo_rangemap.py` | Range map / 点云可视化 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/configs/experiments/cosmos_lidar_tokenizer_waymo.py` | Waymo 后训练实验配置 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/configs/registry.py` | Hydra dataloader 注册 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/datasets/lidar_datasets/configs.py` | Waymo 数据集配置（保守使用 `lidar_length=169`） |
| `cosmos-transfer-lidargen/assets/lidar/waymo_train_split.lst` | 训练集 clip 列表（798 clips） |
| `cosmos-transfer-lidargen/assets/lidar/waymo_val_split.lst` | 验证集 clip 列表（202 clips） |

## 7. 当前实现补充

- 磁盘上保存的 sparse range map 是单通道：`lidar_row` / `lidar_col` / `lidar_range`
- 还原后的原始 shape 为 `128 x 3600`
- tokenizer dataloader 会先做列下采样 `3600 -> 1800`，再把单通道直接复制成 3 通道，并做行 repeat `128 -> 512`
- `CI8x8` 训练时按帧随机采样，单帧输入 shape 约为 `3 x 512 x 896`
- `CV4x8x8` 训练时遵循 causal 形式：输入时间长度是 `1+T`，编码后 latent 时间长度是 `1+T/4`
- 当前 `CV4x8x8` 示例里取 `T=8`，所以模型输入 shape 为 `3 x 9 x 512 x 896`，对应的是 `1` 个真实 context 帧加 `8` 个真实后续帧；latent 时间长度为 `3`
- 点云可视化默认建议使用 `vehicle frame`；如果切回 `lidar frame`，Waymo TOP 数据视觉上可能会出现旋转/倾斜

## 8. 数据路径

| 数据 | 路径 |
|------|------|
| Waymo 原始数据（训练集） | `/data2/rds_hq_waymo/training/` |
| Waymo 原始数据（验证集） | `/data2/rds_hq_waymo/validation/` |
| 转换后 tokenizer 数据（训练集） | `/data2/rds_hq_waymo/lidar_tokenizer/training/` |
| 转换后 tokenizer 数据（验证集） | `/data2/rds_hq_waymo/lidar_tokenizer/validation/` |
| 预训练权重 | `/data2/checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/` → 软链接 `checkpoints/Cosmos-Tokenizer-CI8x8-Lidar` |
| 后训练权重 | `/data2/checkpoints/posttraining/` → 软链接 `checkpoints/posttraining` |
| 推理结果 | `dump_results/lidar_tokenizer/waymo_eval/` |
| 原始数据可视化目录 | `/data2/waymo_visualizations/raw/` |
