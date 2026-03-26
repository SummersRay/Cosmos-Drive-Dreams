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

# 软链接数据到 datasets/ 目录
ln -s /data2/rds_hq_waymo/lidar_tokenizer/training datasets/waymo_lidar_training
ln -s /data2/rds_hq_waymo/lidar_tokenizer/validation datasets/waymo_lidar_validation

# 生成 split 文件
ls datasets/waymo_lidar_training/lidar/ | sed 's/.tar//' > assets/lidar/waymo_train_split.lst
ls datasets/waymo_lidar_validation/lidar/ | sed 's/.tar//' > assets/lidar/waymo_val_split.lst
```

## 3. 后训练 LiDAR Tokenizer (8x A100-80G)

基于预训练的 Cosmos-Tokenizer-CI8x8-Lidar 在 Waymo 数据上 fine-tune。

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
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

**权重存储**：权重实际保存在 `/data2/checkpoints/posttraining/`，通过软链接映射到 `checkpoints/posttraining`。

**训练结果**（Validation Loss，前 5000 iter）：

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

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen

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

**推理参数**：
- `--tokenizer_dtype float32`：**必须指定**，默认 bfloat16 会导致 JIT 模型 dtype 不匹配报错
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
- 因此模型实际输入 shape 为 `3 x 512 x 1800`
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
