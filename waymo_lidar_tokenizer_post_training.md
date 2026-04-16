# Cosmos Drive Dreams - Waymo LiDAR Tokenizer 工作流

## 1. 总览

当前文档默认以 `CI8x8-Waymo + OpenSora-S1/S2/S3` 为主线，不再把 `V3/V4/V5` 作为新的起点。

推荐路线：
- Stage 1：`CI8x8-Waymo`
  先把 2D image tokenizer 适配到 Waymo TOP range map，建立稳定的单帧空间重建能力。
- Stage 2-S1：`OpenSora-S1`
  冻结 2D tokenizer，只训 temporal compressor，在固定 `29 -> 8 -> 29` 设定下做 latent-dominant 训练。
- Stage 2-S2：`OpenSora-S2`
  从 `S1` 最佳 checkpoint 接力，放开 `post_quant_conv + decoder` 做 joint finetune，仍以 latent 重建为主（`latent_recon=1.0, color=0.25`）。
- Stage 2-S3：`OpenSora-S3`
  从 `S2` 最佳 checkpoint 接力，切到像素重建主导（`color=1.0, latent_recon=0.25→0.05`）。

这条线只重新加载
[CI8x8-Waymo iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt)，
不复用历史 3D checkpoint；同时 temporal compressor 已经切到参考
[/root/workspace/Open-Sora/opensora/models/vae/vae_temporal.py](/root/workspace/Open-Sora/opensora/models/vae/vae_temporal.py:325)
的 `opensora_temporal_vae` backend。

### 1.1 当前推荐主线

| 阶段 | 实验名 | 初始化来源 | 说明 | 当前状态 |
|------|--------|------------|------|----------|
| Stage 1 | `cosmos_lidar_tokenizer_waymo` | 通用 `CI8x8-Lidar` | Waymo 2D tokenizer | 已完成 |
| S1 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s1` | `CI8x8-Waymo iter_000020000.pt` | 冻结 2D，只训 Open-Sora 风格 temporal VAE | `2026-04-16` 已完成 `30000` iter；总 loss / latent best 为 `iter_000025500.pt` |
| S2 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s2` | `OpenSora-S1-BypassFix iter_000029500.pt` | 放开 `post_quant_conv + decoder`，latent 主导 joint finetune | `2026-04-16` loss 权重修正后重新启动 |
| S3 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s3` | `OpenSora-S2 best checkpoint` | 切到像素重建主导 | 配置已就绪，待 `S2` 收敛后接力 |

### 1.2 阅读建议

- 如果是新实验，直接看 §2、§5.4、§7.11，不需要从 `V3/V4/V5` 接着训。
- 历史 `T29 / LRDecayFT / V3 / V4 / V5` 仍保留在后文，主要用于对比，不再作为当前推荐初始化来源。
- 当前固定设定只有一条：`29 -> 8 -> 29`，保留首帧旁路，不做变长。

### 1.3 历史实验怎么理解

- `T29 LatentCompressor` 到 `LRDecayFT-V4` 代表旧主线逐步放松 loss 和 decoder 约束的探索过程。
- `LRDecayFT-V4` 是旧主线里更值得回看的对照点，但它不再是新的初始化来源。
- `LRDecayFT-V5` 保留在文档里，主要用于和 `OpenSora-S2/S3` 的 bottleneck 放松思路做对照。

### 1.4 命名速查

| 名称片段 | 含义 |
|----------|------|
| `CI8x8` | image tokenizer，主要做空间压缩，通常对应单帧训练与推理 |
| `CV4x8x8` | video tokenizer，时间压缩 `4x`，空间压缩 `8x` |
| `Waymo` | 数据域切到 Waymo TOP LiDAR range map |
| `T17` | 总输入帧数为 `17 = 1 + 16` |
| `T29` | 总输入帧数为 `29 = 1 + 28` |
| `Streaming` | 启用了严格 `1 + 4k` 的 streaming video 路径 |
| `LatentCompressor` | 冻结 2D tokenizer，在 latent 空间做时序压缩与重建 |
| `Flow` | 在 baseline 基础上额外引入 `flow loss` 的实验分支 |

例子：
- `Waymo-T17-Streaming` 可以读成：
  `Waymo` 数据域 + `CV4x8x8` 视频 tokenizer + `17` 帧总输入 + streaming 训练/推理路径。

## 2. 运行环境与启动

统一环境：
- 工作目录：`/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen`
- conda 环境：`cosmos-predict1`
- 输出根目录：`checkpoints/`
- 当前推荐实验：`cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s1`

每次启动训练前先执行：

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints
```

### 2.1 单卡 smoke

推荐先用下面这条命令确认训练代码、数据和 loss 都能正常推进 iter：

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
  -m cosmos_predict1.tokenizer.training.train \
  --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
  experiment=cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s1 \
  job.group=debug \
  job.name=Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-Smoke1GPU-BypassFix \
  job.wandb_name=Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-Smoke1GPU-BypassFix \
  trainer.max_iter=2 \
  trainer.validation_iter=1000 \
  trainer.max_val_iter=1 \
  trainer.logging_iter=1 \
  checkpoint.save_iter=1000
```

已验证结果：
- `2026-04-16` 修复首帧 bypass 语义后，单卡 smoke 正常完成 `iter 0` 验证并推进到 `iter 2`。
- `2026-04-16` `OpenSora-S2` 从 `S1-BypassFix iter_000029500.pt` 接力的单卡 smoke 正常完成 2 个 iter。
- 输出目录：[Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-Smoke1GPU-BypassFix](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/debug/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-Smoke1GPU-BypassFix)

### 2.2 正式 8 卡启动

`OpenSora-S1-BypassFix` 的复现启动命令：

```bash
tmux new-session -d -s waymo_tok_t29_opensora_s1_bypassfix '/bin/zsh -lc "
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
export OUTPUT_ROOT=checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix
torchrun --standalone --nnodes=1 --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
  --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
  experiment=cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s1 \
  job.name=Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix \
  job.wandb_name=Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix \
  2>&1 | tee -a checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/stdout.log
"'
```

当前状态：
- `tmux` 会话 `waymo_tok_t29_opensora_s1_bypassfix` 已结束，训练已完成 `30000/30000`。
- 日志文件：[stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/stdout.log)

`OpenSora-S2` 当前正式启动命令：

```bash
tmux new-session -d -s waymo_tok_t29_opensora_s2 '/bin/zsh -lc "
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
export OUTPUT_ROOT=checkpoints
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/matplotlib
torchrun --master_port=29652 --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
  --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
  experiment=cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s2 \
  job.name=Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2 \
  job.wandb_name=Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2 \
  2>&1 | tee -a checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2/stdout.log
"'
```

当前状态：
- `tmux` 会话 `waymo_tok_t29_opensora_s2` 正在运行。
- CSV sidecar 会话 `waymo_tok_t29_opensora_s2_loss` 每 `300s` 刷新一次 `validation_loss_history.csv`。
- 日志文件：[stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2/stdout.log)
- CSV 文件：[validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2/validation_loss_history.csv)

### 2.3 续训与查看

自动续训条件：
- 同一 `job.name`
- 同一输出目录
- 目录里存在 `checkpoints/latest_checkpoint.txt`

常用命令：

```bash
tmux ls
tmux attach -t waymo_tok_t29_opensora_s2
tail -f checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2/stdout.log
```

补充说明：
- 只有历史 `flow` 分支才需要提前缓存 RAFT 权重；当前 `OpenSora-S1/S2` 配置里 `flow=off`，不依赖这一步。
- `S2` 建议优先看 `validation_loss_history.csv`、`stdout.log` 和每 `500` iter 的 checkpoint。

## 3. 数据转换

将 Waymo `rds_hq` 的 `lidar_raw` 转为 tokenizer 训练所需的 sparse range map。

输出格式：
- `metadata/{clip_id}.npz`
- `lidar/{clip_id}.tar`

Range map 规格：
- `128 x 3600`
- 使用 Waymo TOP LiDAR 的真实 64 条非均匀 beam inclination，插值到 128 行

```bash
cd /root/workspace/Cosmos-Drive-Dreams
conda activate cosmos-predict1

# 训练集
python cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py \
    --input_root /data2/rds_hq_waymo/training \
    --output_root /data2/rds_hq_waymo/lidar_tokenizer/training \
    --num_workers 16

# 验证集
python cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py \
    --input_root /data2/rds_hq_waymo/validation \
    --output_root /data2/rds_hq_waymo/lidar_tokenizer/validation \
    --num_workers 16
```

注意：
- 原始 `xyz` 在 vehicle frame；转换后 range map 位于 Waymo TOP LiDAR frame。
- metadata 中保存的 `pose_list` 是和输出 LiDAR frame 对齐后的 `lidar_to_world`。
- 当前数据规模：
  - train `798` clips
  - val `202` clips

## 4. 软链接与 Split

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

ln -s /data2/rds_hq_waymo/lidar_tokenizer/training datasets/waymo_lidar_training
ln -s /data2/rds_hq_waymo/lidar_tokenizer/validation datasets/waymo_lidar_validation

ls datasets/waymo_lidar_training/lidar/ | sed 's/.tar//' > assets/lidar/waymo_train_split.lst
ls datasets/waymo_lidar_validation/lidar/ | sed 's/.tar//' > assets/lidar/waymo_val_split.lst
```

## 5. 训练

### 5.1 Stage 1: `CI8x8-Waymo`

用途：
- 先做 Waymo 域适配
- 学好单帧 spatial reconstruction
- 为后续 3D streaming 提供更稳的初始化

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints

torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py \
    -- \
    experiment=cosmos_lidar_tokenizer_waymo
```

当前配置要点：
- 初始化权重：
  `checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/Cosmos-0.1-Tokenizer-CI8x8/autoencoder.pt`
- `max_iter=20000`
- `validation_iter=500`
- `max_val_iter=5`
- `precision=float32`
- train / val 都对齐到 `512 x 896`
- `strict_resume=False`

最终产物目录：
- [/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo)

### 5.2 Stage 2: `T29 LatentCompressor`

用途：
- 冻结 `CI8x8-Waymo` 的 2D tokenizer
- 只训练 latent-side temporal compressor
- 当前固定语义是 `29 -> 8 -> 29`
- 其中第 `1` 帧旁路保留，压缩的只有后 `28` 帧

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py \
    -- \
    experiment=cosmos_lidar_tokenizer_waymo_t29_latent_compressor \
    trainer.max_iter=40000
```

当前配置要点：
- 初始化权重：
  [iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt)
- 输入总帧数：`29 = 1 + 28`
- 时间压缩：`4x`
- latent 时间长度：`8 = 1 + 7`
- 空间尺寸：`512 x 896`
- `precision=bfloat16`
- 冻结模块：`encoder / quant_conv / post_quant_conv / decoder`
- 优化器只更新 temporal compressor 参数
- `strict_resume=False`

当前 loss 设计：
- `color + latent_recon + delayed flow + small kl`
- `latent_recon` 从 `iter 0` 生效，权重 `1.0`
- `flow` 在 `iter 5000` 后以 `0.002` 小权重开启（首次启用时会 lazy 下载 RAFT，见 §2.1 注意事项）
- `kl=1e-5`（对 posterior 做温和约束，避免 latent 长期漂移，方便后续接 diffusion prior）
- `perceptual.enabled=False`
- `video_consistency.enabled=False`
- `ema.enabled=False`

这是一版分离式 baseline，目标是先保住 2D 重建能力，再单独学习时序压缩。

当前产物目录：
- [/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor)

当前 smoke 结果：
- `1GPU` 训练 smoke 已通过：
  [/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-Debug1GPU](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-Debug1GPU)
- `8GPU` 训练 smoke 已通过：
  [/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-Smoke8GPU](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-Smoke8GPU)
- 固定 `29` 帧推理 smoke 已通过：
  [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_debug1gpu/range_map_video/10203656353524179475_7625_000_7645_000.mp4)
  / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_debug1gpu/histogram/10203656353524179475_7625_000_7645_000.png)

当前正式长训的 loss 走势和已回填推理结果统一收在 §7.5。日常追踪主要看这两个文件：
- [stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/stdout.log)
- [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/validation_loss_history.csv)

瘦身后 checkpoint 单文件 `49 MB`（对比 CI8x8-Waymo `iter_000020000.pt` 的 `916 MB`，见 §11）。

后期平台期微调分支：

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py \
    -- \
    experiment=cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft
```

`LRDecayFT` 配置要点：
- 从 [iter_000027000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/checkpoints/iter_000027000.pt) 只加载模型权重
- `checkpoint.load_training_state=False`
- `optimizer.lr=1e-5`
- `scheduler=warmup_cosine`
- `scheduler.warmup_iters=500`
- `scheduler.lr_decay_iters=13000`
- `scheduler.min_lr=1e-6`
- `trainer.max_iter=13000`
- `trainer.max_val_iter=10`
- `flow` 从微调起点即保持 `0.002`

当前 `LRDecayFT` 目录：
- [/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT)
- 最新稳定 checkpoint：
  [iter_000009000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/checkpoints/iter_000009000.pt)
- 当前 loss 记录：
  [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/validation_loss_history.csv)

### 5.3 历史实验

仓库中仍保留以下历史或分支实验配置，便于回看：
- `cosmos_lidar_tokenizer_cv4x8x8_waymo`
- `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29`
- `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_flow`
- `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_streaming`
- `cosmos_lidar_tokenizer_cv4x8x8_waymo_t15_streaming`

这些配置仍保留在仓库里供历史回看；当前推荐新主线见 §5.4 的 `OpenSora-S1 / S2 / S3`。

### 5.4 Open-Sora 重开版：`OpenSora-S1 / S2 / S3`

这是当前唯一推荐的新开线。它只复用
[CI8x8-Waymo iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt)，
不再承接历史 3D checkpoint。

直接参考：
- [Open-Sora VAE 报告](/root/workspace/Open-Sora/docs/zh_CN/vae.md)
- [stage1.py](/root/workspace/Open-Sora/configs/vae/train/stage1.py)
- [stage2.py](/root/workspace/Open-Sora/configs/vae/train/stage2.py)
- [stage3.py](/root/workspace/Open-Sora/configs/vae/train/stage3.py)
- [vae_temporal.py](/root/workspace/Open-Sora/opensora/models/vae/vae_temporal.py:325)

固定约束：
- 固定 `29 -> 8 -> 29`
- 保留首帧 bypass，但只在 decode 输出端覆盖首帧
- 不做变长
- 2D tokenizer 仍沿用 `CI8x8-Waymo`

| 子阶段 | 实验名 | 可训练模块 | 主要 loss | 当前状态 |
|--------|--------|------------|-----------|----------|
| S1 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s1` | `temporal_compressor` | `latent_recon=1.0`，`color=0.1`，`flow=off` | `2026-04-16` 已完成 `30000` iter；总 loss / latent best 为 `iter_000025500.pt`，S2 主接力用 `iter_000029500.pt` |
| S2 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s2` | `quant_conv + post_quant_conv + decoder + temporal_compressor` | `color=1.0`，`latent_recon=0.25`，`temporal_delta=0.25`，`flow=off` | 已启动 8 卡训练，`iteration 0` validation 通过 |
| S3 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_opensora_s3` | 同 S2 | `color=1.0`，`latent_recon=0.05 -> 0.0`，`temporal_delta=0.4`，`flow=off` | 配置已就绪，等待 `S2` 最佳 checkpoint |

补充：
- 当前真正“从零重开”的只有 `S1`，并已完成一轮完整 `30000` iter 训练。
- `S2` 的 `checkpoint.load_path` 已替换成 `S1-BypassFix iter_000029500.pt`；`S3` 仍保留 `REPLACE_WITH_BEST_S2_CHECKPOINT.pt`，等待 `S2` 收敛后再替换。
- 这条线迁移的是 Open-Sora 的 `stage-wise objective switch + temporal VAE structure`，不是它的变长训练策略。

## 6. 推理

### 6.1 `CI8x8-Waymo`

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

python -m cosmos_predict1.tokenizer.inference.lidar_cli \
    --sample_path="/data2/rds_hq_waymo/lidar_tokenizer/validation/lidar/<clip_id>.tar" \
    --enc_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000_enc.jit" \
    --dec_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000_dec.jit" \
    --output_folder="waymo_eval_ci8x8_waymo_iter20000" \
    --tokenizer_dtype float32 \
    --max_frames 20 \
    --waymo_top \
    --display_frame vehicle
```

### 6.2 `T29 LatentCompressor`

该方案只支持完整 `.pt + config.yaml` 推理，不走 `enc/dec` 分离路径，也不支持变长输入。

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
conda activate cosmos-predict1

python -m cosmos_predict1.tokenizer.inference.lidar_cli \
    --sample_path="/data2/rds_hq_waymo/lidar_tokenizer/validation/lidar/<clip_id>.tar" \
    --model_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/checkpoints/<iter>.pt" \
    --tokenizer_config_path="checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/config.yaml" \
    --output_folder="waymo_eval_t29_latent_<iter>" \
    --tokenizer_type video \
    --temporal_window 29 \
    --tokenizer_dtype bfloat16 \
    --max_frames 29 \
    --vis_pcd 0 \
    --waymo_top \
    --display_frame vehicle
```

推理参数要点：
- `--temporal_window=29`
- `--max_frames=29`
- 固定语义是 `29 -> 8 -> 29`
- 第 `1` 帧是精确保留的 2D latent，不参与时间压缩
- 当前 v1 不支持变长和长序列 streaming
- 点云可视化当前使用 Plotly，并恢复为并行渲染路径

## 7. 实验记录（训练 + 推理合并）

统一说明：
- 当前已回填的历史样本统一使用 `10203656353524179475_7625_000_7645_000`
- smoke 推理结果不放进正式实验记录
- 每个实验都按“训练设置 / 推理记录”放在一起，方便横向比较

### 7.1 `CI8x8-Waymo`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo` |
| 任务形态 | 2D image tokenizer，单帧空间重建 |
| 初始化来源 | 通用 `CI8x8-Lidar` `autoencoder.pt` |
| 训练输入 | clip 内采样 `10` 帧，train / val 统一 `512 x 896` |
| 压缩语义 | spatial `8x` |
| 精度 | `float32` |
| 训练轮次 | `max_iter=20000`，`validation_iter=500`，`max_val_iter=5`，`save_iter=1000` |
| 关键设置 | `strict_resume=False` |
| 当前状态 | 已完成，作为 stage-1 最终权重 |
| 权重目录 | [Cosmos-LidarTokenizer-CI8x8-Waymo](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo) |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000020000` | `20` 帧，image tokenizer | `0.20 m` | `0.09 m` | `0.00` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_ci8x8_waymo_iter20000/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_ci8x8_waymo_iter20000/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_ci8x8_waymo_iter20000/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | Stage-1 最终结果 |

### 7.2 旧版 `CV4x8x8-Waymo`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_cv4x8x8_waymo` |
| 任务形态 | 端到端 3D video tokenizer，旧版 causal 语义 |
| 初始化来源 | 通用 `CI8x8-Lidar` |
| 训练输入 | 旧版总输入 `9` 帧 |
| 压缩语义 | `9 -> 3 -> 9` |
| 精度 | `bfloat16` |
| 训练轮次 | `max_iter=20000` |
| 当前状态 | 历史完成 |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000020000` | 旧版 video eval，默认 `20` 帧 | `10.16 m` | `4.20 m` | `0.14` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_cv4x8x8_iter20000/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_cv4x8x8_iter20000/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_cv4x8x8_iter20000/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | 旧版 `9` 帧 causal 语义 |

### 7.3 `T15 Streaming`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_cv4x8x8_waymo_t15_streaming` |
| 任务形态 | streaming video tokenizer，legacy ragged-tail |
| 初始化来源 | 通用 `CI8x8-Lidar` |
| 训练输入 | 总输入 `15 = 1 + 14` |
| 压缩语义 | `15 -> 4 -> 15` |
| 精度 | `bfloat16` |
| 关键设置 | 非严格 `1 + 4k`，尾部存在 ragged tail |
| 当前状态 | 历史暂停 |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000012000` | `29` 帧推理（`15` 训 `29` 推） | `13.52 m` | `7.25 m` | `0.27` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t15_streaming_latest_29f/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t15_streaming_latest_29f/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t15_streaming_latest_29f/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | 后半段质量偏弱 |

### 7.4 `T17 Streaming`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_cv4x8x8_waymo_t17_streaming` |
| 任务形态 | 严格 `1 + 4k` streaming video tokenizer |
| 初始化来源 | `CI8x8-Waymo iter_000020000.pt` |
| 训练输入 | 总输入 `17 = 1 + 16`，`512 x 896` |
| 压缩语义 | `17 -> 5 -> 17` |
| 精度 | `bfloat16` |
| 关键设置 | `streaming_raw_chunk_size=4`，`streaming_latent_chunk_size=1`，`streaming_detach_cache=True` |
| loss 取向 | 保守 baseline，主要验证 streaming 结构 |
| 当前状态 | 历史暂停，最新稳定 checkpoint 为 `iter_000017000.pt` |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000007000` | `29` 帧推理（`17` 训 `29` 推） | `11.54 m` | `6.23 m` | `0.23` | 无固定产物目录 | 历史终端记录 |
| `iter_000017000` | `29` 帧推理（`17` 训 `29` 推） | `8.17 m` | `4.03 m` | `0.15` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t17_streaming_iter17000_29f/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t17_streaming_iter17000_29f/histogram/10203656353524179475_7625_000_7645_000.png) | 当前最新稳定对比结果；点云视频未成功生成 |

### 7.5 `T29 LatentCompressor`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor` |
| 任务形态 | 冻结 2D tokenizer + latent-side temporal compressor |
| 初始化来源 | [CI8x8-Waymo iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt) |
| 训练输入 | 固定总输入 `29 = 1 + 28`，空间 `512 x 896` |
| 压缩语义 | `29 -> 8 -> 29`，第 `1` 帧旁路保留、不参与压缩 |
| 结构设置 | 冻结 `encoder / quant_conv / post_quant_conv / decoder`，只训练 temporal compressor |
| 精度 | `bfloat16` |
| 训练轮次 | 当前续训总目标 `max_iter=40000`，`validation_iter=500`，`max_val_iter=5`，`save_iter=1000` |
| loss 设置 | `color=1.0`，`latent_recon=1.0`，`flow` 在 `iter 5000` 后开启且权重 `0.002`，`kl=1e-5`，`perceptual/video_consistency/ema=off` |
| 关键设置 | `strict_resume=False`，checkpoint 瘦身后单文件约 `49 MB` |
| 日志记录 | [stdout.log](./cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/stdout.log) / [validation_loss_history.csv](./cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/validation_loss_history.csv) |
| 当前状态 | 历史主线；`2026-04-12` 从 `iter_000017000.pt` 续训到总 `40000` iter |

训练记录（节选）：

| iter | val loss | 备注 |
|------|----------|------|
| 0 | `3.930` | 初始验证 |
| 2500 | `1.156` | 前期快速下降 |
| 4500 | `0.909` | flow 启用前最低点之一 |
| 5000 | —— | validation 阶段因 RAFT 下载中断 |
| 5500 | `0.893` | resume 后 flow 首次生效 |
| 17000 | `0.582` | 第一次 SIGKILL 前最新 checkpoint |
| 17500 | `0.572` | 从 iter_000017000 续训后首个 validation |
| 18500 | `0.533` | |
| 19500 | `0.519` | 当前最佳 |
| 20000 | `0.556` | |
| 21000 | `0.527` | |
| 22000 | `0.533` | 最新（训练仍在进行中，目标 40000） |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000017000` | 固定 `29 -> 8 -> 29` | `7.60 m` | `3.20 m` | `0.11` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_iter17000/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_iter17000/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_iter17000/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | 当前正式长训阶段的 `17000` 轮结果；点云恢复到最开始的 Plotly 并行渲染观感 |

### 7.6 `T29 LatentCompressor LRDecayFT`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft` |
| 任务形态 | `T29 LatentCompressor` 的后期降 LR 微调分支 |
| 初始化来源 | [T29 LatentCompressor iter_000027000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/checkpoints/iter_000027000.pt) |
| 训练输入 | 固定总输入 `29 = 1 + 28`，空间 `512 x 896` |
| 压缩语义 | `29 -> 8 -> 29`，第 `1` 帧旁路保留、不参与压缩 |
| 精度 | `bfloat16` |
| 优化设置 | `optimizer.lr=1e-5`，`scheduler=warmup_cosine`，`warmup_iters=500`，`lr_decay_iters=13000`，`min_lr=1e-6` |
| 验证设置 | `validation_iter=500`，`max_val_iter=10` |
| loss 设置 | 沿用 `color + latent_recon + flow + small kl`；`flow=0.002` 从微调起点即开启 |
| 关键设置 | `checkpoint.load_training_state=False`，只加载模型权重；用于处理 base run 后期平台 |
| 日志记录 | [stdout.log](./cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/stdout.log) / [validation_loss_history.csv](./cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/validation_loss_history.csv) |
| 当前状态 | 当前最新稳定 checkpoint 为 `iter_000009000.pt`；后续续训多次在 validation 后被 `SIGKILL` |

训练记录（节选）：

| iter | val loss | 备注 |
|------|----------|------|
| 0 | `0.499023` | 从 `iter_000027000.pt` 只加载模型权重开始微调 |
| 2000 | `0.478271` | 较起点已有改善 |
| 5000 | `0.477881` | 持续缓慢下降 |
| 7000 | `0.477783` | 局部平台 |
| 8500 | `0.484204` | 小幅反弹 |
| 9000 | `0.461572` | 当前最佳；之后在 `9500` validation 后被 `SIGKILL` |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000009000` | 固定 `29 -> 8 -> 29` | `6.79 m` | `2.51 m` | `0.08` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_lrdecayft_iter9000/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_lrdecayft_iter9000/histogram/10203656353524179475_7625_000_7645_000.png) | 当前这条微调分支的最新稳定单样本结果；本次未渲染点云视频 |

### 7.7 `T29 LatentCompressor LRDecayFT-V2`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v2` |
| 任务形态 | `LRDecayFT` 的诊断分支，进一步降低 LR / flow，并关闭 validation media |
| 初始化来源 | [LRDecayFT iter_000010000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/checkpoints/iter_000010000.pt) |
| 训练输入 | 固定总输入 `29 = 1 + 28`，空间 `512 x 896` |
| 压缩语义 | `29 -> 8 -> 29`，第 `1` 帧旁路保留、不参与压缩 |
| 精度 | `bfloat16` |
| 优化设置 | `optimizer.lr=5e-6`，`scheduler=warmup_cosine`，`warmup_iters=200`，`lr_decay_iters=10000`，`min_lr=5e-7` |
| 验证设置 | `validation_iter=500`，`max_val_iter=10`，`job.wandb_log_validation_media=False` |
| loss 设置 | `color + latent_recon + flow + small kl`，其中 `flow=0.001` |
| 日志记录 | [stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V2/stdout.log) / [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V2/validation_loss_history.csv) |
| 当前状态 | 最新稳定 checkpoint 为 [iter_000005000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V2/checkpoints/iter_000005000.pt)；`5500` 轮 validation 后被 `SIGKILL` |

训练记录（节选）：

| iter | val loss | 备注 |
|------|----------|------|
| 0 | `0.473596` | 从 `LRDecayFT iter_000010000.pt` 只加载模型权重开始 |
| 1000 | `0.468445` | |
| 1500 | `0.467749` | |
| 2000 | `0.464075` | 当前已记录最佳点 |
| 3000 | `0.488928` | 中途有明显震荡 |
| 4000 | `0.479651` | |
| 5000 | `0.466650` | 当前最新稳定 checkpoint |
| 5500 | —— | validation 结束后 `SIGKILL`，未写入新 checkpoint |

推理记录：

| Checkpoint | 推理设置 | RMSE | MAE | Rel | 产物 | 备注 |
|------------|----------|------|-----|-----|------|------|
| `iter_000005000` | 固定 `29 -> 8 -> 29` | `6.77 m` | `2.48 m` | `0.08` | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_lrdecayft_v2_iter5000/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_lrdecayft_v2_iter5000/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_t29_latent_compressor_lrdecayft_v2_iter5000/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | 当前诊断分支的最新稳定单样本结果；已补齐点云视频 |

### 7.8 `T29 LatentCompressor LRDecayFT-V3`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v3` |
| 任务形态 | 平台期优化分支：`masked latent_recon + temporal_delta` |
| 初始化来源 | [LRDecayFT iter_000009000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT/checkpoints/iter_000009000.pt) |
| 训练输入 | 固定总输入 `29 = 1 + 28`，空间 `512 x 896` |
| 压缩语义 | `29 -> 8 -> 29`，第 `1` 帧旁路保留、不参与压缩 |
| 精度 | `bfloat16` |
| 优化设置 | `optimizer.lr=5e-6`，`scheduler=warmup_cosine`，`warmup_iters=200`，`lr_decay_iters=12000`，`min_lr=5e-7` |
| 验证设置 | `validation_iter=500`，`max_val_iter=5`，合并导出 `validation_loss + depth metrics` 到同一个 CSV |
| loss 设置 | `color=1.0`，`latent_recon` 在 `iter 5000` 后从 `1.0` 降到 `0.25`，`temporal_delta=0.25`，`flow=0.0005`，`small kl` 保留 |
| 关键设置 | latent mask 缩放改为 conservative max-pooling；validation media 关闭；更建议按聚合 `depth_mae / depth_rmse / depth_relative_error` 选 checkpoint |
| 日志记录 | [stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V3/stdout.log) / [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V3/validation_loss_history.csv) |
| 当前状态 | 长训中；当前按聚合 depth 指标综合最优点为 `iter_000007500.pt` |

训练记录（节选）：

| iter | val loss | depth_mae | depth_rmse | depth_rel | 备注 |
|------|----------|-----------|------------|-----------|------|
| 0 | `0.472730` | `1.888650` | `5.086497` | `0.075810` | 从 `LRDecayFT iter_000009000.pt` 只加载模型权重开始 |
| 500 | `0.510107` | `2.140552` | `5.522050` | `0.081036` | 初期波动较大 |
| 1000 | `0.471826` | `1.807986` | `4.916105` | `0.073225` | |
| 2000 | `0.461426` | `1.976791` | `5.347620` | `0.076936` | |
| 3500 | `0.472705` | `1.893045` | `5.074972` | `0.073880` | |
| 5000 | `0.146094` | `1.896370` | `5.140749` | `0.070589` | `latent_recon` 权重从 `1.0` 切到 `0.25`，总 loss 出现阶跃下降 |
| 6500 | `0.160889` | `2.126555` | `5.586620` | `0.076774` | 平台区内局部最差点之一 |
| 7000 | `0.146411` | `1.831076` | `5.017861` | `0.071688` | 指标回落到较好区间 |
| 7500 | `0.142371` | `1.740864` | `4.709203` | `0.071459` | 当前综合最佳点；`depth_mae / depth_rmse / val loss` 都在这里最优 |
| 8000 | `0.148547` | `1.820969` | `4.907019` | `0.071134` | `depth_relative_error` 接近最佳，但 `mae/rmse` 略回弹 |
| 8500 | `0.150757` | `1.934805` | `5.231071` | `0.074168` | 进入平台震荡 |

说明：

- `5000` 之后总 `validation_loss` 的大幅下降，主要来自 loss 权重切换，不应直接等价理解成模型质量同幅度提升。
- 这条线当前更应该看聚合 `depth_mae / depth_rmse / depth_relative_error`，因为它们和推理脚本里的 `MAE / RMSE / Rel error` 是同口径指标。
- 到 `8500` 为止，`V3` 已经体现出提升，但 `7000 ~ 8500` 更像平台+抖动区，而不是继续稳定下降。

### 7.9 `T29 LatentCompressor LRDecayFT-V4`

训练设置：

| 项目 | 设置 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v4` |
| 任务形态 | 方案 A：保持 latent compressor 架构，轻量 joint finetune decoder 侧 |
| 初始化来源 | [V3 iter_000008000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V3/checkpoints/iter_000008000.pt) |
| 解冻模块 | `post_quant_conv`、`decoder` |
| 学习率策略 | compressor 主 LR=`5e-6`，解冻的 image tokenizer 模块使用 `lr_scale=0.2` |
| 训练输入 | 固定总输入 `29 = 1 + 28`，空间 `512 x 896` |
| 压缩语义 | `29 -> 8 -> 29`，第 `1` 帧旁路保留、不参与压缩 |
| 精度 | `bfloat16` |
| 验证设置 | `validation_iter=500`，`max_val_iter=5`，`wandb offline`，validation media 关闭，`dataloader_val.prefetch_factor=2` |
| loss 设置 | 继承 `V3`：`color + masked latent_recon + temporal_delta + low-weight flow + small kl` |
| 日志记录 | [stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/stdout.log) / [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/validation_loss_history.csv) |
| 当前状态 | 已稳定写出多个 checkpoint；当前 `depth_mae / depth_rmse` 最优点是 [iter_000005500.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/checkpoints/iter_000005500.pt)，`depth_relative_error` 最优点是 [iter_000008000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/checkpoints/iter_000008000.pt) |

说明：

- `V4` 不是重新发明结构，而是在现有 `T29 LatentCompressor` 上把“只训 temporal compressor”升级成“冻结 encoder/quant_conv，轻量 joint finetune decoder 侧”的版本。
- 第一轮启动暴露了一个代码问题：优化器分组参数经过 `LazyConfig` 后混入了 `DictConfig`，导致 `TypeError: optimizer can only optimize Tensors`。该问题已在 [training/model.py](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/model.py) 修复。
- 采用更干净的启动方式并把 `dataloader_val.prefetch_factor` 降到 `2` 后，`V4` 已能稳定跑过多个 validation 节点；系统 `SIGKILL` 风险有所缓解，但底层根因仍未完全定位。
- `V4` 当前更应该按聚合 `depth_mae / depth_rmse / depth_relative_error` 选 checkpoint，而不是只盯总 `validation_loss`；从曲线看它已经进入平台震荡区，但仍整体优于 `V3`。

关键训练节点：

| iter | val loss | depth_mae | depth_rmse | depth_rel | 备注 |
|------|----------|-----------|------------|-----------|------|
| 4500 | `0.480957` | `1.646758` | `4.710167` | `0.060560` | `5000` 前的旧权重 regime |
| 5000 | `0.148096` | `1.668455` | `4.757467` | `0.062871` | `latent_recon` 权重切到 `0.25` 后的首个点，loss 存在阶跃 |
| 5500 | `0.134204` | `1.375925` | `4.167069` | `0.055805` | 当前 `depth_mae / depth_rmse` 最优点 |
| 6500 | `0.133508` | `1.523177` | `4.618411` | `0.059177` | 总 loss 最低点，但 depth 指标不如 `5500` |
| 8000 | `0.135413` | `1.417164` | `4.270668` | `0.054947` | 当前 `depth_relative_error` 最优点 |
| 9000 | `0.139575` | `1.458441` | `4.320888` | `0.058258` | 仍在平台震荡区内，未明显刷新 best |

判断：

- `V4` 的 decoder-side finetune 是有效的，尤其在 `depth_mae / depth_rmse / depth_relative_error` 上都优于 `V3`。
- 但这条线到 `5500 ~ 9000` 已经很像平台期：不同 depth 指标的最优点开始分裂，说明继续单纯加 iter 的边际收益在变小。
- 因此下一步更合适的是基于 `V4` 的 best checkpoint 继续放松 bottleneck 约束，而不是继续在 `V4` 上无脑堆轮数。

### 7.10 `T29 LatentCompressor LRDecayFT-V5`（设计中）

设计目标：

- 在 `V4` 证明“轻量 joint finetune decoder 侧”有效之后，进一步验证 plateau 是否来自 **冻结 `quant_conv` 让 bottleneck 分布过于僵硬**。
- 继续冻结 `encoder`，但把可训练模块从 `post_quant_conv + decoder` 扩展到 `quant_conv + post_quant_conv + decoder`。
- 让目标函数更偏向最终 range 重建和时序一致性，而不是持续被 `latent_recon` 强约束住。

拟定训练设置：

| 项目 | 设计 |
|------|------|
| 实验名 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v5` |
| 初始化来源 | `V4 iter_000008000.pt` |
| 冻结模块 | `encoder` |
| 可训练模块 | `quant_conv`、`post_quant_conv`、`decoder`、`temporal_compressor` |
| 学习率策略 | compressor 主 LR=`3e-6`；image tokenizer 模块 `lr_scale=0.1` |
| checkpoint 保存 | `save_iter=500` |
| loss 设置 | `color + latent_recon + temporal_delta + small kl`，关闭 `flow` |
| `latent_recon` | 先保持 `0.25`，在 `iter 6500` 后降到 `0.1` |
| `temporal_delta` | 从 `0.25` 提到 `0.35` |
| 验证设置 | `validation_iter=500`，`max_val_iter=5`，`prefetch_factor=2`，`wandb offline` |
| 选模指标 | 以聚合 `depth_mae / depth_rmse / depth_relative_error` 为主，不再以总 `validation_loss` 为主 |

设计动机：

- `V4` 已经说明只解冻 decoder 侧可以继续改善 depth 指标，但平台期来得很快。
- 当前 `latent_recon` 仍然主导总 loss，而 target latent 又来自冻结的 `quant_conv` 路径，这会限制 temporal compressor 和 decoder 为最终 range 重建做更有利的偏移。
- `V5` 的核心假设是：**在仍然冻结 encoder 的前提下，放开 `quant_conv` 可以让 bottleneck 映射与 decoder 一起协同适配时序压缩后的 latent 分布**。
- 同时关闭 `flow` 可以去掉一个域不完全匹配、但会增加验证负载的项，让训练目标更聚焦。

### 7.11 `OpenSora-S1 / S2 / S3`（推荐的新开线）

这条线只重新加载 `CI8x8-Waymo` 的 2D 权重，并把 stage-2 明确拆成 `S1 / S2 / S3` 三段；目标是把旧主线里“训练目标切换过慢”的问题拆开处理。

| 项目 | 记录 |
|------|------|
| 2D 初始化 | [CI8x8-Waymo iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt) |
| 3D 结构 | `opensora_temporal_vae` |
| 固定设定 | `29 -> 8 -> 29`，首帧 bypass，不做变长 |
| S1 训练模块 | `temporal_compressor` |
| S2 / S3 训练模块 | `post_quant_conv + decoder + temporal_compressor`（`quant_conv` 只在 encode 路径使用，encode 在 `no_grad` 下运行，解冻无效） |
| 当前状态 | `2026-04-16` 已完成 `30000` iter；后半段进入平台震荡 |
| 正式输出目录 | [Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix) |
| 正式日志 | [stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/stdout.log) |
| Loss CSV | [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/validation_loss_history.csv) |
| Loss 曲线 | [validation_loss_curve.png](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/validation_loss_curve.png) |
| S1 latent best checkpoint | [iter_000025500.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/checkpoints/iter_000025500.pt) |
| S2 主接力 checkpoint | [iter_000029500.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/checkpoints/iter_000029500.pt) |
| S2 输出目录 | [Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S2) |
| 当前结论 | `S1` 已进入平台期；`S2` 已从 `iter_000029500.pt` 接力启动 |

S1 validation 关键节点：

| iter | val loss | latent recon | color | depth mae | depth rmse | depth rel | 备注 |
|------|----------|--------------|-------|-----------|------------|-----------|------|
| 0 | `3.170703` | `3.092578` | `0.076831` | `27.610565` | `30.429682` | `2.157027` | 初始点 |
| 500 | `1.676172` | `1.658594` | `0.017456` | `6.464605` | `11.057816` | `0.289883` | 早期快速收敛 |
| 5000 | `0.640625` | `0.635352` | `0.005368` | `3.696425` | `7.856123` | `0.138406` | 进入稳定下降段 |
| 16000 | `0.484302` | `0.481348` | `0.003145` | `2.205369` | `5.215471` | `0.093866` | depth rmse 最优 |
| 25500 | `0.448047` | `0.445093` | `0.003004` | `2.183936` | `5.502246` | `0.084676` | 总 loss / latent recon / color 最优 |
| 29500 | `0.452832` | `0.449951` | `0.003064` | `2.165261` | `5.509322` | `0.084322` | depth mae 最优 |
| 30000 | `0.457764` | `0.454492` | `0.003161` | `2.223384` | `5.615626` | `0.083239` | depth rel 最优 |

选模建议：
- `S2` 主接力使用 [iter_000029500.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/checkpoints/iter_000029500.pt)，因为 `S2` 会转向 raw/depth reconstruction，`29500` 是 S1 validation `depth_mae` 最低点，并且单条 29 帧样本推理优于 `25500/30000`。
- [iter_000025500.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-OpenSora-S1-BypassFix/checkpoints/iter_000025500.pt) 继续保留为 latent best 对照；如果 `S2` 从 `29500` 接力不稳定，再回退到该 checkpoint。
- `25500` 之后 `validation_loss` 基本平台震荡，最终 `30000` 比 best 高约 `2.17%`，没有明显 collapse。

S2 配置修正记录（`2026-04-16`）：
- **移除 `quant_conv`**：`quant_conv` 只在 encode 路径使用，encode 跑在 `torch.no_grad()` 下，永远不会收到梯度。标记为 trainable 只会浪费 optimizer 内存和 checkpoint 空间。
- **loss 重心改为 latent 主导**：对齐 OpenSora 分阶段思路，S2 仍以 `latent_recon=1.0` 为主，`color=0.25` 辅助，`temporal_delta=0.1`。像素主导推到 S3（`color=1.0`）。
- **S3 同步调整**：显式设 `color=1.0`，`latent_recon` 从 `0.25` 在 iter 1000 后衰减到 `0.05`（不完全丢弃 latent 约束），`temporal_delta=0.25`。
- 之前基于旧 loss 权重的 S2 试跑数据已清除，重新从 `iter_000029500.pt` 启动。

三阶段 loss 对比（修正后）：

| | S1 | S2 | S3 |
|---|---|---|---|
| color | 0.1 | 0.25 | **1.0** |
| latent_recon | **1.0** | **1.0** | 0.25→0.05 |
| temporal_delta | off | 0.1 | 0.25 |
| kl | 1e-5 | 1e-5 | 1e-5 |
| 解冻模块 | 无 | post_quant_conv + decoder | post_quant_conv + decoder |

渐进思路：S1 纯 latent → S2 latent 主导 + decoder 适应 → S3 像素主导 + latent 收尾。

Open-Sora 对应参考：
- [Open-Sora VAE 报告](/root/workspace/Open-Sora/docs/zh_CN/vae.md)
- [stage1.py](/root/workspace/Open-Sora/configs/vae/train/stage1.py)
- [stage2.py](/root/workspace/Open-Sora/configs/vae/train/stage2.py)
- [stage3.py](/root/workspace/Open-Sora/configs/vae/train/stage3.py)
- [vae_temporal.py](/root/workspace/Open-Sora/opensora/models/vae/vae_temporal.py:325)

## 8. 原始数据可视化

```bash
cd /root/workspace/Cosmos-Drive-Dreams
conda activate cosmos-predict1

python cosmos-drive-dreams-toolkits/visualize_waymo_rangemap.py \
    --tar_path="<lidar_tar_path>" \
    --output_dir="/data2/waymo_visualizations/raw"

python cosmos-drive-dreams-toolkits/visualize_waymo_rangemap.py \
    --tar_path="<lidar_tar_path>" \
    --output_dir="/data2/waymo_visualizations/raw" \
    --vis_pcd \
    --camera_view front_view \
    --display_frame vehicle \
    --max_frames 20
```

常用参数：
- `--camera_view front_view` 或 `top_down_view`
- `--display_frame vehicle`
- `--max_frames 20`

## 9. 当前已知限制

- `OpenSora-S1` 当前以 `BypassFix` 版本作为正式基线，已有稳定 checkpoint 和 29 帧离线推理可视化结果。
- 当前新主线是**有意保持固定 `29` 帧**，不引入变长；这不是遗漏，而是为了先把 Open-Sora 的 stage-wise objective switch 在 fixed-29 设定里验证清楚。
- `OpenSora-S2` 已指向 `S1-BypassFix iter_000029500.pt`，loss 权重已修正为 latent 主导（`2026-04-16`）；`OpenSora-S3` 仍是 fail-fast 占位值，等 `S2` best 出来后再替换。
- 历史 `flow` 分支首次进入相关 validation 时仍依赖 RAFT 权重缓存；当前 `OpenSora-S1` 因为 `flow=off` 不受这件事影响。
- 历史实验里“单条样本推理指标”和“聚合 validation depth 指标”不是同一口径，横向比较时要区分。
- 当前环境已补齐 `ffmpeg` 和 Kaleido/Chrome 依赖，点云视频渲染可以跑通；常规快速检查仍建议先看 `range_map_video + histogram`。

## 10. 关键文件

| 文件 | 说明 |
|------|------|
| `cosmos-drive-dreams-toolkits/convert_waymo_lidar_to_tokenizer_format.py` | Waymo → tokenizer 格式转换 |
| `cosmos-drive-dreams-toolkits/visualize_waymo_rangemap.py` | Range map / 点云可视化 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/configs/experiments/cosmos_lidar_tokenizer_waymo.py` | Waymo 训练实验配置 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/datasets/lidar_datasets/configs.py` | Waymo 数据配置 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/networks/latent_temporal_compressor_video.py` | 新的冻结 2D + latent temporal compressor 主体 |
| `cosmos-transfer-lidargen/cosmos_predict1/tokenizer/networks/continuous_video.py` | Streaming video tokenizer |
| `cosmos-transfer-lidargen/assets/lidar/waymo_train_split.lst` | train split |
| `cosmos-transfer-lidargen/assets/lidar/waymo_val_split.lst` | val split |

## 11. 本轮代码审阅改动（2026-04-11）

一次集中评审后对 T29 LatentCompressor 主线做的 4 处小改动，都已随长训启动生效：

1. **`train()` 每次都重新 freeze 2D backbone**
   [latent_temporal_compressor_video.py](cosmos-transfer-lidargen/cosmos_predict1/tokenizer/networks/latent_temporal_compressor_video.py) 的 `train()` override 改为每次切换 train/eval 模式都调一次 `_freeze_image_tokenizer()`，杜绝构造后 requires_grad 被意外打开。

2. **`decode()` 接收可选 `exact_context_latent`**
   `_decode_temporal_latents` / `decode` 新增 `exact_context_latent=None` 参数；`forward()` 里显式传入 `target_latents[:, :, :1]`，保证即便调用方只拿到 `temporal_compressor.encode` 的输出也能正确还原首帧旁路。缺省行为完全向后兼容。

3. **Checkpoint 瘦身：不再存冻结的 2D 权重**
   `LatentTemporalCompressorVideoTokenizer.frozen_state_dict_prefixes()` 返回 `("image_tokenizer.",)`，[training/model.py](cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/model.py) 的 `TokenizerModel.state_dict` 检测到这个 hook 后把对应前缀从保存的 state_dict 里剔除。实测：T29 LatentCompressor 的 ckpt 从预计 ~950 MB 压到 **49 MB**，20k iter 全程预估少写 ~17 GB。Resume 路径同步生效：瘦身后的 ckpt 不含 2D 权重，`image_tokenizer` 在 `__init__` 里从 `frozen_image_tokenizer_ckpt` 重新加载，`TokenizerModel.load_state_dict` 的 `own_state - filtered_state_dict` 集合里也不会再出现 `image_tokenizer.*`，不会报 missing key。已在本次 resume（iter 5000）验证过。

4. **`kl` 权重从 `0.0` 改为 `1e-5`**
   给 posterior 一个温和约束，避免 latent 分布长期漂移；为后续接 diffusion prior 省一版重训。对当前 val loss 曲线影响观测不到（量级远低于 color + latent_recon）。

LTCV 单测 `pytest cosmos-transfer-lidargen/cosmos_predict1/tokenizer/networks/latent_temporal_compressor_video_test.py -q` 改动后仍然 3/3 绿。

## 12. 数据路径

| 数据 | 路径 |
|------|------|
| Waymo 原始训练集 | `/data2/rds_hq_waymo/training/` |
| Waymo 原始验证集 | `/data2/rds_hq_waymo/validation/` |
| tokenizer 训练集 | `/data2/rds_hq_waymo/lidar_tokenizer/training/` |
| tokenizer 验证集 | `/data2/rds_hq_waymo/lidar_tokenizer/validation/` |
| 预训练权重 | `/data2/checkpoints/Cosmos-Tokenizer-CI8x8-Lidar/` |
| 后训练权重 | `/data2/checkpoints/posttraining/` |
| 推理结果 | `/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/` |
| 原始数据可视化 | `/data2/waymo_visualizations/raw/` |
