# Cosmos Drive Dreams - Waymo LiDAR Tokenizer 工作流

## 1. 总览

当前推荐路线是一个两阶段方案：
- Stage 1：`CI8x8-Waymo`
  先把 2D image tokenizer 适配到 Waymo TOP range map，建立稳定的单帧空间重建能力。
- Stage 2：`T29 LatentCompressor`
  冻结 `CI8x8-Waymo` 的 2D tokenizer，仅训练 latent-side temporal compressor，在 latent 空间做 `1 + 28 -> 1 + 7 -> 1 + 28`。

### 1.1 当前主线

| 阶段 | 实验名 | 作用 | 初始化来源 | 当前状态 |
|------|--------|------|------------|----------|
| Stage 1 | `cosmos_lidar_tokenizer_waymo` | Waymo 2D 域适配 | 通用 `CI8x8-Lidar` | 已完成 |
| Stage 2 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor` | 冻结 2D tokenizer + latent-side temporal compressor | `CI8x8-Waymo iter_000020000.pt` | 当前主线，长训中 |
| Stage 2-FT | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft` | 后期降 LR 微调变体 | `T29 LatentCompressor iter_000027000.pt` | 长训中，当前从 `iter_000009000.pt` 续到总 `20000` iter |
| Stage 2-FT-V2 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v2` | 更小 LR + 更低 flow + validation media 关闭 | `LRDecayFT iter_000010000.pt` | 当前最新稳定 checkpoint 为 `iter_000005000.pt`；`5500` 轮 validation 后被 `SIGKILL` |
| Stage 2-FT-V3 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v3` | `masked latent_recon + temporal_delta` 优化分支 | `LRDecayFT iter_000009000.pt` | 8 卡长训中；`validation_iter=500`，当前按聚合 depth 指标的最佳点是 `iter_000007500.pt` |
| Stage 2-FT-V4 | `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v4` | 方案 A：轻量 joint finetune decoder 分支 | `V3 iter_000008000.pt` | 已实现并启动；当前在初始 validation `4/5` 处再次被 `SIGKILL`，尚无稳定 checkpoint |

### 1.2 关键结论

- stage-1 已完成，最终权重是 [iter_000020000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/checkpoints/iter_000020000.pt)
- 当前主线固定语义是 `29 -> 8 -> 29`，其中第 `1` 帧旁路保留，不参与时间压缩
- 当前主线已通过 `1GPU / 8GPU` 训练 smoke 和固定 `29` 帧推理 smoke
- 当前主线正式长训已于 `2026-04-11` 启动，并在 `2026-04-12` 从 `iter_000017000.pt` 续训到总 `40000` iter
- 针对 `22500+` 后验证 loss 震荡平台，新增了 `LRDecayFT` 变体：从 `iter_000027000.pt` 只加载模型权重续训，改用较小 LR + cosine decay + 更稳的 val 统计
- `LRDecayFT` 到 `iter_000009000.pt` 的单条验证样本推理结果为 `RMSE 6.79 / MAE 2.51 / Rel 0.08`，优于当前已记录的 base `T29` `iter_000017000`
- `LRDecayFT-V2` 到 `iter_000005000.pt` 的单条验证样本推理结果为 `RMSE 6.77 / MAE 2.48 / Rel 0.08`，并已补齐 range / histogram / point-cloud 三类产物
- 当前平台期判断更偏向“优化目标瓶颈”，不是单纯训练时长不足：已新增 `V3` 分支，改为 `masked latent_recon + temporal_delta`，并把聚合 `depth mae/rmse/rel` 输出到日志与独立 CSV
- `V3` 的 latent mask 缩放不再用 `nearest` 单点采样，而改成 conservative max-pooling；不过 latent_recon 上的 masking 仍只是“权重再分配”的近似，因为 2D encoder 的感受野会跨越多个原始像素
- `V3` 这条线当前更应该按聚合 `depth_mae / depth_rmse / depth_relative_error` 选 checkpoint，而不是只看总 `validation_loss`；到目前为止综合最佳点是 `iter_000007500.pt`
- 方案 A（`V4`）已经落地：保持 latent compressor 架构不变，解冻 `post_quant_conv + decoder`，并对解冻的 2D decoder 侧使用更小 LR；当前代码接线与优化器分组已验证通过
- 历史对比主线 `T17 Streaming` 仍保留，最新稳定 checkpoint 是 [iter_000017000.pt](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CV4x8x8-Waymo-T17-Streaming/checkpoints/iter_000017000.pt)

### 1.3 实验状态速览

| 实验 | 类型 | 状态 | 初始化来源 | 备注 |
|------|------|------|------------|------|
| `cosmos_lidar_tokenizer_waymo` | `CI8x8-Waymo` | 已完成 | 通用 `CI8x8-Lidar` | stage-1 主线 |
| `cosmos_lidar_tokenizer_waymo_t29_latent_compressor` | `T29 LatentCompressor` | 长训中 | `CI8x8-Waymo iter_000020000.pt` | 当前主线；详见 §7.5 |
| `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft` | `T29 LatentCompressor LRDecayFT` | 长训中 | `T29 LatentCompressor iter_000027000.pt` | 后期降 LR 微调分支；当前从 `iter_000009000.pt` 续到总 `20000` iter；详见 §7.6 |
| `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v2` | `T29 LatentCompressor LRDecayFT-V2` | 历史暂停 | `LRDecayFT iter_000010000.pt` | validation media 关闭 + `wandb offline` 诊断分支；最新稳定 checkpoint 为 `iter_000005000.pt`；详见 §7.7 |
| `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v3` | `T29 LatentCompressor LRDecayFT-V3` | 长训中 | `LRDecayFT iter_000009000.pt` | 推荐的新平台期优化分支：`masked latent_recon + temporal_delta`；已改回 `validation_iter=500`，latent mask 改为 conservative pooling；当前按聚合 depth 指标最优点是 `iter_000007500.pt` |
| `cosmos_lidar_tokenizer_waymo_t29_latent_compressor_lrdecay_ft_v4` | `T29 LatentCompressor LRDecayFT-V4` | 历史诊断中 | `V3 iter_000008000.pt` | 方案 A：解冻 `post_quant_conv + decoder`；优化器分组 LR 已接通，但初始 validation `4/5` 时再次 `SIGKILL`；详见 §7.9 |
| `cosmos_lidar_tokenizer_cv4x8x8_waymo_t17_streaming` | `T17 Streaming` | 历史暂停 | `CI8x8-Waymo iter_000020000.pt` | 旧 stage-2 主线 |
| `cosmos_lidar_tokenizer_cv4x8x8_waymo` | `CV4x8x8-Waymo` | 历史完成 | 通用 `CI8x8-Lidar` | 旧版 `9` 帧 causal 语义 |
| `cosmos_lidar_tokenizer_cv4x8x8_waymo_t15_streaming` | `T15 Streaming` | 历史暂停 | 通用 `CI8x8-Lidar` | legacy ragged-tail 实验 |
| `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29` | `T29` | 历史暂停 | 通用 `CI8x8-Lidar` | 固定 `29` 帧窗口 baseline |
| `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_flow` | `T29 + flow` | 历史暂停 | 通用 `CI8x8-Lidar` | 时序 loss 探索分支 |
| `cosmos_lidar_tokenizer_cv4x8x8_waymo_t29_streaming` | `T29 Streaming` | 历史暂停 | 通用 `CI8x8-Lidar` | 早期 streaming 探索分支 |

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

## 2. 运行环境与外部启动

统一环境：
- 工作目录：`/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen`
- conda 环境：`cosmos-predict1`
- checkpoint 根目录：`checkpoints/`，实际软链接到 `/data2/checkpoints/`

每次启动训练前建议统一执行：

```bash
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
export OUTPUT_ROOT=checkpoints
```

### 2.1 外部 tmux 启动

长训建议都在外部 `tmux` 里启动，并把 stdout 同步写到实验目录下的 `stdout.log`。

`CI8x8-Waymo`：

```bash
tmux new-session -d -s waymo_tok_ci8x8 '/bin/zsh -lc "
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
export OUTPUT_ROOT=checkpoints
torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
  --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
  experiment=cosmos_lidar_tokenizer_waymo \
  2>&1 | tee checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-CI8x8-Waymo/stdout.log
"'
```

`T29 LatentCompressor`：

```bash
tmux new-session -d -s waymo_tok_t29_latent '/bin/zsh -lc "
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
export OUTPUT_ROOT=checkpoints
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1,::1
torchrun --nproc_per_node=8 -m cosmos_predict1.tokenizer.training.train \
  --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
  experiment=cosmos_lidar_tokenizer_waymo_t29_latent_compressor \
  trainer.max_iter=40000 \
  2>&1 | tee -a checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/stdout.log
"'
```

**注意**：flow loss 在 iter 5000 首次启用时会 lazy-download RAFT-large 权重（torchvision hub，21 MB）。机器没外网会直接 `ConnectionRefusedError` 炸掉训练。上面 tmux 启动注入了本地 socks/http 代理作为兜底；更稳妥的做法是提前把权重放进 torch hub cache：

```bash
mkdir -p /root/.cache/torch/hub/checkpoints
curl -L -o /root/.cache/torch/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth \
  https://download.pytorch.org/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth
```

一旦权重落盘，后续所有 run / resume 都走 cache，不依赖网络。续训用 `tee -a` 追加而不是覆盖 `stdout.log`。

常用 tmux 操作：
- 查看会话：`tmux ls`
- 进入会话：`tmux attach -t waymo_tok_ci8x8`
- 退出不停止：`Ctrl-b d`
- 结束会话：`tmux kill-session -t <session_name>`

### 2.2 续训

这套训练默认按实验目录自动续训：
- 同一 `job.name`
- 同一输出目录
- 目录里存在 `checkpoints/latest_checkpoint.txt`

继续训练时，只要重新用同一个实验名启动，训练器会自动从最近一次保存的 checkpoint 恢复。

常看两个文件：
- `checkpoints/.../checkpoints/latest_checkpoint.txt`
- `checkpoints/.../stdout.log`

### 2.3 日志与 Loss 记录

当前推荐做法是同时保留：
- 本地 `stdout.log + validation_loss_history.csv`
- 在线 `wandb`

当前 `LRDecayFT` 已切到在线 `wandb`：
- entity：`leishuangming1103-zhejiang-university`
- project：`Lidar Tokenizer`
- run name：`Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT`
- run id：`u9e21vjg`

当前代码已支持：
- `wandb_entity`
- `wandb_project`
- 续训时复用已有 `wandb_id.txt`，避免同一实验被拆成多条 run

如果本机没有登录态，可先执行：

```bash
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
wandb login
```

建议直接看：

```bash
tail -f checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/stdout.log
```

当前主线另外开了一个 sidecar tmux 会话，专门把 validation 指标导成单个 CSV：

```bash
tmux new-session -d -s waymo_tok_t29_latent_loss '/bin/zsh -lc "
source /root/miniforge3/etc/profile.d/conda.sh
conda activate cosmos-predict1
cd /root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen
while true; do
  python scripts/export_validation_loss_csv.py \
    --log_path checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/stdout.log \
    --output_csv checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor/validation_loss_history.csv
  sleep 300
done
"'
```

日志里重点关注：
- `Validation loss (iteration N): ...`
- `Loading checkpoint ...`
- `Done with loading the checkpoint ...`
- `Done with training.`

说明：
- 训练进度条基于 `tqdm`，tmux pane 有时会看起来像“没输出”，但 `stdout.log` 通常还在持续写。
- 当前实验主要以 `Validation loss` 作为阶段性对比信号。
- `validation_loss_history.csv` 现在是单个合并表：会同时记录 `validation_loss`、聚合分项 loss（如 `color / latent_recon / temporal_delta / flow / kl`）以及聚合 `depth_mae / depth_rmse / depth_relative_error`。
- 这个 CSV 仍然是从 `stdout.log` 提取的去重版本，便于后续画曲线或比对 resume 前后的走势。
- checkpoint 默认每 `1000` iter 保存一次，validation 默认每 `500` iter 执行一次。
- `wandb` 会单独记录：
  - `train/loss`、`val/loss`
  - `train/color`、`train/latent_recon`、`train/flow`、`train/kl`
  - `val/color`、`val/latent_recon`、`val/flow`、`val/kl`
  - `val/depth_mae`、`val/depth_rmse`、`val/depth_relative_error`

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

当前默认主线不是这些分支，而是 `T29 LatentCompressor`。

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
| `iter_000011000` | `20` 帧，image tokenizer | 未回填 | 未回填 | 未回填 | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_iter11000_clip1/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_iter11000_clip1/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_iter11000_clip1/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | 只保留了产物目录，终端指标未单独记录 |
| `iter_000013000` | `20` 帧，image tokenizer | 未回填 | 未回填 | 未回填 | [range](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_iter13000/range_map_video/10203656353524179475_7625_000_7645_000.mp4) / [hist](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_iter13000/histogram/10203656353524179475_7625_000_7645_000.png) / [pcd](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/dump_results/lidar_tokenizer/waymo_eval_iter13000/point_cloud/10203656353524179475_7625_000_7645_000.mp4) | 只保留了产物目录，终端指标未单独记录 |
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
| 当前状态 | 当前主线；`2026-04-12` 从 `iter_000017000.pt` 续训到总 `40000` iter |

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
| 验证设置 | `validation_iter=500`，`max_val_iter=5`，`wandb offline`，validation media 关闭 |
| loss 设置 | 继承 `V3`：`color + masked latent_recon + temporal_delta + low-weight flow + small kl` |
| 日志记录 | [stdout.log](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/stdout.log) / [validation_loss_history.csv](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/checkpoints/posttraining/tokenizer/Cosmos-LidarTokenizer-Waymo-T29-LatentCompressor-LRDecayFT-V4/validation_loss_history.csv) |
| 当前状态 | 已修复优化器参数组接线；8 卡 run 已能正常进入 `Starting training...` 并跑到初始 validation，但在 `4/5` 时再次被系统 `SIGKILL`，尚未写出 `iteration 0` 聚合指标或稳定 checkpoint |

说明：

- `V4` 不是重新发明结构，而是在现有 `T29 LatentCompressor` 上把“只训 temporal compressor”升级成“冻结 encoder/quant_conv，轻量 joint finetune decoder 侧”的版本。
- 第一轮启动暴露了一个代码问题：优化器分组参数经过 `LazyConfig` 后混入了 `DictConfig`，导致 `TypeError: optimizer can only optimize Tensors`。该问题已在 [training/model.py](/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen/cosmos_predict1/tokenizer/training/model.py) 修复。
- 修复后，`V4` 已确认能越过优化器初始化、加载 checkpoint，并进入初始 validation；当前新的阻塞点不在方案 A 代码本身，而仍然是这台机器在 validation 尾段的 `SIGKILL` 老问题。

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

- 当前 `T29 LatentCompressor`、`LRDecayFT` 与 `LRDecayFT-V2` 都已回填单条验证样本推理结果；`V3` 目前先以聚合 validation depth 指标作为主判断口径，整套验证集平均推理指标仍未回填。
- 该方案只支持固定 `29` 帧，不支持变长和 streaming 长序列。
- 当前 `T17 Streaming` 虽然支持 `29` 帧推理，但质量仍明显弱于 `CI8x8-Waymo` 的单帧重建。
- 当前新主线使用 `color + latent_recon + delayed flow + small kl`，仍属于 v1 baseline。
- `V3` 虽然把 latent mask 下采样改成了 conservative pooling，但 latent_recon 上的 gating 仍不是“严格剔除 invalid receptive field”，而更接近对有效区域的近似重加权。
- `V3` 在 `iter 5000` 处有一次明确的 loss 权重切换，因此总 `validation_loss` 曲线存在阶跃，不宜简单与 `5000` 前做同尺度比较。
- 点云视频渲染在当前环境里不够稳定，`range_map_video + histogram` 更适合作为常规评估输出。
- 当前 `T17` 训练多次在 validation 后被 `SIGKILL`；最新稳定 checkpoint 是 `iter_000017000.pt`。
- **Flow loss 在 iter 5000 首次启用时依赖网络下载 RAFT 权重**：`torchvision.models.optical_flow.raft_large(pretrained=True)` 会 lazy 拉取 `raft_large_C_T_SKHT_V2-ff5fadd5.pth`。离线环境必须提前把权重放进 `~/.cache/torch/hub/checkpoints/`，否则 iter 5000 的首个 validation 会直接崩溃。具体操作见 §2.1。本次长训就是在 iter 5000 踩到了这个坑，通过预下载 + 代理 env 双兜底恢复。
- `V3` 把 `max_val_iter` 收回到 `5` 以减轻 validation 阶段的系统负载；代价是 val 曲线会比 `V1/V2` 更抖，最好结合离线推理或聚合 depth 指标一起看。
- `V4` 已证明方案 A 的代码接线和优化器分组没有问题，但 validation 阶段的系统 `SIGKILL` 仍未解除；下一步如果继续推进 V4，更可能需要继续减轻 validation 负载，而不是再改 decoder finetune 结构本身。

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
