#!/bin/zsh

set -euo pipefail

REPO_DIR="/root/workspace/Cosmos-Drive-Dreams/cosmos-transfer-lidargen"
LAUNCH_LOG="${REPO_DIR}/waymo_t29_launcher.log"
BASELINE_JOB="Cosmos-LidarTokenizer-CV4x8x8-Waymo-T29"
BASELINE_SMOKE_JOB="Cosmos-LidarTokenizer-CV4x8x8-Waymo-T29-Smoke8GPU"

log() {
    local message="$1"
    printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "${LAUNCH_LOG}"
}

gpu_query() {
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
}

wait_for_free_gpus() {
    while true; do
        local gpu_users
        gpu_users="$(gpu_query)"
        if [[ -z "${gpu_users//[[:space:]]/}" ]]; then
            break
        fi
        log "Detected running GPU workloads; waiting 60s before retrying."
        printf '%s\n' "${gpu_users}" | tee -a "${LAUNCH_LOG}"
        sleep 60
    done
}

run_training() {
    local experiment_name="$1"
    local job_name="$2"
    local master_port="$3"
    local extra_args=("${@:4}")
    local job_log="${REPO_DIR}/checkpoints/posttraining/tokenizer/${job_name}.stdout.log"

    log "Launching ${job_name} with experiment=${experiment_name}"
    : > "${job_log}"
    (
        cd "${REPO_DIR}"
        set +u
        source /root/miniforge3/etc/profile.d/conda.sh
        conda activate cosmos-predict1
        set -u
        export OUTPUT_ROOT=checkpoints
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        export MPLCONFIGDIR=/tmp/matplotlib
        torchrun --nproc_per_node=8 --master_port="${master_port}" \
            -m cosmos_predict1.tokenizer.training.train \
            --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
            experiment="${experiment_name}" \
            job.name="${job_name}" \
            "${extra_args[@]}"
    ) >> "${job_log}" 2>&1
}

main() {
    : > "${LAUNCH_LOG}"
    log "Queued Waymo T29 training launcher."
    wait_for_free_gpus

    if run_training \
        "cosmos_lidar_tokenizer_cv4x8x8_waymo_t29" \
        "${BASELINE_SMOKE_JOB}" \
        "29631" \
        trainer.max_iter=1 \
        trainer.run_validation=False \
        trainer.validation_iter=100000 \
        checkpoint.save_iter=100000 \
        trainer.logging_iter=1; then
        log "T29 baseline smoke test passed. Starting full baseline training."
        run_training \
            "cosmos_lidar_tokenizer_cv4x8x8_waymo_t29" \
            "${BASELINE_JOB}" \
            "29632"
    else
        log "T29 baseline smoke test failed. Not launching full training."
    fi
}

main "$@"
