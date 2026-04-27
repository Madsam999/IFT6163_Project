#!/usr/bin/env bash
set -euo pipefail

PROFILE="puppersimMJX/tasks/apriltag_walls/config/training_profiles/apriltag_walls_camera_nopriv_hl_level2.json"

TOTAL_STEPS=30000000
EPISODE_LENGTH=1500
EVAL_INTERVAL=1000000
VIDEO_INTERVAL=2000000
REPORT_INTERVAL=2000000
VIDEO_STEPS=1500
REPORT_STEPS=1500
REPORT_NUM_ROLLOUTS=5
UNROLL_LENGTH=128
CHECKPOINT_INTERVAL=2000000

ICM_START_WEIGHT=0.005
ICM_FINAL_WEIGHT=0.0
ICM_ANNEAL_STEPS="${TOTAL_STEPS}"

COMMON_ARGS=(
  --env-name pupper_v2
  --backend mjx
  --profile "${PROFILE}"
  --num-timesteps "${TOTAL_STEPS}"
  --episode-length "${EPISODE_LENGTH}"
  --eval-interval "${EVAL_INTERVAL}"
  --video-eval-interval "${VIDEO_INTERVAL}"
  --report-eval-interval "${REPORT_INTERVAL}"
  --video-steps "${VIDEO_STEPS}"
  --report-steps "${REPORT_STEPS}"
  --report-num-rollouts "${REPORT_NUM_ROLLOUTS}"
  --unroll-length "${UNROLL_LENGTH}"
  --checkpoint-interval "${CHECKPOINT_INTERVAL}"
  --capture-video-during-training
  --capture-report-during-training
  --no-log-training-metrics
)

run_baseline() {
  local seed="$1"
  python puppersimMJX/pupper_train_ppo_brax.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --task-name "level2_baseline_seed${seed}"
}

run_icm() {
  local seed="$1"
  python puppersimMJX/pupper_train_ppo_brax.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --task-name "level2_icm_decay_0p005_seed${seed}" \
    --icm-enabled \
    --icm-feature-dim 64 \
    --icm-hidden-dim 128 \
    --icm-learning-rate 0.0003 \
    --icm-beta 0.2 \
    --icm-eta 1.0 \
    --icm-reward-weight "${ICM_START_WEIGHT}" \
    --icm-reward-weight-final "${ICM_FINAL_WEIGHT}" \
    --icm-anneal-schedule linear \
    --icm-anneal-steps "${ICM_ANNEAL_STEPS}" \
    --icm-reward-clip 0.0
}

for seed in 1 2 3; do
  echo "=== level2 baseline seed ${seed} ==="
  run_baseline "${seed}"

  if [[ "${seed}" == "1" ]]; then
    echo "=== skipping level2 ICM decay 0.005->0.0 seed 1; already ran ==="
  else
    echo "=== level2 ICM decay 0.005->0.0 seed ${seed} ==="
    run_icm "${seed}"
  fi
done
