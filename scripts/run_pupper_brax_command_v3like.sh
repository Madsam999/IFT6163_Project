#!/usr/bin/env bash
set -euo pipefail

python puppersim/pupper_train_ppo_brax.py \
  --env-name pupper_v2 \
  --backend mjx \
  --env-kwargs puppersim/config/pupper_brax_env_kwargs.command_locomotion.json \
  --randomization-module puppersim.pupper_brax_domain_randomization \
  --randomization-fn domain_randomize \
  --randomization-kwargs puppersim/config/pupper_brax_randomization_kwargs.v3like.json \
  --num-timesteps 1000000000 \
  --episode-length 500 \
  --num-evals 11 \
  --reward-scaling 1.0 \
  --normalize-observations \
  --action-repeat 1 \
  --unroll-length 20 \
  --num-minibatches 32 \
  --num-updates-per-batch 4 \
  --discounting 0.97 \
  --learning-rate 3e-5 \
  --entropy-cost 1e-2 \
  --num-envs 8192 \
  --batch-size 256 \
  --network-hidden-sizes 256,128,128,128 \
  --activation elu \
  --track \
  --wandb-project-name "${WANDB_PROJECT:-pupper-v2-brax}" \
  --wandb-entity "${WANDB_ENTITY:-}"
