# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import sys
import random
import time
from collections import deque
from dataclasses import dataclass

# Ensure local package imports when this script is executed as:
# `python puppersim/pupper_train_ppo_cont_action.py`
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import puppersim

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from puppersim.icm import ICMModule


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "pupper"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    save_checkpoints: bool = True
    """whether to periodically save training checkpoints"""
    checkpoint_interval: int = 1000000
    """save a checkpoint every N environment steps"""
    checkpoint_dirname: str = "checkpoints"
    """subdirectory under runs/{run_name} for periodic checkpoints"""
    reward_log_interval: int = 500
    """log reward/* scalars every N environment steps"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # ICM arguments
    use_icm: bool = False
    """if toggled, use the Intrinsic Curiosity Module to augment rewards"""
    icm_feature_dim: int = 64
    """size of the ICM feature embedding"""
    icm_lr: float = 3e-4
    """learning rate for the ICM optimiser"""
    icm_eta: float = 0.01
    """scales the forward-error intrinsic reward (r_int = eta/2 * ||phi_hat - phi||^2)"""
    icm_lam: float = 0.2
    """balance between inverse and forward loss: L=(1-lam)*L_fwd + lam*L_inv"""
    icm_beta: float = 1.0
    """initial weight of the intrinsic reward: r = r_ext + beta * r_int"""
    icm_beta_final: float = 0.0
    """final value of beta after decay (set > 0 to keep some curiosity throughout)"""
    icm_beta_decay_frac: float = 0.7
    """fraction of total iterations over which beta decays from icm_beta to icm_beta_final"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode='rgb_array')
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), observation_space=env.observation_space)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_run_name(args: Args) -> str:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{args.env_id}__{args.exp_name}__{timestamp}"


def save_checkpoint(path: str, agent: nn.Module, optimizer: optim.Optimizer, global_step: int, iteration: int, args: Args):
    checkpoint = {
        "global_step": global_step,
        "iteration": iteration,
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def log_reward_terms_from_infos(
    infos,
    writer: SummaryWriter,
    global_step: int,
):
    for key, value in infos.items():
        if not isinstance(key, str):
            continue
        if not key.startswith("reward/"):
            continue
        if key.startswith("_"):
            continue
        try:
            values = np.asarray(value, dtype=np.float64)
        except Exception:
            continue
        if values.size == 0:
            continue
        step_value = float(np.nanmean(values))
        if not np.isfinite(step_value):
            continue
        writer.add_scalar(key, step_value, global_step)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


def compute_beta(iteration: int, num_iterations: int,
                 beta_start: float, beta_final: float, decay_frac: float) -> float:
    """Linearly decay beta from beta_start to beta_final over the first decay_frac of training."""
    decay_steps = int(num_iterations * decay_frac)
    if iteration >= decay_steps:
        return beta_final
    t = iteration / decay_steps
    return beta_start + t * (beta_final - beta_start)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = build_run_name(args)
    if args.track:
        try:
            import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    checkpoint_dir = os.path.join("runs", run_name, args.checkpoint_dirname)
    if args.save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)
    next_checkpoint_step = max(1, int(args.checkpoint_interval))

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))

    # ICM setup
    icm = None
    icm_optimizer = None
    if args.use_icm:
        icm = ICMModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=args.icm_feature_dim,
            eta=args.icm_eta,
            lam=args.icm_lam,
        ).to(device)
        icm_optimizer = optim.Adam(icm.parameters(), lr=args.icm_lr, eps=1e-5)
        print(f"ICM enabled: feature_dim={args.icm_feature_dim}, beta={args.icm_beta} -> {args.icm_beta_final}")

    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    action_dim = int(np.prod(envs.single_action_space.shape))

    # ICM setup
    icm = None
    icm_optimizer = None
    if args.use_icm:
        icm = ICMModule(
            obs_dim=obs_dim,
            action_dim=action_dim,
            feature_dim=args.icm_feature_dim,
            eta=args.icm_eta,
            lam=args.icm_lam,
        ).to(device)
        icm_optimizer = optim.Adam(icm.parameters(), lr=args.icm_lr, eps=1e-5)
        print(f"ICM enabled: feature_dim={args.icm_feature_dim}, beta={args.icm_beta} -> {args.icm_beta_final}")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    next_obs_buf = torch.zeros_like(obs)  # stores s_{t+1} for each step (needed by ICM)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    recent_ep_returns = deque(maxlen=100)
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Current beta for this iteration
        beta = compute_beta(iteration - 1, args.num_iterations,
                            args.icm_beta, args.icm_beta_final, args.icm_beta_decay_frac)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Store next_obs for ICM (done after converting to tensor)
            next_obs_buf[step] = next_obs

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episodic_return = float(info["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={episodic_return}")
                        writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        recent_ep_returns.append(episodic_return)
                        writer.add_scalar(
                            "charts/avg_episode_return_last_100",
                            float(np.mean(recent_ep_returns)),
                            global_step,
                        )

        # --- ICM: augment rewards with intrinsic signal ---
        if icm is not None and beta > 0:
            b_obs_icm = obs.reshape(-1, obs_dim)
            b_next_obs_icm = next_obs_buf.reshape(-1, obs_dim)
            b_actions_icm = actions.reshape(-1, action_dim)

            r_int = icm.compute_intrinsic_reward(b_obs_icm, b_next_obs_icm, b_actions_icm)
            r_int = r_int.reshape(args.num_steps, args.num_envs)
            rewards = rewards + beta * r_int

            writer.add_scalar("icm/intrinsic_reward_mean", r_int.mean().item(), global_step)
            writer.add_scalar("icm/intrinsic_reward_max", r_int.max().item(), global_step)
            writer.add_scalar("icm/beta", beta, global_step)

        # --- ICM: augment rewards with intrinsic signal ---
        if icm is not None and beta > 0:
            b_obs_icm = obs.reshape(-1, obs_dim)
            b_next_obs_icm = next_obs_buf.reshape(-1, obs_dim)
            b_actions_icm = actions.reshape(-1, action_dim)

            r_int = icm.compute_intrinsic_reward(b_obs_icm, b_next_obs_icm, b_actions_icm)
            r_int = r_int.reshape(args.num_steps, args.num_envs)
            rewards = rewards + beta * r_int

            writer.add_scalar("icm/intrinsic_reward_mean", r_int.mean().item(), global_step)
            writer.add_scalar("icm/intrinsic_reward_max", r_int.max().item(), global_step)
            writer.add_scalar("icm/beta", beta, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # --- ICM: update networks on the collected rollout (one pass over full batch) ---
        if icm is not None:
            icm_optimizer.zero_grad()
            icm_loss, fwd_loss, inv_loss = icm.compute_loss(
                obs.reshape(-1, obs_dim),
                next_obs_buf.reshape(-1, obs_dim),
                actions.reshape(-1, action_dim),
            )
            icm_loss.backward()
            icm_optimizer.step()

            writer.add_scalar("losses/icm_loss", icm_loss.item(), global_step)
            writer.add_scalar("losses/icm_forward_loss", fwd_loss.item(), global_step)
            writer.add_scalar("losses/icm_inverse_loss", inv_loss.item(), global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if args.save_checkpoints and global_step >= next_checkpoint_step:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"{args.exp_name}_step_{global_step:012d}.pt",
            )
            save_checkpoint(
                path=checkpoint_path,
                agent=agent,
                optimizer=optimizer,
                global_step=global_step,
                iteration=iteration,
                args=args,
            )
            print(f"checkpoint saved to {checkpoint_path}")
            while next_checkpoint_step <= global_step:
                next_checkpoint_step += max(1, int(args.checkpoint_interval))

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        state_dict = {
            "agent": agent.state_dict(),
        }
        if icm is not None:
            state_dict["icm"] = icm.state_dict()
        torch.save(state_dict, model_path)
        print(f"model saved to {model_path}")
        from cleanrl.cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
