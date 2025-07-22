# =========================
# Imports and Dependencies
# =========================

# Standard Python libraries
import os
import random
import time
from dataclasses import dataclass

# Custom or local simulation package (could be your own code)
import puppersim

# Third-party libraries for RL, ML, and logging
import gymnasium as gym  # Provides RL environments
import numpy as np       # Numerical operations, like arrays and math
import torch             # PyTorch: main deep learning library
import torch.nn as nn    # Neural network components
import torch.optim as optim  # Optimizers for training neural networks
import tyro              # For parsing command-line arguments
from torch.distributions.normal import Normal  # For continuous action sampling
from torch.utils.tensorboard import SummaryWriter  # For logging training progress
from typing import Optional

# =========================
# Hyperparameters and Config
# =========================

@dataclass
class Args:
    # Experiment and logging settings
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Name of this experiment (used for logging and saving results)"""
    seed: int = 1
    """Random seed for reproducibility (ensures same results each run)"""
    torch_deterministic: bool = True
    """If True, makes PyTorch operations deterministic (slower, but reproducible)"""
    cuda: bool = True
    """If True, use GPU for training if available"""
    track: bool = False
    """If True, log experiment to Weights & Biases (wandb) for visualization"""
    wandb_project_name: str = "cleanRL"
    """Project name for wandb logging"""
    wandb_entity: Optional[str] = None
    """Team/entity for wandb logging"""
    capture_video: bool = False
    """If True, record videos of the agent's performance"""
    video_trigger: int = 0
    """Trigger videos at every x episodes"""
    save_model: bool = False
    """If True, save the trained model to disk"""
    eval_episodes: int = 10
    """Number of eval episodes"""
    upload_model: bool = False
    """If True, upload the model to HuggingFace Hub"""
    hf_entity: str = ""
    """User or organization name for HuggingFace Hub uploads"""

    # RL algorithm settings
    env_id: str = "HalfCheetah-v4"
    """Which environment to train on (from Gymnasium)"""
    total_timesteps: int = 1000000
    """Total number of environment steps to train for"""
    learning_rate: float = 3e-4
    """Learning rate for the optimizer (how fast the model learns)"""
    num_envs: int = 1
    """Number of parallel environments (for faster data collection)"""
    num_steps: int = 2048
    """Number of steps to run in each environment before updating the model"""
    anneal_lr: bool = True
    """If True, decrease learning rate over time (can help training stability)"""
    gamma: float = 0.99
    """Discount factor for future rewards (how much to care about future vs. immediate rewards)"""
    gae_lambda: float = 0.95
    """Lambda for Generalized Advantage Estimation (controls bias-variance tradeoff)"""
    num_minibatches: int = 32
    """How many mini-batches to split the data into for each update"""
    update_epochs: int = 10
    """How many times to update the model with the same batch of data"""
    norm_adv: bool = True
    """If True, normalize advantages (helps with training stability)"""
    clip_coef: float = 0.2
    """Clipping parameter for PPO (prevents too large policy updates)"""
    clip_vloss: bool = True
    """If True, clip value loss (helps prevent value function from changing too much)"""
    ent_coef: float = 0.0
    """Weight for entropy bonus (encourages exploration)"""
    vf_coef: float = 0.5
    """Weight for value function loss (balances policy and value learning)"""
    max_grad_norm: float = 0.5
    """Maximum gradient norm (prevents exploding gradients)"""
    target_kl: Optional[float] = None
    """Target KL divergence (can be used to stop updates early if policy changes too much)"""

    # These are calculated at runtime
    batch_size: int = 0
    """Total batch size (num_envs * num_steps)"""
    minibatch_size: int = 0
    """Size of each mini-batch"""
    num_iterations: int = 0
    """Number of training iterations (total_timesteps // batch_size)"""

# =========================
# Environment Creation
# =========================

def make_env(env_id, idx, capture_video, run_name, gamma):
    """
    Returns a function that creates a single environment instance with all necessary wrappers.
    Wrappers add features like video recording, observation normalization, reward normalization, etc.
    """
    def thunk():
        # If capturing video, only record from the first environment
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode='rgb_array')
            video_trigger = args.video_trigger
            if video_trigger > 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}",episode_trigger=lambda episode_id: episode_id % video_trigger == 0)
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")                
        else:
            env = gym.make(env_id)
        # Flatten observations (useful if the environment returns a dictionary of observations)
        env = gym.wrappers.FlattenObservation(env)
        # Record episode statistics (like total reward and episode length)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # Clip actions to the valid range for the environment
        env = gym.wrappers.ClipAction(env)
        # Normalize observations (helps learning)
        env = gym.wrappers.NormalizeObservation(env)
        # Actually clip observations to [-10, 10]
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        # Normalize rewards (helps learning)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        # Clip rewards to [-10, 10] to avoid outliers
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env
    return thunk

# =========================
# Neural Network Utilities
# =========================

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Helper function to initialize neural network layers with orthogonal weights and constant bias.
    Orthogonal initialization can help with stable learning.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# =========================
# Agent Definition
# =========================

class Agent(nn.Module):
    """
    The Agent contains two neural networks:
    - The Critic: estimates the value of a state (how good it is)
    - The Actor: outputs the mean of a Gaussian distribution for actions (for continuous control)
    """
    def __init__(self, envs):
        super().__init__()
        # Critic network: predicts a single value for each state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # Actor network: predicts the mean of the action distribution
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        # The log standard deviation of the action distribution (learned parameter, not a network)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        """
        Passes the input through the critic network to get the value estimate.
        """
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
        Passes the input through the actor to get the action distribution,
        samples an action (if not provided), and returns:
        - action
        - log probability of the action
        - entropy of the action distribution (for exploration)
        - value estimate from the critic
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# =========================
# Main Training Loop
# =========================

if __name__ == "__main__":
    # Parse command-line arguments and set up experiment parameters
    args = tyro.cli(Args)
    # Calculate batch sizes and number of iterations based on user settings
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # Create a unique name for this run (for logging and saving)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time()))}"

    # If tracking with wandb, initialize it
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    # Set up TensorBoard logging
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Set random seeds for reproducibility (so results can be repeated)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Choose device: GPU if available and requested, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create multiple environments for parallel data collection
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    # Ensure the environment uses continuous actions (not discrete)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Create the agent and optimizer
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Allocate storage for observations, actions, rewards, etc. for each step and environment
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize environment and episode state
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)  # Get initial observation from all environments
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)  # Track which environments are done

    # Main training loop: repeat for each policy update
    for iteration in range(1, args.num_iterations + 1):
        # Optionally decrease learning rate over time (annealing)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect experience from the environments for num_steps
        for step in range(0, args.num_steps):
            global_step += args.num_envs  # Count total environment steps
            if global_step % 10000 == 0:
                print(f"Global step: {global_step}")
                print(f"Reward: {reward}")
            obs[step] = next_obs          # Store current observation
            dones[step] = next_done       # Store done flags

            # Get action and value prediction from the agent (no gradients needed here)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Step the environment with the chosen action
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # Determine if episode ended due to termination or truncation
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # Convert next_obs and next_done to tensors for next step
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # If episode finished, log episodic return and length to TensorBoard
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # =========================
        # Compute Advantages and Returns (GAE)
        # =========================
        # This helps the agent understand how good its actions were, considering future rewards
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
                # Temporal difference error
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                # GAE advantage calculation
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values  # The "target" for the value function

        # =========================
        # Prepare Data for Training
        # =========================
        # Flatten the batch so we can shuffle and split into mini-batches
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # =========================
        # PPO Policy and Value Update
        # =========================
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)  # Shuffle indices for mini-batch sampling
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # Get new logprobs, entropy, and value for this mini-batch
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()  # Probability ratio for PPO

                with torch.no_grad():
                    # KL divergence diagnostics (measures how much the policy changed)
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    # Normalize advantages for stability
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (PPO surrogate objective, clipped)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (how well the value function predicts returns)
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

                # Entropy loss (encourages exploration)
                entropy_loss = entropy.mean()
                # Total loss combines policy, value, and entropy losses
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Backpropagation and optimization step
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            # Early stopping if policy changed too much (KL divergence)
            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # =========================
        # Logging and Diagnostics
        # =========================
        # Explained variance: how well the value function predicts returns (1.0 is perfect)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log various statistics to TensorBoard for visualization
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))  # Steps per second
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # =========================
    # Save and Evaluate Model
    # =========================
    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        state_dict = agent.state_dict()
        torch.save(state_dict, model_path)
        print(f"model saved to {model_path}")
        from cleanrl.cleanrl_utils.evals.ppo_eval import evaluate

        # Evaluate the trained model
        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=args.eval_episodes,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )

        # Log evaluation results
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        # Optionally upload model to HuggingFace Hub for sharing
        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    # Clean up: close environments and logging
    envs.close()
    writer.close()
