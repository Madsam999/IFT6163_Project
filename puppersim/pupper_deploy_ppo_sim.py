"""
Robot Deployment Script for PyTorch PPO Models - Simulation Version

This script loads a trained PPO model and runs it in simulation.

Usage:
python puppersim/pupper_deploy_ppo_sim.py --model_path runs/model.cleanrl_model
"""

import os
import time
import numpy as np
import pickle
import argparse
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd
import torch
import gymnasium as gym

# Import existing class Agent
from pupper_train_ppo_cont_action import Agent

def create_pupper_env(render=True):
    """
    Create Pupper environment for simulation.
    """
    CONFIG_DIR = puppersim.getPupperSimPath()
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg_standing.gin")
    
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
    gin.parse_config_file(_CONFIG_FILE)
    gin.bind_parameter("SimulationParameters.enable_rendering", render)
    env = env_loader.load()
    
    return env


def deploy_policy(model_path, num_episodes=10, max_steps=10000, 
                 device='cpu', render=True, log_to_file=False):
    """
    Deploy a trained PPO policy in simulation.
    
    Args:
        model_path: Path to the .cleanrl_model file
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        device: PyTorch device
        render: Whether to render
        log_to_file: Whether to log data to file
    """
    print(f"Loading model from {model_path}")
    
    # Set up device
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    # Create environment
    print("Creating environment...")
    env = create_pupper_env(render=render)
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found:", model_path)
    
    # Create a mock envs object for Agent initialization (matching training script)
    mock_envs = type('MockEnvs', (), {
        'single_observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        'single_action_space': gym.spaces.Box(low=-1, high=1, shape=(action_dim,))
    })()
    
    agent = Agent(mock_envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Deployment loop
    episode_returns = []
    episode_lengths = []
    
    # Logging setup
    log_dict = {
        't': [],
        'IMU': [],
        'MotorAngle': [],
        'action': [],
        'obs': [],
        'reward': []
    } if log_to_file else None
    
    try:
        for episode in range(num_episodes):
            print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
            
            # Reset environment
            obs = env.reset()
            
            episode_return = 0.0
            episode_length = 0
            done = False
            
            # Episode loop for simulation
            while not done:
                
                # Get action from policy
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action_tensor = agent.actor_mean(obs_tensor)
                    action = action_tensor.cpu().numpy().flatten()
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                # Update episode stats
                episode_return += reward
                episode_length += 1
                
                # Logging
                if log_to_file and log_dict:
                    log_dict['MotorAngle'].append(obs[0:12].copy())
                    log_dict['IMU'].append(obs[12:16].copy())
                    log_dict['action'].append(action.copy())
                    log_dict['obs'].append(obs.copy())
                    log_dict['reward'].append(reward)
                
                # Safety check for max steps
                if episode_length >= max_steps:
                    print(f"Episode terminated: reached max steps ({max_steps})")
                    break
                
                # Print periodic updates
                if episode_length % 100 == 0:
                    print(f"Step {episode_length}, Return: {episode_return:.2f}")
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {episode_length}")
            print(f"  Return: {episode_return:.2f}")
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save logs
        if log_to_file and log_dict:
            log_filename = "pupper_ppo_simulation_log_"+str(int(time.time()))+".pkl"
            print(f"Saving log to {log_filename}")
            with open(log_filename, "wb") as f:
                pickle.dump(log_dict, f)
        
        # Close environment
        env.close()
    
    # Results summary
    if episode_returns:
        print(f"\n--- Deployment Summary ---")
        print(f"Episodes completed: {len(episode_returns)}")
        print(f"Mean return: {np.mean(episode_returns):.2f}")
        print(f"Std return: {np.std(episode_returns):.2f}")
        print(f"Min return: {np.min(episode_returns):.2f}")
        print(f"Max return: {np.max(episode_returns):.2f}")
        print(f"Mean episode length: {np.mean(episode_lengths):.1f}")
    
    return episode_returns, episode_lengths


def main():
    parser = argparse.ArgumentParser(description="Deploy PPO policy in simulation")
    
    # Model and deployment settings
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the .cleanrl_model file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device to run model on')
    
    # Episode settings
    parser.add_argument('--num_episodes', type=int, default=10,
                       help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum steps per episode')
    
    # Visualization and logging
    parser.add_argument('--render', action='store_true', 
                       help='Render simulation')
    parser.add_argument('--no-render', dest='render', action='store_false',
                       help='Disable rendering')
    parser.set_defaults(render=True)  # Default to rendering enabled
    parser.add_argument('--log_to_file', action='store_true',
                       help='Log data to file')
    
    args = parser.parse_args()
    
    print("=== Pupper PPO Policy Deployment - Simulation ===")
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Render: {args.render}")
    
    # Deploy policy
    episode_returns, episode_lengths = deploy_policy(
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=args.device,
        render=args.render,
        log_to_file=args.log_to_file
    )


if __name__ == "__main__":
    main()
