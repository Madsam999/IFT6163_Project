"""
Robot Deployment Script for TensorFlow Lite PPO Models - Robot Version

This script loads a trained PPO model and runs it on the Pupper robot.

Usage:
python puppersim/pupper_deploy_ppo_robot.py --model_path runs/model.tflite
"""

import os
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter
import pickle
import argparse
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

def create_pupper_env():
    """
    Create Pupper environment for robot deployment.
    """
    CONFIG_DIR = puppersim.getPupperSimPath()
    _CONFIG_FILE = os.path.join(CONFIG_DIR, "config", "pupper_pmtg_robot.gin")
    
    gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
    gin.parse_config_file(_CONFIG_FILE)
    gin.bind_parameter("SimulationParameters.enable_rendering", False)  # No rendering on robot
    env = env_loader.load()
    
    return env


def deploy_policy(model_path, max_steps=10000, log_to_file=False):
    """
    Deploy a trained PPO policy on the robot.
    
    Args:
        model_path: Path to the .tflite file
        max_steps: Maximum steps per episode
        log_to_file: Whether to log data to file
    """
    # Check if TFLite file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model file not found: {model_path}")
    
    print(f"Loading TFLite model from {model_path}")
    
    # Create environment
    print("Creating robot environment...")
    env = create_pupper_env()
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Load the TFLite model and allocate tensors
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensors details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully!")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
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
        print("\n--- Starting Robot Deployment ---")
        print("Press Ctrl+C to stop")
        
        # Reset environment
        obs = env.reset()
        
        episode_return = 0.0
        episode_length = 0
        done = False
        
        # Timing setup
        start_time = time.time()
        env_start_time_wall = time.time()
        last_spammy_log = 0.0
        
        # Episode loop for robot - runs continuously until stopped
        while not done:
            
            # Get action from TFLite model
            obs_input = obs.reshape(1, -1).astype(np.float32)  # Reshape to [1, obs_dim]
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], obs_input)
            
            # Run inference
            interpreter.invoke()
            
            # Get output tensor
            action = interpreter.get_tensor(output_details[0]['index']).flatten()
            
            # Step environment
            obs, reward, done, _ = env.step(action)
            
            # Real-time synchronization for robot deployment just like the ARS script.
            # Sync to real time.
            wall_elapsed = time.time() - env_start_time_wall
            sim_elapsed = env.env_step_counter * env.env_time_step
            sleep_time = sim_elapsed - wall_elapsed
            if sleep_time > 0:
                print(sleep_time)
                time.sleep(sleep_time)
            elif sleep_time < -1 and time.time() - last_spammy_log > 1.0:
                print(f"Cannot keep up with realtime. {-sleep_time:.2f} sec behind, "
                      f"sim/wall ratio {(sim_elapsed/wall_elapsed):.2f}.")
                last_spammy_log = time.time()
            
            # Update episode stats
            episode_return += reward
            episode_length += 1
            
            # Logging
            if log_to_file and log_dict:
                log_dict['t'].append(env.robot.GetTimeSinceReset())
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
        
        # Episode summary
        episode_duration = time.time() - start_time
        
        print(f"\nRobot deployment completed:")
        print(f"  Steps: {episode_length}")
        print(f"  Return: {episode_return:.2f}")
        print(f"  Duration: {episode_duration:.2f}s")
        print(f"  Avg step time: {episode_duration/episode_length:.4f}s")
    
    except KeyboardInterrupt:
        print("\nDeployment interrupted by user")
        episode_duration = time.time() - start_time
        print(f"Ran for {episode_duration:.2f}s")
    except Exception as e:
        print(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save logs
        if log_to_file and log_dict:
            log_filename = "pupper_ppo_robot_log_"+str(int(time.time()))+".pkl"
            print(f"Saving log to {log_filename}")
            with open(log_filename, "wb") as f:
                pickle.dump(log_dict, f)
        
        # Close environment
        env.close()
        print("Robot deployment finished.")


def main():
    parser = argparse.ArgumentParser(description="Deploy PPO policy on Pupper robot")
    
    # Model and deployment settings
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the .tflite file')
    
    # Episode settings
    parser.add_argument('--max_steps', type=int, default=10000,
                       help='Maximum steps before stopping')
    
    # Logging
    parser.add_argument('--log_to_file', action='store_true',
                       help='Log data to file')
    
    args = parser.parse_args()
    
    print("=== Pupper PPO Policy Deployment - Robot ===")
    print(f"Model: {args.model_path}")
    print(f"Max steps: {args.max_steps}")
    print(f"Logging: {args.log_to_file}")
    
    # Deploy policy
    deploy_policy(
        model_path=args.model_path,
        max_steps=args.max_steps,
        log_to_file=args.log_to_file
    )


if __name__ == "__main__":
    main()
