"""
Save PyTorch model as ONNX to run on robot.

Usage: 
python puppersim/save_as_onnx.py --model_path runs/model.cleanrl_model --env_id PupperGymEnvLong-v0
"""

import os
import numpy as np
import torch
import gymnasium as gym
import argparse

# Import existing classes to avoid duplication
from pupper_train_ppo_cont_action import Agent

#Function to Convert to ONNX 
def Convert_ONNX(output_path, obs_dim, agent): 

    # set the model to inference mode 
    agent.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, obs_dim, requires_grad=True)

    # Export the model   
    torch.onnx.export(agent.actor_mean,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         output_path,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['observation'],   # the model's input names 
         output_names = ['action'], # the model's output names 
         dynamic_axes={'observation' : {0 : 'batch_size'},    # variable length axes 
                       'action' : {0 : 'batch_size'}}) 
    print('Model has been converted to ONNX')
    

def make_deployment_env(env_id, run_on_robot=False, render=True, gamma=0.99):
    """
    Create environment with same wrappers as training.
    Uses the existing PupperGymEnv class.
    """
    def thunk():
        # Use the existing PupperGymEnv class with proper rendering setup
        if render and not run_on_robot:
            env = gym.make(env_id, render_mode="human", render=True)
        else:
            env = gym.make(env_id, render_mode=None, render=False)
        
        # Apply same wrappers as training for consistency
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        
        # Observation normalization
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        
        return env
    return thunk


def main():
    parser = argparse.ArgumentParser(description="Deploy PPO policy to Pupper robot")

    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the .cleanrl_model file')
    parser.add_argument('--env_id', type=str, default="PupperGymEnvLong-v0",
                       help='Environment ID')
    
    args = parser.parse_args()
    
    model_path = args.model_path
    env_id = args.env_id
    
    # Set up output path
    output_path = os.path.splitext(model_path)[0]+'.onnx'
    print(f"ONNX model path: {output_path}")
    
    # Create environment
    print("Creating environment...")
    env_fn = make_deployment_env(env_id=env_id)
    env = env_fn()
    
    # Get environment dimensions
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found:", model_path)
    
    print(f"Loading model from {model_path}")
    # Create a mock envs object for Agent initialization (matching training script)
    mock_envs = type('MockEnvs', (), {
        'single_observation_space': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,)),
        'single_action_space': gym.spaces.Box(low=-1, high=1, shape=(action_dim,))
    })()
    
    agent = Agent(mock_envs)
    agent.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")
    
    # Converting model to ONNX
    Convert_ONNX(output_path=output_path, obs_dim=obs_dim, agent=agent)


if __name__ == "__main__":
    main()
