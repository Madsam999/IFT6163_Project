"""Quick visual test for PupperNavEnv-v0.

Runs the robot with random actions so you can see:
  - The red goal sphere spawns at a random position
  - The observation space is 19-dim (16 base + 3 goal)
  - dist_to_goal decreases/increases as the robot stumbles around
  - A new goal spawns on each reset

Usage:
    python puppersim/test_nav_env.py
"""

import time
import numpy as np
import gymnasium as gym
import puppersim  # registers PupperNavEnv-v0


def main():
    print("Creating PupperNavEnv-v0 ...")
    env = gym.make("PupperNavEnv-v0", render_mode="human")

    # Base env has 18 obs (12 motor + 4 IMU + 2 PMTG phase) + 3 goal = 21
    # Action space is 16: 12 motor residuals + 4 PMTG gait params
    print(f"Observation space: {env.observation_space.shape}  (expected: (21,))")
    print(f"Action space:      {env.action_space.shape}  (expected: (16,))")
    print()

    for episode in range(3):
        obs, info = env.reset(seed=episode)
        print(f"--- Episode {episode + 1} ---")
        print(f"  Initial obs shape : {obs.shape}")
        print(f"  Goal obs (last 3) : dx={obs[-3]:.3f}  dy={obs[-2]:.3f}  dist={obs[-1]:.3f}")

        ep_return = 0.0
        for step in range(300):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            if step % 50 == 0:
                print(f"  step={step:3d}  dist={info['dist_to_goal']:.3f}  "
                      f"reward={reward:+.3f}  success={info['success']}")

            time.sleep(0.01)  # slow down to roughly real-time

            if terminated or truncated:
                status = "SUCCESS" if info["success"] else "fell/timeout"
                print(f"  Done at step {step} — {status}  ep_return={ep_return:.3f}")
                break

    env.close()
    print("\nTest complete.")


if __name__ == "__main__":
    main()
