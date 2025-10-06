import os
import time
import argparse
import subprocess
from src.cfd_environment import CFDEnv
from src.ppo_agent import PPO
from src.utils.case_generator import create_cavity_case

# Define the path to the OpenFOAM environment setup script.
OPENFOAM_BASH_PATH = "/opt/openfoam12/etc/bashrc"

def run_baseline(case_path):
    """Runs the baseline simulation to establish a benchmark."""
    print("Running baseline simulation...")
    command = f"source {OPENFOAM_BASH_PATH} && blockMesh && simpleFoam"
    log_path = os.path.join(case_path, "log.baseline")
    with open(log_path, 'w') as log_file:
        subprocess.run(command, shell=True, check=True, executable='/bin/bash', cwd=case_path, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Baseline simulation complete. Log saved to '{log_path}'")

def main(args):
    # --- Setup ---
    if not os.path.exists(OPENFOAM_BASH_PATH):
        raise FileNotFoundError(f"OpenFOAM bashrc not found at {OPENFOAM_BASH_PATH}. Please run setup.sh first.")

    base_case_path = "cavity_base"
    create_cavity_case(base_case_path)
    
    if args.baseline_only:
        run_baseline(base_case_path)
        return

    # --- RL Training ---
    env = CFDEnv(openfoam_bash=OPENFOAM_BASH_PATH, base_case_path=base_case_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.k_epochs, args.eps_clip, args.action_std)

    print("Starting training...")
    start_time = time.time()
    time_step = 0
    i_episode = 0

    while time_step <= args.max_timesteps:
        state, _ = env.reset()
        current_ep_reward = 0

        for t in range(1, env.max_steps + 2):
            action = ppo_agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(terminated or truncated)

            time_step += 1
            current_ep_reward += reward

            if time_step % args.update_timestep == 0:
                ppo_agent.update()

            if terminated or truncated:
                break
        
        i_episode += 1
        if i_episode % 10 == 0:
            print(f"Episode: {i_episode}, Timesteps: {time_step}, Reward: {current_ep_reward:.2f}")

    env.close()
    print(f"\nTraining finished in {time.time() - start_time:.2f} seconds.")
    ppo_agent.save(args.model_path)
    print(f"Model saved as {args.model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a PPO agent to optimize CFD convergence.")
    parser.add_argument("--max-timesteps", type=int, default=5000, help="Total timesteps for training.")
    parser.add_argument("--update-timestep", type=int, default=100, help="Timesteps to collect before an update.")
    parser.add_argument("--k-epochs", type=int, default=20, help="Number of epochs to update the policy.")
    parser.add_argument("--lr-actor", type=float, default=0.0001, help="Learning rate for the actor.")
    parser.add_argument("--lr-critic", type=float, default=0.001, help="Learning rate for the critic.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--eps-clip", type=float, default=0.2, help="PPO clip parameter.")
    parser.add_argument("--action-std", type=float, default=0.1, help="Initial action standard deviation.")
    parser.add_argument("--model-path", type=str, default="PPO_CFD_Solver.pth", help="Path to save the trained model.")
    parser.add_argument("--baseline-only", action="store_true", help="Only run the baseline simulation and exit.")
    
    args = parser.parse_args()
    main(args)
