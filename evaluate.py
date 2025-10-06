import os
import re
import torch
import argparse
from src.cfd_environment import CFDEnv
from src.ppo_agent import PPO

# Define the path to the OpenFOAM environment setup script.
OPENFOAM_BASH_PATH = "/opt/openfoam12/etc/bashrc"

def parse_iteration_count(log_file_path):
    """Parses an OpenFOAM log file to find the final iteration count."""
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
        time_steps = re.findall(r'^Time = (\d+)', log_content, re.MULTILINE)
        return int(time_steps[-1]) if time_steps else None
    except (FileNotFoundError, IndexError):
        return None

def main(args):
    # --- Setup ---
    if not os.path.exists(OPENFOAM_BASH_PATH):
        raise FileNotFoundError(f"OpenFOAM bashrc not found at {OPENFOAM_BASH_PATH}. Please run setup.sh first.")

    # We assume the base case and baseline log were created by train.py
    base_case_path = "cavity_base"
    if not os.path.exists(base_case_path):
        raise FileNotFoundError("Base case not found. Please run 'python train.py --baseline-only' first.")

    env = CFDEnv(openfoam_bash=OPENFOAM_BASH_PATH, base_case_path=base_case_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Agent parameters must match the trained agent's
    ppo_agent_eval = PPO(state_dim, action_dim, 0, 0, 0, 0, 0, 0.1)
    ppo_agent_eval.load(args.model_path)
    print(f"Loaded trained model from {args.model_path}")

    # --- Run Evaluation ---
    state, _ = env.reset()
    done = False
    eval_step = 0

    print("\n| Step |      Action (U, p)      |  Reward  | Converged/Done |")
    print("|------|-------------------------|----------|----------------|")

    while not done and eval_step < env.max_steps:
        eval_step += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_raw = ppo_agent_eval.policy_old.actor(state_tensor).numpy()
            action = ((action_raw + 1) / 2) * 0.9 + 0.1 # Tanh scaling

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        action_str = f"({action[0]:.3f}, {action[1]:.3f})"
        print(f"| {eval_step:<4} | {action_str:<23} | {reward:<8.2f} | {str(done):<14} |")

    # --- Automated Performance Comparison ---
    print("\n====================== Evaluation Summary ======================")
    baseline_steps = parse_iteration_count(os.path.join(base_case_path, "log.baseline"))
    agent_steps = eval_step

    print(f"Trained agent finished in: {agent_steps} steps.")

    if baseline_steps is not None:
        print(f"Baseline (default solver) finished in: {baseline_steps} steps.")
        print("-" * 60)
        if agent_steps < baseline_steps:
            improvement = ((baseline_steps - agent_steps) / baseline_steps) * 100
            print(f"SUCCESS: The agent was {improvement:.2f}% faster than the baseline. 🚀")
        else:
            slowdown = ((agent_steps - baseline_steps) / baseline_steps) * 100
            print(f"REGRESSION: The agent was {slowdown:.2f}% slower than the baseline.")
    else:
        print("Could not determine baseline performance to compare against.")
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth) file.")
    args = parser.parse_args()
    main(args)
