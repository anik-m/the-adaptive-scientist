


# The Adaptive Scientist



This project uses a Deep Reinforcement Learning (RL) based agent to intelligently control the relaxation factors in an OpenFOAM CFD simulation, with the goal of reaching a converged solution faster than the default solver settings.



## 🚀 Project Structure

- **`/src`**: Contains the core Python source code.
  - `cfd_environment.py`: The custom `gymnasium` environment that interfaces with OpenFOAM.
  - `ppo_agent.py`: The PPO agent implementation.
  - `utils/case_generator.py`: A utility to create the OpenFOAM test case.
- **`train.py`**: The main script to train the RL agent.
- **`evaluate.py`**: A script to evaluate a trained agent and compare it to the baseline.
- **`setup.sh`**: A shell script to install all dependencies (OpenFOAM and Python packages) in a Debian/Ubuntu-based environment like Google Colab.
- **`requirements.txt`**: A list of Python packages required for this project.

## ⚙️ Setup and Installation

### In a Cloud Environment (like Google Colab)

1.  **Clone the repository:**
    ```bash
    !git clone [https://github.com/your-username/ppo-openfoam-optimizer.git](https://github.com/your-username/ppo-openfoam-optimizer.git)
    !cd ppo-openfoam-optimizer
    ```

2.  **Run the setup script:**
    This will install OpenFOAM and all Python dependencies. This will take several minutes.
    ```bash
    !bash setup.sh
    ```



## 📈 Usage

### 1. Run the Baseline Simulation
First, establish the performance benchmark using the default OpenFOAM solver settings.

```bash
!python train.py --baseline-only
```
This will create a `cavity_base/` directory containing the CFD case and a `log.baseline` file with the results.

### 2. Train the RL Agent
To train the agent, run the `train.py` script.

```bash
!python train.py
```
You can customize the training process with command-line arguments. For example, to run a longer training session:
```bash
!python train.py --max-timesteps 20000 --lr-actor 0.00005
```
The trained model will be saved as `PPO_CFD_Solver.pth` by default.



### 3. Evaluate the Trained Agent
Once training is complete, evaluate your saved model to see if it outperforms the baseline.

```bash
!python evaluate.py --model-path PPO_CFD_Solver.pth
```
The script will run a full simulation controlled by the agent and print a summary comparing the number of steps taken by the agent versus the baseline.

## Local Setup

For a local installation, it's highly recommended to use a virtual environment to manage project dependencies.

1.  **Create and Activate a Virtual Environment:**
    Open your terminal in the project directory and run:
    ```bash
    # Create a virtual environment named 'venv'
    python3 -m venv venv

    # Activate the environment (on Linux/macOS)
    source venv/bin/activate
    ```
Or if you prefer, use conda/mamba

2.  **Run the Setup Script:**
    The `setup.sh` script will install all system and Python dependencies. Execute it from your terminal using `bash`:
    ```bash
    bash setup.sh
    ```

3.  **Run the Project:**
    Once the setup is complete, you can run the project scripts.
    ```bash
    python train.py
    ```
    *Note: Remember to deactivate the environment in your terminal when you are finished.*




## 📈 Simulation & Results

This project simulates **lid-driven cavity flow**, a classic CFD benchmark problem. The case involves a fluid within a square cavity where the top "lid" moves at a constant speed, generating a primary vortex within the fluid.

The RL agent was trained for **5,000 timesteps** to control the solver's under-relaxation factors. It successfully learned a policy that dramatically outperformed the baseline solver, reaching convergence **90% faster**.

| Method                  | Iterations to Converge |
| ----------------------- | ---------------------- | 
| Baseline (Default URFs) | 1000                   | 
| **Trained RL Agent** | **100** |

### Motivation

My primary goal was to see if connecting a learning agent to a complex, "black-box" environment like a physics simulator, which is a common real-world application of RL, would have any effect. This project serves as my sandbox for exploring the practical challenges of defining state and action spaces, designing reward functions, and implementing a complete agent-environment loop from scratch. This helped me gain useful knowledge in RL, PPO, and to apply it to a real word problem.



#### Core Technologies

*   **CFD Solver**: [**OpenFOAM**](https://www.openfoam.com/), the leading open-source C++ toolbox for CFD. 
*   **RL Environment**: [**Gymnasium**](https://gymnasium.farama.org/), the standard toolkit for building RL environments. 
*   **RL Algorithm**: [**Proximal Policy Optimization (PPO)**](https://spinningup.openai.com/en/latest/algorithms/ppo.html) [1], a robust policy gradient algorithm. 

#### Reinforcement Learning Formulation (MDP)

A major part was framing the problem as a Markov Decision Process (MDP) [2]:

*   **State ($S$):** To represent the simulation's status, I designed a state vector composed of the history of the last *N* solver residuals for pressure (`p`) and velocity (`U`). This gives the agent a sense of the convergence trend.
*   **Action ($A$):** The agent's action is a continuous vector representing the under-relaxation factors (URFs) for the solver. The PPO agent's policy network outputs a distribution from which these continuous values are sampled.
*   **Reward ($R$):** I engineered a reward function to guide the agent. The reward at each step is the **negative logarithm of the current pressure residual**. This provides a dense signal that strongly encourages the agent to make the error smaller. A large negative penalty is given if the simulation diverges, teaching the agent to maintain stability.

#### Agent-Environment Interaction Loop

RL agent communicates with the C++-based OpenFOAM solver using a simple but effective file-based I/O loop:

1.  **Agent Selects Action**: The PPO agent chooses a new set of URFs.
2.  **Environment Updates Config**: My custom environment script programmatically modifies the `system/fvSolution` file with the new URFs.
3.  **Environment Executes Step**: The environment calls the `simpleFoam` solver for a fixed number of iterations using Python's `subprocess` module.
4.  **Environment Parses State**: The script then parses the solver's log file using regular expressions to extract the new residuals, which form the next state.
5.  **Reward Calculation**: The reward is calculated from the new residuals, and the `(state, reward, done)` tuple is returned to the agent to continue the learning loop.

### Project Status & Learning Goals


*   **Phase 0: Establishing a proof of concept for RL - [Completed]**
    *   Learn about the underlying compoenents necessary for establishing justification for RL and PPO, and implementing it.



*   **Phase 1: Exploring High-Performance Computing (HPC) Integration**
    *   Learn how to scale this experiment by running multiple simulations in parallel.
    *   Investigate frameworks like **SmartSim** [3] or **DRLinFluids** [4] that enable efficient, in-memory data exchange between Python and HPC simulations, moving beyond the slow file-based method used here.
*   **Phase 2: Deepening the Research**
    *   Apply the agent to more complex 2D and 3D CFD problems.
    *   Experiment with more sophisticated state representations and reward-shaping techniques, inspired by recent work. [5]


### Feedback and Collaborations Welcome!

Further development of this project, such as applying the agent to more complex simulations or exploring other RL algorithms, is dependent on finding interested collaborators. Please reach out if you would like to contribute. 

As this is a personal learning project, I am very open to feedback, suggestions, and advice from others. 


### References and Further Reading

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.

[2] Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction." *MIT Press*.

[3] Kurz, M., et al. (2022). "Deep Reinforcement Learning for Computational Fluid Dynamics on HPC Systems." *arXiv preprint arXiv:2205.06502*.

[4] Wang, Q., et al. (2022). "DRLinFluids--An open-source python platform of coupling Deep Reinforcement Learning and OpenFOAM." *Physics of Fluids*.

[5] Maulik, R., et al. (2021). "Reinforcement learning for the runtime control of physics-based simulations." *Journal of Open Source Software*.


