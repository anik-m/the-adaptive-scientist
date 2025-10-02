

# The Adaptive Scientist



This repository documents my personal project aimed at learning and applying Deep Reinforcement Learning (RL) to a challenging and interesting problem: controlling a scientific simulation in real-time. The goal is to train an RL agent that can dynamically tune the hyperparameters of a Computational Fluid Dynamics (CFD) solver to help it converge faster.

### Motivation

I started this project to move beyond textbook examples and gain hands-on experience with the entire RL workflow. My primary goal was to understand how to connect a learning agent to a complex, "black-box" environment like a physics simulator, which is a common real-world application of RL. This project serves as my sandbox for exploring the practical challenges of defining state and action spaces, designing reward functions, and implementing a complete agent-environment loop from scratch.



### Technical Approach 

This proof-of-concept is built entirely within a notebook, making it self-contained and easy to reproduce. Here are the key technical components I implemented:

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

I will try to continue the project and will update milestones appropriately.

*   **Phase 0: Establishing a proof of concept for RL (completed)**
    *   Learn about all the underlying compoenents necessary for establishing justification for RL and PPO, and implementing it.

*   **Phase 1: Exploring High-Performance Computing (HPC) Integration**
    *   Learn how to scale this experiment by running multiple simulations in parallel.
    *   Investigate frameworks like **SmartSim** or **DRLinFluids** that enable efficient, in-memory data exchange between Python and HPC simulations, moving beyond the slow file-based method used here.
*   **Phase 2: Deepening the Research**
    *   Apply the agent to more complex 2D and 3D CFD problems.
    *   Experiment with more sophisticated state representations and reward-shaping techniques.


### Getting Started: Run the Demo

You can run my proof-of-concept yourself directly in your browser.

[Open in colab](https://colab.research.google.com/github/anik-m/YOUR_REPO_NAME/blob/main/adaptive_scientist_colab_demo.ipynb)

**Instructions:**

1.  Click the "Open in Colab" button above.
2.  In the notebook, click `Runtime` -> `Run all`.
3.  The notebook will install OpenFOAM, run the baseline simulation, train the RL agent, and evaluate its performance. The entire process takes about [e.g., 1-2 hours] to complete.

### Feedback Welcome!

As this is a personal learning project, I am very open to feedback, suggestions, and advice from others. If you have any ideas for improvement or spot something that could be done better, please feel free to open an issue!

### References and Further Reading

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal Policy Optimization Algorithms." *arXiv preprint arXiv:1707.06347*.

[2] Sutton, R. S., & Barto, A. G. (2018). "Reinforcement Learning: An Introduction." *MIT Press*.

[3] Kurz, M., et al. (2022). "Deep Reinforcement Learning for Computational Fluid Dynamics on HPC Systems." *arXiv preprint arXiv:2205.06502*.

[4] Wang, Q., et al. (2022). "DRLinFluids--An open-source python platform of coupling Deep Reinforcement Learning and OpenFOAM." *Physics of Fluids*.

[5] Maulik, R., et al. (2021). "Reinforcement learning for the runtime control of physics-based simulations." *Journal of Open Source Software*.




