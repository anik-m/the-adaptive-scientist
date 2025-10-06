import os
import shutil
import subprocess
import re
import glob
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CFDEnv(gym.Env):
    def __init__(self, openfoam_bash, base_case_path): # <<< MODIFIED: Added 'base_case_path' argument
        super(CFDEnv, self).__init__()
        self.openfoam_bash = openfoam_bash
        self.base_case_path = base_case_path # <<< MODIFIED: Use the passed argument

        # Unique instance path to avoid conflicts
        self.instance_id = f"instance_{time.time()}_{np.random.randint(1000)}"
        self.case_path = os.path.join("/tmp", self.instance_id)

        # File paths
        self.fv_solution_path = os.path.join(self.case_path, 'system', 'fvSolution')
        self.control_dict_path = os.path.join(self.case_path, 'system', 'controlDict')
        self.log_path = os.path.join(self.case_path, 'log.simpleFoam')

        self.action_space = spaces.Box(low=0.1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.float32)

        self.residual_history = np.zeros(10, dtype=np.float32)
        self.step_count = 0
        self.max_steps = 100 # Max iterations per episode

    def _get_latest_time(self):
        """Finds the latest time step directory in the case."""
        time_dirs = glob.glob(os.path.join(self.case_path, '[0-9]*'))
        if not time_dirs:
            return 0
        return max([int(os.path.basename(d)) for d in time_dirs if os.path.basename(d).isdigit()])

    def _set_control_dict(self, start_time, end_time):
        """Dynamically sets the start and end time in controlDict."""
        with open(self.control_dict_path, 'r') as f:
            content = f.read()
        content = re.sub(r'startFrom\s+\w+;', f'startFrom\tstartTime;', content)
        content = re.sub(r'startTime\s+[\d\.]+;', f'startTime\t{start_time};', content)
        content = re.sub(r'endTime\s+[\d\.]+;', f'endTime\t{end_time};', content)
        with open(self.control_dict_path, 'w') as f:
            f.write(content)

    def _run_openfoam_command(self, command):
        """Runs an OpenFOAM command."""
        full_cmd = f"source {self.openfoam_bash} && {command}"
        try:
            # Use a single log file, overwriting it for each command
            with open(self.log_path, 'w') as log_file:
                subprocess.run(
                    full_cmd, shell=True, check=True, cwd=self.case_path,
                    executable='/bin/bash', stdout=log_file, stderr=subprocess.STDOUT
                )
            return True
        except subprocess.CalledProcessError:
            return False

    def _parse_residuals(self):
        """Parses the final residuals from the log file."""
        residuals = {'Ux': None, 'p': None}
        try:
            with open(self.log_path, 'r') as f:
                log_content = f.read()
            # Find the last occurrence of 'Solving for' blocks
            u_match = re.findall(r'Solving for Ux.*Initial residual = ([\d.e-]+)', log_content)
            p_match = re.findall(r'Solving for p.*Initial residual = ([\d.e-]+)', log_content)
            if u_match: residuals['Ux'] = float(u_match[-1])
            if p_match: residuals['p'] = float(p_match[-1])
        except (FileNotFoundError, IndexError):
            return None, True
        if residuals['Ux'] is None or residuals['p'] is None or np.isnan(residuals['Ux']) or np.isnan(residuals['p']):
            return None, True # Diverged if residuals are bad
        return np.array([residuals['Ux'], residuals['p']]), False

    def _update_relaxation_factors(self, action):
        """Updates relaxation factors in fvSolution."""
        u_urf, p_urf = np.clip(action[0], 0.1, 1.0), np.clip(action[1], 0.1, 1.0)
        with open(self.fv_solution_path, 'r') as f:
            content = f.read()
        content = re.sub(r'(U\s+)[0-9\.]+', fr'\g<1>{u_urf:.3f}', content)
        content = re.sub(r'(p\s+)[0-9\.]+', fr'\g<1>{p_urf:.3f}', content)
        with open(self.fv_solution_path, 'w') as f:
            f.write(content)

    def step(self, action):
        self.step_count += 1
        self._update_relaxation_factors(action)

        # Get current time and set controlDict to run for ONE step
        current_time = self._get_latest_time()
        self._set_control_dict(start_time=current_time, end_time=current_time + 1)

        command_successful = self._run_openfoam_command('simpleFoam')
        new_residuals, diverged = (None, True) if not command_successful else self._parse_residuals()

        info, terminated, truncated = {}, False, False
        if diverged:
            reward = -200.0
            terminated = True
        else:
            # Reward is negative log of pressure residual (we want to minimize it)
            reward = -np.log10(new_residuals[1] + 1e-10)
            self.residual_history = np.roll(self.residual_history, 2)
            self.residual_history[0:2] = new_residuals

            # Check for convergence
            if new_residuals[1] < 1e-5:
                reward += 100.0 # Large bonus for converging
                terminated = True

        if self.step_count >= self.max_steps:
             truncated = True

        return self.residual_history, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.residual_history.fill(0)

        # Create a unique, clean case directory for this episode
        if os.path.exists(self.case_path):
            shutil.rmtree(self.case_path)
        shutil.copytree(self.base_case_path, self.case_path)

        # We need a meshed case but don't run the solver here.
        # The first step will handle the first iteration.
        self._run_openfoam_command('blockMesh')

        # The initial state is a vector of zeros, as no simulation has run.
        info = {}
        return self.residual_history, info

    def close(self):
        if os.path.exists(self.case_path):
            shutil.rmtree(self.case_path)
