# # SUBMISSION: Agent
# This will be the Agent class we run in the 1v1. We've started you off with a functioning RL agent (`SB3Agent(Agent)`) and if-statement agent (`BasedAgent(Agent)`). Feel free to copy either to `SubmittedAgent(Agent)` then begin modifying.
# 
# Requirements:
# - Your submission **MUST** be of type `SubmittedAgent(Agent)`
# - Any instantiated classes **MUST** be defined within and below this code block.
# 
# Remember, your agent can be either machine learning, OR if-statement based. I've seen many successful agents arising purely from if-statements - give them a shot as well, if ML is too complicated at first!!
# 
# Also PLEASE ask us questions in the Discord server if any of the API is confusing. We'd be more than happy to clarify and get the team on the right track.
# Requirements:
# - **DO NOT** import any modules beyond the following code block. They will not be parsed and may cause your submission to fail validation.
# - Only write imports that have not been used above this code block
# - Only write imports that are from libraries listed here
# We're using PPO by default, but feel free to experiment with other Stable-Baselines 3 algorithms!

import os
import numpy as np
import gdown
from typing import Optional
from gymnasium import spaces, ActionWrapper
from stable_baselines3 import PPO
from environment.agent import Agent

class BoxToMultiBinary10(ActionWrapper):
    """
    Expose MultiBinary(10) to the algo, while the underlying env still accepts Box(0,1, (10,))
    with threshold 0.5. We feed exact 0/1 floats to avoid nondifferentiable, off-policy thresholds.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiBinary(10)  # 10 buttons

    def action(self, action: np.ndarray) -> np.ndarray:
        # PPO will give 0/1 ints for MultiBinary; convert to float for the Box env
        # No extra thresholding needed downstream.
        return action.astype(np.float32)

    def reverse_action(self, action):
        # Not strictly needed, but keeps interface complete
        return (action > 0.5).astype(np.int8)


class SubmittedAgent(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
        self,
        file_path: Optional[str] = None,
    ):
        super().__init__(file_path)

        # To run a TTNN model, you must maintain a pointer to the device and can be done by 
        # uncommmenting the line below to use the device pointer
        # self.mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1,1))

    def _initialize(self) -> None:
        if self.file_path is None:
            raise RuntimeError("Please give it a filepath.")
        else:
            # For inference: load trained model
            self.model = PPO.load(self.file_path)
            self.model.policy.set_training_mode(False)     # SB3 helper
            self.model.policy.eval()                       # PyTorch safety belt
            for p in self.model.policy.parameters():
                p.requires_grad_(False)

    def _gdown(self) -> str:
        # Option 1: Use a local checkpoint after training (QRDQN models)
        # Try latest QRDQN checkpoints first (num4, num3, num2, num1)
        for exp_name in ['num4', 'num3', 'num2', 'num1', 'e1', 'e']:
            experiment_dir = f"checkpoints/{exp_name}"
            if os.path.isdir(experiment_dir):
                # Find the latest checkpoint
                files = [f for f in os.listdir(experiment_dir) if f.endswith('.zip')]
                if files:
                    # Get the checkpoint with the highest step count
                    # Filename format: rl_model_2150042_steps.zip
                    files.sort(key=lambda x: int(x.split('_')[-2]))
                    latest_checkpoint = os.path.join(experiment_dir, files[-1])
                    if os.path.isfile(latest_checkpoint):
                        print(f"Loading local checkpoint: {latest_checkpoint}")
                        return latest_checkpoint
        
        # Option 2: Download from Google Drive
        data_path = "rl-model.zip"
        if not os.path.isfile(data_path):
            print("downloading from internet")
            print(f"Downloading {data_path}...")
            # Place a link to your PUBLIC model data here. This is where we will download it from on the tournament server.
            url = "https://drive.google.com/file/d/1JIokiBOrOClh8piclbMlpEEs6mj3H1HJ/view?usp=sharing"
            gdown.download(url, output=data_path, fuzzy=True)
        return data_path

    def predict(self, obs):
        act, _ = self.model.predict(obs, deterministic=True)
        return act.astype(np.float32)

    def save(self, file_path: str) -> None:
        self.model.save(file_path)

    # If modifying the number of models (or training in general), modify this
    def learn(self, env, total_timesteps, log_interval: int = 4):
        self.model.set_env(BoxToMultiBinary10(env))
        self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)
