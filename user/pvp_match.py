from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import HardHardCodedBot, UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent #add anymore custom Agents (from train_agent.py) here as needed
from user.my_agent import SubmittedAgent
import pygame
pygame.init()

my_agent = UserInputAgent()

#Input your file path here in SubmittedAgent if you are loading a model:
opponent = SubmittedAgent(file_path=r'checkpoints\z3\rl_model_1773387_steps.zip')
# opponent = HardHardCodedBot()


match_time = 99999

# Run a single real-time match 
run_real_time_match(
    agent_1=opponent,  # Your AI
    agent_2=my_agent,  # You
    max_timesteps=30 * 999990000,  # Match time in frames (adjust as needed)
    resolution=CameraResolution.LOW,
)
