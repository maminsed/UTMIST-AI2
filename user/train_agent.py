'''
TRAINING: AGENT

This file contains all the types of Agent classes, the Reward Function API, and the built-in train function from our multi-agent RL API for self-play training.
- All of these Agent classes are each described below. 

Running this file will initiate the training function, and will:
a) Start training from scratch
b) Continue training from a specific timestep given an input `file_path`
'''

# -------------------------------------------------------------------
# ----------------------------- IMPORTS -----------------------------
# -------------------------------------------------------------------

from types import NoneType
import torch; torch.set_float32_matmul_precision('high')
import gymnasium as gym
from torch.nn import functional as F
from torch import nn as nn
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, SAC, DQN, DDPG, TD3, HER 
from sb3_contrib import QRDQN, RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment.agent import *
from typing import Optional, Type, List, Tuple

# -------------------------------------------------------------------------
# ----------------------------- AGENT CLASSES -----------------------------
# -------------------------------------------------------------------------

# class SB3Agent(Agent):
#     '''
#     SB3Agent:
#     - Defines an AI Agent that takes an SB3 class input for specific SB3 algorithm (e.g. PPO, SAC)
#     Note:
#     - For all SB3 classes, if you'd like to define your own neural network policy you can modify the `policy_kwargs` parameter in `self.sb3_class()` or make a custom SB3 `BaseFeaturesExtractor`
#     You can refer to this for Custom Policy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
#     '''
#     def __init__(
#             self,
#             sb3_class: Optional[Type[BaseAlgorithm]] = PPO,
#             file_path: Optional[str] = None
#     ):
#         self.sb3_class = sb3_class
#         super().__init__(file_path)

#     def _initialize(self) -> None:
#         if self.file_path is None:
#             self.model = self.sb3_class("MlpPolicy", self.env, verbose=0, n_steps=30*90*3, batch_size=128, ent_coef=0.01)
#             del self.env
#         else:
#             self.model = self.sb3_class.load(self.file_path)

#     def _gdown(self) -> str:
#         # Call gdown to your link
#         return

#     #def set_ignore_grad(self) -> None:
#         #self.model.set_ignore_act_grad(True)

#     def predict(self, obs):
#         action, _ = self.model.predict(obs)
#         return action

#     def save(self, file_path: str) -> None:
#         self.model.save(file_path, include=['num_timesteps'])

#     def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
#         self.model.set_env(env)
#         self.model.verbose = verbose
#         self.model.learn(
#             total_timesteps=total_timesteps,
#             log_interval=log_interval,
#         )

# class RecurrentPPOAgent(Agent):
#     '''
#     RecurrentPPOAgent:
#     - Defines an RL Agent that uses the Recurrent PPO (LSTM+PPO) algorithm
#     '''
#     def __init__(
#             self,
#             file_path: Optional[str] = None
#     ):
#         super().__init__(file_path)
#         self.lstm_states = None
#         self.episode_starts = np.ones((1,), dtype=bool)

#     def _initialize(self) -> None:
#         if self.file_path is None:
#             policy_kwargs = {
#                 'activation_fn': nn.ReLU,
#                 'lstm_hidden_size': 512,
#                 'net_arch': [dict(pi=[32, 32], vf=[32, 32])],
#                 'shared_lstm': True,
#                 'enable_critic_lstm': False,
#                 'share_features_extractor': True,

#             }
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             print(f"Using device: {device}")
#             self.model = RecurrentPPO("MlpLstmPolicy",
#                                       self.env,
#                                       verbose=0,
#                                       n_steps=30*90*20,
#                                       batch_size=16,
#                                       ent_coef=0.05,
#                                       policy_kwargs=policy_kwargs,
#                                       device=device)
#             del self.env
#         else:
#             self.model = RecurrentPPO.load(self.file_path)

#     def reset(self) -> None:
#         self.episode_starts = True

#     def predict(self, obs):
#         action, self.lstm_states = self.model.predict(obs, state=self.lstm_states, episode_start=self.episode_starts, deterministic=True)
#         if self.episode_starts: self.episode_starts = False
#         return action

#     def save(self, file_path: str) -> None:
#         self.model.save(file_path)

#     def learn(self, env, total_timesteps, log_interval: int = 2, verbose=0):
#         self.model.set_env(env)
#         self.model.verbose = verbose
#         self.model.learn(total_timesteps=total_timesteps, log_interval=log_interval)

class EasyHardCodedBot(Agent):
    '''
    EasyHardCodedBot:
    - Defines a hard-coded Agent that predicts actions based on if-statements. Interesting behaviour can be achieved here.
    - The if-statement algorithm can be developed within the `predict` method below.
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()

        # If off the edge, come back
        if pos[0] > 10.67/2:
            action = self.act_helper.press_keys(['a'])
        elif pos[0] < -10.67/2:
            action = self.act_helper.press_keys(['d'])
        elif not opp_KO:
            # Head toward opponent
            if (opp_pos[0] > pos[0]):
                action = self.act_helper.press_keys(['d'])
            else:
                action = self.act_helper.press_keys(['a'])

        # Note: Passing in partial action
        # Jump if below map or opponent is above you
        if (pos[1] > 1.6 or pos[1] > opp_pos[1]) and self.time % 2 == 0:
            action = self.act_helper.press_keys(['space'], action)

        # Attack if near #TODO: change
        # if (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
        #     action = self.act_helper.press_keys(['j'], action)
        return action

class HardHardCodedBot(Agent):
    '''
    Input the **file_path** to your agent here for submission!
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.time = 0
        self.prev_pos = None
        self.down = False
        self.recover = False

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_KO = self.obs_helper.get_section(obs, 'opponent_state') in [5, 11]
        action = self.act_helper.zeros()
        facing = self.obs_helper.get_section(obs, 'player_facing')

        if self.prev_pos is not None:
            self.down = (pos[1] - self.prev_pos[1]) > 0
        self.prev_pos = pos

        self.recover = False
        if pos[0] < -6.9:
            action = self.act_helper.press_keys(['d'], action)
            self.recover = True
        elif pos[0] > -1.9 and pos[0] < 0:
            action = self.act_helper.press_keys(['a'], action)
            self.recover = True
        elif pos[0] > 0 and pos[0] < 1.9:
            action = self.act_helper.press_keys(['d'], action)
            self.recover = True
        elif pos[0] > 6.9:
            action = self.act_helper.press_keys(['a'], action)
            self.recover = True

        # Jump if falling
        if self.down or self.obs_helper.get_section(obs, 'player_grounded') == 1:
            '''
            if self.obs_helper.get_section(obs, "player_recoveries_left") > 0:
                action = self.act_helper.press_keys(['k'], action)
            else:
                action = self.act_helper.press_keys(['space'], action)
            '''
            if self.time % 10 == 0:
                action = self.act_helper.press_keys(['space'], action)

        
        if not self.recover:
            if opp_pos[0] > pos[0] and facing == 0:
                action = self.act_helper.press_keys(['d'], action)
            elif opp_pos[0] < pos[0] and facing == 1:
                action = self.act_helper.press_keys(['a'], action)
        
                
        # Attack if near
        if not self.recover and (pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        return action

class UserInputAgent(Agent):
    '''
    UserInputAgent:
    - Defines an Agent that performs actions entirely via real-time player input
    '''
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        action = self.act_helper.zeros()
       
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            action = self.act_helper.press_keys(['w'], action)
        if keys[pygame.K_a]:
            action = self.act_helper.press_keys(['a'], action)
        if keys[pygame.K_s]:
            action = self.act_helper.press_keys(['s'], action)
        if keys[pygame.K_d]:
            action = self.act_helper.press_keys(['d'], action)
        if keys[pygame.K_SPACE]:
            action = self.act_helper.press_keys(['space'], action)
        # h j k l
        if keys[pygame.K_h]:
            action = self.act_helper.press_keys(['h'], action)
        if keys[pygame.K_j]:
            action = self.act_helper.press_keys(['j'], action)
        if keys[pygame.K_k]:
            action = self.act_helper.press_keys(['k'], action)
        if keys[pygame.K_l]:
            action = self.act_helper.press_keys(['l'], action)
        if keys[pygame.K_g]:
            action = self.act_helper.press_keys(['g'], action)

        return action

# class ClockworkAgent(Agent):
#     '''
#     ClockworkAgent:
#     - Defines an Agent that performs sequential steps of [duration, action]
#     '''
#     def __init__(
#             self,
#             action_sheet: Optional[List[Tuple[int, List[str]]]] = None,
#             *args,
#             **kwargs
#     ):
#         super().__init__(*args, **kwargs)

#         self.steps = 0
#         self.current_action_end = 0  # Tracks when the current action should stop
#         self.current_action_data = None  # Stores the active action
#         self.action_index = 0  # Index in the action sheet

#         if action_sheet is None:
#             self.action_sheet = [
#                 (10, ['a']),
#                 (1, ['l']),
#                 (20, ['a']),
#                 (3, ['a', 'j']),
#                 (15, ['space']),
#             ]
#         else:
#             self.action_sheet = action_sheet

#     def predict(self, obs):
#         """
#         Returns an action vector based on the predefined action sheet.
#         """
#         # Check if the current action has expired
#         if self.steps >= self.current_action_end and self.action_index < len(self.action_sheet):
#             hold_time, action_data = self.action_sheet[self.action_index]
#             self.current_action_data = action_data  # Store the action
#             self.current_action_end = self.steps + hold_time  # Set duration
#             self.action_index += 1  # Move to the next action

#         # Apply the currently active action
#         action = self.act_helper.press_keys(self.current_action_data)
#         self.steps += 1  # Increment step counter
#         return action



# class MLPExtractor(BaseFeaturesExtractor):
#     '''
#     Class that defines an MLP Base Features Extractor
#     '''
#     def __init__(self, observation_space: gym.Space, features_dim: int = 64, hidden_dim: int = 64):
#         super(MLPExtractor, self).__init__(observation_space, features_dim)
        
#         self.model = MLPPolicy(
#             obs_dim=observation_space.shape[0], 
#             action_dim=10,
#             hidden_dim=hidden_dim,
#         )
    
#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         return self.model(obs)
    
#     @classmethod
#     def get_policy_kwargs(cls, features_dim: int = 64, hidden_dim: int = 64) -> dict:
#         return dict(
#             features_extractor_class=cls,
#             features_extractor_kwargs=dict(features_dim=features_dim, hidden_dim=hidden_dim) #NOTE: features_dim = 10 to match action space output
#         )  

# class MLPPolicy(nn.Module):
#     def __init__(self, obs_dim: int = 64, action_dim: int = 10, hidden_dim: int = 64):
#         """
#         A 3-layer MLP policy:
#         obs -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim) -> ReLU -> Linear(action_dim)
#         """
#         super(MLPPolicy, self).__init__()

#         # Input layer
#         self.fc1 = nn.Linear(obs_dim, hidden_dim, dtype=torch.float32)
#         # Hidden layer
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32)
#         # Output layer
#         self.fc3 = nn.Linear(hidden_dim, action_dim, dtype=torch.float32)

#     def forward(self, obs):
#         """
#         obs: [batch_size, obs_dim]
#         returns: [batch_size, action_dim]
#         """
#         x = F.relu(self.fc1(obs))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)



class BoxToMultiBinary10(gym.ActionWrapper):
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

class MLPWithLayerNorm(BaseFeaturesExtractor):
    def __init__(self, observation_space:gym.Space, features_dim:int = 256):
        super().__init__(observation_space, features_dim)
        in_dim = observation_space.shape[0]
        self.model = nn.Sequential(
            nn.Linear(in_dim, 256), 
            nn.SiLU(),

            nn.Linear(256,features_dim),
            nn.SiLU()
        )

    def forward(self, obs:torch.Tensor) ->torch.Tensor:
        return self.model(obs)

class CustomAgent(Agent):
    def __init__(self, sb3_class: Optional[Type[BaseAlgorithm]]=PPO, file_path: str = None, extractor: BaseFeaturesExtractor = None):
        self.sb3_class = sb3_class
        self.extractor = extractor
        super().__init__(file_path)

    
    def _initialize(self) -> None:
        print("initializing PPO network: uwu")
        if self.file_path is None:
            # Set device for GPU training
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Wrap env in BoxToMultiBinary10 for proper action space
            wrapped_env = BoxToMultiBinary10(self.env)
            
            policy_kwargs=dict(
                features_extractor_class=MLPWithLayerNorm,
                features_extractor_kwargs=dict(features_dim=256)
            )
            self.model = self.sb3_class(
                "MlpPolicy", 
                    wrapped_env, 
                    n_steps=2048,          # rollout length per update
                    batch_size=512,        # minibatch size (2048 disables minibatching!)
                    n_epochs=10,       # updates per batch
                    gamma=0.995,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,         # encourage button exploration
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    learning_rate=3e-4,
                    verbose=0,
                    policy_kwargs=policy_kwargs,
                    device=device,  # Use GPU if available
                )
            del self.env
        else:
            self.model = self.sb3_class.load(self.file_path)

    def _gdown(self) -> str:
        # Call gdown to your link
        return

    #def set_ignore_grad(self) -> None:
        #self.model.set_ignore_act_grad(True)

    def predict(self, obs):
        act, _ = self.model.predict(obs)
        return act.astype(np.float32)

    def save(self, file_path: str) -> None:
        self.model.save(file_path, include=['num_timesteps'])

    def learn(self, env, total_timesteps, log_interval: int = 1, verbose=0):
        self.model.set_env(BoxToMultiBinary10(env))
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            progress_bar=False
        )

# --------------------------------------------------------------------------------
# ----------------------------- REWARD FUNCTIONS API -----------------------------
# --------------------------------------------------------------------------------

'''
Example Reward Functions:
- Find more [here](https://colab.research.google.com/drive/1qMs336DclBwdn6JBASa5ioDIfvenW8Ha?usp=sharing#scrollTo=-XAOXXMPTiHJ).
'''

def base_height_l2(
    env: WarehouseBrawl,
    target_height: float,
    obj_name: str = 'player'
) -> float:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # Extract the used quantities (to enable type-hinting)
    obj: GameObject = env.objects[obj_name]

    # Compute the L2 squared penalty
    return (obj.body.position.y - target_height)**2

class RewardMode(Enum):
    ASYMMETRIC_OFFENSIVE = 0
    SYMMETRIC = 1
    ASYMMETRIC_DEFENSIVE = 2

def damage_interaction_reward(
    env: WarehouseBrawl,
    mode: RewardMode = RewardMode.SYMMETRIC,
) -> float:
    """
    Computes the reward based on damage interactions between players.

    Modes:
    - ASYMMETRIC_OFFENSIVE (0): Reward is based only on damage dealt to the opponent
    - SYMMETRIC (1): Reward is based on both dealing damage to the opponent and avoiding damage
    - ASYMMETRIC_DEFENSIVE (2): Reward is based only on avoiding damage

    Args:
        env (WarehouseBrawl): The game environment
        mode (DamageRewardMode): Reward mode, one of DamageRewardMode

    Returns:
        float: The computed reward.
    """
    # Getting player and opponent from the enviornment
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]

    # Reward dependent on the mode
    damage_taken = player.damage_taken_this_frame
    damage_dealt = opponent.damage_taken_this_frame

    if mode == RewardMode.ASYMMETRIC_OFFENSIVE:
        reward = damage_dealt
    elif mode == RewardMode.SYMMETRIC:
        reward = damage_dealt - damage_taken
    elif mode == RewardMode.ASYMMETRIC_DEFENSIVE:
        reward = -damage_taken
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Clip per-step reward to [-0.2, +0.2]
    reward = reward / 140
    return clip_reward(reward)


# In[ ]:
def clip_reward(reward: float, min_val: float = -0.2, max_val: float = 0.2) -> float:
    """
    Clip per-step rewards to [-0.2, +0.2].
    Terminal rewards are NOT clipped.
    """
    return max(min_val, min(max_val, reward))

def phi_distance(p_x,p_y,o_x,o_y):
    dx, dy = p_x - o_x, p_y - o_y
    dist = (dx*dx + dy*dy) ** 0.5
    return dist

def head_to_opponent(
    env: WarehouseBrawl,
) -> float:
    player = env.objects['player']
    opponent = env.objects['opponent']

    # current absolute separations
    curr_dx = abs(player.body.position.x - opponent.body.position.x)
    curr_dy = abs(player.body.position.y - opponent.body.position.y)

    # previous absolute separations (use prev positions stored on objects)
    prev_dx = abs(player.prev_x - opponent.prev_x)
    prev_dy = abs(player.prev_y - opponent.prev_y)

    eps = 1e-6
    # fractional progress for each axis: positive when closing, negative when opening
    frac_x = (prev_dx - curr_dx) / max(prev_dx, eps)
    frac_y = (prev_dy - curr_dy) / max(prev_dy, eps)

    # weight horizontal closing more than vertical
    w_x, w_y = 0.7, 0.3

    combined = w_x * frac_x + w_y * frac_y

    # shape and scale the signal
    shaped = np.tanh(5.0 * combined)

    return env.dt * shaped

def on_win_reward(env: WarehouseBrawl, agent: str) -> float:
    #favouring early wins and penilizing early looses
    multipier = max(50/ max(env.steps, 1),1)
    if agent == 'player':
        return multipier
    else:
        return -multipier

def on_knockout_reward(env: WarehouseBrawl, agent: str) -> float:
    """
    Two separate events:
    - You KO them: +50
    - You get KO'd: -50
    """
    opponent = env.objects['opponent']
    player = env.objects['player']
    # This signal is emitted for the agent that got KO'd
    # So 'player' means player got KO'd, opponent got KO'd means win
    if agent == 'player':
        if player.damage_taken_this_stock < 15:
            return -1.5  # You just died??
        else:
            return -1.0
    else:
        if opponent.damage_taken_this_stock < 15:
            return 0.7  # Opponent got KO'd, player wins - you didn't do anything
        else:
            return 1.0

def on_equip_reward(env: WarehouseBrawl, agent: str) -> float:
    """
    Simple equip reward to prevent weapon camping.
    No special bonuses for specific weapons.
    """
    if agent == "player":
        # Just acknowledge weapon pickup, no special bonuses
        player = env.objects["player"]
        if player.weapon == "Hammer":
            return 2.0
        elif player.weapon == "Spear":
            return 1.0
    return 0.0

def on_drop_reward(env: WarehouseBrawl, agent: str) -> float:
    if agent == "player":
        if env.objects["player"].weapon == "Punch":
            return -1.0
    return 0.0

def on_combo_reward(env: WarehouseBrawl, agent: str) -> float:
    """Reward per EXTRA hit after the first - prevents dwarfing KO credit"""
    if agent == 'player':
        return env.dt * 0.05  # Per extra hit, scaled by weight
    else:
        return env.dt * -0.05

def whiff_punishment_reward(env: WarehouseBrawl) -> float:
    """
    STRONG penalties for whiffing attacks, especially when far from opponent.
    Stops spam attacking the air.
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # Calculate distance
    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    distance = (dx**2 + dy**2)**0.5
    vx,vy = player.body.velocity.x-opponent.body.velocity.x, player.body.velocity.y-opponent.body.velocity.y

    closing = (dx*vx+dy*vy) < 0
    
    far = min(distance/6.0,1.5)
    attacking = hasattr(player.state, 'move_type') and player.state.move_type not in [MoveType.NONE,MoveType.RECOVERY] 
    whiff = not hasattr(player, 'damage_taken_this_frame') or player.damage_taken_this_frame == 0
    if not attacking or not whiff:
        return 0.0
    
    not_closing = 1.0 if not closing else 0.4
    return clip_reward(-0.03 * env.dt * far * not_closing,-1,1)

def advantage_state_reward(env: WarehouseBrawl) -> float:
    """
    Rewards opponent in hitstun + tiny per-frame hitstun reward.
    Encourages sustained pressure.
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # Base reward for having opponent stunned
    # Check if opponent is in StunState
    if isinstance(opponent.state, StunState):
        base_reward = 0.05 * env.dt  # Advantage state reward
        hitstun_bonus = (0.02 * env.dt) if opponent.damage_taken_this_frame else 0 # Tiny per-frame hitstun reward
        return base_reward + hitstun_bonus
    
    return 0.0

def proximity_to_opponent_reward(env: WarehouseBrawl) -> float:
    """
    Rewards getting close to opponent - encourages chasing and engagement.
    Reward increases as distance decreases.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Calculate distance
    dx = player.body.position.x - opponent.body.position.x
    dy = player.body.position.y - opponent.body.position.y
    distance = (dx**2 + dy**2)**0.5
    
    max_distance = 15.0  # Maximum arena distance
    
    # Don't reward if player is stunned
    if isinstance(player.state, StunState):
        return 0.0
    
    # Reward for being close - closer = more reward
    # Inverse relationship: closer gets more reward
    if distance < max_distance:
        reward = (max_distance - distance) / max_distance
        return reward * 0.005  # INCREASED per-frame reward
    
    return 0.0

def weapon_stability_reward(env: WarehouseBrawl) -> float:
    """
    Small constant reward for having any weapon (discourages constant switching).
    """
    player: Player = env.objects["player"]
    
    # Reward for having a weapon, slightly more for better weapons
    if player.weapon in ["Hammer", "Spear"]:
        return env.dt * 0.001
    else:
        return 0.0
    
def jumping_on_middle(env:WarehouseBrawl):
    """
    reward for getting on middle
    """
    player: Player = env.objects['player']
    platform = env.objects['platform1']

    edge_x = 1  # Platform half-width (from environment code)
    # Get platform position directly from environment object
    platform_x = platform.body.position.x
    platform_y = platform.body.position.y

    x = player.body.position.x
    y = player.body.position.y
    
    if platform_x - edge_x < x < platform_x + edge_x and platform_y > y:
        return 0.5 * env.dt
    return 0.0

def no_input_penalty(env: WarehouseBrawl) -> float:
    """
    Punishes for not moving too much
    """
    player = env.objects["player"]
    x = player.body.position.x
    y = player.body.position.y
    prev_x = player.prev_x
    prev_y = player.prev_y
    dist = (x-prev_x)**2 + (y-prev_y) ** 2
    if dist > 0.1:
        return min(dist * 0.1,0.5) * env.dt
    return -0.1 * env.dt 

state_mapping = [
    'WalkingState',
    'StandingState',
    'TurnaroundState',
    'AirTurnaroundState',
    'SprintingState',
    'StunState',
    'InAirState',
    'DodgeState',
    'AttackState',
    'DashState',
    'BackDashState',
    'KOState',
    'TauntState',
]

def bad_taunt(env: WarehouseBrawl) -> float:
    player = env.objects['player']
    # Check if player is in TauntState
    if isinstance(player.state, TauntState):
        return clip_reward(-1.0 * env.dt)
    return 0.0



def edge_to_ko_bonus(env: WarehouseBrawl, agent: str) -> float:
    """Bonus for converting edge hits into KOs within 2 seconds"""
    if agent != 'player':
        return 0.0
    
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    # Check if opponent was near edge (within 3 units) when KO'd
    edge_x = env.stage_width_tiles // 2
    opponent_dist_to_left = abs(opponent.body.position.x + edge_x)
    opponent_dist_to_right = abs(opponent.body.position.x - edge_x)
    min_edge_dist = min(opponent_dist_to_left, opponent_dist_to_right)
    
    # Bonus if opponent was near edge when KO'd
    if min_edge_dist < 3.0:
        return 8.0
    
    return 0.0

def head_to_middle_reward(
    env: WarehouseBrawl,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    multiplier = -1 if player.body.position.x > 0 else 1
    reward = multiplier * (player.body.position.x - player.prev_x)

    return reward

def danger_zone_reward(
    env: WarehouseBrawl,
    zone_penalty: int = 1,
    zone_height: float = 4.2
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = -zone_penalty if player.body.position.y >= zone_height else 0.0

    return reward * env.dt

def in_state_reward(
    env: WarehouseBrawl,
    desired_state: Type[PlayerObjectState]=BackDashState,
) -> float:
    """
    Applies a penalty for every time frame player surpases a certain height threshold in the environment.

    Args:
        env (WarehouseBrawl): The game environment.
        zone_penalty (int): The penalty applied when the player is in the danger zone.
        zone_height (float): The height threshold defining the danger zone.

    Returns:
        float: The computed penalty as a tensor.
    """
    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is in the danger zone
    reward = 1 if isinstance(player.state, desired_state) else 0.0

    return reward * env.dt

def holding_more_than_3_keys(
    env: WarehouseBrawl,
) -> float:

    # Get player object from the environment
    player: Player = env.objects["player"]

    # Apply penalty if the player is holding more than 3 keys
    a = player.cur_action
    if (a > 0.5).sum() > 3:
        return env.dt
    return 0

def min_speed_reward(env: WarehouseBrawl) -> float:
    """
    Small bonus for having some horizontal speed, 0 when already fast
    """
    vx = abs(env.objects["player"].body.velocity.x)
    return min(vx, 1.0) * 0.02 * env.dt

def idle_penalty(env: WarehouseBrawl) -> float:
    """
    Penalize player for being idle (low velocity)
    """
    p = env.objects["player"]
    vx, vy = p.body.velocity.x, p.body.velocity.y
    if abs(vx) < 0.5 and abs(vy) < 0.5:
        return -0.1 * env.dt
    return 0.0

def forward_progress_reward(env: WarehouseBrawl) -> float:
    """
    Always pay for moving toward opponent this frame
    """
    p = env.objects["player"]
    o = env.objects["opponent"]
    dx = p.body.position.x - p.prev_x
    sign = 1 if p.body.position.x < o.body.position.x else -1
    return sign * dx  # positive if moved toward, negative if moved away

def time_pressure_reward(env: WarehouseBrawl) -> float:
    """
    Tiny constant per-step penalty to encourage decisive action.
    """
    return -0.0002  # Tiny constant per step

def retreat_penalty(env: WarehouseBrawl) -> float:
    """
    Penalizes retreating after grace window, but NOT when kiting for better engage.
    """
    player = env.objects["player"]
    opponent = env.objects["opponent"]
    
    dx = player.body.position.x - opponent.body.position.x
    vel_x = player.body.velocity.x
    
    # Check if moving away from opponent
    is_retreating = (dx > 0 and vel_x > 0.1) or (dx < 0 and vel_x < -0.1)
    
    if is_retreating:
        # Check if kiting toward a better position (moving toward opponent diagonally)
        # For now, simple check: if opponent is in bad position, allow kiting
        opp_in_bad_position = opponent.body.position.y > 4.0 or opponent.body.position.y < -2.0
        
        if not opp_in_bad_position:
            # This is a retreat, apply penalty
            return -0.01 * env.dt  # Grace window handled by small penalty
    
    return 0.0

def edge_avoidance_reward(env: WarehouseBrawl, danger_zone: float = 3.0) -> float:
    """
    Penalizes agent for being near map edges, BUT only when opponent is NOT off-stage.
    Allows edge-guarding when opponent is disadvantaged.
    """
    player: Player = env.objects["player"]
    opponent: Player = env.objects["opponent"]
    
    # Check if opponent is off-stage - if so, edge-guarding is valid
    edge_x = env.stage_width_tiles // 2
    edge_y = env.stage_height_tiles // 2
    
    opponent_dist_to_left = abs(opponent.body.position.x + edge_x)
    opponent_dist_to_right = abs(opponent.body.position.x - edge_x)
    opponent_dist_to_bottom = abs(opponent.body.position.y + edge_y)
    opponent_dist_to_top = abs(opponent.body.position.y - edge_y)
    opponent_min_dist = min(opponent_dist_to_left, opponent_dist_to_right, 
                           opponent_dist_to_bottom, opponent_dist_to_top)
    
    # If opponent is off-stage, allow edge-guarding
    if opponent_min_dist < 2.0 or opponent.body.position.y > 4.5:
        return 0.0  # No penalty when opponent is off-stage
    
    # Get arena boundaries
    edge_x = env.stage_width_tiles // 2
    edge_y = env.stage_height_tiles // 2
    
    # Distance to each edge
    dist_to_left = abs(player.body.position.x + edge_x)
    dist_to_right = abs(player.body.position.x - edge_x)
    dist_to_bottom = abs(player.body.position.y + edge_y)
    dist_to_top = abs(player.body.position.y - edge_y)
    
    # Find minimum distance to any edge
    min_dist_to_edge = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
    
    # Penalty if too close to edges (when opponent is not off-stage)
    # STRONGER penalty that scales more aggressively
    if min_dist_to_edge < danger_zone:
        # Quadratic penalty for being near edge
        penalty_ratio = min_dist_to_edge / danger_zone
        penalty = -((1.0 - penalty_ratio) ** 2) * 2.0  # Stronger near edges
    else:
        penalty = 0.0
    
    # Clip per-step reward but allow stronger penalties
    return clip_reward(penalty * env.dt * 2.0)  # Scale up the penalty

def fall_velocity_penalty(env: WarehouseBrawl, max_safe_velocity: float = 70.0) -> float:
    """
    Penalizes rapid falling ONLY when off-stage AND recovery resources low.
    Does NOT penalize fast-fall confirms on-stage.
    """
    player: Player = env.objects["player"]
    
    edge_x = env.stage_width_tiles // 2
    edge_y = env.stage_height_tiles // 2
    
    # Check if player is off-stage
    dist_to_left = abs(player.body.position.x + edge_x)
    dist_to_right = abs(player.body.position.x - edge_x)
    dist_to_bottom = abs(player.body.position.y + edge_y)
    player_min_dist = min(dist_to_left, dist_to_right, dist_to_bottom)
    
    is_offstage = player_min_dist < 2.0
    
    # Penalize rapid falling both on-stage (near edges) and off-stage
    if player.body.velocity.y < -max_safe_velocity:
        # Check recovery resources (jumps/recoveries left)
        has_recovery = False
        if hasattr(player.state, 'jumps_left'):
            has_recovery = player.state.jumps_left > 0
        if hasattr(player.state, 'recoveries_left'):
            has_recovery = has_recovery or player.state.recoveries_left > 0
        
        # STRONGER penalty if off-stage or near edge
        is_near_edge = player_min_dist < 3.0
        
        if is_offstage or (is_near_edge and not has_recovery):
            velocity_penalty = abs(player.body.velocity.y) / max_safe_velocity - 1.0
            # STRONGER penalty
            return clip_reward(-velocity_penalty * env.dt * 1.0)
    
    return 0.0

def survival_bonus(env: WarehouseBrawl) -> float:
    """
    Small bonus for staying alive and on-stage.
    Encourages not jumping off.
    """
    player: Player = env.objects["player"]
    
    edge_x = env.stage_width_tiles // 2
    edge_y = env.stage_height_tiles // 2
    
    # Check if on-stage
    dist_to_left = abs(player.body.position.x + edge_x)
    dist_to_right = abs(player.body.position.x - edge_x)
    dist_to_bottom = abs(player.body.position.y + edge_y)
    player_min_dist = min(dist_to_left, dist_to_right, dist_to_bottom)
    
    is_onstage = player_min_dist >= 2.0
    
    if is_onstage:
        return 0.01 * env.dt  # Small survival bonus
    
    return 0.0

def whichPlatform(x,y,positionMap):
    """
    1 -> ground1
    2 -> ground2
    3 -> moving platform
    0 -> anything else
    """
    
    for x_start,x_end,y_stage,stage in positionMap:
        if  x_start < x < x_end and y < y_stage:
            return stage
    return 0

def stage_control(env:WarehouseBrawl):
    """
    reward for being in the middle of platform when your enemy is on the edge
    """
    player = env.objects['player']
    opponent = env.objects['opponent']
    middle = env.objects['platform1'].body.position
    positionMap = [
        #(x_start,x_end,y)
        (-7.0,-2.0,2.80,1),
        (middle[0]-1,middle[0]+1,middle[1],2),
        (2.0,7.0,0.85,3)
    ]
    x_player = player.body.position.x
    x_opponent = opponent.body.position.x
    player_plat = whichPlatform(x_player,player.body.position.y,positionMap)
    opponent_plat = whichPlatform(x_opponent,opponent.body.position.y,positionMap)

    if player_plat and player_plat == opponent_plat:
        x_start,x_end,_,_ = positionMap[player_plat-1]
        player_dist = abs(x_player - (x_start + x_end)/2) 
        opponent_dist = abs(x_opponent - (x_start + x_end)/2)
        if player_dist <= opponent_dist:
            return player_dist / (x_end - x_start)
        else:
            return -opponent_dist / (x_end - x_start)
    return 0.0

'''
Add your dictionary of RewardFunctions here using RewTerms
'''
def gen_reward_manager(numCheckpoint:int):
    """
    0: 2_000_000,
    0.5: 4_000_000,
    1: 2_000_000,
    2: 6_000_000,
    3: 10_000_000,
    """
    checkPointDict = {
        0:{
            'damage_interaction_reward': 0.0,
            'advantage_state_reward': 0.0,
            'whiff_punishment_reward': 0.0,
            'weapon_stability_reward': 0.0,
            # 'proximity_to_opponent_reward': 2.0,
            'head_to_opponent': 10.0,
            'jumping_on_middle': 1.0,
            'bad_taunt': 0.0,
            'no_input_penalty': 0.0,
            'on_win_reward': 0.0,
            'on_knockout_reward': 8.0,
            'on_combo_reward': 0.0,
            'on_equip_reward': 0.0,
            'on_drop_reward': 0.0
        },
        1: {
            'damage_interaction_reward': 10.0,
            'advantage_state_reward': 0.0,
            'whiff_punishment_reward': 1.0,
            'weapon_stability_reward': 0.0,
            # 'proximity_to_opponent_reward': 2.0,
            'head_to_opponent': 6.0,
            'jumping_on_middle': 3.0,
            'bad_taunt': 0.05,
            'no_input_penalty': 2.0,
            'on_win_reward': 100,
            'on_knockout_reward': 60,
            'on_combo_reward': 0.0,
            'on_equip_reward': 0.0,
            'on_drop_reward': 0.0
        },
        2: {
            'damage_interaction_reward': 20.0,
            'advantage_state_reward': 0.0,
            'whiff_punishment_reward': 2.0,
            'weapon_stability_reward': 0.0,
            # 'proximity_to_opponent_reward': 2.0,
            'head_to_opponent': 6.0,
            'jumping_on_middle': 1.0,
            'no_input_penalty': 0.1,
            'on_win_reward': 100,
            'on_knockout_reward': 80,
            'on_combo_reward': 0.0,
            'on_equip_reward': 0.0,
            'on_drop_reward': 0.0
        }, 
        3: {
            'damage_interaction_reward': 15.0,
            'advantage_state_reward': 0.0,
            'whiff_punishment_reward': 4.0,
            'weapon_stability_reward': 5.0,
            # 'proximity_to_opponent_reward': 0.5,
            'head_to_opponent': 4.0,
            'jumping_on_middle': 0.5,
            'on_win_reward': 100,
            'on_knockout_reward': 80,
            'on_combo_reward': 0.0,
            'on_equip_reward': 0.1,
            'on_drop_reward': 0.08
        },
        4: {
            'damage_interaction_reward': 15.0,
            'advantage_state_reward': 2.0,
            'whiff_punishment_reward': 1.0,
            'weapon_stability_reward': 5.0,
            # 'proximity_to_opponent_reward': 0.5,
            'head_to_opponent': 4.0,
            'jumping_on_middle': 0.01,
            'on_win_reward': 100,
            'on_knockout_reward': 80,
            'on_combo_reward': 0.0,
            'on_equip_reward': 0.1,
            'on_drop_reward': 0.001
        }
    }

    currCheckPoint = checkPointDict[numCheckpoint]

    reward_functions = {
        'damage_interaction_reward': RewTerm(func=damage_interaction_reward, weight=currCheckPoint['damage_interaction_reward'], params={'mode': RewardMode.SYMMETRIC}), # checkpoint values from currCheckPoint
        
        # Advantage state reward - encourages pressure maintenance
        'advantage_state_reward': RewTerm(func=advantage_state_reward, weight=currCheckPoint.get('advantage_state_reward', 0.0)), #checkpoint 3
        
        # Whiff punishment
        'whiff_punishment_reward': RewTerm(func=whiff_punishment_reward, weight=currCheckPoint.get('whiff_punishment_reward', 0.0)), #checkpoint 1 #checkpoint 3: reduce to 1.0
        'weapon_stability_reward': RewTerm(func=weapon_stability_reward, weight=currCheckPoint.get('weapon_stability_reward', 0.0)), #checkpoint 2
        
        # encourage running at opponent especially for start to get more useful data
        # 'proximity_to_opponent_reward': RewTerm(func=proximity_to_opponent_reward, weight=currCheckPoint['proximity_to_opponent_reward']), 
        'head_to_opponent': RewTerm(func=head_to_opponent, weight=currCheckPoint['head_to_opponent']), 
        'jumping_on_middle': RewTerm(func=jumping_on_middle, weight=currCheckPoint['jumping_on_middle']), # checkpoint-driven weights
        'no_input_penalty': RewTerm(func=no_input_penalty, weight=currCheckPoint.get('no_input_penalty',0.0)),
        'bad_taunt': RewTerm(func=bad_taunt, weight=currCheckPoint.get('bad_taunt',0.0)),
        # Keep these disabled/zero
        # 'stage_control': RewTerm(func=stage_control, weight=currCheckPoint.get('stage_control', 0.0))
        # 'time_pressure_reward': RewTerm(func=time_pressure_reward, weight=currCheckPoint.get('time_pressure_reward', 0.0)),
        # Time pressure - prevent stalling
        # Contextual edge avoidance - allows edge-guarding (STRONG penalty to prevent jumping off)
        # 'edge_avoidance_reward': RewTerm(func=edge_avoidance_reward, weight=currCheckPoint.get('edge_avoidance_reward', 0.0), params={'danger_zone': 0.0}),
        # 'fall_velocity_penalty': RewTerm(func=fall_velocity_penalty, weight=currCheckPoint.get('fall_velocity_penalty', 0.0), params={'max_safe_velocity': 0.0}),
        # Survival bonus - encourage staying alive
        # 'survival_bonus': RewTerm(func=survival_bonus, weight=currCheckPoint.get('survival_bonus', 0.0)),
        # INCREASED: Retreat penalty - stop running away, engage!
        # 'retreat_penalty': RewTerm(func=retreat_penalty, weight=currCheckPoint.get('retreat_penalty', 0.0)),
        # 'danger_zone_reward': RewTerm(func=danger_zone_reward, weight=currCheckPoint.get('danger_zone_reward', 0.0)),
        # 'penalize_attack_reward': RewTerm(func=in_state_reward, weight=currCheckPoint.get('penalize_attack_reward', 0.0)),
        # 'holding_more_than_3_keys': RewTerm(func=holding_more_than_3_keys, weight=currCheckPoint.get('holding_more_than_3_keys', 0.0)),
    }
    signal_subscriptions = {
        # TERMINAL REWARDS
        'on_win_reward': ('win_signal', RewTerm(func=on_win_reward, weight=currCheckPoint.get('on_win_reward', 0.0))),
        'on_knockout_reward': ('knockout_signal', RewTerm(func=on_knockout_reward, weight=currCheckPoint.get('on_knockout_reward', 0.0))),
        
        # Combo per extra hit - prevents dwarfing KO
        'on_combo_reward': ('hit_during_stun', RewTerm(func=on_combo_reward, weight=currCheckPoint.get('on_combo_reward', 0.0))),
        
        # Weapon rewards
        'on_equip_reward': ('weapon_equip_signal', RewTerm(func=on_equip_reward, weight=currCheckPoint.get('on_equip_reward', 0.0))),
        'on_drop_reward': ('weapon_drop_signal', RewTerm(func=on_drop_reward, weight=currCheckPoint.get('on_drop_reward', 0.0))),
        
        # Edge-to-KO conversion bonus
        # 'edge_to_ko_bonus': ('knockout_signal', RewTerm(func=edge_to_ko_bonus, weight=0.0)), #never
    }
    return RewardManager(reward_functions, signal_subscriptions)

# -------------------------------------------------------------------------
# ----------------------------- MAIN FUNCTION -----------------------------
# -------------------------------------------------------------------------
'''
The main function runs training. You can change configurations such as the Agent type or opponent specifications here.
'''
if __name__ == '__main__':
    # Start FRESH with improved rewards (RECOMMENDED)
    # my_agent = CustomAgent(sb3_class=QRDQN, extractor=MLPExtractor)
    
    # OR: Continue from checkpoint (only if you want to try adapting old behavior):
    # my_agent = CustomAgent(sb3_class=PPO, file_path="checkpoints/AMIN_DUMB_AF_2/rl_model_501760_steps.zip", extractor=MLPWithLayerNorm)
    my_agent = CustomAgent(sb3_class=PPO, file_path="checkpoints/AMIN_DUMB_AF_6/rl_model_2000896_steps.zip", extractor=MLPWithLayerNorm)

    # Start here if you want to train from scratch. e.g:
    #my_agent = RecurrentPPOAgent()

    # Start here if you want to train from a specific timestep. e.g:
    #my_agent = RecurrentPPOAgent(file_path='checkpoints/experiment_3/rl_model_120006_steps.zip')

    # Self-play settings
    selfplay_handler = SelfPlayLatest(
        partial(type(my_agent)), # Agent class and its keyword arguments
                                 # type(my_agent) = Agent class
    )

    selfplay_random = SelfPlayRandom(
        partial(type(my_agent))
    )
    
    # Set save settings here:
    save_handler = SaveHandler(
        agent=my_agent, # Agent to save
        save_freq=200_000, # Save frequency - more frequent to catch good models
        max_saved=40, # Maximum number of saved models
        save_path='checkpoints', # Save path
        run_name='AMIN_DUMB_AF_7',  # Fresh training with aggressive chase + no whiffs
        mode=SaveHandlerMode.RESUME  # Start completely fresh
    )

    # Set opponent settings here:
    opponent_spec0 = {
                    'self_play': (0, selfplay_handler),
                    'self_play_random': (1,selfplay_random),
                    'constant_agent': (19, partial(ConstantAgent)),
                    'easy_hard_coded_bot': (0, partial(EasyHardCodedBot)),
                    'hard_hard_coded_bot': (0,partial(HardHardCodedBot))
                }
    
    opponent_spec1 = {
                    'self_play': (6, selfplay_handler),
                    'self_play_random': (0,selfplay_random),
                    'constant_agent': (14, partial(ConstantAgent)),
                    'easy_hard_coded_bot': (0, partial(EasyHardCodedBot)),
                    'hard_hard_coded_bot': (0,partial(HardHardCodedBot))
                }
    
    
    opponent_spec2 = {
        'self_play': (9, selfplay_handler),
        'self_play_random': (2,selfplay_random),
        'constant_agent': (0, partial(ConstantAgent)),
        'easy_hard_coded_bot': (9, partial(EasyHardCodedBot)),
        'hard_hard_coded_bot': (0,partial(HardHardCodedBot))
    }   

    opponent_spec3 = {
        'self_play': (10, selfplay_handler),
        'self_play_random': (2,selfplay_random),
        'constant_agent': (0, partial(ConstantAgent)),
        'easy_hard_coded_bot': (7, partial(EasyHardCodedBot)),
        'hard_hard_coded_bot': (1,partial(HardHardCodedBot))
    }

    opponent_spec4 = {
        'self_play': (12, selfplay_handler),
        'constant_agent': (0, partial(ConstantAgent)),
        'easy_hard_coded_bot': (0, partial(EasyHardCodedBot)),
        'hard_hard_coded_bot': (8,partial(HardHardCodedBot))
    }
    
    
    opponent_specDict = {
        0: opponent_spec0,
        1:opponent_spec1,
        2:opponent_spec2,
        3:opponent_spec3,
        4:opponent_spec4,
    }

    timeStepDict = {
        0: 2_000_000,
        1: 4_000_000,
        2: 6_000_000,
        3: 8_000_000,
        4: 8_000_000,
    }
    #overnight or rightnow
    state = 'overnight' 
    if state == 'overnight':
        for i in range(1,5):
            print(f"line1242: currently at checkpoint: {i}")
            # Reward manager
            reward_manager = gen_reward_manager(i)

            opponent_cfg = OpponentsCfg(opponents=opponent_specDict[i])

            train(my_agent,
                reward_manager,
                save_handler,
                opponent_cfg,
                CameraResolution.LOW,
                train_timesteps=int(timeStepDict[i]),  # Continue training (total 15M from start)
                train_logging=TrainLogging.PLOT
            )
    elif state == 'rightnow':
        RUN = 0
        reward_manager = gen_reward_manager(RUN)

        opponent_cfg = OpponentsCfg(opponents=opponent_specDict[RUN])
        train(my_agent,
            reward_manager,
            save_handler,
            opponent_cfg,
            CameraResolution.LOW,
            train_timesteps=timeStepDict[RUN],  # Continue training (total 15M from start)
            train_logging=TrainLogging.PLOT
        )
