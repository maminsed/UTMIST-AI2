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
import gdown
from typing import Optional
from environment.agent import Agent
from stable_baselines3 import PPO, A2C # Sample RL Algo imports
from sb3_contrib import RecurrentPPO # Importing an LSTM

# To run the sample TTNN model, you can uncomment the 2 lines below: 
# import ttnn
# from user.my_agent_tt import TTMLPPolicy


class SubmittedAgent(Agent):
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
        self.state_mapping = [
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

    def predict(self, obs):
        self.time += 1
        pos = self.obs_helper.get_section(obs, 'player_pos')
        player_grounded = self.obs_helper.get_section(obs,'player_grounded')
        
        opp_pos = self.obs_helper.get_section(obs, 'opponent_pos')
        opp_state = self.state_mapping[int(self.obs_helper.get_section(obs, 'opponent_state')[0])]

        moving_pos:list[int] = self.obs_helper.get_section(obs, 'player_moving_platform_pos')
        moving_vel:list[int] = self.obs_helper.get_section(obs, 'player_moving_platform_vel')

        moving_start,moving_end = [moving_pos[0]-1,moving_pos[1]],[moving_pos[1]-1,moving_pos[1]]

        if self.time % 300 == 0:
            print("grounded")
            print(player_grounded)

        action = self.act_helper.zeros()
        xdist = (opp_pos[0] - pos[0])
        ydist = (opp_pos[1] - pos[1])
        eudist = xdist**2 + ydist**2 #right is positive, left is negative

        if self.prev_pos is not None:
            self.down = (pos[1] - self.prev_pos[1]) > 0.00
        else:
            self.prev_pos = pos

        if pos[0] < -6.9:
            action = self.act_helper.press_keys(['d'], action)
        elif pos[0] > -1.9 and pos[0] < moving_start[0]-2:
            action = self.act_helper.press_keys(['a'], action)
        elif pos[0] >= moving_start[0]-2 and pos[0] < moving_start[0]+0.1:
            action = self.act_helper.press_keys(['d'],action)
        elif pos[0] >= moving_end[0]-0.1 and pos[0] < moving_end[0]+2:
            action = self.act_helper.press_keys(['a'],action)
        elif pos[0] > moving_end[0]+2 and pos[0] < 1.9:
            action = self.act_helper.press_keys(['d'], action)
        elif pos[0] > 6.9:
            action = self.act_helper.press_keys(['a'], action)

        # Jump if falling
        if (self.down and self.time % 11 == 0) or pos[1] > 5:
            action = self.act_helper.press_keys(['space'], action)

        # to face the player
        if self.whichPlatform(pos) and self.whichPlatform(opp_pos) == self.whichPlatform(pos):
            if xdist > 0:
                action = self.act_helper.press_keys(['d'], action)
            elif xdist < 0:
                action = self.act_helper.press_keys(['a'], action)

        # Attack if near
        if eudist < 4.0:
            action = self.act_helper.press_keys(['j'], action)
        if opp_state == 'KOState' and self.whichPlatform(pos):
            action = self.act_helper.press_keys(['g'],action)
        
        return action

    def whichPlatform(self, position:list[str,str])->int:
        if position[0] > -6.9 and position[0] < -1.9:
            return 1
        elif position[0] > 1.9 and position[0] < 6.9:
            return 2
        return 0

    