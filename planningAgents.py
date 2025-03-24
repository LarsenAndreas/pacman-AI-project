from layout import Layout
from pacman import Directions
from game import Agent
import random
import game
import util
from pacman import GameState
from pacmanGymnasiumWithVector import *
import time 

import math
import random
from collections import deque, namedtuple
from itertools import count

from pacman import GameState
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Union

from pacmanGymnasiumWithVector import PacmanEnv, _get_direction, _get_obs
import gymnasium as gym

class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, **kwargs):
        print(layout)
        self.layout = layout

        self._h, self._w = layout.height, layout.width
        self.initial_game_state = GameState()
        self.initial_game_state.initialize(layout, layout.numGhosts)

        self.policy_net = None
        self.offline_planning()

    def offline_planning(self):
        """Train Deep Q learning agent on the layout."""
        
        print("Training complete!")

    def getAction(self, state : GameState) -> Directions:
        return self.policy_net.select_action(state)


