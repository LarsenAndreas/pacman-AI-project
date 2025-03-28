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
from tqdm import tqdm

from pacmanGymnasiumWithVector import PacmanEnv, _get_direction, _get_obs
import gymnasium as gym

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.01
LR = 1e-3
N_HIDDEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, N_HIDDEN)
        self.layer2 = nn.Linear(N_HIDDEN, N_HIDDEN)
        self.layer3 = nn.Linear(N_HIDDEN, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def optimize_model(policy_net: DQN, target_net: DQN, memory: ReplayMemory, optimizer: optim.AdamW):
    if len(memory) < BATCH_SIZE:
        return policy_net, target_net

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return policy_net, target_net

def select_action(policy_net: DQN, obs: torch.Tensor, env: PacmanEnv, steps_done):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(obs).max(1).indices.view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done

def plot_scores(episode_scores, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_scores, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

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
        env = PacmanEnv(self.initial_game_state, self.layout)

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        n_observations = self._h * self._w * 4
        
        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        
        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)
        
        steps_done = 0
        
        episode_scores = []
        start = time.time()
        pbar = tqdm(total=1.0, desc="Training", leave=True, position=0, colour="green")
        pbar_ep = tqdm(desc="Episode", leave=True, position=1)
        while (t_used := time.time() - start) < 60: # Total time budget of 10 minutes
            print(t_used)
            pbar.update(60 / t_used)
            # Initialize the environment and get its state
            obs, info = env.reset()
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            for t in count():
                pbar_ep.update()
                action, steps_done = select_action(policy_net, obs_tensor, env, steps_done)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                pbar_ep.set_postfix({"reward": reward.item(), "action": action.item(), "terminated": terminated, "truncated": truncated})

                if terminated:
                    next_obs_tensor = None
                else:
                    next_obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(obs_tensor, action, next_obs_tensor, reward)

                # Move to the next state
                #state = next_state

                # Perform one step of the optimization (on the policy network)
                policy_net, target_net = optimize_model(policy_net, target_net, memory, optimizer)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_net_state_dict)
                
                if done:
                    episode_scores.append(env.state.getScore())
                    plot_scores(episode_scores)
                    pbar_ep.reset()
                    break

        env.close()
        pbar.close()

        self.policy_net = policy_net
        print("Training complete!")

    def getAction(self, state : GameState) -> Directions:
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #obs_tensor = torch.Tensor(_get_obs(state, self._w, self._h))
            obs = _get_obs(state, self._w, self._h)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            output = self.policy_net(obs_tensor).max(1).indices.view(1, 1)
            action = _get_direction(output.item())
            if action not in state.getLegalActions():
                return random.choice(state.getLegalActions())

            return action


#    def constructInputMatrix(self, state: GameState) -> ArrayLike:
#        """Constructs an input matrix based on the given state."""
#        #matrix = np.zeros((N, M), dtype="float32")
#        matrix = np.array(state.getWalls().data, dtype="float32")
#
#        food = np.array(state.getFood().data, dtype="float32")
#        matrix[:food.shape[0], :food.shape[1]] += 2 * food
#        
#        (i, j) = state.getPacmanPosition()
#        matrix[i, j] = 3
#
#        for (i, j) in state.getGhostPositions():
#            matrix[int(i), int(j)] = 4
#
#        return matrix