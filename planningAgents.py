from layout import Layout

import time 
import math
import random
from collections import deque, namedtuple
from itertools import count
from pprint import pformat

import gymnasium as gym
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import game
import graphicsDisplay
from pacman import ClassicGameRules, Directions, GameState, layout, loadAgent
from pacmanGymnasium import PacmanEnv, _get_obs, _get_direction

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

# Training hyper parameters
REPLAY_MEMORY_CAPACITY = 10_000
BATCH_SIZE = 512
LR = 5e-3
TAU = 0.005
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 100
N_HIDDEN = 64
N_ACT = 5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, **kwargs):
        self.memory = deque([], maxlen=REPLAY_MEMORY_CAPACITY)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    
    def append_transition(self, t: Transition):
        """Appends the transition to the end of the memory"""
        self.memory.append(t)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PacmanLearner(game.Agent):
    def __init__(
        self,
        env: PacmanEnv,
        **kwargs,
    ):
        
        self.env = env
        self.policy_net = nn.Sequential(
            nn.Linear(self.env._h * self.env._w, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_HIDDEN),
            nn.ReLU(),
            nn.Linear(N_HIDDEN, N_ACT)
        ).to(DEVICE)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        self.memory = ReplayMemory()
        self.steps_done = 0
        self.loss_func = nn.SmoothL1Loss()

        # Start by initially network to give close to zero on inputs similar to those found in the dataset
        # making it more succeptible to initial changes down the road.
        for _ in range(1_000):
            # Make a batch which looks like something that it would see from the environment
            batch = torch.rand((1024, self.env._w * self.env._h)).to(DEVICE) * 4 - 2
            loss = self.loss_func(torch.zeros([1024, N_ACT]).to(DEVICE), self.policy_net(batch))
            self.policy_net.zero_grad()
            loss.backward()

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            self.optimizer.step()

        print("Finished initial tuning")
        self.target_net = deepcopy(self.policy_net.to(DEVICE))

        # Plotting
        self.episode_reward = []
        self.episode_loss = []
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        self.fig.suptitle("Training...")

    def select_action(self, obs):
        """Picks the action using the policy network which models a ''Q-table'' hopefully atleast."""
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if np.random.uniform(0, 1) > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(obs))
        else:
            action = torch.tensor(self.env.action_space.sample(), device=DEVICE)
            return action

    def optimize_model(self):
        """Optimizes the model using the methodology outlined in the DQN algorithm algorithm"""
        if len(self.memory) < BATCH_SIZE:
            return None

        transitions = self.memory.sample(BATCH_SIZE)
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        not_terminated_mask = []
        for i, t in enumerate(transitions):
            if t.next_state is not None:
                not_terminated_mask.append(i)
                next_state_batch.append(t.next_state)
            state_batch.append(t.state)
            action_batch.append(t.action)
            reward_batch.append(t.reward)

        state_batch = torch.stack(state_batch, dim=0)
        action_batch = torch.stack(action_batch, dim=0)
        reward_batch = torch.stack(reward_batch, dim=0).to(torch.float32)
        next_state_batch = torch.stack(next_state_batch, dim=0)

        next_state_values = reward_batch
        with torch.no_grad():
            next_state_values[not_terminated_mask] += torch.max(self.target_net(next_state_batch), dim=1).values * GAMMA

        loss = self.loss_func(next_state_values, torch.gather(self.policy_net(state_batch), 1, action_batch.unsqueeze(1)).squeeze())

        #self.optimizer.zero_grad()
        self.policy_net.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def train(self, time_budget: float = 600):
        """Trains the network using the DQN algorithm using the total time budget."""
        print(f"Starting main training loop, with a time budget of {time_budget} seconds")

        start_time = time.time()
        while (time.time() - start_time < time_budget):
            state, info = self.env.reset()
            state = torch.tensor(state, device=DEVICE)
            episode_loss = 0
            for step in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor(reward, device=DEVICE)

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, device=DEVICE)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize_model()
                episode_loss += loss if loss is not None else 0

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                with torch.no_grad():
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

                    self.target_net.load_state_dict(target_net_state_dict)

                if terminated or truncated:
                    self.episode_reward.append(info["score"])
                    self.episode_loss.append(episode_loss / (step + 1))
                    self.plot_durations()
                    break
                
        input("Ready?")
        return self.policy_net

    def plot_durations(self, show_result=False):
        with torch.no_grad():
            rewards = torch.tensor(self.episode_reward, dtype=torch.float)
            losses = torch.tensor(self.episode_loss, dtype=torch.float)

            self.axs[0].cla()
            self.axs[1].cla()

            self.axs[1].set_xlabel("Episode")
            self.axs[0].set_ylabel("Score")
            self.axs[1].set_ylabel("Loss")

            self.axs[0].plot(rewards.numpy())
            self.axs[1].plot(losses.numpy())
            # Take 100 episode averages and plot them too
            if len(rewards) >= 100:
                means = rewards.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                self.axs[0].plot(means.numpy())

                means = losses.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                self.axs[1].plot(means.numpy())
            plt.pause(0.001)  # pause a bit so that plots are updated

class PlanningAgent(game.Agent):

    def __init__(self, layout : Layout, **kwargs):
        self.layout = layout
        self.initial_game_state = GameState()
        self.initial_game_state.initialize(layout, layout.numGhosts)

        self.offline_planning()

    def offline_planning(self):
        """Train Deep Q learning agent on the layout."""
        env = PacmanEnv(self.initial_game_state, self.layout, max_steps = self.layout.width * self.layout.height * 2) # NOTE: Max Steps gets set dynamically.
        learner = PacmanLearner(env)
        self.policy_net = learner.train(time_budget = 60 * 10)

    def getAction(self, state: GameState) -> Directions:
        """Simply pick an action using the neural network."""
        observation = _get_obs(state)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(observation).to(DEVICE))
            action = _get_direction(int(torch.argmax(q).detach().cpu()))

            if not action in (legal_actions := state.getLegalActions()): # Basically works as a SHIELD
                print(f"Got action {action} which is not in legal actions {legal_actions}")
                action = random.choice(state.getLegalPacmanActions()) # TODO: Move away from ghost if posible.

            print(f"Action: {action}")
            return action 



