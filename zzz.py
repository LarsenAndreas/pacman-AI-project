import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity=10000, **kwargs):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PacmanLearner:
    def __init__(
        self, model: nn.modules.Module, env: gym.Env, EPS_START: float = 0.99, EPS_END: float = 0.05, EPS_DECAY: int = 1000, TAU: float = 0.005, GAMMA: float = 0.99, LR: float = 1e-4, BATCH_SIZE: int = 128, **kwargs
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.LR = LR
        self.BATCH_SIZE = BATCH_SIZE

        self.env = env

        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayMemory(**kwargs)
        self.steps_done = 0
        self.loss_func = nn.SmoothL1Loss()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state))
        else:
            return torch.tensor(env.action_space.sample(), device=self.device)

    def optimize_model(self):
        if len(self.memory) < params["BATCH_SIZE"]:
            return

        transitions = self.memory.sample(params["BATCH_SIZE"])
        # batch = Transition(*zip(*transitions))

        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminate_mask = []
        for i, t in enumerate(transitions):
            if t.next_state is not None:
                terminate_mask.append(i)
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
            next_state_values[terminate_mask] += torch.max(self.target_net(next_state_batch), dim=1).values * self.GAMMA
        
        loss = self.loss_func(next_state_values, torch.gather(self.policy_net(state_batch), 1, action_batch.unsqueeze(0)).squeeze())
        
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, num_episodes: int = 50):
        for i in tqdm(range(num_episodes), desc="Episodes"):
            state, info = env.reset()
            state = torch.tensor(state, device=self.device)
            for step in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, info = env.step(action.item())
                reward = torch.tensor(reward, device=self.device)

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, device=self.device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if terminated or truncated:
                    break


if __name__ == "__main__":

    from pacman import ClassicGameRules, layout, loadAgent
    import graphicsDisplay
    from pprint import pformat
    from pacmanGymnasiumWithVector import PacmanEnv
    from pacman import GameState

    params = dict(
        LAYOUT=layout.getLayout("knownSmall"),
        NUM_OBS=7 * 8,
        NUM_ACT=5,
        NUM_GHOSTS=1,
        BATCH_SIZE=8,  # BATCH_SIZE is the number of transitions sampled from the replay buffer
        GAMMA=0.99,  # GAMMA is the discount factor as mentioned in the previous section
        EPS_START=0.9,  # EPS_START is the starting value of epsilon
        EPS_END=0.05,  # EPS_END is the final value of epsilon
        EPS_DECAY=1000,  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        TAU=0.005,  # TAU is the update rate of the target network
        LR=1e-4,  # LR is the learning rate of the ``AdamW`` optimizer
    )

    print(f"Running with parameters:\n{pformat(params)}")

    model = nn.Sequential(
        nn.Linear(params["NUM_OBS"], 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, params["NUM_ACT"]),
    )

    gamestate = GameState()
    gamestate.initialize(params["LAYOUT"], params["NUM_GHOSTS"])
    env = PacmanEnv(gamestate, params["LAYOUT"])
    pacman = PacmanLearner(model, env, **params)
    pacman.train(50)

    # rules = ClassicGameRules(30)

    # args = dict(
    #     layout=layout.getLayout(params["LAYOUT"]),
    #     horizon=-1,
    #     # pacmanAgent = PlanningAgent(layout.getLayout("knownMedium")),
    #     pacmanAgent=loadAgent("PlanningAgent", False)(layout.getLayout(params["LAYOUT"])),
    #     ghostAgents=[loadAgent("RandomGhost", True)(i + 1) for i in range(params["NUM_GHOSTS"])],
    #     display=graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1),
    #     quiet=False,
    #     catchExceptions=False,
    # )
    # game = rules.newGame(**args)
    # game.run()
