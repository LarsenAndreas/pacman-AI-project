import math
import random
from collections import deque, namedtuple
from itertools import count
from pprint import pformat

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import game
import graphicsDisplay
from pacman import ClassicGameRules, Directions, GameState, layout, loadAgent
from pacmanGymnasiumWithVectorBetter import PacmanEnv

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):

    def __init__(self, capacity=10000, **kwargs):
        self.memory = deque([], maxlen=capacity)

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
        model: nn.modules.Module,
        env: gym.Env,
        EPS_START: float = 0.99,
        EPS_END: float = 0.05,
        EPS_DECAY: int = 1000,
        TAU: float = 0.005,
        GAMMA: float = 0.99,
        LR: float = 1e-4,
        BATCH_SIZE: int = 128,
        **kwargs,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}!")

        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.GAMMA = GAMMA
        self.LR = LR
        self.BATCH_SIZE = BATCH_SIZE
        print(self.EPS_START, self.EPS_END)

        self.env = env

        self.policy_net = model.to(self.device)
        self.target_net = model.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(**kwargs)
        self.steps_done = 0
        self.loss_func = nn.SmoothL1Loss()

        self.best_state = {"state_dict": None, "score": -1e16}

        # Plotting
        self.episode_reward = []
        self.episode_loss = []
        self.fig, self.axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        self.fig.suptitle("Training...")

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1.0 * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state))
        else:
            return torch.tensor(self.env.action_space.sample(), device=self.device)

    def optimize_model(self):
        if len(self.memory) < params["BATCH_SIZE"]:
            return None

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
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def train(self, num_episodes: int = 50):
        with tqdm(range(num_episodes), desc="Episodes") as pbar:
            for i in pbar:
                state, info = self.env.reset()
                state = torch.tensor(state, device=self.device)
                episode_loss = 0
                transitions = []
                for step in count():
                    action = self.select_action(state)
                    observation, reward, terminated, truncated, info = self.env.step(action.item())
                    reward = torch.tensor(reward, device=self.device)

                    if terminated:
                        next_state = None
                    else:
                        next_state = torch.tensor(observation, device=self.device)

                    # Store the transition in memory
                    self.memory.push(state, action, next_state, reward)
                    #transitions.append(Transition(state, action, next_state, reward))

                    # Move to the next state
                    state = next_state

                    # Perform one step of the optimization (on the policy network)
                    loss = self.optimize_model()
                    episode_loss += loss if loss is not None else 0

                    if terminated or truncated:
                        self.episode_reward.append(info["score"])
                        self.episode_loss.append(episode_loss / (step + 1))
                        self.plot_durations()
                        #self.best_state = {"state_dict": policy_net_state_dict, "score": info["score"]} if info["score"] > self.best_state["score"] else self.best_state
                        break
                
                # Include the discounted reward into the transition
                #discounted_reward = 0
                #for t in reversed(transitions):
                #    discounted_reward = (t.reward + self.GAMMA * discounted_reward)
                #    self.memory.push(t.state, t.action, t.next_state, discounted_reward)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key]= policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)

                self.target_net.load_state_dict(target_net_state_dict)
                pbar.set_postfix_str(f"loss: {loss}")

        self.target_net = self.policy_net
        #self.target_net.load_state_dict(self.best_state["state_dict"])

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


    def getAction(self, gamestate: GameState) -> Directions:
        observation = self.env._get_obs(gamestate)
        with torch.no_grad():
            q = self.policy_net(torch.tensor(observation).to(self.device))
            action = self.env._get_direction(int(torch.argmax(q).detach().cpu()))

            if action not in (legal_actions := gamestate.getLegalActions()):  # SHIELD
                action = "Stop"
                print(f"Got action {action} which is not in legal actions {legal_actions}")

            print(f"Action: {action}")
            return action


if __name__ == "__main__":

    params = dict(
        LAYOUT=layout.getLayout("knownSmall"),
        NUM_OBS=7 * 8,
        NUM_ACT=5,
        NUM_GHOSTS=1,
        BATCH_SIZE=512,  # BATCH_SIZE is the number of transitions sampled from the replay buffer
        GAMMA=0.95,  # GAMMA is the discount factor as mentioned in the previous section
        EPS_START=0.8,  # EPS_START is the starting value of epsilon
        EPS_END=0.05,  # EPS_END is the final value of epsilon
        EPS_DECAY=300,  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        TAU=0.005,  # TAU is the update rate of the target network
        LR=1e-4,  # LR is the learning rate of the ``AdamW`` optimizer
        N_HIDDEN = 256,  # Number of neurons in the hidden layers.
        N_EPISODES=10_000,  # Number of episodes performed during training.
    )

    print(f"Running with parameters:\n{pformat(params)}")

    model = nn.Sequential(
        nn.Linear(params["NUM_OBS"], params["N_HIDDEN"]),
        nn.ReLU(),
        nn.Linear(params["N_HIDDEN"], params["N_HIDDEN"]),
        nn.ReLU(),
        nn.Linear(params["N_HIDDEN"], params["N_HIDDEN"]),
        nn.ReLU(),
        nn.Linear(params["N_HIDDEN"], params["N_HIDDEN"]),
        nn.ReLU(),
        nn.Linear(params["N_HIDDEN"], params["NUM_ACT"])
    )

    gamestate = GameState()
    gamestate.initialize(params["LAYOUT"], params["NUM_GHOSTS"])
    env = PacmanEnv(gamestate, params["LAYOUT"])
    pacAgent = PacmanLearner(model, env, **params)
    pacAgent.train(params["N_EPISODES"])

    rules = ClassicGameRules(30)

    args = dict(
        layout=params["LAYOUT"],
        horizon=-1,
        # pacmanAgent = PlanningAgent(layout.getLayout("knownMedium")),
        pacmanAgent=pacAgent,
        ghostAgents=[loadAgent("RandomGhost", True)(i + 1) for i in range(params["NUM_GHOSTS"])],
        display=graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1),
        quiet=False,
        catchExceptions=False,
    )
    game = rules.newGame(**args)
    game.run()
