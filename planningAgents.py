from layout import Layout
from pacman import Directions
from game import Agent
import random
import game
import util
from pacman import GameState
from pacmanGymnasiumWithVector import *
from dqnTrainer import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, **kwargs):
        print(layout)
        self.layout = layout
        self.initial_game_state = GameState.initialize(layout, layout.numGhosts)
        self.offline_planning()

    def offline_planning(self):
        """Train Deep Q learning agent on the layout."""
        
        # Compute offline policy and/or value function
        # Time limit: 10 minutes
        BATCH_SIZE = 128
        GAMMA = 0.99
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        TAU = 0.005
        LR = 1e-4

        # TODO: We need to give it an initial state
        env = PacmanEnv(self.initial_game_state)

        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions).to(device)
        target_net = DQN(n_observations, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(10000)

        steps_done = 0

        episode_durations = []

        

        pass


    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?

        print(state.getPacmanState())
        for ghost in state.getGhostStates():
            print(ghost)

        action = random.choice(state.getLegalActions())

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