from copy import deepcopy

import gymnasium as gym

import numpy as np
from numpy.typing import ArrayLike

from planningAgents import PlanningAgent

from layout import Layout

from pacman import GameState, readCommand


class PacmanEnv(gym.Env):

    def __init__(self, initial_state: GameState = None, max_steps: int = 1000):
        self.initial_state = initial_state if initial_state is not None else GameState()
        self.max_steps = max_steps

        self.action_space = gym.spaces.Discrete(5)
        self._action_to_direction = {
            0: "East",
            1: "West",
            2: "North",
            3: "South",
            4: "Stop",
        }

        self._h, self._w = self.initial_state.layout.height, self.initial_state.layout.width
        self._num_ghosts = ...
        self._num_food = ...
        self._num_capsules = ...
        self.observation_space = gym.spaces.MultiBinary(self._h * self._w * 5)
        self.state = None
        self.steps_beyond_terminated = None
        self.steps = 0

    def set_initial_state(self, layout: Layout, numGhostAgents: int = 2):
        self.initial_state = GameState.initialize(layout, numGhostAgents)

    def _get_obs(self, state: GameState) -> ArrayLike:
        data = {
            "pacman": state.getPacmanPosition().data,
            "ghosts": state.getGhostPositions().data,
            "walls": state.getWalls().data,
            "food": state.getFood().data,
            "capsules": state.getCapsules().data,
        }

        return np.hstack((np.array(item, dtype="float32").flatten() for item in data))

    def _get_info(self) -> dict:
        return {}

    def step(self, action):
        direction = self._action_to_direction[action]
        steps += 1

        state = self.state.generateSuccessor(0, direction)
        observation = self._get_obs(state)

        # TODO: Determine which reward function to use
        reward = state.data.scoreChange  # This is the change in score from the action
        # reward = state.data.score # This is the total score after the action

        terminated = self.state.isLose() or self.state.isWin()
        truncated = self.steps >= self.max_steps
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
    ):
        super().reset()
        self.state = deepcopy(self.initial_state)

        return self._get_obs(self.state), self._get_info()

gym.register(
    id="gymnasium_env/PacmanWorld-v0",
    entry_point=PacmanEnv,
)

if __name__ == "__main__":
    from pacman import ClassicGameRules, layout, loadAgent
    import graphicsDisplay

    rules = ClassicGameRules(30)
    
    args = dict(
        layout=layout.getLayout("knownMedium"),
        horizon=-1,
        # pacmanAgent = PlanningAgent(layout.getLayout("knownMedium")),
        pacmanAgent=loadAgent("PlanningAgent", False)(layout.getLayout("knownMedium")),
        ghostAgents=[
            loadAgent("RandomGhost", True)(1),
        ],
        display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1),
        quiet=False,
        catchExceptions=False,
    )
    game = rules.newGame(**args)
    game.run()
