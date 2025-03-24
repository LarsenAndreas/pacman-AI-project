from copy import deepcopy

import gymnasium as gym

import numpy as np
from numpy.typing import ArrayLike

from layout import Layout

from pacman import GameState, readCommand, Directions

def _get_obs(state: GameState, w: int, h: int) -> ArrayLike:
    matrix = -np.array(state.getWalls().data, dtype="float32")

    food = np.array(state.getFood().data, dtype="float32")
    matrix[:food.shape[0], :food.shape[1]] + food
        
    (i, j) = state.getPacmanPosition()
    matrix[i, j] = 2

    for (i, j) in state.getGhostPositions():
        matrix[int(i), int(j)] = -2

    return matrix.flatten()


    (i, j) = state.getPacmanPosition()
    #mask = np.array(state.getFood().data, dtype = "float32")[i - 1 : i + 2, j - 1 : j + 2] - np.array(state.getWalls().data, dtype="float32")[i - 1 : i + 2, j - 1 : j + 2]
    mask = np.array(state.getFood().data, dtype = "float32") - np.array(state.getWalls().data, dtype="float32")
    mask[i, j] = 2
    
    min_dist = np.inf
    min_dir = None
    for (idx, jdx) in state.getGhostPositions():
        mask[idx, jdx] = -2
        dist = np.abs(idx - i) + np.abs(jdx - j)

        if dist < min_dist:
            min_dist = dist
            min_dir = np.array([idx - i, jdx - j], dtype="float32")
            min_dir /= np.linalg.norm(min_dir) + 1e-16

    obs = np.hstack([mask.flatten(), min_dist, min_dir])

    #print(f"{mask=}, {min_dist=}, {min_dir=}")
    #print(f"{obs=}")
    return obs

    #ghost_mask = np.zeros((w, h))
    #for (i, j) in state.getGhostPositions():
    #    ghost_mask[int(i), int(j)] = 1

    #data = {
    #    "pacman": pacman_mask, #state.getPacmanPosition().data,
    #    "ghosts": ghost_mask, #state.getGhostPositions().data,
    #    "walls": state.getWalls().data,
    #    "food": state.getFood().data,
    #    #"capsules": state.getCapsules().data,
    #}

    #return np.hstack([np.array(mask, dtype="float32").flatten() for mask in data.values()])

def _get_direction(action: int) -> Directions:
    return {
            0: "East",
            1: "West",
            2: "North",
            3: "South",
            4: "Stop",
    }[action]

def _get_min_dist_to_food(state: GameState) -> int:
    (i, j) = state.getPacmanPosition()

    min_dist = np.inf
    for w in range(state.data.layout.width):
        for h in range(state.data.layout.height):
            if state.hasFood(w, h):
                min_dist = min(min_dist, np.abs(i - h) + np.abs(j + w))

    return min_dist

def _get_min_dist_to_ghost(state: GameState) -> int:
    (i, j) = state.getPacmanPosition()

    min_dist = np.inf
    for (idx, jdx) in state.getGhostPositions():
        min_dist = min(min_dist, np.abs(i - idx) + np.abs(j - jdx))
    
    return min_dist


class PacmanEnv(gym.Env):

    def __init__(self, initial_state: GameState = None, layout: Layout = None, max_steps: int = 1000):
        self.initial_state = initial_state if initial_state is not None else GameState()
        self.max_steps = max_steps

        self.action_space = gym.spaces.Discrete(5)
        self._h, self._w = layout.height, layout.width
        self._num_ghosts = ...
        self._num_food = ...
        self._num_capsules = ...
        #self.observation_space = gym.spaces.MultiBinary()
        self.state = None
        self.steps_beyond_terminated = None
        self.steps = 0

    def set_initial_state(self, layout: Layout):
        self.initial_state = GameState.initialize(layout, layout.getNumGhosts)

    def _get_info(self) -> dict:
        return {}

    def step(self, action):
        direction = _get_direction(action)
        if direction not in self.state.getLegalActions():
            direction = "Stop"

        self.steps += 1

        #original_min_dist_to_food = _get_min_dist_to_food(self.state) 
        #original_min_dist_to_ghost = _get_min_dist_to_ghost(self.state)

        self.state = self.state.generatePacmanSuccessor(direction)
        observation = _get_obs(self.state, self._w, self._h)

        terminated = self.state.isLose() or self.state.isWin()

        #new_min_dist_to_food = _get_min_dist_to_food(self.state) 
        #new_min_dist_to_ghost = _get_min_dist_to_ghost(self.state)
        
        reward = self.state.data.scoreChange
        #reward = 10 * self.state.data.scoreChange if self.state.data.scoreChange > 0 or terminated else 3 * (original_min_dist_to_food - new_min_dist_to_food) + (new_min_dist_to_ghost - original_min_dist_to_ghost) # This is the change in score from the action
        # reward = state.data.score # This is the total score after the action

        truncated = self.steps >= self.max_steps
        if truncated:
            print(f"Food left: {self.state.getNumFood()}")
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
    ):
        super().reset()
        self.steps = 0
        self.state = deepcopy(self.initial_state)

        return _get_obs(self.state, self._w, self._h), self._get_info()

gym.register(
    id="gymnasium_env/PacmanWorld-v0",
    entry_point=PacmanEnv,
)

if __name__ == "__main__":
    from pacman import ClassicGameRules, layout, loadAgent
    import graphicsDisplay

    rules = ClassicGameRules(30)
    
    args = dict(
        layout=layout.getLayout("knownSmall"),
        horizon=-1,
        # pacmanAgent = PlanningAgent(layout.getLayout("knownMedium")),
        pacmanAgent=loadAgent("PlanningAgent", False)(layout.getLayout("knownSmall")),
        ghostAgents=[
            loadAgent("RandomGhost", True)(1),
        ],
        display = graphicsDisplay.PacmanGraphics(1.0, frameTime=0.1),
        quiet=False,
        catchExceptions=False,
    )
    game = rules.newGame(**args)
    game.run()
