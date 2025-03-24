from copy import deepcopy

import gymnasium as gym

from typing import Tuple, Set

import numpy as np
from numpy.typing import ArrayLike
from itertools import count
from layout import Layout
from functools import reduce

from pacman import GameState, readCommand, Directions

def generate_neighbour_cells(state: GameState, i: int, j: int) -> Set[Tuple[int, int]]:
    return {(x, y) for x, y in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)] if not state.hasWall(x, y)}

def compute_length_modifier (state: GameState) -> Tuple[int, int]:
    """length modifier is: distance to closest food - distance to closest ghost, computed using BFS."""
    if np.sum(np.array(state.getFood().data, dtype="int64")) == 0:
        return 0 # We have won, so we dont care anymore!
    
    min_distance_to_food, min_distance_to_ghost = None, None

    fronteir = {state.getPacmanPosition()}
    visited = set()
    for d in count():
        # Check if there is a ghost and or food in the current fronteir.
        if (min_distance_to_food == None) and any([state.hasFood(i, j) for (i, j) in fronteir]):
            min_distance_to_food = d 

        if (min_distance_to_ghost == None) and len(set(map(lambda pos: (int(pos[0]), int(pos[1])), state.getGhostPositions())) & fronteir) != 0:
            min_distance_to_ghost = d

        if (min_distance_to_food != None) and (min_distance_to_ghost != None):
            break

        visited |= fronteir
        # Generate next fronteir
        fronteir = reduce(lambda x, y: x | y, [generate_neighbour_cells(state, i, j) for (i, j) in fronteir]) - visited

    return min_distance_to_food - min_distance_to_ghost

class PacmanEnv(gym.Env):

    def __init__(self, initial_state: GameState = None, layout: Layout = None, max_steps: int = 1000):
        self.initial_state = initial_state if initial_state is not None else GameState()
        self.max_steps = max_steps

        self.action_space = gym.spaces.Discrete(5)
        self._h, self._w = layout.height, layout.width
        self.state = None
        self.length_modifier = None
        self.steps = 0

    def _get_info(self) -> dict:
        return {"score": self.state.data.score}
    
    @staticmethod
    def _get_obs(state: GameState) -> ArrayLike:
        gameboard = np.array(state.getWalls().data, dtype="float32") # Set the walls
        gameboard[*state.getPacmanPosition()] = 2 # Set pacmans position
        gameboard[list(map(lambda x,y: (int(x), int(y)), *zip(*state.getGhostPositions())))] = 3 # Set the ghosts positions
        gameboard[state.getFood().data] = 4 # Set the ghosts positions

        # if len(capsules := state.getCapsules()) > 0:
        #     gameboard[tuple(zip(*capsules))] = 5

        return gameboard.flatten()
    
    @staticmethod
    def _get_direction(action: int) -> Directions:
        return {
                0: "East",
                1: "West",
                2: "North",
                3: "South",
                4: "Stop",
        }[action]

    def set_initial_state(self, layout: Layout):
        self.initial_state = GameState.initialize(layout, layout.getNumGhosts)

    def step(self, action):
        direction = self._get_direction(action)
        if direction not in self.state.getLegalActions():
            direction = "Stop"

        self.steps += 1

        self.state = self.state.generatePacmanSuccessor(direction) 
        observation = self._get_obs(self.state)

        length_modifier_change = compute_length_modifier(self.state) - self.length_modifier
        reward = self.state.data.scoreChange + length_modifier_change # This is the change in score from the action
        self.length_modifier += length_modifier_change

        terminated = self.state.isLose() or self.state.isWin()
        truncated = self.steps >= self.max_steps
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def reset(
        self,
    ):
        super().reset()
        self.steps = 0
        self.state = deepcopy(self.initial_state)
        self.length_modifier = compute_length_modifier(self.state)

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
