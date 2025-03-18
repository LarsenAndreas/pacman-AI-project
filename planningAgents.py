from layout import Layout
from pacman import Directions
from game import Agent
import random
import game
import util
from pacman import GameState
import numpy as np 
from numpy.typing import ArrayLike

class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, **kwargs):
        print(layout)
        self.layout = layout
        self.offline_planning()


    def offline_planning(self):
        # Compute offline policy and/or value function
        # Time limit: 10 minutes

        pass


    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?

        print(state.getPacmanState())
        for ghost in state.getGhostStates():
            print(ghost)

        return Directions.STOP

    def constructInputMatrix(self, state: GameState) -> ArrayLike:
        """Constructs an input matrix based on the given state."""
        #matrix = np.zeros((N, M), dtype="float32")
        matrix = np.array(state.getWalls().data, dtype="float32")

        food = np.array(state.getFood().data, dtype="float32")
        matrix[:food.shape[0], :food.shape[1]] += 2 * food
        
        (i, j) = state.getPacmanPosition()
        matrix[i, j] = 3

        for (i, j) in state.getGhostPositions():
            matrix[int(i), int(j)] = 4

        return matrix