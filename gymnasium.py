"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from copy import deepcopy
from pacman import GameState
import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class CartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    | Num | Action                 |
    |-----|------------------------|
    | 0   | Push cart to the left  |
    | 1   | Push cart to the right |

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ### Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ### Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ### Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ### Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ### Arguments

    ```
    gym.make('CartPole-v1')
    ```

    No additional arguments are currently supported.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, initial_state: GameState, render_mode: Optional[str] = None):
        self.initial_state = initial_state
        self.action_space = spaces.Discrete(5)

        self.state = None
        self.steps_beyond_terminated = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        move = None # TODO: Generate a pacman move from the action.
        food_count_before_move = self.state.getNumFood()
        self.state.generatePacmanSuccessor(move)
        did_pacman_eat = food_count_before_move - self.state.getNumFood() != 0

        terminated = self.state.isLose() or self.state.isWin()

        if not terminated and did_pacman_eat:
            reward = 5.0
        
        elif not terminated and (not did_pacman_eat):
            reward = -1.0

        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            if self.state.isLose():
                reward = -1_000
            else:
                reward = 1 # Maybe this should be changed aswell.
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        return self.constructInputMatrix(self.state), reward, terminated, False, {}

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

    def reset(
        self,
        state: GameState,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed) # Might not be needed, but was included in the original code.
        self.state = deepcopy(state)

        return self.constructInputMatrix(state).flatten(), {}