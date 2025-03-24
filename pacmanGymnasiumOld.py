from copy import deepcopy

import gymnasium as gym

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

        # self._h, self._w = self.initial_state.layout.height, self.initial_state.layout.width
        # self._num_ghosts = ...
        # self._num_food = ...
        # self._num_capsules = ...
        self._h, self._w = 2, 3
        self._num_ghosts = 4
        self._num_food = 5
        self._num_capsules = 6
        self.observation_space = gym.spaces.Dict(
            {
                "pacman": gym.spaces.MultiDiscrete([self._h, self._w]),
                "ghosts": gym.spaces.Tuple([gym.spaces.MultiDiscrete([self._h, self._w]) for _ in range(self._num_ghosts)]),
                "walls": gym.spaces.MultiDiscrete([self._h, self._w]),
                "food": gym.spaces.Tuple([gym.spaces.MultiDiscrete([self._h, self._w]) for _ in range(self._num_food)]),
                "capsules": gym.spaces.Tuple([gym.spaces.MultiDiscrete([self._h, self._w]) for _ in range(self._num_capsules)]),
            }
        )
        self.state = None
        self.steps_beyond_terminated = None
        self.steps = 0

    def _get_obs(self, state: GameState) -> dict:
        return {
            "pacman": state.getPacmanPosition(),
            "ghosts": state.getGhostPositions(),
            "walls": state.getWalls().data,
            "food": state.getFood().data,
            "capsules": state.getCapsules().data,
        }

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
    from gymnasium.wrappers import FlattenObservation
    env = gym.make("gymnasium_env/PacmanWorld-v0")
    print(env.observation_space.shape)
    env = FlattenObservation(env)
    print(env.observation_space.shape)


    # args = readCommand("python pacman.py -p PlanningAgent -l knownLarge")
    print("Done!")
    ...
