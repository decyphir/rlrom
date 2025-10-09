from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces import Box
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
import numpy as np
import pprint
from gymnasium.envs.registration import register

class UnlockEnvV1(RoomGrid):
    """
    ## Description

    The agent has to open a locked door. This environment can be solved without
    relying on language.

    ## Mission Space

    "open the door"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Unlock-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        room_size = 6
        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 8 * room_size**2

        super().__init__(
            mission_space=mission_space,
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=max_steps,
            **kwargs,
        )

        # Initialize observation_space after creating a sample obs
        sample_obs, _ = super().reset()
        flat_sample = self._get_flat_obs(sample_obs)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(flat_sample.size,),
            dtype=np.uint8,
        )

    @staticmethod
    def _gen_mission():
        return "open the door"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)
        self.place_agent(0, 0)
        self.door = door
        self.key = False
        self.mission = "open the door"

    def _get_flat_obs(self, obs):
        """
        Flatten the observation into a 1D array like FlatObsWrapper.
        obs: the original dict observation from RoomGrid
        """
        if isinstance(obs, dict) and "image" in obs:
            return obs["image"].ravel()
        else:
            # If your environment returns a raw array already
            return np.array(obs).ravel()

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward()
                terminated = True
        # Flatten the observation
        obs = self._get_flat_obs(obs)
        self.key = True if self.carrying and self.carrying.type == "key" else False
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        flat_obs = self._get_flat_obs(obs).astype(np.uint8)
        return flat_obs, info
    
register(
    id="MiniGrid-CustomUnlock-v1",
    entry_point=__name__ + ":UnlockEnvV1",
)