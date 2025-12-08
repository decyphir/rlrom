from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from minigrid.core.roomgrid import RoomGrid
from minigrid.core.world_object import Ball
from gymnasium.envs.registration import register
from typing import Optional
from gymnasium.spaces import Box
from minigrid.core.world_object import Key
import numpy as np

# For manual control
import pygame
from minigrid.core.actions import Actions


class BlockedUnlockPickupBoxEnv(RoomGrid):
    """

    ## Description

    The agent has to pick up a box which is placed in another room, behind a
    locked door. The door is also blocked by a ball which the agent has to move
    before it can unlock the door. Hence, the agent has to learn to move the
    ball, pick up the key, open the door and pick up the object in the other
    room. This environment can be solved without relying on language.

    ## Mission Space

    "pick up the {color} {type}"

    {color} is the color of the box. Can be "red", "green", "blue", "purple",
    "yellow" or "grey".

    {type} is the type of the object. Can be "box" or "key".

    ## Action Space

    | Num | Name         | Action            |
    |-----|--------------|-------------------|
    | 0   | left         | Turn left         |
    | 1   | right        | Turn right        |
    | 2   | forward      | Move forward      |
    | 3   | pickup       | Pick up an object |
    | 4   | drop         | Unused            |
    | 5   | toggle       | Unused            |
    | 6   | done         | Unused            |

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

    1. The agent picks up the correct box.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-BlockedUnlockPickup-v0`

    """

    def __init__(self, max_steps: int | None = None, **kwargs):
        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )

        room_size = 6
        if max_steps is None:
            max_steps = 16 * room_size**2

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
        self.has_ball = False

    @staticmethod
    def _gen_mission(color: str, obj_type: str):
        return f"pick up the {color} {obj_type}"

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=False)
        self.door = door
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0] - 2, pos[1], Ball(color))
        # Add a key to unlock the door
        #self.add_object(0, 0, "key", door.color)

        # Compute agent position (one tile left of the door)
        ax, ay = pos[0] , pos[1]
        
        self.place_agent(0,0)
        self.agent_pos = (ax-1, ay)
        self.agent_dir = 0
        self.door = door

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

    
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
        
        
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Original obs is a dict: {"image": ..., "direction": ..., "mission": ...}
        image = obs["image"]  # shape: (7, 7, 3)
        
        # Flatten just the object type layer (if you don’t care about color or state)
        # The first channel is the object type (0=empty, 1=wall, 2=door, 3=key, etc.)
        flat_obs = image.flatten().astype(np.float32)  # image[:, :, 0].flatten().astype(np.float32) 
        self.has_ball = False
        self.carrying = Key(self.door.color)
        self.p_obs = flat_obs
        self.door.is_open = True
        return flat_obs, info
    
    def step(self, action):
        prev_obs = self._get_flat_obs(self.p_obs)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Flatten the observation
        obs = self._get_flat_obs(obs)
        self.p_obs = obs
        if  action == self.actions.pickup and self.carrying is not None and self.carrying.type == "box" and prev_obs[78] == 7: #prev_obs[78] == 7 and 
            reward = self._reward()
            terminated = True   
        return obs, reward, terminated, truncated, info
    
    def manual_control(self, obs):
        action=None
        for event in pygame.event.get():                        
            if event.type == pygame.KEYDOWN:                
                if event.key == pygame.K_LEFT:
                    action = Actions.left
                    break
                elif event.key == pygame.K_RIGHT:
                    action = Actions.right
                    break
                elif event.key == pygame.K_UP:
                    action = Actions.forward
                    break
                elif event.key == pygame.K_SPACE:
                    action = Actions.toggle       # open door / pickup / drop
                    break
                elif event.key == pygame.K_p or event.key == pygame.K_RSHIFT:
                    action = Actions.pickup
                    break
                elif event.key == pygame.K_d or event.key == pygame.K_RCTRL:
                    action = Actions.drop
                    break
                elif event.key == pygame.K_b:
                    action = Actions.done                    
                    break
        return action
   
register(
    id="MiniGrid-BlockedUnlockPickup-v4",
    entry_point=__name__ + ":BlockedUnlockPickupBoxEnv",
)