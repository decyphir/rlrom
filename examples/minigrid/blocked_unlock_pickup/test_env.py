import env_task3
import minigrid
import gymnasium as gym
import os
from rlrom.wrappers.reward_machine import RewardMachine
from rlrom.wrappers.stl_wrapper import STLWrapper
from rlrom.wrappers.stl_wrapper import stl_wrap_env
from rlrom import utils
from gymnasium.wrappers import FlattenObservation
import pygame
from minigrid.core.actions import Actions
import yaml


with open('./examples/minigrid/blocked_unlock_pickup/cfg_rm_task3.yml', 'r') as f:
    cfg_rm = yaml.load(f, Loader=yaml.SafeLoader)
    

file = './examples/minigrid/blocked_unlock_pickup/cfg_specs.yml'
#cfg = utils.load_cfg(file)

env = gym.make("MiniGrid-BlockedUnlockPickup-v4",render_mode="human")
env = FlattenObservation(env) 
# print("cfg", cfg)
# env = stl_wrap_env(env, cfg)
# env = RewardMachine(env, cfg_rm) 

obs, info = env.reset()
done = False

pygame.init()
running = True

while running:
#    for event in pygame.event.get():
#        if event.type == pygame.QUIT:
#            running = False
#            break
#
#        if event.type == pygame.KEYDOWN:
#            action = None
#
#            if event.key == pygame.K_LEFT:
#                action = Actions.left
#            elif event.key == pygame.K_RIGHT:
#                action = Actions.right
#            elif event.key == pygame.K_UP:
#                action = Actions.forward
#            elif event.key == pygame.K_SPACE:
#                action = Actions.toggle       # open door / pickup / drop
#            elif event.key == pygame.K_p or event.key == pygame.K_RSHIFT:
#                action = Actions.pickup
#            elif event.key == pygame.K_d or event.key == pygame.K_RCTRL:
#                action = Actions.drop
#            elif event.key == pygame.K_b:
#                action = Actions.done
#            elif event.key == pygame.K_ESCAPE:
#                running = False
#                break
    manual_control = env.get_wrapper_attr('manual_control')
    action = manual_control(obs)
    if action is not None:
        obs, reward, terminated, truncated, info = env.step(action)
        # if env.unwrapped.carrying is not None:
        #     print(env.unwrapped.carrying.type)
        env.render()
        
        if terminated or truncated:
            print("Episode done!")
            break
            obs, _ = env.reset()
            env.render()

env.close()