from rlrom.testers import RLTester
from rlrom.trainers import RLTrainer
import rlrom.utils as utils
from bokeh.io import output_notebook
from bokeh.plotting import show
import minigrid
#from blockedunlockpickupenv
from rlrom.wrappers.reward_machine import RewardMachine
output_notebook()
from pprint import pprint
import functools
import os

file = './examples/minigrid/blocked_unlock_pickup/cfg_ppo_specs.yml'

train = False
if train:
    cfg = utils.load_cfg(file)
    print("file", file)
    T = RLTrainer(cfg)
    train = T.train()
else:
    Tt = RLTester(file)
    Tres = Tt.run_cfg_test()


