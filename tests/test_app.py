from context import *
from pprint  import pprint

import rlrom.app

env   = "Pendulum-v1"
model = "sb3/ppo-Pendulum-v1"

if __name__ == '__main__':
    dropdown_env.value = env
 #   status.value = "Environment " + env + " drooply loaded"
    [specs.value, plot_prompt.value, status.value]  = callback_env(env)
    rlrom.app.web_gui.launch()