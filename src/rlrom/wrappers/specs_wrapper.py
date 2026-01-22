from rlrom.wrappers.stl_wrapper import stl_wrap_env
from rlrom.wrappers.reward_machine import RewardMachineWrapper
from gymnasium.wrappers import FlattenObservation


def wrap_env_specs(env,cfg):
    
    cfg_specs = cfg.get('cfg_specs', None)            
    if cfg_specs is not None:
        env = stl_wrap_env(env, cfg)
        
        cfg_rm = cfg_specs.get('cfg_rm', None)            
        if cfg_rm is not None:
            env = RewardMachineWrapper(env, cfg_rm)  
        else:
            env = FlattenObservation(env)          
    
    return env
