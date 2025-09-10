import gymnasium as gym
import highway_env  # noqa: F401

# Instantiate environment
def make_env_train(cfg):
    
    # env configuration    
    cfg_env = cfg['cfg_env']            
    if cfg_env.get('manual_control', False):
        print("WARNING: manual_control was set to True. I'm setting it back to False")
        cfg_env['manual_control'] = False    
    
    env = gym.make("highway-fast-v0")    
    env.unwrapped.configure(cfg_env)
                    
    return env


def make_env_test(cfg):
    
    # env configuration    
    cfg_env = cfg['cfg_env']                    
    cfg_test = cfg.get('cfg_test',{})
    
    if 'render_mode' in cfg_test:
        render_mode = cfg.get('render_mode', 'human')
    else: 
        render_mode='human'
    env = gym.make("highway-v0", render_mode=render_mode)    
    env.unwrapped.configure(cfg_env)
                    
    return env
