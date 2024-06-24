from context import *

def test_pendul():
    env_name = 'Pendulum-v1'
    T = RLModelTester(env_name=env_name)
    print('Testing ', env_name)    
    T.test_seed(1)
    T.test_seed(2)
    T.specs = rlrom.cfg_envs[env_name]['specs']

    T.eval_spec('phi_goal')
    
    print(T.evals)

def test_highway():
    env_name = 'highway-v0'
    T = RLModelTester(env_name=env_name, render_mode='human')
    print('Testing ', env_name)    
    T.test_seed(1)
    

def test_supported_envs():
    for env in rlrom.supported_envs:
        print('Testing ', env)
        T = RLModelTester(env)
        T.test_seed(1)
    

if __name__ == '__main__':
    #test_pendul()
    test_highway()