from context import *


def test_supported_envs():
    for env in rlrom.supported_envs:
        print('Testing ', env)
        T = RLModelTester(env)
        T.test_seed(1)
    

if __name__ == '__main__':
    test_supported_envs()