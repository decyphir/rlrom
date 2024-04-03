from context import *

env  = "Pendulum-v1"
model = "sb3/ppo-Pendulum-v1"

def test_human_render():
    T = rlrom.RLModelTester(env_name=env)
    T.load_hf_model(model)
    T.test_seed(5, render_mode='human')
    T.test_seed(5, render_mode='human')

def test_find_models():
    T = rlrom.RLModelTester(env_name=env)
    T.find_hf_models()

def test_get_dataframe():
    T = rlrom.RLModelTester(env_name=env)
    T.load_hf_model(model)
    T.test_seed(5)
    T.get_dataframe_from_trace()


if __name__ == '__main__':
    test_get_dataframe()