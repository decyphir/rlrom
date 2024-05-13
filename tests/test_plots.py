from context import *

lay= """
torque,
 cos_theta reward
reward
"""

env  = "Pendulum-v1"
model = "sb3/ppo-Pendulum-v1"


def test_parse_signal_spec():
    s_spec = 'torque'
    out = hf.parse_signal_spec(s_spec)
    print(out)

def test_plot():
    T = rlrom.RLModelTester(env_name=env)
    T.load_hf_model(model)
    T.test_seed(5)
    
    fig, status = T.get_fig(lay)
    print(status)
    fig.show()

    
if __name__ == '__main__':
    test_plot()
    