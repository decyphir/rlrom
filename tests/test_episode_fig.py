"""Tests for the cfg_test.plot pipeline (matplotlib + stlrom Signal plotting)."""
import os

import matplotlib
import matplotlib.figure
import numpy as np
import pytest

from rlrom.testers import RLTester

# Same specs as tests/test_smoke.py (no file loading, no model download)
STL_CFG = {
    "env_name": "CartPole-v1",
    "model_name": "random",
    "cfg_test": {"num_ep": 1, "num_steps": 50},
    "cfg_specs": {
        # Note: avoid bare negative literals (stlrom grammar limitation,
        # e.g. `x[t] < -0.2` fails to parse).
        "specs": "signal x, theta, env_reward\ncart_left := x[t] < 0.5 and x[t] > 0.1",
        "obs_names": {"x": "obs[0]", "theta": "obs[2]"},
        "reward_formulas": {
            "cart_left": {"online": True, "past_horizon": 0, "weight": 5.0},
        },
        "flatten_obs": False,
    },
}


@pytest.fixture(scope="module")
def tester():
    T = RLTester(dict(STL_CFG))
    T.run_cfg_test()
    return T


def test_get_signal_from_str_returns_signal(tester):
    T = tester
    T.init_env(render_mode=None)
    set_ep = T.env.get_wrapper_attr('set_episode_data')
    set_ep(T.test_results[-1]['episodes'][0])
    g = T.env.get_wrapper_attr('get_signal_from_str')
    import stlrom
    for name in ('x', 'theta', 'reward', 'env_reward', 'cart_left',
                 'sat(cart_left)', 'reward_formulas/cart_left:weighted'):
        S = g(name)
        assert isinstance(S, stlrom.Signal), name
        assert len(S.get_samples_list()) > 0, name
    T.env.close()


def test_get_fig_returns_matplotlib_figure(tester):
    fig, status = tester.get_fig("x, theta\nreward, env_reward\nsat(cart_left), cart_left")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert not status.startswith("Warning"), status
    assert len(fig.axes) == 3
    # each axis carries the requested signals
    for ax, expected in zip(fig.axes, [['x', 'theta'], ['reward', 'env_reward'],
                                        ['sat(cart_left)', 'cart_left']]):
        labels = ax.get_legend_handles_labels()[1]
        assert expected == labels, (labels, expected)
    # savefig works (headless)
    out = "/tmp/rlrom_test_get_fig.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    assert os.path.getsize(out) > 0


def test_get_fig_unknown_formula_is_warning(tester):
    fig, status = tester.get_fig("sat(does_not_exist)\nx")
    assert status.startswith("Warning")
    assert isinstance(fig, matplotlib.figure.Figure)


def test_cfg_test_plot_saves_figures(tmp_path):
    cfg = dict(STL_CFG)
    out_file = str(tmp_path / "plot0.png")
    cfg['cfg_test'] = {
        "num_ep": 1, "num_steps": 50, "render_mode": None,
        "plot": [
            {"title": "Observations", "out": out_file,
             "ax": ["x, theta", "reward"]},
        ],
    }
    T = RLTester(cfg)
    T.run_cfg_test()
    assert os.path.exists(out_file)
    assert os.path.getsize(out_file) > 0


def test_live_plot_headless_noop(tmp_path):
    """live: True items must not break headless runs: no window is shown,
    episodes still run to completion, static items still get displayed,
    and live items still honor out: (saved from the final episode)."""
    import matplotlib
    import matplotlib.figure

    out_file = str(tmp_path / "live.png")
    cfg = {
        "env_name": "CartPole-v1",
        "model_name": "random",
        "cfg_test": {"num_ep": 1, "num_steps": 30, "render_mode": None,
                     "plot": [
                         {"title": "Live obs", "live": True, "live_window": 10,
                          "out": out_file,
                          "ax": ["x, theta", "reward"]},
                         {"title": "Static obs", "ax": ["x", "reward"]},
                     ]},
        "cfg_specs": {
            "specs": "signal x, theta, env_reward\ncart_left := x[t] < 0.5 and x[t] > 0.1",
            "obs_names": {"x": "obs[0]", "theta": "obs[2]"},
            "reward_formulas": {"cart_left": {"online": True, "past_horizon": 0,
                                              "weight": 5.0}},
            "flatten_obs": False,
        },
    }
    import matplotlib.pyplot as plt
    plt.close("all")  # don't count figures left over from other tests
    before = set(plt.get_fignums())
    T = RLTester(cfg)
    Tres = T.run_cfg_test()
    assert Tres is not None
    # episode ran (random policy falls early, but still collects steps)
    assert len(Tres["episodes"][0]["rewards"]) > 0
    # live item's out: file was saved
    assert os.path.exists(out_file) and os.path.getsize(out_file) > 0
    # live figures were closed; only the static figure stays open
    assert len(set(plt.get_fignums()) - before) == 1
    plt.close("all")


def test_live_plot_items_helper():
    T = RLTester(dict(STL_CFG))
    assert T._live_plot_items() is None
    T.cfg['cfg_test']['plot'] = [
        {"title": "A", "live": True, "ax": ["x"]},
        {"title": "B", "ax": ["x"]},
    ]
    items = T._live_plot_items()
    assert len(items) == 1 and items[0]["title"] == "A"
    T.cfg['cfg_test']['plot'] = 'None'
    assert T._live_plot_items() is None


def test_live_line_update_sets_data():
    """live_line.update must actually plot.
    Regression: reset_plot() once returned None and left live_line.line as
    None ('NoneType' object has no attribute 'set_xdata'); it only surfaced
    on interactive backends, so force the update path here under Agg."""
    import matplotlib.pyplot as plt
    from rlrom.plots import LivePlot

    plt.close("all")
    T = RLTester(dict(STL_CFG))
    T.init_env()
    T.env.reset(seed=0)
    for _ in range(3):
        T.env.step(1)

    lp = LivePlot(T, [{"title": "L", "live": True, "ax": ["x, theta", "reward"]}],
                  live_window=4)
    lp.create()
    lp._interactive = True  # force the real update path even under Agg
    lp.step()
    lp.step()
    n_steps = len(T.env.get_wrapper_attr('get_time')())
    t_end = T.env.get_wrapper_attr('get_time')()[-1]
    assert len(lp.lines_by_fig) == 1
    assert len(lp.lines_by_fig[0]) == 3  # x, theta on ax0; reward on ax1
    for l in lp.lines_by_fig[0]:
        assert l.line is not None
        assert len(l.line.get_xdata()) == n_steps
        assert len(l.line.get_ydata()) == n_steps
        # env/reward signals are zero-order-hold -> rendered as steps
        assert l.line.get_drawstyle() == 'steps-post'
    # trailing window: x spans (t[-1]-live_window, t[-1])
    assert lp.figs[0].axes[0].get_xlim() == (max(0.0, t_end - 4), t_end)
    lp.close()

    # no live_window: x spans 0..end of episode (regression: it was stuck at
    # matplotlib's empty-data sentinel (-0.05, 0.05))
    lp2 = LivePlot(T, [{"title": "L2", "live": True, "ax": ["x"]}])
    lp2.create()
    lp2._interactive = True
    lp2.step()
    assert lp2.figs[0].axes[0].get_xlim() == (0.0, t_end)
    lp2.close()
    plt.close("all")
