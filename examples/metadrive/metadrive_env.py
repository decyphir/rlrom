"""MetaDrive environments for rlrom.

MetaDrive drives a Panda3D engine: ``use_render`` decides whether a window is
opened, so rlrom's ``render_mode`` is mapped onto it here rather than passed as
an ``env_kwargs`` entry (MetaDrive has no ``render_mode`` key).

The simulator assets (~170 MB) live inside the installed package; the first
run pulls them (``python -m metadrive.pull_asset``).
"""
from metadrive.envs import MetaDriveEnv


def _make_env(cfg, render_mode=None):
    env_cfg = dict(cfg.get('cfg_env') or {})
    env_cfg['use_render'] = render_mode not in (None, 'None', False)
    return MetaDriveEnv(env_cfg)


def make_env_train(cfg):
    return _make_env(cfg, render_mode=None)


def make_env_test(cfg):
    return _make_env(cfg, render_mode=(cfg.get('cfg_test') or {}).get('render_mode'))
