"""MetaDrive environments for rlrom.

MetaDrive drives a Panda3D engine: ``use_render`` decides whether a window is
opened, so rlrom's ``render_mode`` is mapped onto it here rather than passed as
an ``env_kwargs`` entry (MetaDrive has no ``render_mode`` key).

The simulator assets (~170 MB) live inside the installed package; the first
run pulls them (``python -m metadrive.pull_asset``).

Gymnasium compatibility: MetaDrive 0.4.3's ``BaseEnv.reset(seed=None)`` predates
gymnasium's ``options`` argument, while the SB3 vector-env / Monitor chain calls
``reset(seed=..., options=...)``. ``MetaDriveRLEnv`` accepts and drops ``options``
so the env can sit under the standard wrappers; ``step`` already returns the
5-tuple. ``vehicle`` and the rest of the MetaDrive API stay reachable, which is
how ``cfg_specs.aux_sig_names`` reads the driving features.
"""
import json
import os

from metadrive.envs import MetaDriveEnv
import numpy as np


# ---------------------------------------------------------------------------
# Driving features for the STL spec + shield (single source of truth).
# `cfg_specs.aux_sig_names` calls these through `self.import_module.<fn>(self.env)`
# and `metadrive_shield` calls the same functions, so the monitored signals and
# the shield's region test cannot drift apart.
#   heading_err: heading_diff - 0.5, in [-0.5, 0.5]  (0 = aligned with the lane)
#   lat_dev:     signed lateral offset from the current lane centre, divided by
#                the lane half-width -> about [-1, 1] inside the lane
#   v:           longitudinal speed / max speed -> about [0, 1]
# ---------------------------------------------------------------------------

def _lane(vehicle):
    try:
        return vehicle.navigation.current_lane
    except Exception:
        return None


def speed_norm(env):
    v = env.vehicle
    return float(v.speed / v.max_speed_m_s)


def heading_err(env):
    v = env.vehicle
    lane = _lane(v)
    if lane is None:
        return 0.0
    return float(v.heading_diff(lane) - 0.5)


def lat_dev(env):
    v = env.vehicle
    lane = _lane(v)
    if lane is None:
        return 2.0  # off the lane -> far outside the lateral band
    lat = lane.local_coordinates(v.position)[1]
    return float(lat / (lane.width / 2.0))


# PROVISIONAL Phase-2 safe action: a lane-keeping steer (corrects lateral and
# heading error) plus a target-speed controller (slower when off-centre / heading
# off). A constant piecewise table cannot steer a curve that bends both ways, so
# a state-dependent feedback is the default. Phase 3 (learn_shield.py) learns the
# paper's piecewise-constant table, available opt-in via action_shield.use_table.
# Steering sign verified on a straight lane: positive steer drives lat_dev
# negative, so a1 = +K_LAT*lat_dev + K_HEAD*heading_err pulls to the centre.
K_LAT = 1.2
K_HEAD = 1.6
K_V = 3.0
V0 = 0.95        # target speed (normalised) when centred
V_OFF = 0.35     # target-speed reduction at full off-centre / heading error
V_MIN = 0.2
DANGER_ACCEL_MAX = -0.3

# Learned piecewise-constant table (Phase 3, learn_shield.py). Loaded lazily; if
# absent, metadrive_shield falls back to the provisional feedback law above.
_TABLE = None


def _load_table():
    global _TABLE
    if _TABLE is None:
        path = os.path.join(os.path.dirname(__file__), 'shield_table.json')
        _TABLE = False
        if os.path.exists(path):
            with open(path) as f:
                _TABLE = json.load(f)
    return _TABLE or None


def _region_key(he, lat, e1, e2):
    # Keep in sync with learn_shield.region_of (signed regions).
    if abs(lat - he) >= e2:
        return 'danger'
    side = 'left' if lat >= 0 else 'right'
    return ('center_' if abs(he) < e1 else 'notcenter_') + side


def _bucket(v, edges=(0.2, 0.5, 0.8)):
    for i, edge in enumerate(edges):
        if v < edge:
            return i
    return len(edges)


def metadrive_shield(wrapper, action):
    """rlrom continuous-action shield hook (cfg_specs.action_shield.function).

    `wrapper` is the SpecWrapper; thresholds are read from the STL driver's
    `param` values (`e2`) so the spec and the shield share one definition.
    Returns a replacement action (np.ndarray) or None to leave it unchanged.
    """
    env = wrapper.env
    he = heading_err(env)
    lat = lat_dev(env)
    v = speed_norm(env)
    # Learned table is opt-in (action_shield.use_table): with this baseline its
    # mean-safe-action constants ignore steering and stall the car, so the
    # feedback law below is the default. See learn_shield.py for the table.
    if wrapper.action_shield.get('use_table'):
        table = _load_table()
        if table is not None:
            e1 = float(wrapper.stl_driver.get_param('e1'))
            e2 = float(wrapper.stl_driver.get_param('e2'))
            region = _region_key(he, lat, e1, e2)
            if region == 'danger':
                a = np.asarray(action, dtype=np.float32).copy()
                a[0] = float(table['danger']['steer_scale']) * a[0]
                a[1] = float(table['danger']['accel'])
                return a
            return np.array(table[region][_bucket(v)], dtype=np.float32)
    e2 = float(wrapper.stl_driver.get_param('e2'))
    # Safe action: lane-keeping steer (corrects lateral + heading error) and a
    # target-speed controller (slow down when off-centre / heading off).
    a1 = float(np.clip(K_LAT * lat + K_HEAD * he, -1.0, 1.0))
    off = max(abs(lat), abs(he) / 0.2)
    v_target = max(V_MIN, V0 - V_OFF * min(off, 1.0))
    a2 = float(np.clip(K_V * (v_target - v), -1.0, 1.0))
    if abs(lat - he) >= e2:      # R_danger: brake harder
        a2 = min(a2, DANGER_ACCEL_MAX)
    return np.array([a1, a2], dtype=np.float32)



class MetaDriveRLEnv(MetaDriveEnv):
    """``MetaDriveEnv`` with a gymnasium-compatible ``reset`` signature."""

    def reset(self, seed=None, options=None):
        # gymnasium wrappers pass ``seed=`` values outside MetaDrive's scenario
        # range (SB3's vector env seeds the workers with its own counters), and
        # MetaDrive asserts ``start_seed <= seed < start_seed + num_scenarios``.
        # Map any integer seed into the scenario range; leave ``None`` alone so
        # MetaDrive samples a random scenario itself.
        if seed is not None:
            num = int(self.config.get('num_scenarios', 1))
            start = int(self.config.get('start_seed', 0))
            seed = int(seed)
            if not (start <= seed < start + num):
                seed = start + seed % num
        return super().reset(seed=seed)


def _make_env(cfg, render_mode=None):
    env_cfg = dict(cfg.get('cfg_env') or {})
    env_cfg['use_render'] = render_mode not in (None, 'None', False)
    return MetaDriveRLEnv(env_cfg)


def make_env_train(cfg):
    return _make_env(cfg, render_mode=None)


def make_env_test(cfg):
    return _make_env(cfg, render_mode=(cfg.get('cfg_test') or {}).get('render_mode'))
