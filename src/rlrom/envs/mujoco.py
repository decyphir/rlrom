"""MuJoCo environments for rlrom, as an add-on.

rlrom reaches MuJoCo through gymnasium's own ``mujoco`` environments, so this
module adds no physics dependency to the core: it supplies the two environment
factories rlrom looks for in a config (``make_env_test`` / ``make_env_train``)
and resolves the environment id.

Why an id has to be resolved at all: the policies published on the Hugging Face
Hub by the Stable-Baselines3 project (``sb3/td3-Hopper-v3``, ...) are named after
the Gym-era ``-v3`` ids, and rlrom derives both the checkpoint filename and the
lookup from the config's ``env_name``. Gymnasium registers the ``-v4`` and ``-v5``
ports and rejects ``-v3`` with an error asking for ``mujoco_py``. The ``-v4``
environment is the port of ``-v3`` onto the ``mujoco`` bindings with identical
observation and action spaces, so the published policy transfers unchanged.

Config usage:

    env_name: Hopper-v3            # the name the policy was published under
    model_path: huggingface
    model_name: sb3/td3-Hopper-v3
    import_module: rlrom.envs.mujoco
    make_env_test: make_env_test

Optional keys, read from the top-level config:

    env_id: Hopper-v4              # instantiate a different id than env_name
    env_kwargs: {...}              # forwarded to gym.make (e.g. forward_reward_weight)

Both factories fall back to ``gym.make`` on the resolved id, so any gymnasium
environment works through them, not only MuJoCo ones.
"""

import gymnasium as gym

# The families gymnasium ships with a `mujoco` port whose observation and
# action spaces match the Gym-era version. Explicit rather than "any -v3",
# because plenty of -v2/-v3 ids have no port and rewriting them would turn a
# clear error into a confusing one.
_PORTED_FAMILIES = frozenset({
    "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup",
    "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher",
    "Swimmer", "Walker2d",
})
_LEGACY_VERSIONS = ("v2", "v3")
_PORT = "v4"


def resolve_env_id(env_name):
    """Map a policy-era environment name to an id gymnasium can instantiate.

    ``Hopper-v3`` -> ``Hopper-v4``. Ids that are not a legacy version of a ported
    family are returned unchanged, so ordinary gymnasium ids pass through.
    """
    base, _, version = env_name.rpartition("-")
    if base in _PORTED_FAMILIES and version in _LEGACY_VERSIONS:
        return f"{base}-{_PORT}"
    return env_name


def _env_id(cfg):
    env_id = cfg.get("env_id")
    if env_id:
        return env_id
    env_name = cfg.get("env_name")
    if not env_name:
        raise ValueError("cfg has no 'env_name' nor 'env_id'")
    return resolve_env_id(env_name)


def _make(cfg, render_mode):
    kwargs = dict(cfg.get("env_kwargs") or {})
    return gym.make(_env_id(cfg), render_mode=render_mode, **kwargs)


def make_env_test(cfg):
    """Factory for ``make_env_test`` in a config: instantiate with the test render mode."""
    cfg_test = cfg.get("cfg_test") or {}
    render_mode = cfg_test.get("render_mode")
    if render_mode == "None":
        render_mode = None
    return _make(cfg, render_mode)


def make_env_train(cfg):
    """Factory for ``make_env_train`` in a config: instantiate without rendering."""
    return _make(cfg, None)
