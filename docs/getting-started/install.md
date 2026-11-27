# Install

## Requirements

- **Python 3.12 or newer.**
- A **C++ toolchain**. The monitoring engine, `stlrom`, is published on PyPI as
  source only, so pip compiles it during the install. On Debian/Ubuntu:

    ```bash
    sudo apt install build-essential
    ```

    CMake is provided by the Python build backend and does not have to be
    installed separately. Expect the first install to take a few minutes longer
    than a pure-Python package.

## From PyPI

```bash
pip install rlrom
```

This installs the `rlr` command and the library's runtime dependencies
(gymnasium, stable-baselines3, sb3-contrib, polars, matplotlib, …).

Verify the install:

```bash
rlr --help
```

## Optional extras

Some case studies need an environment that is too heavy to be a base
dependency. Install the matching extra:

| Extra | Adds | Used by |
|---|---|---|
| `notebooks` | `jupyter`, `plotly`, `kaleido`, `ipympl` | the `.ipynb` examples |
| `highway-env` | [highway-env](https://github.com/Farama-Foundation/HighwayEnv) | [Highway-env example](../examples/highway-env.md) |
| `mujoco` | MuJoCo and the Gymnasium MuJoCo environments | [MuJoCo example](../examples/mujoco.md) |
| `metadrive` | [MetaDrive](https://metadrive-simulator.readthedocs.io/) driving simulator | [MetaDrive example](../examples/metadrive.md) |
| `jax` | SBX (JAX) algorithms and the TensorFlow stack | algorithm coverage |
| `morl` | multi-objective algorithms (`morl-baselines`, `mo-gymnasium`) | multi-objective configs |
| `old_gym` | legacy `gym` 0.26 | old environments |

```bash
pip install "rlrom[mujoco]"
```

!!! warning "MetaDrive"

    The `metadrive` extra resolves the simulator from git, not from PyPI: the
    0.4.3 release pins `panda3d==1.10.13` and `requires-python <3.12`, while
    the same version on `main` accepts newer Python. Its ~170 MB of assets are a
    separate download that happens on first use.

## From a source checkout

The repository is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone git@gricad-gitlab.univ-grenoble-alpes.fr:donzeal/rlrom.git
cd rlrom
uv sync                      # base install, or:
uv sync --extra mujoco       # with an extra
uv run rlr --help
uv run pytest tests/         # 192 tests, no downloads needed
```

!!! tip

    `uv sync` installs into `.venv/` at the repository root, and every example
    command in this documentation then runs as `uv run rlr …` from that root.
