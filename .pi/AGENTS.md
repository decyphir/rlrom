# AGENTS.md — RLRom repository

## Session start (do this first)

1. Read `docs/CODEBASE_OVERVIEW.md`. Do **not** re-read the whole code base from scratch;
   the overview is the reference map (package structure, wrapper pipeline, config system,
   canonical example configs, known issues).
2. From the request, identify the relevant area and read **only those files**, e.g.:
   - STL monitoring / reward shaping / obs augmentation → `src/rlrom/wrappers/stl_wrapper.py`
   - Reward machines / FSMs → `src/rlrom/wrappers/reward_machine.py`,
     `examples/highway_env/cfg_specs_rmf.yml`, `examples/minigrid/blocked_unlock_pickup/`
   - Training flow / algorithms / checkpoints → `src/rlrom/trainers.py`, `src/rlrom/utils.py`
   - Testing / evaluation / plotting → `src/rlrom/testers.py`, `src/rlrom/plots.py`
   - Config format / CLI → `src/rlrom/rlrom_run.py`, `src/rlrom/utils.py` (`load_cfg`),
     `examples/cartpole/cfg0tr_ppo_specs.yml` (simplest complete example)
3. Keep `docs/CODEBASE_OVERVIEW.md` up to date: after any change that alters structure,
   APIs, config keys, or the known-issues list, update the corresponding section in the
   same change.

## Working style

- **Plan before acting**: for any non-trivial request, present a short plan (steps,
  files touched, how it will be verified) and wait for confirmation before implementing.
  Trivial one-liner fixes may skip this, but say so.
- **Incremental steps**: make small, reviewable changes; after each step report what was
  done and what remains.
- No small talk. Be concise and factual. State assumptions explicitly.
- **Safe by default**: do not delete data, models, or experiment artifacts
  (`examples/*/models/*__training*`, `tb_logs`, notebooks). Do not overwrite existing
  trained models or result files. Never edit `.venv/` or `uv.lock` unless asked.
- Prefer config/YAML changes over code changes for experiment variations.
- Do not run long trainings without explicit approval (even short runs can be slow);
  prefer small, targeted verifications.

## Environment & verification

- Python env: `uv`-managed, venv at repo root (`.venv`). Use `uv run ...` for commands
  (e.g. `uv run python ...`, `uv run rlr test <cfg.yml>`).
- **Run the test suite before committing logic changes:** `uv run pytest tests/`
  (fast, no models/downloads needed). See `docs/CODEBASE_OVERVIEW.md`, "Test suite".
- Quick smoke test with a real model: `uv run rlr test examples/cartpole/cfg0tr_ppo_specs.yml`
  (uses the locally trained `examples/cartpole/models/ppo_specs.zip`; override with
  `--set-params num_ep=1 num_steps=50 render_mode=None`).
- Do not run long trainings without explicit approval; prefer small, targeted verifications.
- Note: `load_cfg` chdir's to the config file's directory — run commands accordingly.
