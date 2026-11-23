"""Static checks over the example configurations.

These guard the class of regression that made the MiniGrid reward machine inert:
a config that names a reward machine through a key the wrapper no longer reads
(``cfg_specs.cfg_rm``), or a machine whose transition conditions are not
declared as monitor formulas, so the FSM never leaves its initial state.

Nothing here imports an example module or builds an environment: the YAML is
parsed directly and ``.yml`` references are resolved by hand. That keeps the
test free of the optional extras (metadrive, mujoco, highway-env) and outside
``load_cfg``'s ``import_module`` side effect.
"""
from pathlib import Path

import pytest
from ruamel.yaml import YAML

yaml = YAML(typ='safe')

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
SKIP_DIRS = {'models', 'tb_logs'}
OBSOLETE_KEYS = ['cfg_rm']


def _example_configs():
    files = []
    for path in sorted(EXAMPLES.rglob('*.yml')):
        if SKIP_DIRS & set(path.relative_to(EXAMPLES).parts):
            continue
        files.append(path)
    return files


def _load(path):
    with open(path) as f:
        return yaml.load(f)


def _resolve(value, base):
    """Resolve a single ``.yml`` reference the way ``load_cfg`` does."""
    if isinstance(value, str) and value.endswith('.yml'):
        target = base / value
        if not target.exists():
            pytest.skip(f'{value} not next to {base}')  # symlinked/absent model cfg
        return _load(target)
    return value


def _reward_machines(specs, base):
    machines = specs.get('reward_machines') or []
    return [_resolve(m, base) for m in machines]


def _condition_names(machine):
    machine = _resolve(machine, Path('.')) if isinstance(machine, str) else machine
    fsm = machine.get('reward_machine') or machine.get('fsm') or {}
    return {t.get('condition') for t in fsm.get('transitions', [])}


def test_no_obsolete_cfg_specs_keys():
    offenders = []
    for path in _example_configs():
        cfg = _load(path)
        if not isinstance(cfg, dict):
            continue
        specs = cfg.get('cfg_specs')
        if isinstance(specs, dict):
            for key in OBSOLETE_KEYS:
                if key in specs:
                    offenders.append(f'{path.relative_to(EXAMPLES)}: cfg_specs.{key}')
    assert not offenders, (
        'reward machines must be declared under cfg_specs.reward_machines, not '
        'cfg_specs.cfg_rm (which the wrapper no longer reads):\n  '
        + '\n  '.join(offenders))


def test_reward_machines_have_a_machine_and_declared_labels():
    problems = []
    for path in _example_configs():
        cfg = _load(path)
        if not isinstance(cfg, dict):
            continue
        specs = cfg.get('cfg_specs')
        if not isinstance(specs, dict) or not specs.get('reward_machines'):
            continue
        base = path.parent
        labels = set((specs.get('monitor_formulas') or {}).keys())
        for entry in specs['reward_machines']:
            entry = _resolve(entry, base)
            fsm = entry.get('reward_machine') or entry.get('fsm')
            if not fsm:
                problems.append(f'{path.name}: a reward_machines entry has neither '
                                f'reward_machine nor fsm')
                continue
            fsm = _resolve(fsm, base)
            for condition in _condition_names({'fsm': fsm}):
                if condition not in labels:
                    problems.append(f'{path.name}: transition condition {condition!r} '
                                    f'is not declared in cfg_specs.monitor_formulas')
    assert not problems, '\n  ' + '\n  '.join(problems)


def test_every_example_config_parses():
    """A smoke check that the configs are YAML at all (catches bad merges)."""
    bad = []
    for path in _example_configs():
        try:
            _load(path)
        except Exception as exc:  # noqa: BLE001 - report the file, not the parser
            bad.append(f'{path.relative_to(EXAMPLES)}: {exc}')
    assert not bad, '\n  ' + '\n  '.join(bad)
