"""
Lightweight configuration helpers for the ``rlr`` command line.

These functions were extracted from :mod:`rlrom.utils` so that commands like
``rlr show`` can load and display a configuration without importing the whole
heavy dependency stack (stable-baselines3, morl-baselines, jax, torch, ...).

Only standard library (plus numpy and ruamel.yaml) are imported here on purpose.
"""

import os
import sys
import importlib

import numpy as np
from ruamel.yaml import YAML

yaml = YAML(typ='safe')
# Define a representer for NumPy arrays
yaml.representer.add_representer(np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist()))

# Define a representer for NumPy floats
yaml.representer.add_representer(np.float64, lambda dumper, data: dumper.represent_float(float(data)))


def set_rec_cfg_field(cfg, **kargs):
    def rec_set(cfg, key, value):
        for item in cfg:
            if item == key:
                cfg[item] = value
            elif isinstance(cfg[item], dict):
                cfg[item] = rec_set(cfg[item], key, value)
        return cfg
    for item in kargs:
        cfg = rec_set(cfg, item, kargs[item])
    return cfg


_MISSING = object()


def find_subcfg(cfg, key):
    """Return the sub-configuration selected by ``key``.

    ``key`` may be either a dot-separated path (e.g. ``cfg_train.a.b``) or a
    single key name, in which case the first occurrence found at any depth is
    returned. Raises ``KeyError`` if nothing matches.
    """
    def get_path(cfg, parts):
        node = cfg
        for part in parts:
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                raise KeyError(part)
        return node

    if isinstance(key, str) and '.' in key:
        return get_path(cfg, key.split('.'))

    # single key: search recursively for the first occurrence at any depth
    def search(node, name):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == name:
                    return v
                found = search(v, name)
                if found is not _MISSING:
                    return found
        return _MISSING

    found = search(cfg, key)
    if found is _MISSING:
        raise KeyError(key)
    return found


def load_cfg(cfg, verbose=1):
    exclude_load_file = ['res_file']

    def load_yml_file(field, path):
        """Import the YAML file at ``path`` and recursively process its content.

        Shared by dict values (subsections) and list items (e.g. the
        ``reward_machines`` section) so that both are imported the same way.
        """
        print('loading field [', field, '] from YAML file [', path, ']')
        with open(path, 'r') as f:
            content = yaml.load(f)
        if content is None:
            print('WARNING: YAML file [', path, '] is empty; field [', field, '] is set to None.')
        return recursive_load(content)

    def load_list(key, items):
        """Process the items of a list-valued section (e.g. ``reward_machines``).

        An item that is a single string pointing to an existing ``.yml`` file is
        imported the same way as a subsection. Dict and list items are processed
        recursively; any other item is left untouched.
        """
        for i, item in enumerate(items):
            if isinstance(item, str) and item.endswith('.yml'):
                if verbose >= 1 and os.path.exists(item):
                    items[i] = load_yml_file(f'{key}[{i}]', item)
                elif verbose >= 1:
                    print('WARNING: field [', f'{key}[{i}]',
                          '] points to YAML file [', item, '] which does not exist!')
            elif isinstance(item, dict):
                items[i] = recursive_load(item)
            elif isinstance(item, list):
                items[i] = load_list(f'{key}[{i}]', item)
        return items

    def recursive_load(cfg):
        if cfg is None:
            return None
        if isinstance(cfg, list):
            return load_list('', cfg)
        for key, value in cfg.items():
            #print('reading', key, 'with value', value)
            if isinstance(value, str) and value.endswith('.yml'):
                if verbose >= 1:
                    if key not in exclude_load_file:
                        cfg[key] = load_yml_file(key, value)
                    else:
                        cfg[key] = value
                else:
                    cfg[key] = value
                    print('WARNING: file', value, 'not found!')
            elif isinstance(value, str) and value.endswith('.stl'):
                if verbose >= 1:
                    print('loading field [', key, '] from STL file [', value, ']')
                with open(value, 'r') as F:
                    cfg[key] = F.read()
            elif key == 'import_module':
                to_import = cfg.get('import_module')
                if to_import is not None:
                    imported = importlib.import_module(to_import)
                    print(f'Imported module {to_import}')

            elif isinstance(value, dict):
                cfg[key] = recursive_load(value)
            elif isinstance(value, list):
                cfg[key] = load_list(key, value)

        return cfg

    if isinstance(cfg, str) and os.path.exists(cfg):
        # here cfg is a path and we found it so we can already load it at the top level
        cfg_file = cfg
        with open(cfg_file, 'r') as f:
            cfg = yaml.load(f)

        # now we should check if it has a path where we should be
        this_cfg_pathdir = cfg.get('this_cfg_pathdir', '')

        if this_cfg_pathdir == '':
            # no cfg_pathdir is explicitly specified. We try the folder of cfg file
            this_cfg_pathdir, _ = os.path.split(cfg_file)
            if this_cfg_pathdir == '':
                this_cfg_pathdir = '.'

        this_cfg_pathdir = os.path.abspath(this_cfg_pathdir)
        cfg['this_cfg_pathdir'] = this_cfg_pathdir
    elif not isinstance(cfg, dict):
        raise TypeError(f"Expected file name or dict.")
    else:
        this_cfg_pathdir = cfg.get('this_cfg_pathdir', '.')
    # now we have defined this_cfg_pathdir. Might be that it don't exist. We issue warning in that case
    if os.path.exists(this_cfg_pathdir):
        os.chdir(this_cfg_pathdir)
    else:
        print(f'WARNING: {this_cfg_pathdir} does not exist, assume current folder for working directory.')

    if '' not in sys.path:
        sys.path.append('')
    if '.' not in sys.path:
        sys.path.append('.')

    return recursive_load(cfg)
