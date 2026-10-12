import os
import sys
import argparse
from pprint import pprint


def main():
    # rlr [test|train|show] cfg_main.yml [--cfg-train cfg_train.yml] [--cfg-test cfg_test.yml]

    parser = argparse.ArgumentParser(description='Run a configuration file in YAML format for testing or training.')
    parser.add_argument('action', type=str, help='action should be either "test", "train" or "show"')
    parser.add_argument('main_cfg', type=str, default='cfg_main.yml', help='Path to main configuration file in YAML format.')
    parser.add_argument('--cfg-train', type=str, help='Override cfg_train section in main with content of a YAML file.')
    parser.add_argument('--cfg-test', type=str, help='Override cfg_test with content of a YAML file.')
    parser.add_argument('--cfg-specs', type=str, help='Override cfg_specs with content of a YAML file.')
    parser.add_argument('--set-params', nargs='*', action=ParseKwargs, help='list of key=value overriding keys existing in the cfg file.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('--expand', action='store_true',
                        help='[show] Recursively expand nested .yml/.stl/import_module references. '
                             'By default the raw top-level config is shown.')
    parser.add_argument('--subcfg', type=str, default=None, metavar='KEY',
                        help='[show] Display only a sub-configuration selected by KEY '
                             '(dot-separated path, or first occurrence found at any depth). '
                             'The --expand option applies.')
    args = parser.parse_args()
    
    # Start with default configuration
    custom_cfg = dict()

    # Load main config file
    if os.path.exists(args.main_cfg):
        if args.action == 'show' and not args.expand:
            # 'show' by default prints only the top-level YAML: load it raw
            # without resolving nested .yml/.stl/import_module references.
            from rlrom.cfg_utils import yaml
            with open(args.main_cfg, 'r') as f:
                custom_cfg = yaml.load(f) or {}
        else:
            from rlrom.cfg_utils import load_cfg
            custom_cfg = load_cfg(args.main_cfg)
    else:
        print(f"Error: Config file {args.main_cfg} was not found.")
        sys.exit(1)

    # Override with train config if specified
    if args.cfg_train:        
        if os.path.exists(args.cfg_train):
            custom_cfg['cfg_train'] = args.cfg_train
            print(f"Using training config from {args.cfg_train}")
        else:
            print(f"Warning: Training config file {args.cfg_train} not found.")

    # Override with test config if specified
    if args.cfg_test:        
        if os.path.exists(args.cfg_test):
            custom_cfg['cfg_test'] = args.cfg_test
            print(f"Using training config from {args.cfg_test}")
        else:
            print(f"Warning: Training config file {args.cfg_test} not found.")

    # Override with specs config if specified
    if args.cfg_specs:        
        if os.path.exists(args.cfg_specs):
            custom_cfg['cfg_specs'] = args.cfg_specs
            print(f"Using training config from {args.cfg_specs}")
        else:
            print(f"Warning: Training config file {args.cfg_specs} not found.")

    if args.set_params is not None:
        from rlrom.cfg_utils import set_rec_cfg_field
        custom_cfg = set_rec_cfg_field(custom_cfg, **args.set_params)
        
    if args.verbose>=1:
        pprint(custom_cfg)
                    
    if args.action=='test':
        from rlrom.rlrom_test import main_test
        main_test(custom_cfg)    
    elif args.action=='train':    
        from rlrom.rlrom_train import main_train
        main_train(custom_cfg)
    elif args.action=='show':
        main_show(custom_cfg, subcfg=args.subcfg)
    else: 
        print(f'Unrecognized action {args.action}. It has to be "test", "train" or "show".')

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value



def main_show(custom_cfg, subcfg=None):
    if subcfg is not None:
        from rlrom.cfg_utils import find_subcfg
        try:
            custom_cfg = find_subcfg(custom_cfg, subcfg)
        except KeyError:
            print(f"Error: no sub-configuration found for '{subcfg}'.")
            sys.exit(1)
    pprint(custom_cfg)

if __name__ == "__main__":
    main()