import os
import sys
import argparse
from pprint import pprint

from rlrom.rlrom_plot import add_results_arguments, add_plot_options, main_plot, _merge_negative_values
from rlrom.rlrom_select_model import add_select_model_options, main_select_model


def main():
    # rlr <test|train|show|plot> cfg_main.yml [action-specific options, see --help]

    parser = argparse.ArgumentParser(description='Run a configuration file in YAML format: '
                                                 'test an agent, train, show the config, plot training results, '
                                                 'or select a trained model.')
    parser.add_argument('action', type=str, help='the action to perform: "test", "train", "show", "plot" or "select-model"')
    parser.add_argument('main_cfg', type=str, default='cfg_main.yml',
                        help='Path to the main configuration file in YAML format (default: cfg_main.yml).')
    parser.add_argument('--cfg-train', type=str, help='[test, train] Override cfg_train section in main with content of a YAML file.')
    parser.add_argument('--cfg-test', type=str, help='[test, train] Override cfg_test with content of a YAML file.')
    parser.add_argument('--cfg-specs', type=str, help='[test, train] Override cfg_specs with content of a YAML file.')
    parser.add_argument('--set-params', nargs='*', action=ParseKwargs, help='[test, train] list of key=value overriding keys existing in the cfg file.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('--expand', action='store_true',
                        help='[show] Recursively expand nested .yml/.stl/import_module references. '
                             'By default the raw top-level config is shown.')
    parser.add_argument('--subcfg', type=str, default=None, metavar='KEY',
                        help='[show] Display only a sub-configuration selected by KEY '
                             '(dot-separated path, or first occurrence found at any depth). '
                             'The --expand option applies.')
    # Options shared by the 'plot' and 'select-model' actions, then the
    # action-specific ones (must not be registered twice in this flat parser)
    add_results_arguments(parser, prefix='[plot, select-model] ')
    add_plot_options(parser, prefix='[plot] ')
    add_select_model_options(parser, prefix='[select-model] ')
    args = parser.parse_args(_merge_negative_values(sys.argv[1:]))
    
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
        
    # Note: for 'plot' and 'select-model', --verbose is handled by their
    # main functions (list the training folders)
    if args.verbose>=1 and args.action not in ('plot', 'select-model'):
        pprint(custom_cfg)
                    
    if args.action=='test':
        from rlrom.rlrom_test import main_test
        main_test(custom_cfg)    
    elif args.action=='train':    
        from rlrom.rlrom_train import main_train
        main_train(custom_cfg)
    elif args.action=='show':
        main_show(custom_cfg, subcfg=args.subcfg)
    elif args.action=='plot':
        sys.exit(main_plot(custom_cfg, trainings=args.trainings, metrics=args.metric,
                           envelope=args.envelope, output=args.output, title=args.title,
                           log_y=args.log_y, list_metrics=args.list_metrics, verbose=args.verbose))
    elif args.action=='select-model':
        sys.exit(main_select_model(custom_cfg, trainings=args.trainings, metric=args.metric,
                                   min_=args.min, step=args.step, top=args.show_top,
                                   show_selected=args.show_selected,
                                   list_metrics=args.list_metrics, verbose=args.verbose))
    else: 
        print(f'Unrecognized action {args.action}. It has to be "test", "train", "show", "plot" or "select-model".')

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