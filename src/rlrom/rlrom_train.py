import os
import sys
import argparse
from rlrom import rlrom_run, utils
from pprint import pprint

def main():
    # rlrom_test cfg_main.yml [--cfg_test cfg_test.yml]    [--cfg_specs cfg_specs.yml]    
    
    parser = argparse.ArgumentParser(description='Run a configuration file in YAML format for testing.')
    parser.add_argument('main_cfg', type=str, default='cfg_main.yml', help='Path to main configuration file in YAML format.')
    parser.add_argument('--cfg_specs', type=str, help='Override cfg_specs with content of a YAML file.')
    parser.add_argument('--cfg_train', type=str, help='Override cfg_train section in main with content of a YAML file.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level') 
    args = parser.parse_args()
    
    # Start with default configuration
    custom_cfg = dict()        
    
    # Load main config file
    if os.path.exists(args.main_cfg):
        custom_cfg = utils.load_cfg(args.main_cfg)
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

    # Override with specs config if specified
    if args.cfg_specs:        
        if os.path.exists(args.cfg_specs):
            custom_cfg['cfg_specs'] = args.cfg_specs
            print(f"Using training config from {args.cfg_specs}")
        else:
            print(f"Warning: Training config file {args.cfg_specs} not found.")

    if args.verbose>=1:
        pprint(custom_cfg)
        
    rlrom_run.main_train(custom_cfg)

if __name__ == "__main__":
    main()