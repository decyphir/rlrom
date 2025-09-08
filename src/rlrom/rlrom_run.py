import os
import sys
import argparse

from rlrom.testers import RLTester
from rlrom import utils
from rlrom.trainers import RLTrainer
from pprint import pprint
import numpy as np


def main():
    # rlrom_run [test|train] cfg_main.yml [--cfg-train cfg_train.yml]    
    
    parser = argparse.ArgumentParser(description='Run a configuration file in YAML format for testing or training.')
    parser.add_argument('action', type=str, help='action should be either "test" or "train"')
    parser.add_argument('main_cfg', type=str, default='cfg_main.yml', help='Path to main configuration file in YAML format.')
    parser.add_argument('--cfg_train', type=str, help='Override cfg_train section in main with content of a YAML file.')
    parser.add_argument('--cfg_test', type=str, help='Override cfg_test with content of a YAML file.')
    parser.add_argument('--cfg_specs', type=str, help='Override cfg_specs with content of a YAML file.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level') 
    parser.add_argument('--num-trainings',type=int, default=1, help='Number of repeats of training') #TODO 
    args = parser.parse_args()
    
    # Start with default configuration
    custom_cfg = dict()        
    
    # Load main config file
    if os.path.exists(args.main_cfg):
        # we change dir to where the cfg file is, so that the file can use relative path
        dirname, basename = os.path.split(args.main_cfg)
        if dirname != '':
            os.chdir(dirname)            
        custom_cfg = utils.load_cfg(basename)
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


    if args.verbose>=1:
        pprint(custom_cfg)
            
    if args.num_trainings:
        num_trainings= args.num_trainings
    else:
        num_trainings = 1
        
    if args.action=='test':
        tester = RLTester(custom_cfg)
        Tres= tester.run_cfg_test()
        tester.print_res_all_ep(Tres)
    elif args.action=='train':    
        trainer = RLTrainer(custom_cfg)
        trainer.train()
    else: 
        print(f'Unrecognized action {args.action}. Should be either "test" or "train"')

if __name__ == "__main__":
    main()