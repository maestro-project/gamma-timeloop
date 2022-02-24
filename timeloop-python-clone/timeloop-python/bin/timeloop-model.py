#!/usr/bin/env python3
from yaml import parse
from pytimeloop.app import Model
from pytimeloop import Config

import argparse
import glob
import logging
import os
import sys

parser = argparse.ArgumentParser(
    description='Run Timeloop given architecture, workload, and mapping.')
parser.add_argument('configs', nargs='+', help='Config files to run Timeloop.')
parser.add_argument('--output_dir', default='.',
                    help='Directory to dump output.')
parser.add_argument('--verbosity', type=int, default=1,
                    help='0 is only error; 1 adds warning; 2 is everyting.')


def load_configs(input_fnames):
    input_files = []
    for fname in input_fnames:
        input_files += glob.glob(fname)
    yaml_str = ''
    for fname in input_files:
        with open(fname, 'r') as f:
            yaml_str += f.read()
        yaml_str += '\n'
    config = Config.load_yaml(yaml_str)
    return config


if __name__ == '__main__':
    args = parser.parse_args()
    config = load_configs(args.configs)

    log_level = logging.INFO
    if args.verbosity == 0:
        log_level = logging.INFO
    elif args.verbosity == 1:
        log_level = logging.WARNING
    elif args.verbosity == 2:
        log_level = logging.INFO
    else:
        raise ValueError('Verbosity level unrecognized.')

    # Print logs from pytimeloop to console
    logger = logging.getLogger('pytimeloop')
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)

    out_dir = args.output_dir
    out_stats_fname = os.path.join(out_dir, 'timeloop-model.stats.txt')
    out_mapping_fname = os.path.join(out_dir, 'timeloop-model.map.txt')

    app = Model(config, out_dir, log_level=log_level)
    eval_stats = app.run()
    logger.info('Evaluation status: {}'.format(eval_stats.eval_status))
    logger.info('Pre-evaluation status: {}'.format(eval_stats.pre_eval_status))

    with open(out_stats_fname, 'w+') as f:
        f.write(eval_stats.pretty_print_stats())

    with open(out_mapping_fname, 'w+') as f:
        f.write(eval_stats.pretty_print_mapping())
