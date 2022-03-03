from gamma_timeloop_env import GammaTimeloopEnv
import argparse
import os
import pandas as pd
import numpy as np
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="energy", help='1st order fitness objective')
    parser.add_argument('--fitness2', type=str, default=None, help='2nd order fitness objective')
    parser.add_argument('--fitness3', type=str, default=None, help='3rd order fitness objective')
    parser.add_argument('--num_pops', type=int, default=5,help='number of populations')
    parser.add_argument('--epochs', type=int, default=5, help='number of generations/epochs')
    parser.add_argument('--arch_path', type=str, default='./in_config/arch.yaml', help='Input architecture file path')
    parser.add_argument('--problem_path', type=str, default='./in_config/problem.yaml', help='Input problem file path')
    parser.add_argument('--sparse_path', type=str, default=None, help='Input sparse configuration file path')
    parser.add_argument('--report_dir', type=str, default='report', help='The report directory')
    parser.add_argument('--save_chkpt', action='store_true', default=False)
    parser.add_argument('--disable_pytimeloop', action='store_true', default=False)
    opt = parser.parse_args()
    opt.num_gens = opt.epochs
    fitness = [opt.fitness1]
    fitness.append(opt.fitness2) if opt.fitness2 is not None else None
    fitness.append(opt.fitness3) if opt.fitness3 is not None else None
    print(f'Fitness Objective: {fitness}')
    gamma_timeloop = GammaTimeloopEnv( fitness_obj=fitness, report_dir=opt.report_dir, in_arch_file=opt.arch_path,
                                       in_problem_file=opt.problem_path, in_sparse_file=opt.sparse_path,
                                        save_chkpt=opt.save_chkpt, disable_pytimeloop=opt.disable_pytimeloop)

    gamma_timeloop.run(dimension=None, num_pops=opt.num_pops, num_gens=opt.num_gens)
