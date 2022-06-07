from gamma_timeloop_env import GammaTimeloopEnv
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="energy", help='1st order fitness objective')
    parser.add_argument('--fitness2', type=str, default=None, help='2nd order fitness objective')
    parser.add_argument('--fitness3', type=str, default=None, help='3rd order fitness objective')
    parser.add_argument('--num_pops', type=int, default=5,help='number of populations')
    parser.add_argument('--epochs', type=int, default=5, help='number of generations/epochs')
    parser.add_argument('--config_path', type=str, default='./in_config',
                        help='Configuration path, should include arch.yaml and optionally problem.yaml')
    parser.add_argument('--report_dir', type=str, default='report', help='The report directory')
    parser.add_argument('--density', type=str, default='0.5,1,1', help='The density of Input, Output, Weight Tenor.')
    parser.add_argument('--save_chkpt', action='store_true', default=False)
    parser.add_argument('--use_sparse', action='store_true', default=False, help='Execute Map Space Exploration on sparse accelerator')
    parser.add_argument('--explore_bypass', action='store_true', default=False,
                        help='Enable it can add bypass buffer option in to the search space')
    opt = parser.parse_args()
    opt.num_gens = opt.epochs
    fitness = [opt.fitness1]
    fitness.append(opt.fitness2) if opt.fitness2 is not None else None
    fitness.append(opt.fitness3) if opt.fitness3 is not None else None
    print(f'Fitness Objective: {fitness}')
    density = opt.density.split(',')
    density = {'Inputs': float(density[0]), 'Outputs': float(density[1]), 'Weights': float(density[2])}
    gamma_timeloop = GammaTimeloopEnv(fitness_obj=fitness, report_dir=opt.report_dir, use_pool=True, use_IO=False,
                                      debug=False, in_config_dir=opt.config_path, density=density,
                                      save_chkpt=opt.save_chkpt, use_sparse=opt.use_sparse,
                                      explore_bypass=opt.explore_bypass)


    gamma_timeloop.run(dimension=None, num_pops=opt.num_pops, num_gens=opt.num_gens)
