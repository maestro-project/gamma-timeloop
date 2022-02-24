'''
V2 use Timeloop representation and is not compatible to MAESTRO
'''
import numpy as np
import yaml
import os, sys
import copy
from functools import reduce
import random
from timeloop_env import TimeloopEnv
from multiprocessing.pool import Pool
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import shutil
from functools import cmp_to_key, partial
from collections import defaultdict, OrderedDict
from utils import timing, is_pareto
import math
import re
import glob
import pickle
from datetime import datetime

class GammaTimeloopEnv(object):
    def __init__(self, in_config_dir='./in_config', fitness_obj=['latency'], report_dir='./report',
                 use_pool=True, disable_pytimeloop=False, log_level=0, debug=False, init_random_tile=False, to_par_RS=False,
                 save_chkpt=False, use_sparse=False, density=None,  emulate_random=False):
        self.debug = bool(debug)
        self.fitness_obj = fitness_obj
        self.dim_note = ['N', 'K', 'C', 'Y', 'X', 'R', 'S']
        self.parallizable_dim_note = ['N', 'K', 'C', 'Y', 'X'] if bool(to_par_RS ) is False else self.dim_note
        self.parallizable_dim_note_set = set(self.parallizable_dim_note)
        self.len_dimension = len(self.dim_note)
        self.timeloop_configfile_path = f'/tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
        self.report_dir = report_dir
        self.use_sparse = use_sparse
        self.density = self.get_default_density() if density is None else density
        self.timeloop_env = TimeloopEnv(config_path=self.timeloop_configfile_path, in_config_dir=in_config_dir, debug=self.debug,
                                            use_sparse=self.use_sparse, density=self.density)
        self.num_buf_levels = self.timeloop_env.get_num_buffer_levels()
        print(f'Number of buffer levels: {self.num_buf_levels}')
        self.buf_spmap_cstr = self.timeloop_env.get_buffer_spmap_cstr()
        self.buffers_with_spmap = list(self.timeloop_env.get_buffers_with_spmap())
        self.use_pool = bool(use_pool)
        self.use_IO = bool(disable_pytimeloop)
        self.log_level = log_level
        self.init_random_tile = bool(init_random_tile)
        self.idealperf = {}
        self.save_chkpt = save_chkpt
        self.fitness_record = []
        self.all_fitness_record = []
        self.sol_record = []
        self.all_sol_record = []
        self.emulate_random = emulate_random


    def get_default_density(self):
        density = {'Weights': 1,
                   'Inputs': 1,
                   'Outputs': 1}
        return density

    def set_dimension(self, dimension=None):
        if dimension is None:
            self.dimension, self.dimension_dict = self.timeloop_env.get_problem_info()
        else:
            self.dimension = dimension
            self.dimension_dict = self.get_dimension_dict(dimension)
        self.dimension_factor = self.get_dimension_factors(self.dimension_dict)
        self.dimension_prime =  {key: self.get_prime_factors(self.dimension_dict[key]) for key in self.dim_note}
        self.idealperf['edp'], self.idealperf['latency'], self.idealperf['energy'] = self.timeloop_env.get_ideal_perf(self.dimension)
        self.fitness_record = []
        self.all_fitness_record = []
        self.sol_record = []
        self.all_sol_record = []

    def get_dimension_dict(self, dim_value):
        return {note: value for note, value in zip(self.dim_note, dim_value)}


    def get_prime_factors(self, n):
        primes = defaultdict(int)
        while n % 2 == 0:
            primes['2'] += 1
            n = n // 2
        for i in range(3,int(math.sqrt(n))+1,2):
            while n % i== 0:
                primes[f'{i}'] += 1
                n = n // i
        if n > 2:
            primes[f'{n}'] += 1
        return primes

    def get_factors(self, n):
        return list(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimension_dict):
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            factors = self.get_factors(value)
            dimension_factors[key] = factors
        return dimension_factors

    def update_tile_for_buf_cstr(self, indv, max_trial=None):
        is_valid_tile = self.timeloop_env.check_tile_fit_buffer(indv)
        trial_id = 0
        if max_trial is None:
            max_trial = float('Inf')
        while(is_valid_tile is False and trial_id < max_trial):
            indv = self.mutate_tiles(indv, alpha=1, beta=0)
            is_valid_tile = self.timeloop_env.check_tile_fit_buffer(indv)
            trial_id += 1
        return indv

    def mutate_tiles(self, indv, alpha=1.0, beta=0.0, num_mu_loc=2, gen=1, max_trial=100):
        if random.random() < alpha:
            pick_dim = np.random.choice(['N','K', 'C', 'Y', 'X', 'R', 'S'], p = [0.1, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05])
            if random.random()< beta:
                pick_level1, pick_level2 = np.random.choice(np.arange(1, self.num_buf_levels+1), 2, replace=False)
                for _ in range(num_mu_loc):
                    from_genome = indv[f'l{pick_level1}']['tile_size']
                    to_genome = indv[f'l{pick_level2}']['tile_size']
                    if len(from_genome[pick_dim]) == 0 and len(to_genome[pick_dim])==0:
                        break
                    elif len(from_genome[pick_dim]) == 0:
                        from_genome, to_genome = to_genome, from_genome
                    values = list(from_genome[pick_dim].keys())
                    pick_prime = np.random.choice(values)

                    from_genome[pick_dim][pick_prime] -= 1
                    if from_genome[pick_dim][pick_prime] == 0:
                        del from_genome[pick_dim][pick_prime]
                    to_genome[pick_dim][pick_prime] += 1
                    pick_dim = np.random.choice(['K', 'C', 'Y', 'X', 'R', 'S'], p = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
            else:
                # random mutation
                random_tiles = self.init_random_tile_size(dims=pick_dim)
                for level in indv.keys():
                    indv[level]['tile_size'][pick_dim] = random_tiles[level][pick_dim]
        return indv


    def mutate_bypass(self, indv, alpha=0.5, beta=1.0):
        if random.random() < alpha:
            pick_level = np.random.choice(np.arange(1, self.num_buf_levels))
            if random.random() < beta:
                pick_tensor = np.random.choice(['Weights', 'Inputs', 'Outputs'])
                indv[f'l{pick_level}']['bypass'][pick_tensor] = not indv[f'l{pick_level}']['bypass'][pick_tensor]
            else:
                indv[f'l{pick_level}']['bypass'] = {'Inputs':random.choice([True, False]), 'Weights': random.choice([True, False]), 'Outputs': random.choice([True, False])}
        return indv

    def mutate_par(self, indv, alpha=0.1, beta=1.0):
        if random.random() < alpha:
            pick_level = np.random.choice(self.buffers_with_spmap)
            if random.random() < beta:
                par_dims = indv[pick_level]['par_dims']
                if len(par_dims)>0 and random.random() < 0.5:
                    del_dim = np.random.choice(list(par_dims))
                    indv[pick_level]['par_dims'].remove(del_dim)
                else:
                    if len(par_dims) == len(self.parallizable_dim_note_set):
                        return indv
                    new_dim = np.random.choice(list(self.parallizable_dim_note_set - par_dims))
                    indv[pick_level]['par_dims'].add(new_dim)
            else:
                indv[pick_level]['par_dims'] = set(np.random.choice(self.parallizable_dim_note, random.randint(1, len(self.parallizable_dim_note)), replace=False))
        return indv

    def mutate_thread(self, indv, alpha=0.5, beta=0, gen=1):
        indv = self.mutate_order(indv)
        indv = self.mutate_par(indv)
        indv = self.mutate_bypass(indv)
        indv = self.mutate_level(indv)
        indv = self.mutate_tiles(indv)
        indv = self.update_tile_for_buf_cstr(indv, max_trial=None)
        indv = self.update_for_spmap_cstr(indv)
        return indv

    def mutate(self, pops, alpha, gen):
        for i in range(len(pops)):
            indv = pops[i]
            indv = self.mutate_order(indv, alpha=0.1)
            indv = self.mutate_par(indv, alpha=0.1)
            indv = self.mutate_bypass(indv, alpha=alpha)
            indv = self.mutate_level(indv, alpha=alpha)
            indv = self.mutate_tiles(indv, alpha=1,  gen=gen)
            indv = self.update_tile_for_buf_cstr(indv)
            indv = self.update_for_spmap_cstr(indv)
            pops[i] = indv
        return pops

    def mutate_order(self, indv, alpha=0.1, beta=1.0):
        if random.random() < alpha:
            pick_level = np.random.choice(np.arange(1, self.num_buf_levels+1))
            if random.random() < beta:
                loop_order = indv[f'l{pick_level}']['loop_order']
                loop_order = list(loop_order)
                idxs = random.sample(set(np.arange(0, self.len_dimension)), 2)
                loop_order[idxs[0]], loop_order[idxs[1]] = loop_order[idxs[1]], loop_order[idxs[0]]
                indv[f'l{pick_level}']['loop_order'] = ''.join(loop_order)
            else:
                indv[f'l{pick_level}']['loop_order'] = "".join(np.random.permutation(['N', 'K', 'C', 'Y', 'X', 'R', 'S']))
        return indv

    def mutate_level(self, indv, alpha=0.5):
        if random.random() < alpha:
            pick_level1, pick_level2 = np.random.choice(np.arange(1, self.num_buf_levels+1), 2, replace=False)
            indv[f'l{pick_level1}']['tile_size'], indv[f'l{pick_level2}']['tile_size'] = indv[f'l{pick_level2}']['tile_size'], indv[f'l{pick_level1}']['tile_size']
        return indv

    def crossover(self, pops, parents, num_injects=0, alpha=0.5):
        if len(parents) ==1:
            for idx in range(len(pops)):
                pops[idx] = copy.deepcopy(parents[0])
        else:
            for idx in range(0,len(pops)-num_injects,2):
                dad, mom = parents[random.randint(0, len(parents)-1)], parents[random.randint(0, len(parents)-1)]
                dad = copy.deepcopy(dad)
                mom = copy.deepcopy(mom)
                if random.random() < alpha:
                    length = min(len(dad), len(mom))
                    change_item = np.random.choice(['tile_size', 'loop_order', 'par_dims', 'bypass'])
                    pick_dim = np.random.choice(['K', 'C', 'Y', 'X'])
                    for l in range(1, length+1):
                        level = f'l{l}'
                        if change_item == 'tile_size':
                            dad[level][change_item][pick_dim], mom[level][change_item][pick_dim] = mom[level][change_item][pick_dim], dad[level][change_item][pick_dim]
                        else:
                            dad[level][change_item], mom[level][change_item] = mom[level][change_item], dad[level][change_item]
                pops[idx] = dad
                if idx + 1 < len(pops):
                    pops[idx+1] = mom
            for idx in range(len(pops)-num_injects,len(pops)):
                pops[idx] = self.init_random_indv()
        return pops

    def get_prod(self, dicts):
        ret_value = 1
        for k, v in dicts.items():
            ret_value *= ((int(k))**v)
        return ret_value

    def update_for_spmap_cstr(self, indv):
        to_append_to_toplevel = {k:defaultdict(int) for k in 'NKCYXRS'}
        for level in range(1, self.num_buf_levels+1):
            par_dims = indv[f'l{level}']['par_dims']
            tile_sizes = OrderedDict({key: indv[f'l{level}']['tile_size'][key] for key in par_dims})
            num_pars = [self.get_prod(v) for v in tile_sizes.values()]

            idx = 0
            for key, value in tile_sizes.items():
                if np.prod(num_pars) > self.buf_spmap_cstr[f'l{level}']:
                    for k, v in value.items():
                        to_append_to_toplevel[key][k] += v
                    indv[f'l{level}']['tile_size'][key] = defaultdict(int)
                    num_pars[idx] = 1
                    idx += 1
                else:
                    break
        for key, value in to_append_to_toplevel.items():
            if len(value)>0:
                cur_value = indv[f'l{self.num_buf_levels}']['tile_size'][key]
                for k, v in value.items():
                    cur_value[k]+= v
                indv[f'l{self.num_buf_levels}']['tile_size'][key] = cur_value
                indv[f'l{self.num_buf_levels}']['par_dims'].remove(key) if key in indv[f'l{self.num_buf_levels}']['par_dims'] else None
        return indv



    def init_random_tile_size(self, dims='NKCYXRS'):
        tile_hierachy = {f'l{level}': {k:defaultdict(int) for k in dims} for level in range(1, 1+self.num_buf_levels)}
        for key in dims:
            tile_budget = self.dimension_prime[key]
            for k, v in tile_budget.items():
                for _ in range(v):
                    level = random.randint(1, self.num_buf_levels)
                    tile_hierachy[f'l{level}'][key][k] +=1
        return tile_hierachy

    def init_random_single_level(self,buffer_level=1):
        genome = { 'tile_size': {key: defaultdict(int) for key in 'NKCYXRS'},
                   'loop_order': "".join(np.random.permutation(['N', 'K', 'C', 'Y', 'X', 'R', 'S'])),
                   'par_dims': set(np.random.choice(self.parallizable_dim_note, random.randint(1, len(self.parallizable_dim_note)), replace=False)) if f'l{buffer_level}' in self.buffers_with_spmap else set(),
                   'bypass':{'Inputs':random.choice([True, False]), 'Weights': random.choice([True, False]), 'Outputs': random.choice([True, False])},
                   }
        if buffer_level==self.num_buf_levels:
            genome = { 'tile_size':{key: defaultdict(int) for key in 'NKCYXRS'},
                       'loop_order': "".join(np.random.permutation(['N', 'K', 'C', 'Y', 'X', 'R', 'S'])),
                       'par_dims': set(),
                       'bypass':{'Inputs':False, 'Weights': False, 'Outputs': False}
                       }
        return genome

    def init_random_indv(self):
        indv = {f'l{i}': self.init_random_single_level(buffer_level=i) for i in range(1, 1+self.num_buf_levels)}
        tile_hierachy = self.init_random_tile_size()
        for i in range(1, 1+self.num_buf_levels):
            indv[f'l{i}']['tile_size'] = tile_hierachy[f'l{i}']
        return indv

    def init_single_level(self, buffer_level=1):
        genome = { 'tile_size': {key: defaultdict(int) for key in 'NKCYXRS'},
                   'loop_order': 'NKCYXRS',
                   # 'par_dims': {np.random.choice(self.parallizable_dim_note)} if f'l{buffer_level}' in self.buffers_with_spmap else set(),
                   'par_dims': {'K', 'X','Y'} if f'l{buffer_level}' in self.buffers_with_spmap else set(),
                   'bypass':{'Inputs':False, 'Weights': False, 'Outputs': False}
                   }
        if buffer_level==self.num_buf_levels:
            genome = { 'tile_size': {key: self.get_prime_factors(self.dimension_dict[key]) for key in 'NKCYXRS'},
                       'loop_order': 'NKCYXRS',
                       'par_dims': set(),
                       'bypass':{'Inputs':False, 'Weights': False, 'Outputs': False}
                       }
        return genome

    def init_indv(self,):
        indv = {f'l{i}': self.init_single_level(buffer_level=i) for i in range(1, 1+self.num_buf_levels)}
        if self.init_random_tile:
            tile_hierachy = self.init_random_tile_size()
            for i in range(1, 1+self.num_buf_levels):
                indv[f'l{i}']['tile_size'] = tile_hierachy[f'l{i}']
        return indv

    def init_pops(self, num_pops, random=False):
        if random:
            pops = [self.init_random_indv() for _ in range(num_pops//2)] + [self.init_random_indv() for _ in range(num_pops - num_pops//2)]
        else:
            pops = [self.init_indv() for _ in range(num_pops//2)] + [self.init_random_indv() for _ in range(num_pops - num_pops//2)]
        return pops, np.ones((num_pops, len(self.fitness_obj))) * np.NINF

    def get_random_indv(self, num_indvs=2):
        new_pops = []
        for i in range(num_indvs):
            indv = self.init_random_indv()
            new_pops.append(indv)
        return new_pops

    def sort_rank_func(self, cand1, cand2, delta=0.1):
        def helper(item1, item2, is_last=False):
            margin = abs(((item1+item2) /2) * delta) if not is_last else 0
            if margin == float('Inf'):
                margin = 0
            if item1 > item2 + margin:
                return 1
            elif item1 +margin < item2:
                return -1
            else:
                return 0
        fitness_len = len(cand1) - 1
        for i in range(fitness_len):
            ret = helper(cand1[i], cand2[i], is_last=(i==0 or i==fitness_len-1))
            if ret != 0:
                return ret
        return ret




    def select_parents(self, pops, fitness, num_parents, num_elites, num_pops, use_soft_margin=False, use_pareto=True):
        if use_pareto:
            parereto_masks, num_paretros = is_pareto(fitness, return_mask=True)
            fitness_list = [tuple([m]+list(ar)+[-i]) for i, (m, ar) in enumerate(zip(parereto_masks, fitness))]
        else:
            num_paretros = 1
            fitness_list = [tuple(list(ar)+[-i]) for i, ar in enumerate(fitness)]
        if not use_soft_margin:
            sort_rank_func = partial(self.sort_rank_func, delta=0)
        else:
            sort_rank_func = self.sort_rank_func
        fitness_list = sorted(fitness_list, key=cmp_to_key(sort_rank_func), reverse=True)
        idx = [int(-ar[-1]) for ar in fitness_list]
        new_pop = [pops[i] for i in idx][:num_pops]
        new_fitness = fitness[idx][:num_pops]
        num_parents = min(num_pops, max(num_paretros, num_parents))
        num_elites =  min(num_pops, max(num_paretros, num_elites))
        parents = copy.deepcopy(new_pop[:num_parents])
        elites = copy.deepcopy(new_pop[:num_elites])
        elites_fitness = copy.deepcopy(new_fitness[:num_elites])
        return new_pop, new_fitness, parents, elites, elites_fitness, num_parents, num_elites

    def thread_fun(self, args, do_mutate=True):
        indv, pool_idx = args
        if do_mutate:
            indv = self.mutate_thread(indv, alpha=self.alpha, beta=self.beta, gen=self.gen)
        fit = self.timeloop_env.run_timeloop( self.dimension, indv, pool_idx=pool_idx, use_IO=self.use_IO,
                                              fitness_obj=self.fitness_obj)
        if do_mutate:
            return indv, fit
        else:
            return fit

    def evaluate(self, pops, fitness, pool, num_pops=10):
        if not pool:
            for i, indv in enumerate(pops):
                ret = self.thread_fun((indv, 0))
                indv, fit = ret
                pops[i] = indv
                fitness[i] = fit
        else:
            while(1):
                try:
                    rets = list(pool.map(self.thread_fun, zip(pops, np.arange(len(pops)))))
                    for i, ret in enumerate(rets):
                        indv, fit = ret
                        pops[i] = indv
                        fitness[i] = fit
                    break
                except Exception as e:
                    if self.log_level>2:
                        print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(num_pops)
        return pool, fitness, pops

    def create_timeloop_report(self, indv, dir_path='./report'):
        fitness = self.thread_fun((indv, 0), do_mutate=False)
        os.makedirs(dir_path, exist_ok=True)
        if self.use_IO is False:
            self.timeloop_env.dump_timeloop_config_files(self.dimension, indv, dir_path)
        else:
            os.system(f'cp -d -r {os.path.join(self.timeloop_configfile_path, "pool-0")}/* {dir_path}')
        with open(os.path.join(dir_path,'Gamma-Timeloop.txt'), 'w') as fd:
            value = [f'{v:.5e}' for v in fitness]
            fd.write(f'Achieved Fitness: {value}\n')
            # fd.write(f'Achieved NormFitness: {self.get_norm_fitness(fitness)}')

    def get_norm_fitness(self, fit_value):
        norm_fitness = [abs(v/self.idealperf[self.fitness_obj[i]]) for i, v in enumerate(fit_value)]
        return [f"{n:.1f}" for n in norm_fitness]

    def run(self, dimension=None, num_pops=100, num_gens=100, elite_ratio=0.05, parents_ratio=0.5, inject_ratio=0.1):
        self.set_dimension(dimension)
        num_injects = max(1, int(num_pops*inject_ratio))
        num_parents = int(num_pops*parents_ratio)
        num_elites = max(1, int(num_pops*elite_ratio))
        pops, fitness = self.init_pops(num_pops)
        if self.use_pool:
            pool = ProcessPoolExecutor(num_pops)
            self.timeloop_env.create_pool_env(num_pools=num_pops, dimension=self.dimension, indv=pops[0], use_IO=self.use_IO)
        else:
            pool = None
            self.timeloop_env.create_pool_env(num_pools=1,  dimension=self.dimension, indv=pops[0],  use_IO=self.use_IO)

        for g in range(num_gens):
            if self.emulate_random:
                pops, fitness = self.init_pops(num_pops, random=True)
            if g == 0:
                pops, fitness, parents, elites, elites_fitness, num_parents, num_elites = self.select_parents(pops, fitness, num_parents, num_elites, num_pops)
            if g == 0:
                alpha = 1
            else:
                alpha = 0.5
            self.alpha = alpha
            self.beta = 0.5
            self.gen = g
            pops = self.crossover(pops, parents=parents, num_injects=num_injects, alpha=alpha)

            pool, fitness, pops = self.evaluate(pops, fitness, pool, num_pops)
            pops = elites + pops
            fitness = np.concatenate((elites_fitness, fitness), axis=0)

            pops, fitness, parents, elites, elites_fitness, num_parents, num_elites = self.select_parents(pops, fitness, num_parents, num_elites, num_pops)
            if g > 30:
                a=1
            best_idx = 0
            best_sol = pops[best_idx]
            print(f'[Gen{g}] fitness: {fitness[best_idx]}')
            self.record_chkpt(pops, fitness, best_idx, g, num_gens, num_pops)
        print(f'Achieved Fitness: {fitness[best_idx]}')
        self.create_timeloop_report(best_sol, dir_path=self.report_dir)
        self.clean_timeloop_output_files()


    def record_chkpt(self, pops, fitness, best_idx, gen, num_gens, num_pops):
        if self.save_chkpt:
            self.all_fitness_record.append(copy.deepcopy(fitness))
            self.all_sol_record.append(copy.deepcopy(pops))
            self.fitness_record.append(copy.deepcopy(fitness[best_idx]))
            self.sol_record.append(copy.deepcopy(pops[best_idx]))
            cur_gen = gen+1
            if cur_gen == num_gens:
                with open(os.path.join(self.report_dir, 'gamma_chkpt.plt'), 'wb') as fd:
                    chkpt = {
                             'fitness_record': self.fitness_record,
                             'all_fitness_record':self.all_fitness_record,
                             'all_sol_record':self.all_sol_record,
                             'sol_record':self.sol_record,
                             'best_fitness': self.fitness_record[-1],
                             # 'norm_best_fitness': self.get_norm_fitness(self.fitness_record[-1]),
                             'num_gens': num_gens,
                             'num_pops': num_pops,
                             'sampled_points': num_gens * num_pops}
                    pickle.dump(chkpt, fd)

    def clean_timeloop_output_files(self):
        shutil.rmtree(self.timeloop_configfile_path)
        out_prefix = "./timeloop-model."
        output_file_names = []
        output_file_names.append( "tmp-accelergy.yaml")
        output_file_names.append(out_prefix + "accelergy.log")
        output_file_names.extend(glob.glob("*accelergy.log"))
        output_file_names.extend(glob.glob("*tmp-accelergy.yaml"))
        output_file_names.append(out_prefix + ".log")
        output_file_names.append(out_prefix + "ART.yaml")
        output_file_names.append(out_prefix + "ART_summary.yaml")
        output_file_names.append(out_prefix + "ERT.yaml")
        output_file_names.append(out_prefix + "ERT_summary.yaml")
        output_file_names.append(out_prefix + "flattened_architecture.yaml")
        output_file_names.append(out_prefix + "map+stats.xml")
        output_file_names.append(out_prefix + "map.txt")
        output_file_names.append(out_prefix + "stats.txt")
        for f in output_file_names:
            if os.path.exists(f):
                os.remove(f)








