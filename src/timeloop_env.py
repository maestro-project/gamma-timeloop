import logging

import numpy as np
import yaml
import os, sys
import copy
from subprocess import Popen, PIPE, call
from parse_timeloop_output import parse_timeloop_stats
from pytimeloop.app import Model
from pytimeloop import ConfigDict
from utils import *
import re
class TimeloopEnv(object):
    def __init__(self, config_path='./out_config', in_config_dir= './in_config', debug=False, use_sparse=False, density=None):

        self.config_path = config_path
        self.use_sparse = use_sparse
        with open(os.path.join(in_config_dir, 'arch.yaml'), 'r') as fd:
            self.arch = yaml.load(fd, Loader = yaml.SafeLoader)
        with open(os.path.join(in_config_dir, 'problem.yaml'), 'r') as fd:
            self.problem = yaml.load(fd,Loader = yaml.SafeLoader)
        if self.use_sparse:
            with open(os.path.join(in_config_dir, 'sparse.yaml'), 'r') as fd:
                self.sparse = yaml.load(fd,Loader = yaml.SafeLoader)

        buffer_name_list, buffer_size_list, buffer_spmap_cstr, user_specified_spmaps, num_buffer_levels, num_pes = self.get_buffer_info()
        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.user_specified_spmaps = user_specified_spmaps
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels
        self.num_pes = num_pes
        self._executable = 'timeloop-model'
        self.debug = debug
        self.buf_energy_cost = self.get_default_buffer_energy_cost()
        self.density = density

    def get_default_buffer_energy_cost(self):
        buf_energy_cost = {'DRAM': 200,
                           'l2': 2.2,
                           'l1': 1.12,
                           'MAC': 1.0,
        }
        return buf_energy_cost

    def get_num_buffer_levels(self):
        return self.num_buffer_level

    def get_buffer_spmap_cstr(self):
        return self.buffer_spmap_cstr

    def get_buffers_with_spmap(self):
        return self.buffers_with_spmap


    def get_problem_info(self):
        dim_note = 'NKCYXRS'
        problem = copy.deepcopy(self.problem)
        dimension = []
        dimension_dicts = {}
        for key in dim_note:
            value = problem['problem']['instance'][self.get_timeloop_notation(key)]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def get_buffer_info(self):
        arch = copy.deepcopy(self.arch)
        num_instances = []
        buffer_name_list = []
        buffer_size_list = []
        num_buffer_levels = 0
        user_specified_spmaps = []
        arch = arch['architecture']
        num_pe = 0
        while 1:
            try:
                user_specified_spmap = False
                instances = 1
                arch = arch['subtree'][0]
                attrubutes = arch['local'][0]['attributes']
                depth = attrubutes['depth'] if 'depth' in attrubutes else float('Inf')
                word_bits = attrubutes['word-bits'] if 'word-bits' in attrubutes else 8
                width =  attrubutes['width'] if 'width' in attrubutes else 8
                buffer_size = depth * width / word_bits
                buffer_name = arch['local'][0]['name']
                macc = arch['local'][1]['name'] if len(arch['local'])>1 else None
                re_ret = re.search('.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                    user_specified_spmap = True
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                user_specified_spmaps.append(user_specified_spmap)
                num_buffer_levels += 1
            except:
                instances = int(macc.split('..')[1].split(']')[0]) + 1
                num_pe = instances
                num_instances.append(instances)
                break
        sp_cstr = []
        for i in range(len(num_instances)-1):
            allowed_sp_size = num_instances[i+1]//num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i+1] % num_instances[i] !=0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')
        user_specified_spmaps.pop(0)
        user_specified_spmaps.append(False)
        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
               {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
               set([f'l{level}' for level, user_sp in zip(np.arange(num_buffer_levels, 0, -1), user_specified_spmaps) if user_sp]), \
               num_buffer_levels, \
               num_pe

    def get_timeloop_notation(self, g):
        timeloop_dict = {'N': 'N', 'K': 'M', 'C': 'C', 'Y': 'P', 'X': 'Q', 'R': 'R', 'S': 'S'}
        return timeloop_dict[g]

    def get_gamma_notation(self, t):
        gamma_dict = {'N': 'N','M': 'K','C': 'C','P': 'Y','Q': 'X','R': 'R','S': 'S'}
        return gamma_dict[t]

    def get_dimension_dict(self, dim_value):
        dim_note = 'NKCYXRS'
        return {note: value for note, value in zip(dim_note, dim_value)}

    def init_tp_tile_size(self):
        series =  [f'{self.get_timeloop_notation(note)}={1}' for note in 'NKCYXRS']
        return ' '.join(series)

    def get_tp_tile_size(self, dim_value):
        series =  [f'{self.get_timeloop_notation(note)}={value}' for note, value in dim_value.items()]
        return ' '.join(series)

    def get_tp_sp_tile_size(self, dim_value, sp_dim, timeloop_notation=True):
        if timeloop_notation:
            temporal_series = [f'{self.get_timeloop_notation(note)}={value if note not in sp_dim else 1}' for note, value in dim_value.items()]
            spatial_series =  [f'{self.get_timeloop_notation(note)}={value if note in sp_dim else 1}' for note, value in dim_value.items()]
            return ' '.join(temporal_series), ' '.join(spatial_series)
        else:
            temporal_series = [dim_value[note] if note not in sp_dim else 1 for note in 'NKCYXRS']
            spatial_series =  [dim_value[note] if note in sp_dim else 1 for note in 'NKCYXRS']
            return np.array(temporal_series), np.array(spatial_series)

    def get_loop_order(self, loop_order):
        series = [self.get_timeloop_notation(g) for g in loop_order]
        return ''.join(series)

    def get_implicit_l3_tile_size(self, dim_value, l2_tile_size, l1_tile_size):
        l3_tile_size = [int(d/(l2*l1)) for d, l2, l1 in zip(dim_value, l2_tile_size, l1_tile_size)]
        l3_tile_size_mode = [d%(l2*l1) for d, l2, l1 in zip(dim_value, l2_tile_size, l1_tile_size)]
        if np.sum(l3_tile_size_mode) == 0:
            return l3_tile_size
        else:
            print('Tile size not divisible')
            return None


    def create_pool_env(self, num_pools, dimension, indv, use_IO=False):
        os.makedirs(self.config_path, exist_ok=True)
        if use_IO:
            arch_paths, problem_paths, map_paths, pool_paths = [], [], [], []
            for i in range(num_pools):
                pool_dir = os.path.join(self.config_path, f'pool-{i}')
                os.makedirs(pool_dir, exist_ok=True)
                pool_paths.append(pool_dir)
                arch_paths.append(os.path.abspath(os.path.join(pool_dir, 'arch.yaml')))
                problem_paths.append(os.path.abspath(os.path.join(pool_dir, 'problem.yaml')))
                map_paths.append(os.path.abspath(os.path.join(pool_dir, 'map.yaml')))
            self.arch_path, self.problem_path, self.map_path, self.pool_path =  arch_paths, problem_paths, map_paths, pool_paths
        else:
            arch, problem, map = self.get_configs(dimension, indv)
            cfg = {}
            cfg.update(arch)
            cfg.update(map)
            cfg.update(problem)
            if self.use_sparse:
                cfg.update({'sparse_optimizations': self.sparse})
            config = ConfigDict(cfg)
            with stdout_redirected():
                timeloop_app = Model(config, self.config_path)
            with open(os.path.join(self.config_path, 'timeloop-model.ART.yaml'), 'r') as fd:
                art = yaml.load(fd, Loader = yaml.SafeLoader)
            with open(os.path.join(self.config_path, 'timeloop-model.ERT.yaml'), 'r') as fd:
                ert = yaml.load(fd, Loader = yaml.SafeLoader)
            cfg.update(art)
            cfg.update(ert)
            self.art = art
            self.ert = ert
            self.shared_cfg = cfg

    def get_arch_configs(self, l2_size, l1_size, num_pes):
        arch = copy.deepcopy(self.arch)
        arch['architecture']['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] = l2_size
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['name']=f'RegisterFile[0..{num_pes}]'
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][0]['attributes']['depth'] = l1_size
        arch['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']=f'MACC[0..{num_pes}]'
        return arch

    def get_problem_configs(self, dimension):
        problem =  copy.deepcopy(self.problem)
        dimension_dict = self.get_dimension_dict(dimension)
        for key, value in dimension_dict.items():
            problem['problem']['instance'][self.get_timeloop_notation(key)] = value
        if self.use_sparse:
            problem['problem']['instance']['density'] = {}
            for key in ['Inputs', 'Weights', 'Outputs']:
                cur_density = self.density[key]
                if cur_density < 1:
                    problem['problem']['instance']['density'][key] = {}
                    problem['problem']['instance']['density'][key]['distribution'] = 'fixed-structured'
                    problem['problem']['instance']['density'][key]['density'] = cur_density
        return problem

    def get_prod(self, dicts):
        ret_value = 1
        for k, v in dicts.items():
            ret_value *= ((int(k))**v)
        return ret_value

    def get_bypass(self, bypass):
        to_pass = [k  for k, v in bypass.items() if v]
        to_keep = [k  for k, v in bypass.items() if not v]
        return to_pass, to_keep

    def get_input_weight_output_tile(self, tiles):
        N, K, C, Y, X, R, S = tiles
        input_tile, weight_tile, output_tile = N*(Y+R-1)*(X+S-1)*C, K*R*S*C, Y*X*K*N
        return input_tile, weight_tile, output_tile

    def get_ideal_perf(self, dimension):
        N, K, C, Y, X, R, S = dimension
        input_size, weight_size, output_size = [N*Y*X*C, R*S*C*K, N*Y*X*K] # Input, weight, output
        num_flops = N*R*S*C*Y*X*K
        energys = {}
        for level in range(1, self.num_buffer_level+1):
            if level == 1:
                buf_energy_cost = self.buf_energy_cost['l1']
            elif level == self.num_buffer_level:
                buf_energy_cost = self.buf_energy_cost['DRAM']
            else:
                buf_energy_cost = self.buf_energy_cost['l2']
            energys[f'l{level}-Inputs'] = input_size * buf_energy_cost
            energys[f'l{level}-Weights'] = weight_size * buf_energy_cost
            energys[f'l{level}-Outputs'] = output_size * buf_energy_cost
        energys['compute'] = num_flops * self.buf_energy_cost['MAC']
        energy = sum(e for e in energys.values()) * 1e-6  # energy_uJ
        # cycles = num_flops/self.num_pes
        cycles = num_flops/(self.num_pes-1)
        edp = cycles * energy
        return edp, cycles, energy


    def check_tile_fit_buffer(self, indv):
        len_dim = len('NKCYXRS')
        tile_prods = {}
        tile_prod = np.ones((len_dim,))
        for level in range(1, self.num_buffer_level):
            tile_sizes = {dim_note:self.get_prod(values) for dim_note, values in indv[f'l{level}']['tile_size'].items()}
            par_dims = indv[f'l{level}']['par_dims']
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes, par_dims, timeloop_notation=False)
            tile_prod = (tile_prod * tp_tile_sizes * sp_tile_sizes)
            tile_prods[f'l{level}'] = tile_prod
        for level in range(1, self.num_buffer_level):
            input_tile, weight_tile, output_tile = self.get_input_weight_output_tile(tile_prods[f'l{level}'])
            total_tile = 0
            total_tile += input_tile if indv[f'l{level}']['bypass']['Inputs'] is False else 0
            total_tile += weight_tile if indv[f'l{level}']['bypass']['Weights'] is False else 0
            total_tile += output_tile if indv[f'l{level}']['bypass']['Outputs'] is False else 0
            if total_tile > self.buffer_size_list[f'l{level}']:
                return False
        return True

    def get_tile_buf_size(self, indv):
        len_dim = len('NKCYXRS')
        tile_prods = {}
        tile_prod = np.ones((len_dim,))
        for level in range(1, self.num_buffer_level+1):
            tile_sizes = {dim_note:self.get_prod(values) for dim_note, values in indv[f'l{level}']['tile_size'].items()}
            par_dims = indv[f'l{level}']['par_dims']
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes, par_dims, timeloop_notation=False)
            tile_prod = (tile_prod * tp_tile_sizes * sp_tile_sizes)
            tile_prods[f'l{level}'] = tile_prod
        ret = {}
        for level in range(1, self.num_buffer_level+1):
            input_tile, weight_tile, output_tile = self.get_input_weight_output_tile(tile_prods[f'l{level}'])
            total_tile = input_tile + weight_tile + output_tile
            ret[f'l{level}'] = {'Inputs': input_tile,
                                'Weights': weight_tile,
                                'Outputs':output_tile,
                                'Total':total_tile}
        return ret


    def check_tile_fit_buffer_temp(self, indv):
        len_dim = len('NKCYXRS')
        tile_prods = {}
        tile_prod = np.ones((len_dim,))
        for level in range(1, self.num_buffer_level+1):
            tile_sizes = {dim_note:self.get_prod(values) for dim_note, values in indv[f'l{level}']['tile_size'].items()}
            par_dims = indv[f'l{level}']['par_dims']
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes, par_dims, timeloop_notation=False)
            tile_prod = (tile_prod * tp_tile_sizes * sp_tile_sizes)
            tile_prods[f'l{level}'] = tile_prod
        ret = {}
        for level in range(1, self.num_buffer_level+1):
            input_tile, weight_tile, output_tile = self.get_input_weight_output_tile(tile_prods[f'l{level}'])
            total_tile = 0
            total_tile += input_tile if indv[f'l{level}']['bypass']['Inputs'] is False else 0
            total_tile += weight_tile if indv[f'l{level}']['bypass']['Weights'] is False else 0
            total_tile += output_tile if indv[f'l{level}']['bypass']['Outputs'] is False else 0
            ret[f'l{level}'] = total_tile
        return ret


    def get_map_config(self, indv):
        mapping = []
        for level in range(1, self.num_buffer_level+1):
            target = self.buffer_name_list[f'l{level}']
            permutation = self.get_loop_order(indv[f'l{level}']['loop_order'])
            tile_sizes = {dim_note:self.get_prod(values) for dim_note, values in indv[f'l{level}']['tile_size'].items()}
            par_dims = indv[f'l{level}']['par_dims']
            bypass = indv[f'l{level}']['bypass']
            to_pass, to_keep = self.get_bypass(bypass)
            bypass_map = {'target': target,
                          'type': 'bypass',
                          'keep': to_keep,
                          'bypass': to_pass
                        }
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes, par_dims)
            cur_map = {'target': target,
                       'type': 'temporal',
                        'factors': tp_tile_sizes,
                       'permutation': permutation,
                       }
            mapping.append(cur_map)
            if f'l{level}' in self.buffers_with_spmap:
                cur_map = {'target': target,
                           'type': 'spatial',
                           'factors': sp_tile_sizes,
                           'permutation': permutation,
                           }
                mapping.append(cur_map)
            mapping.append(bypass_map)
        return {'mapping': mapping}


    def get_configs(self, dimension,  indv,):
        arch = self.arch
        problem = self.get_problem_configs(dimension)
        map = self.get_map_config(indv)
        return arch, problem, map

    def write_config(self, arch, problem, map, arch_path, problem_path, map_path, sparse_path=None):
        with open(arch_path, 'w') as fd:
            yaml.dump(arch, fd)
        with open(problem_path, 'w') as fd:
            yaml.dump(problem, fd)
        with open(map_path, 'w') as fd:
            yaml.dump(map, fd)
        if self.use_sparse:
            with open(sparse_path, 'w') as fd:
                yaml.dump(self.sparse, fd)

    def dump_timeloop_config_files(self, dimension, indv, out_dir):
        arch, problem, map = self.get_configs(dimension, indv)
        self.write_config(arch, problem, map, arch_path=os.path.join(out_dir, 'arch.yaml'),
                          problem_path=os.path.join(out_dir, 'problem.yaml'), map_path=os.path.join(out_dir, 'map.yaml'),
                          sparse_path=os.path.join(out_dir, 'sparse.yaml'),)


    def run_timeloop(self, dimension,  indv,
                               pool_idx=0, use_IO=False, fitness_obj=['latency']):
        arch, problem, map = self.get_configs(dimension, indv)
        if use_IO:
            self.write_config(arch, problem, map, arch_path=self.arch_path[pool_idx],
                              problem_path=self.problem_path[pool_idx], map_path=self.map_path[pool_idx],)
            command = [self._executable, self.arch_path[pool_idx], self.problem_path[pool_idx], self.map_path[pool_idx]]
            process = Popen(command, stdout=PIPE, stderr=PIPE, cwd=self.pool_path[pool_idx])
            stdout, stderr = process.communicate()
            process.wait()
            if stderr:
                return [-float('Inf')] * len(fitness_obj)
            else:
                try:
                    stats = parse_timeloop_stats(self.pool_path[pool_idx])
                    fitness = self.judge_IO(stats, fitness_obj)
                except:
                    fitness = [-float('Inf')] * len(fitness_obj)
                return fitness
        else:
            cfg = copy.deepcopy(self.shared_cfg)
            cfg.update(map)
            config = ConfigDict(cfg)
            if not self.debug:
                with stdout_redirected():
                    try:
                        timeloop_app = Model(config,'.')
                        eval_stats = timeloop_app.run()
                        fitness = self.judge(eval_stats, fitness_obj)
                    except:
                        fitness = [-float('Inf')] * len(fitness_obj)
            else:
                print(indv)
                self.dump_timeloop_config_files(dimension, indv, './report/')
                timeloop_app = Model(config,'.')
                eval_stats = timeloop_app.run()
                fitness = self.judge(eval_stats, fitness_obj)
                print(fitness)
            return fitness



    def judge_IO(self, stats, fitness_obj):
        ret = []
        for f in fitness_obj:
            if f == 'edp':
                ret.append(-stats['cycles'] * stats['energy_pJ'] * 1E-6) # energy_uJ
            if f == 'latency':
                ret.append(-stats['cycles'])
            if f == 'utilization':
                ret.append(stats['utilization'])
            if f == 'energy':
                ret.append(-stats['energy_pJ'] * 1E-6) # energy_uJ
        return ret


    def judge(self, stats, fitness_obj):
        ret = []
        for f in fitness_obj:
            if f == 'edp':
                ret.append(-stats.cycles * stats.energy * 1E-6) # energy_uJ
            if f == 'latency':
                ret.append(-stats.cycles)
            if f == 'area':
                ret.append(-stats.area)
            if f == 'utilization':
                ret.append(-stats.utilization)
            if f == 'energy':
                ret.append(-stats.energy * 1E-6) # energy_uJ
        return ret

if __name__ == '__main__':
    l2_size = 2**14
    l1_size = 2**12
    num_pes = 64
    dimension = [32, 32, 16, 16, 3, 3]
    K,C,Y,X,R,S = dimension
    l2_tile_size = [8, 8, 2, 2, 3, 3]
    l1_tile_size = [4, 4, 8, 8, 1, 1]
    l2_loop_order = 'KCYXRS'
    l1_loop_order = 'YXKCRS'
    par_dims = 'KC'
    config_path = '/home/felix/Documents/my_code/timeloop-accelergy-exercises/workspace/exercises/2020.ispass/timeloop/04-model-conv1d+oc-3levelspatial/config'
    timeloop = TimeloopEnv()
    timeloop.create_timeloop_config(dimension, l2_size, l1_size, num_pes, l2_tile_size, l1_tile_size, l2_loop_order, l1_loop_order, par_dims)
    timeloop.run_timeloop()
