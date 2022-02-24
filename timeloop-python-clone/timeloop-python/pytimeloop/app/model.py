from bindings import ArchProperties
from pytimeloop.config import Config
from pytimeloop.engine import Accelerator
from pytimeloop.model import ArchSpecs, SparseOptimizationInfo
from pytimeloop.mapping import ArchConstraints, Mapping
from pytimeloop.problem import Workload
import os
import logging






class Model:
    def __init__(self, cfg: Config, out_dir: str='.', auto_bypass_on_failure=False,
                 out_prefix='', log_level=logging.WARNING, dump_file=True ):
        # Setup logger
        self.log_level = log_level
        self.model_logger = logging.getLogger('pytimeloop.app.Model')
        self.model_logger.setLevel(log_level)

        semi_qualified_prefix = 'timeloop-model'
        semi_qualified_prefix = semi_qualified_prefix + out_prefix
        out_prefix = os.path.join(out_dir, semi_qualified_prefix)
        # Architecture configuration
        self.arch_specs = ArchSpecs(cfg['architecture'])
        if dump_file:
            self.arch_specs.generate_tables(
                cfg, semi_qualified_prefix, out_dir, out_prefix, self.log_level)


        # Problem configuration
        self.workload = Workload(cfg['problem'])
        self.model_logger.info('Problem configuration complete.')

        # self.arch_props = ArchProperties(self.arch_specs)

        # Architecture constraints
        # self.constraints = ArchConstraints(
        #     self.arch_props, self.workload, cfg['architecture_constraints'])
        # self.model_logger.info('Architecture configuration complete.')

        # Mapping configuration
        self.mapping = Mapping(cfg['mapping'], self.arch_specs, self.workload)
        self.model_logger.info('Mapping construction complete.')

        # Validate mapping against architecture constraints
        # if not self.constraints.satisfied_by(self.mapping):
        #     self.model_logger.error(
        #         'Mapping violates architecture constraints.')
        #     raise ValueError('Mapping violates architecture constraints.')

        # Sparse optimizations
        if 'sparse_optimizations' in cfg:
            sparse_opt_cfg = cfg['sparse_optimizations']
        else:
            sparse_opt_cfg = Config()
        self.sparse_optimizations = SparseOptimizationInfo(
            sparse_opt_cfg, self.arch_specs)



    def run(self):
        try:
            engine = Accelerator(self.arch_specs)

            eval_stat = engine.evaluate(self.mapping,
                                        self.workload,
                                        self.sparse_optimizations,
                                        log_level=self.log_level)
            return eval_stat
        except:
            return None


