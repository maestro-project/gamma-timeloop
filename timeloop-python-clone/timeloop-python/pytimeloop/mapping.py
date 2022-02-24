from bindings import ArchProperties, NativeArchConstraints, NativeMapping
from .config import Config
from .problem import Workload
from .model import ArchSpecs


class ArchConstraints(NativeArchConstraints):
    def __init__(self, arch_prop: ArchProperties, workload: Workload,
                 config: Config):
        _, native_config_node = config.get_native()
        super().__init__(arch_prop, workload, native_config_node)


class Mapping(NativeMapping):
    def __init__(self, config: Config, arch_specs: ArchSpecs,
                 workload: Workload):
        _, workload_config_node = config.get_native()
        super().__init__(workload_config_node, arch_specs, workload)
