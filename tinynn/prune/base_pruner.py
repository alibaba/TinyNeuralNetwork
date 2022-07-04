import collections
import typing
from abc import ABC, abstractmethod
from inspect import getsource
from pprint import pformat

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from tinynn.graph.modifier import is_dw_conv
from tinynn.graph.tracer import TraceGraph, model_tracer, trace
from tinynn.util.train_util import DLContext, get_module_device
from tinynn.util.util import conditional, get_actual_type, get_logger

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

log = get_logger(__name__)


class BasePruner(ABC):
    required_params = ()
    required_context_params = ()
    default_values = {}
    context_from_params_dict = {}
    condition_dict = {}

    model: nn.Module
    dummy_input: torch.Tensor
    config: typing.Union[dict, collections.OrderedDict]
    graph: TraceGraph
    context: DLContext

    def __init__(self, model, dummy_input, config):
        self.model = model
        self.config = config
        self.dummy_input = dummy_input
        self.graph = self.trace()

    @abstractmethod
    def prune(self):
        """The main function for pruning"""
        pass

    @abstractmethod
    def register_mask(self):
        """Computes the mask for the parameters in the model and register them through the maskers"""
        pass

    @abstractmethod
    def apply_mask(self):
        """Applies the masks for the parameters and updates the shape and properties of the tensors and modules"""
        pass

    def parse_config(self):
        """Parses the config and init the parameters of the pruner"""

        if isinstance(self.config, str):
            self.config = self.load_config(self.config)

        if not isinstance(self.config, dict):
            raise Exception('The `config` argument requires a parsed json object (e.g. dict or OrderedDict)')

        missing_params = set(self.required_params) - set(self.config)
        if len(missing_params) != 0:
            missing_params_str = ', '.join(missing_params)
            raise Exception(f'Missing param {missing_params_str} for {type(self).__name__}')

        for param_key, default_value in self.default_values.items():
            if param_key not in self.config:
                self.config[param_key] = default_value

        type_dict = {}
        for cls in reversed(type(self).__mro__):
            if hasattr(cls, '__annotations__') and cls != BasePruner:
                type_dict.update(cls.__annotations__)

        for param_key, param_type in type_dict.items():
            if param_key not in self.config:
                continue

            type_expected = False
            param_types = get_actual_type(param_type)
            for type_cand in param_types:
                if isinstance(self.config[param_key], type_cand):
                    type_expected = True
                    break

            if not type_expected:
                raise Exception(f'The type of `{param_key}` in {type(self).__name__} should be {param_type}')

        for param_key, predicate in self.condition_dict.items():
            if predicate(self.config[param_key]) is False:
                raise Exception(f'The value of `{param_key}` doesn\'t meet the requirement: {getsource(predicate)}')

    def parse_context(self, context: DLContext):
        """Parses the context and copy the needed items to the pruner"""

        for context_key, param_keys in self.context_from_params_dict.items():
            for param_key in param_keys:
                if param_key in self.config:
                    setattr(context, context_key, self.config[param_key])
                    break

        filtered_context = dict(filter(lambda x: x[1], context.__dict__.items()))
        missing_context_items = list(set(self.required_context_params) - set(filtered_context))
        if len(missing_context_items) != 0:
            missing_context_items_str = ', '.join(missing_context_items)
            raise Exception(f'Missing context items {missing_context_items_str} for {type(self).__name__}')

        self.context = context

    def summary(self):
        """Dumps the parameters and possibly the related context items of the pruner"""

        if len(self.required_params) > 0:
            log.info('-' * 80)
            log.info(f'params ({type(self).__name__}):')
            for k, v in self.__dict__.items():
                if k in self.required_params or k in self.default_values:
                    log.info(f'{k}: {pformat(v)}')

        if len(self.required_context_params) > 0:
            log.info('-' * 80)
            log.info(f'context ({type(self).__name__}):')
            log.info('\n'.join((f'{k}: {pformat(v)}' for k, v in self.context.__dict__.items())))

    @classmethod
    def load_config(cls, path: str) -> dict:
        """Loads the configuration file and returns it as a dictionary"""

        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.RoundTripLoader)
        return config

    @conditional(lambda: not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
    def generate_config(self, path: str, config: dict = None) -> None:
        """Generates a new copy the updated configuration with the given path"""

        if config is None:
            config = self.config

        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, Dumper=yaml.RoundTripDumper)

    def trace(self) -> TraceGraph:
        with torch.no_grad():
            if isinstance(self.model, DataParallel) or isinstance(self.model, DistributedDataParallel):
                model = self.model.module
            else:
                model = self.model

            old_device = get_module_device(model)

            model.cpu()
            graph = trace(model, self.dummy_input)

            if old_device is not None:
                model.to(device=old_device)

            return graph

    def reset(self):
        """Regenerate the TraceGraph when it is invalidated"""

        self.graph = self.trace()

    def calc_flops(self) -> int:
        """Calculate the flops of the given model"""

        # If graph is invalidated, then we need to regenerate the graph
        if not self.graph.inited:
            self.reset()

        graph: TraceGraph = self.graph
        total_ops = 0
        for node in graph.forward_nodes:
            m = node.module
            remove_in_channel_count = 0
            remove_out_channel_count = 0
            if hasattr(m, 'masker'):
                if m.masker.in_remove_idx is not None:
                    remove_in_channel_count = len(m.masker.in_remove_idx)

                if m.masker.ot_remove_idx is not None:
                    remove_out_channel_count = len(m.masker.ot_remove_idx)

            if type(m) in (nn.Conv2d, nn.ConvTranspose2d):
                kernel_ops = torch.zeros(m.weight.size()[2:]).numel()
                bias_ops = 1 if m.bias is not None else 0

                in_channels = m.in_channels - remove_in_channel_count
                out_channels = m.out_channels - remove_out_channel_count

                out_elements = node.next_tensors[0].nelement() // m.out_channels * out_channels

                if is_dw_conv(m):
                    groups = in_channels
                else:
                    groups = m.groups
                total_ops += out_elements * (in_channels // groups * kernel_ops + bias_ops)
            elif type(m) in (nn.BatchNorm2d,):
                channels = m.num_features - remove_in_channel_count
                nelements = node.prev_tensors[0].numel() // m.num_features * channels

                total_ops += 2 * nelements
            elif type(m) in (nn.AvgPool2d, nn.AdaptiveAvgPool2d):
                channels = node.prev_tensors[0].size(1) - remove_in_channel_count

                kernel_ops = 1
                num_elements = node.prev_tensors[0].numel() // node.prev_tensors[0].size(1) * channels

                total_ops += kernel_ops * num_elements
            elif type(m) in (nn.ReLU,):
                channels = node.prev_tensors[0].size(1) - remove_in_channel_count

                kernel_ops = 1
                num_elements = node.prev_tensors[0].numel() // node.prev_tensors[0].size(1) * channels

                total_ops += kernel_ops * num_elements
            elif type(m) in (nn.Linear,):
                in_channels = m.in_features - remove_in_channel_count
                out_channels = m.out_features - remove_out_channel_count

                total_mul = in_channels
                num_elements = node.next_tensors[0].numel() // m.out_features * out_channels

                total_ops += total_mul * num_elements

        return total_ops
