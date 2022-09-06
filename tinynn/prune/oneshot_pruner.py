import copy
import sys
import typing
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from tinynn.graph import modifier
from tinynn.graph.tracer import ignore_mod_param_update_warning
from tinynn.util.util import get_logger
from tinynn.prune.base_pruner import BasePruner
from tinynn.graph.modifier import is_dw_conv

log = get_logger(__name__)


class OneShotChannelPruner(BasePruner):
    required_params = ('sparsity', 'metrics')

    metrics: str
    sparsity: typing.Union[typing.Dict[str, float], float]
    default_sparsity: float
    metric_func: typing.Callable[[torch.Tensor, torch.nn.Module], float]
    skip_last_fc: bool
    bn_compensation: bool
    exclude_ops: list

    def __init__(self, model, dummy_input, config):
        """Constructs a new OneShotPruner (including random, l1_norm, l2_norm, fpgm)

        Args:
            model: The model to be pruned
            dummy_input: A viable input to the model
            config (dict, str): Configuration of the pruner (could be path to the json file)

        Raises:
            Exception: If a model without parameters or prunable ops is given, the exception will be thrown
        """

        super(OneShotChannelPruner, self).__init__(model, dummy_input, config)

        self.center_nodes = []
        self.sparsity = {}
        self.bn_compensation = False
        self.parse_config()

        for n in self.graph.forward_nodes:
            # Only prune the specific operators
            if (
                (
                    n.type() in [nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d]
                    and not is_dw_conv(n.module)
                )
                or (n.type() in [nn.Linear])
                or (n.type() in [nn.RNN, nn.GRU, nn.LSTM] and len(n.prev_tensors) == 1)
            ):
                self.center_nodes.append(n)
                if n.unique_name not in self.sparsity:
                    self.sparsity[n.unique_name] = self.default_sparsity

        last_center_node = self.center_nodes[-1] if len(self.center_nodes) > 0 else None
        if last_center_node is None:
            raise Exception("No operations to prune")

        # 除非人工指定了剪枝率，否则当最后一个中心节点是linear时不对其进行剪枝（通常网络最后的全连接层与类别数量有关）
        if self.skip_last_fc and last_center_node.type() in [nn.Linear]:
            if self.sparsity[last_center_node.unique_name] == self.default_sparsity:
                self.sparsity[last_center_node.unique_name] = 0.0

        self.graph_modifier = modifier.GraphChannelModifier(self.graph, self.center_nodes, self.bn_compensation)

        for sub_graph in self.graph_modifier.sub_graphs.values():
            exclude = False
            for m in sub_graph.modifiers:
                if m.node.type() in self.exclude_ops:
                    exclude = True
                    break

                if m.node.type() in (nn.RNN, nn.GRU, nn.LSTM) and len(m.node.prev_tensors) > 1:
                    exclude = True
                    break

            if exclude:
                sub_graph.skip = True

    def parse_config(self):
        """Parses the context and copy the needed items to the pruner"""

        super().parse_config()

        all_param_keys = list(self.required_params) + list(self.default_values.keys())
        for param_key in all_param_keys:
            if param_key not in ['sparsity', 'metrics', 'skip_last_fc', 'exclude_ops', "bn_compensation"]:
                setattr(self, param_key, self.config[param_key])

        sparsity = self.config['sparsity']
        metrics = self.config['metrics']
        self.skip_last_fc = self.config.get('skip_last_fc', True)
        self.bn_compensation = self.config.get('bn_compensation', self.bn_compensation)
        self.exclude_ops = self.config.get('exclude_ops', [])

        if isinstance(sparsity, float):
            self.default_sparsity = sparsity
        elif isinstance(sparsity, dict):
            if 'default' in sparsity:
                self.default_sparsity = sparsity['default']
                del sparsity['default']
            self.sparsity.update(sparsity)
        else:
            raise Exception(f'The type of `sparsity` should either be float or dict for {type(self).__name__}')

        if not hasattr(self, 'default_sparsity'):
            raise Exception(f'Please specify a default sparsity using the key `default` for {type(self).__name__}')

        if hasattr(modifier, metrics):
            self.metric_func = getattr(modifier, metrics)
        else:
            raise Exception(f'{metrics} is not a known metrics for {type(self).__name__}')

    def prune(self):
        """The main function for pruning.
        As for oneshot pruning, it is simply calculating the masks (`register_mask`) and them apply them (`apply_mask`).
        """

        self.register_mask()

        with ignore_mod_param_update_warning():
            self.apply_mask()

    def register_mask(self):
        """Computes the mask for the parameters in the model and register them through the maskers"""
        log.info("Register a mask for each operator")

        importance = {}

        for sub_graph in self.graph_modifier.sub_graphs.values():
            if sub_graph.skip:
                log.info(f"skip subgraph {sub_graph.center}")
                continue

            for m in sub_graph.modifiers:
                if m.node in self.center_nodes and m in sub_graph.dependent_centers:
                    modifier.update_weight_metric(importance, self.metric_func, m.module(), m.unique_name())

            sub_graph.calc_prune_idx(importance, self.sparsity)
            log.info(f"subgraph [{sub_graph.center}] compute over")

        for m in self.graph_modifier.modifiers.values():
            m.register_mask(self.graph_modifier.modifiers, importance, self.sparsity)

    def apply_mask(self):
        """Applies the masks for the parameters and updates the shape and properties of the tensors and modules"""

        log.info("Apply the mask of each operator")

        for m in self.graph_modifier.modifiers.values():
            m.apply_mask(self.graph_modifier.modifiers)

        self.graph_modifier.unregister_modifier()

        # Sync parameters after the size of the tensors has been shrunk
        if dist.is_available() and dist.is_initialized():
            for ps in self.model.parameters():
                dist.broadcast(ps, 0)

    def generate_config(self, path: str) -> None:
        """Generates a new copy the updated configuration with the given path"""

        config = copy.deepcopy(self.config)
        config['sparsity'] = dict()
        config['sparsity']['default'] = self.default_sparsity
        config['sparsity'].update(self.sparsity)

        super().generate_config(path, config)

    def reset(self) -> None:
        """Regenerate the TraceGraph and the Graph Modifier when they are invalidated"""

        self.graph = self.trace()
        self.center_nodes.clear()
        for n in self.graph.forward_nodes:
            if (n.type() in [nn.Conv2d, nn.ConvTranspose2d] and not is_dw_conv(n.module)) or (n.type() in [nn.Linear]):
                self.center_nodes.append(n)
                if n.unique_name not in self.sparsity:
                    self.sparsity[n.unique_name] = self.default_sparsity

        self.graph_modifier = modifier.GraphChannelModifier(self.graph, self.center_nodes, self.bn_compensation)
