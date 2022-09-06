import copy
import sys
import typing
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from tinynn.prune import OneShotChannelPruner

from tinynn.graph import modifier
from tinynn.graph.tracer import ignore_mod_param_update_warning
from tinynn.util.util import get_logger
from tinynn.prune.base_pruner import BasePruner
from tinynn.graph.modifier import is_dw_conv

log = get_logger(__name__)


class IdentityChannelPruner(OneShotChannelPruner):
    required_params = ('sparsity', 'metrics')

    bn_compensation: bool
    exclude_ops: list

    def __init__(self, model, dummy_input, config=None):
        """Constructs a new IdentityChannelPruner

        Args:
            model: The model to be pruned
            dummy_input: A viable input to the model
            config (dict, str): Configuration of the pruner (could be path to the json file)

        Raises:
            Exception: If a model without parameters or prunable ops is given, the exception will be thrown
        """
        self.bn_compensation = True
        if config is None:
            config = {"metrics": "l2_norm", "sparsity": 0.5}
        super(IdentityChannelPruner, self).__init__(model, dummy_input, config)

    def register_mask(self):
        """Computes the mask for the parameters in the model and register them through the maskers"""
        log.info("Register a mask for each operator")

        for sub_graph in self.graph_modifier.sub_graphs.values():
            if sub_graph.skip:
                log.info(f"skip subgraph {sub_graph.center}")
                continue

            sub_graph.calc_prune_idx(None, self.sparsity)
            log.info(f"subgraph [{sub_graph.center}] compute over")

        for m in self.graph_modifier.modifiers.values():
            m.register_mask(self.graph_modifier.modifiers, None, self.sparsity)
