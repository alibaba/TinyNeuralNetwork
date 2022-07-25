import os

import torch
import torch.distributed as dist

from tinynn.graph import modifier
from tinynn.util.util import get_logger
from tinynn.prune.oneshot_pruner import OneShotChannelPruner

log = get_logger(__name__)


class ADMMPruner(OneShotChannelPruner):
    required_params = ('sparsity', 'metrics', 'admm_iterations', 'admm_epoch', 'rho', 'admm_lr')
    required_context_params = ('val_loader', 'train_loader', 'train_func', 'validate_func', 'optimizer', 'criterion')
    default_values = {'admm_save_freq': 1, 'admm_valid_freq': 1, 'admm_dir': 'admm_train/'}
    context_from_params_dict = {
        'optimizer': ['admm_optimizer', 'optimizer'],
        'criterion': ['admm_criterion', 'criterion'],
    }
    condition_dict = {
        'admm_iterations': lambda x: 0 < x,
        'admm_epoch': lambda x: 0 < x,
        'rho': lambda x: 0 < x < 1,
        'admm_lr': lambda x: 0 < x < 1,
    }

    admm_iterations: int
    admm_epoch: int
    rho: float
    admm_lr: float
    admm_dir: str
    admm_save_freq: int
    admm_valid_freq: int

    def __init__(self, model, dummy_input, config, context):
        super().__init__(model, dummy_input, config)

        self.parse_context(context)

        self.Z = {}
        self.U = {}

    def prune(self):
        """The main function for pruning"""

        log.info('Start ADMM training')
        old_criterion = self.context.criterion
        self.register_mask()
        for iteration in range(1, self.admm_iterations + 1):
            for epoch in range(1, self.admm_epoch + 1):
                self.context.epoch = epoch + (iteration - 1) * self.admm_epoch
                self.adjust_learning_rate()
                self.context.criterion = self.construct_admm_criterion(old_criterion)
                if dist.is_available() and dist.is_initialized():
                    self.context.train_loader.sampler.set_epoch(self.context.epoch)
                self.context.train_func(self.model, self.context)

                if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                    if self.context.epoch % self.admm_save_freq == 0:
                        save_path = os.path.join(self.admm_dir, f'epoch_{self.context.epoch}.pth')
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        log.info("Saving model to {}".format(save_path))
                        torch.save(self.model.state_dict(), save_path)

                    if self.context.validate_func is not None and self.context.epoch % self.admm_valid_freq == 0:
                        # According to https://github.com/pytorch/pytorch/issues/54059, when validating via DDP,
                        # it needs to be done on the original module.
                        if dist.is_available() and dist.is_initialized():
                            self.context.validate_func(self.model.module, self.context)
                        else:
                            self.context.validate_func(self.model, self.context)

                self.context.best_epoch = self.context.epoch

            self.admm_params_update()

        self.context.criterion = old_criterion
        # Need to reset the masker before final pruning
        self.graph_modifier.reset_masker()
        super().prune()

    def adjust_learning_rate(self):
        epoch = self.context.epoch
        admm_epoch = self.admm_epoch
        if (epoch - 1) % admm_epoch == 0:
            lr = self.admm_lr
        else:
            admm_epoch_offset = (epoch - 1) % admm_epoch
            # LR is updated roughly every 1/3 admm_epoch.
            admm_step = admm_epoch / 3
            lr = self.admm_lr * (0.1 ** (admm_epoch_offset // admm_step))

        for param_group in self.context.optimizer.param_groups:
            param_group['lr'] = lr

    def construct_admm_criterion(self, old_criterion):
        def criterion_func(output, target):
            loss = old_criterion(output, target)
            for n in self.center_nodes:
                if n.unique_name not in self.Z or n.unique_name not in self.U:
                    continue

                loss += (
                    0.5
                    * self.rho
                    * (torch.norm(n.module.weight - self.Z[n.unique_name] + self.U[n.unique_name], p=2) ** 2)
                )
            return loss

        return criterion_func

    def register_mask(self):
        super().register_mask()

        for m in self.graph_modifier.modifiers.values():
            # Disable mask here so that we will use them to update U and Z.
            # They won't be applied so that the training process won't be affected.
            m.disable_mask()
            if m.node in self.center_nodes and m.dim_changes_info.pruned_idx_o:
                device = m.module().weight.data.device
                self.Z[m.unique_name()] = m.module().weight.detach() * m.masker().get_mask('weight').to(device=device)
                self.U[m.unique_name()] = torch.zeros_like(self.Z[m.unique_name()], device=device)

    def admm_params_update(self):
        self.graph_modifier.reset_masker()
        importance = {}
        for sub_graph in self.graph_modifier.sub_graphs.values():
            for m in sub_graph.modifiers:
                if m.node in self.center_nodes and m in sub_graph.dependent_centers:
                    if m.unique_name() not in self.U:
                        continue
                    self.Z[m.unique_name()] = (m.module().weight + self.U[m.unique_name()]).detach()
                    importance[m.unique_name()] = self.metric_func(self.Z[m.unique_name()], m.module())

            sub_graph.calc_prune_idx(importance, self.sparsity)
            log.info(f"subgraph [{sub_graph.center}] compute over")

        for m in self.graph_modifier.modifiers.values():
            m.register_mask(self.graph_modifier.modifiers, importance, self.sparsity)

            # Disable mask here so that we will use them to update U and Z.
            # They won't be applied so that the training process won't be affected.
            m.disable_mask()
            if m.node in self.center_nodes and m.dim_changes_info.pruned_idx_o:
                weight = m.module().weight
                self.Z[m.unique_name()] = self.Z[m.unique_name()] * m.masker().get_mask('weight')
                self.U[m.unique_name()] = weight - self.Z[m.unique_name()] + self.U[m.unique_name()]

        # Sync ADMM parameters
        if dist.is_available() and dist.is_initialized():
            for state in (self.U, self.Z):
                for param in state.values():
                    dist.broadcast(param, 0)
