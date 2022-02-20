import copy
import math
import multiprocessing as mp
import os
import shutil
import sys
import typing

import torch
from tinynn.graph import modifier
from tinynn.util.util import get_logger
from tinynn.prune import OneShotChannelPruner

if sys.version_info.major == 3 and sys.version_info.minor < 7:
    from futures3.process import ProcessPoolExecutor
else:
    from concurrent.futures import ProcessPoolExecutor

log = get_logger(__name__)


def device_init(device_ids):
    if torch.cuda.is_available():
        device_id = device_ids.get()
        log.info(f'Init pool process with cuda id {device_id}')
        torch.cuda.set_device(device_id)


class NetAdaptPruner(OneShotChannelPruner):
    required_params = (
        'budget_type',
        'metrics',
        'netadapt_max_iter',
        'budget_reduce_rate_init',
        'budget_reduce_rate_decay',
        'netadapt_lr',
    )
    required_context_params = ('val_loader', 'train_loader', 'train_func', 'validate_func', 'optimizer', 'criterion')
    default_values = {'netadapt_dir': 'netadapt_train/', 'netadapt_max_rounds': -1, 'netadapt_min_feature_size': 8}
    context_from_params_dict = {
        'optimizer': ['netadapt_optimizer', 'optimizer'],
        'criterion': ['netadapt_criterion', 'criterion'],
    }
    condition_dict = {
        'netadapt_max_iter': lambda x: 0 < x,
        'budget_reduce_rate_init': lambda x: 0 <= x <= 1,
        'budget_reduce_rate_decay': lambda x: 0 <= x <= 1,
        'netadapt_lr': lambda x: 0 < x < 1,
        'netadapt_max_rounds': lambda x: x >= -1,
        'budget_type': lambda x: x in ('flops', 'weights', 'latency'),
    }

    budget: int
    budget_ratio: float
    budget_type: str
    budget_reduce_rate_init: float
    budget_reduce_rate_decay: float
    netadapt_max_iter: int
    netadapt_lr: float
    netadapt_max_rounds: int
    netadapt_min_feature_size: int
    netadapt_dir: str
    init_flops: int

    def __init__(self, model, dummy_input, config, context):
        self.default_sparsity = 0.0
        self.original_channels = {}
        super().__init__(model, dummy_input, config)

        self.parse_context(context)
        self.iteration = 1

        for node in self.center_nodes:
            self.original_channels[node.unique_name] = node.module.weight.shape[0]

    def parse_config(self):
        """Parses the context and copy the needed items to the pruner"""

        super(OneShotChannelPruner, self).parse_config()

        all_param_keys = list(self.required_params) + list(self.default_values.keys())
        for param_key in all_param_keys:
            if param_key not in ['sparsity', 'metrics']:
                setattr(self, param_key, self.config[param_key])

        metrics = self.config['metrics']
        if hasattr(modifier, metrics):
            self.metric_func = getattr(modifier, metrics)
        else:
            raise Exception(f'{metrics} is not a known metrics for {type(self).__name__}')

        budget = None
        budget_ratio = None

        if self.budget_type != 'flops':
            # TODO: Implement budget type: weights, latency
            raise NotImplementedError('Only `budget_type == "flops"` is supported')

        if 'budget' in self.config:
            budget = self.config['budget']

            if not isinstance(budget, int):
                raise Exception('The type of `budget` should be int')

            if budget < 0:
                raise Exception('The value of `budget` doesn\'t meet the requirement: x >= 0')

        if 'budget_ratio' in self.config:
            budget_ratio = self.config['budget_ratio']

            if not isinstance(budget_ratio, float):
                raise Exception('The type of `budget_ratio` should be float')

            if budget_ratio < 0 or budget_ratio > 1:
                raise Exception('The value of `budget_ratio` doesn\'t meet the requirement: 0 <= x <= 1')

        if (budget is not None) != (budget_ratio is not None):
            self.init_flops = self.calc_flops()
            if budget_ratio is not None:
                self.budget = int(budget_ratio * self.init_flops)
            else:
                self.budget = budget
            log.info(f'Global Target/Initial FLOPS: {self.budget}/{self.init_flops}')
        else:
            raise Exception('You should define either `budget` or `budget_ratio` for NetAdaptPruner, not both of them')

    def generate_config(self, path: str) -> None:
        """Generates a new copy the updated configuration with the given path"""

        super(OneShotChannelPruner, self).generate_config(path)

    def get_sparsity_state(self) -> typing.Dict[str, float]:
        """Calculate the sparsity of the subgraphs in the original model"""

        sparsity = {}
        for node in self.center_nodes:
            sparsity[node.unique_name] = (
                self.original_channels[node.unique_name] - node.module.weight.shape[0]
            ) / self.original_channels[node.unique_name]
        return sparsity

    def get_pruned_subgraph_info(
        self, iteration: int, subgraph_id: int, current_flops: float, target_flops: float
    ) -> typing.Tuple[float, typing.Dict[str, float], int]:
        """Prunes the given subgraph so that the flops of the model <
        target flops and finetunes model with the pruned subgraph"""

        if torch.cuda.is_available():
            self.context.device = torch.device("cuda", torch.cuda.current_device())
        else:
            self.context.device = torch.device("cpu")

        # Copies the model, otherwise the one in the main process will be updated as well
        self.model = copy.deepcopy(self.model)
        self.model.eval()
        self.reset()
        log.info(f'Processing subgraph index {subgraph_id} at iteration: {iteration}, device: {self.context.device}')
        sparsity, new_flops = self.find_prune_plan(subgraph_id, current_flops, target_flops)
        if len(sparsity) == 0:
            # Cannot find a plan, early stop
            log.warning(f'Subgraph: {subgraph_id}, Iteration: {iteration}, cannot find a plan')
            return -1, {}, -1

        # Apply the mask to get a pruned model
        super().apply_mask()

        # Regenerate the optimizer, since the model has changed
        self.context.optimizer = type(self.context.optimizer)(
            self.model.parameters(), **self.context.optimizer.defaults
        )

        # Use fork to speed up data loading in child processes
        mp_context = mp.get_context('fork')
        self.context.train_loader.multiprocessing_context = mp_context
        self.context.val_loader.multiprocessing_context = mp_context

        num_iter_per_epoch = len(self.context.train_loader)
        max_epoch = math.ceil(self.netadapt_max_iter / num_iter_per_epoch)
        max_iter_last_epoch = (self.netadapt_max_iter - 1) % num_iter_per_epoch + 1
        for epoch in range(1, max_epoch + 1):
            self.context.epoch = epoch
            if epoch == max_epoch:
                old_max_iter = self.context.max_iteration
                self.context.max_iteration = max_iter_last_epoch

            self.context.train_func(self.model, self.context)

            if epoch == max_epoch:
                self.context.max_iteration = old_max_iter

        save_path = os.path.join(self.netadapt_dir, f'iter_{iteration}_subgraph_{subgraph_id}.pth')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        log.info("Saving model to {}".format(save_path))
        torch.save(self.model.state_dict(), save_path)

        acc = self.context.validate_func(self.model, self.context)
        self.context.best_epoch = self.context.epoch

        log.info(f'Subgraph: {subgraph_id}, Iteration: {iteration}, FLOPS: {new_flops}, Accuracy: {acc}')

        del self.model

        return acc, sparsity, new_flops

    def find_prune_plan(
        self, subgraph_id: int, pre_flops: float, target_flops: float
    ) -> typing.Tuple[typing.Dict[str, float], int]:
        """Figures out the best plan to prune the given subgraph so that the flops of the model < target flops"""

        nodes = []

        for m in self.graph_modifier.sub_graphs[subgraph_id]:
            if m.node in self.center_nodes and m.output_modify_:
                nodes.append(m.node)

        num_out_channels = nodes[0].next_tensors[0].shape[1]

        # All possible number of channels according to `netadapt_min_feature_size`
        candidate_channels = list(
            range(
                num_out_channels // self.netadapt_min_feature_size * self.netadapt_min_feature_size,
                self.netadapt_min_feature_size - 1,
                -self.netadapt_min_feature_size,
            )
        )
        # Possible sparsity for pruning the model
        sparsity_per_step = list(
            map(
                lambda x: (num_out_channels - x[0]) / num_out_channels,
                zip(candidate_channels[1:], candidate_channels[:-1]),
            )
        )
        # Init sparsity dictionary
        self.sparsity = self.sparsity.fromkeys(self.sparsity, 0.0)

        post_flops = pre_flops
        diff_flops = None
        possible_diff_flops = None
        for idx, s in enumerate(sparsity_per_step):
            for node in nodes:
                self.sparsity[node.unique_name] = s

            # Fast-forward when possible
            if diff_flops is not None:
                post_flops -= diff_flops
                if not post_flops < target_flops:
                    continue
                target_out_channels = candidate_channels[idx + 1]
                for node in nodes:
                    self.sparsity[node.unique_name] = (num_out_channels - target_out_channels) / num_out_channels

            # Reset masks
            if idx != 0:
                self.graph_modifier.unregister_masker()
                self.graph_modifier.reset_masker()

            # Get the updated masks
            super().register_mask()

            # Update flops for the current graph
            if diff_flops is None:
                next_flops = self.calc_flops()

                if idx == 1:
                    # Skip further calculations if FLOPs varies proportional to output channel size
                    num_cur_flops = post_flops
                    num_next_flops = next_flops
                    if (
                        num_cur_flops % candidate_channels[idx] == 0
                        and num_next_flops % candidate_channels[idx + 1] == 0
                        and num_cur_flops // candidate_channels[idx] == num_next_flops // candidate_channels[idx + 1]
                    ):
                        diff_flops = num_cur_flops - num_next_flops
                    else:
                        possible_diff_flops = num_cur_flops - num_next_flops
                elif idx == 2:
                    # Skip further calculations if FLOPS(n) - FLOP(n-1) is constant
                    if post_flops - next_flops == possible_diff_flops:
                        diff_flops = possible_diff_flops
                        possible_diff_flops = None
                post_flops = next_flops

            log.info(
                f'Subgraph: {subgraph_id}, '
                f'Channels: {candidate_channels[idx + 1]}/{num_out_channels}, '
                f'FLOPS(pre/post/target): {pre_flops}/{post_flops}/{target_flops:.2f}'
            )

            # Early stop if we get the desired sparsity
            if post_flops < target_flops:
                sparsity = copy.deepcopy(self.sparsity)
                return sparsity, post_flops

        return {}, -1

    def prune_subgraph(self, sparsity: typing.Dict[str, float]) -> None:
        """Prunes the model with the sparsity dictionary given"""

        self.sparsity = copy.deepcopy(sparsity)

        super().prune()

    def prune(self):
        """The main function for pruning"""

        # PyTorch forbids initialization CUDA in the default fork settings
        # So `spawn` is used here instead
        mp_context = mp.get_context('spawn')
        if torch.cuda.is_available():
            max_workers = torch.cuda.device_count()
        else:
            max_workers = 1

        available_cores = mp_context.Queue()

        for i in range(max_workers):
            available_cores.put(i)

        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp_context, initializer=device_init, initargs=(available_cores,)
        ) as pool:

            cur_flops = self.calc_flops()

            while (
                self.netadapt_max_rounds == -1 or self.iteration <= self.netadapt_max_rounds
            ) and cur_flops > self.budget:

                log.info(f"Start iteration {self.iteration}")

                # Acquire target FLOPs for the current ratio
                target_flops = cur_flops - self.budget_reduce_rate_init * cur_flops * (
                    self.budget_reduce_rate_decay ** (self.iteration - 1)
                )

                # Regenerate modifier, graph and center nodes before sending to child processes
                if self.iteration != 1:
                    self.reset()

                # Prepare jobs for child processes
                num_subgraphs = len(self.graph_modifier.sub_graphs)
                results = pool.map(
                    self.get_pruned_subgraph_info,
                    [self.iteration] * num_subgraphs,
                    range(num_subgraphs),
                    [cur_flops] * num_subgraphs,
                    [target_flops] * num_subgraphs,
                )

                # Get the step we should take by comparing the best accuracy among all sub jobs
                max_idx, (best_acc, best_sparsity, best_flops) = max(enumerate(results), key=lambda x: x[1][0])

                if len(best_sparsity) == 0:
                    log.error('All subgraphs yield invalid result, stopping')
                    break

                # Sync model back to the main process
                self.prune_subgraph(best_sparsity)

                load_path = os.path.join(self.netadapt_dir, f'iter_{self.iteration}_subgraph_{max_idx}.pth')
                self.model.load_state_dict(torch.load(load_path, map_location='cpu'))

                global_sparsity = self.get_sparsity_state()
                global_sparsity['best_flops'] = best_flops
                global_sparsity['iteration'] = self.iteration

                sparsity_path = os.path.join(self.netadapt_dir, f'iter_{self.iteration}_sparsity.yml')
                super(OneShotChannelPruner, self).generate_config(sparsity_path, global_sparsity)

                save_path = os.path.join(self.netadapt_dir, f'iter_{self.iteration}.pth')
                shutil.copy(load_path, save_path)

                # Print info and update current flops
                log.info(f'Iter: {self.iteration}, Acc: {best_acc}, FLOPS: {best_flops}, Subgraph: {max_idx}')
                cur_flops = best_flops
                self.iteration += 1

    def restore(self, iteration: int) -> torch.nn.Module:
        """Restores a model at specific iteration"""

        sparsity_path = os.path.join(self.netadapt_dir, f'iter_{iteration}_sparsity.yml')
        weights_path = os.path.join(self.netadapt_dir, f'iter_{iteration}.pth')

        for path in (sparsity_path, weights_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f'{path} is required for restoring the model')

        config = self.load_config(sparsity_path)

        self.prune_subgraph(config)
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))

        self.iteration = config['iteration'] + 1

        log.info(f"Restored from iteration {iteration}")

        return self.model
