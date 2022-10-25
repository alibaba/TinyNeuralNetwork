from collections import OrderedDict
from copy import deepcopy
import torch
import torch.nn as nn
import typing
from tinynn.graph.tracer import TraceNode, TraceGraph
from tinynn.util.util import get_logger
from tinynn.graph import masker
import numpy as np

log = get_logger(__name__)


def complementary_list(a, b):
    return list(set(a).difference(set(b)))


def get_smallest_k(lst, k, offset=0):
    idx_lst = [(i, float(lst[i])) for i in range(len(lst))]
    sorted_lst = sorted(idx_lst, key=lambda x: x[1])
    sorted_lst_k = sorted_lst[:k]
    idx = [sorted_lst_k[i][0] + offset for i in range(len(sorted_lst_k))]

    return sorted(idx)


def rnn_gate_size(module: nn.Module) -> int:
    """the gate size of the recurrent modules"""
    if isinstance(module, nn.RNN):
        return 1
    elif isinstance(module, nn.GRU):
        return 3
    elif isinstance(module, nn.LSTM):
        return 4
    else:
        raise AttributeError(f'gate size of {type(module)} is unknown')


def update_weight_metric(importance, metric_func, module, name):
    if type(module) in [nn.Linear, nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d, nn.ConvTranspose1d]:
        importance[name] = metric_func(module.weight, module)
    elif type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
        num_directions = 2 if module.bidirectional else 1
        has_proj = hasattr(module, 'proj_size') and module.proj_size > 0

        gs = rnn_gate_size(module)

        weights = []

        if has_proj:
            for i in range(module.num_layers):
                weight_hrs = []

                for j in range(num_directions):
                    suffix = '_reverse' if j > 0 else ''
                    weight_hr = getattr(module, f'weight_hr_l{i}{suffix}')
                    weight_hrs.append(weight_hr)

                weights.append(torch.cat(weight_hrs, dim=0))

            importance[name] = metric_func(weights, module)

            weights.clear()
            name = f'{name}:h'

        for i in range(module.num_layers):
            weight_ihs = []
            weight_hhs = []

            for j in range(num_directions):
                suffix = '_reverse' if j > 0 else ''
                weight_ih = getattr(module, f'weight_ih_l{i}{suffix}')
                weight_hh = getattr(module, f'weight_hh_l{i}{suffix}')

                weight_ihs.append(weight_ih)
                weight_hhs.append(weight_hh)

            if gs == 1:
                weights.append(torch.cat(weight_ihs, dim=0))
                weights.append(torch.cat(weight_hhs, dim=0))
            else:
                w_ih_splits = zip(*[torch.unbind(x.view(gs, module.hidden_size, -1)) for x in weight_ihs])
                w_hh_splits = zip(*[torch.unbind(x.view(gs, module.hidden_size, -1)) for x in weight_hhs])

                ih_gate_weights = [torch.cat(x) for x in w_ih_splits]
                hh_gate_weights = [torch.cat(x) for x in w_hh_splits]

                weights.extend(ih_gate_weights)
                weights.extend(hh_gate_weights)

            importance[name] = metric_func(weights, module)
    else:
        raise AttributeError(f'{type(module).__name__}({name}) is not supported for importance calculation')


def random(tensor, module):
    if type(module) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        return torch.randperm(tensor.shape[0])
    if type(module) in [nn.ConvTranspose2d, nn.ConvTranspose1d]:
        return torch.randperm(tensor.shape[1])
    if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
        assert isinstance(tensor, (tuple, list))
        return torch.randperm(tensor[0].shape[1])


def l1_norm(tensor, module):
    """Calculate the L1-normalization of each channel"""
    if type(module) in [nn.Conv2d]:
        return torch.norm(tensor, p=1, dim=[1, 2, 3])
    if type(module) in [nn.Conv1d]:
        return torch.norm(tensor, p=1, dim=[1, 2])
    if type(module) in [nn.Linear]:
        return torch.norm(tensor, p=1, dim=[1])
    if type(module) in [nn.ConvTranspose2d]:
        return torch.norm(tensor, p=1, dim=[0, 2, 3])
    if type(module) in [nn.ConvTranspose1d]:
        return torch.norm(tensor, p=1, dim=[0, 2])
    if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
        assert isinstance(tensor, (tuple, list))
        return torch.sum(torch.stack([torch.norm(t, p=1, dim=[1]) for t in tensor]), dim=0)


def l2_norm(tensor, module):
    """Calculate the L2-normalization of each channel"""
    if type(module) in [nn.Conv2d]:
        return torch.norm(tensor, p=2, dim=[1, 2, 3])
    if type(module) in [nn.Conv1d]:
        return torch.norm(tensor, p=2, dim=[1, 2])
    if type(module) in [nn.Linear]:
        return torch.norm(tensor, p=2, dim=[1])
    if type(module) in [nn.ConvTranspose2d]:
        return torch.norm(tensor, p=2, dim=[0, 2, 3])
    if type(module) in [nn.ConvTranspose1d]:
        return torch.norm(tensor, p=2, dim=[0, 2])
    if type(module) in [nn.GRU, nn.LSTM, nn.RNN]:
        assert isinstance(tensor, (tuple, list))
        return torch.sum(torch.stack([torch.norm(t, p=2, dim=[1]) for t in tensor]), dim=0)


def fpgm(tensor, module):
    """Calculate the geometric median (Filter Pruning via Geometric Median for Deep Convolutional Neural
    Networks Acceleration, https://arxiv.org/abs/1811.00250)"""
    assert type(module) in [nn.Linear, nn.Conv2d]
    num_channels = tensor.shape[0]
    batched_weight = tensor.view(num_channels, -1)
    return torch.cdist(batched_weight, batched_weight, p=2).abs().sum(0)


def is_dw_conv(module):
    """Check whether the model is depth-wise convolution"""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d)):
        if module.in_channels == module.groups == module.out_channels:
            return True
    return False


def merge_group(group: list):
    new_group = []

    while len(group) > 0:
        loop = False

        a = group[0]
        for b in group[1:]:
            if len(a & b) > 0:
                k = a & b
                m = a - k
                n = b - k
                group.remove(a)
                group.remove(b)
                group += [i for i in [m, n, k] if len(i) > 0]
                loop = True
                break
        if loop:
            continue

        new_group.append(a)
        group.remove(a)

    group[:] = new_group[:]

    for i in range(len(group)):
        gi = group[i]
        for gj in group[i + 1 :]:
            if gi == gj and gi is not gj:
                new_group.remove(gi)
                break

    group[:] = new_group[:]
    group.sort()
    return group


def merge_constraint(constraint: typing.List[typing.Set]):
    """Merge all constraints with intersection"""
    if {-1.0} in constraint:
        constraint.remove({-1.0})

    value_mapping = dict()

    # remove empty constraint
    for idx in reversed(range(len(constraint))):
        if len(constraint[idx]) == 0:
            del constraint[idx]
            continue

    # build value_map，used to quickly find which sets the value is in
    for idx in range(len(constraint)):
        for value in constraint[idx]:
            value_mapping[value] = value_mapping.get(value, [])
            value_mapping[value].append(idx)

    reindex_list = [i for i in range(len(constraint))]

    # for quickly finding all sets that redirect to the same set,
    homology_idx = {i: [i] for i in range(len(constraint))}

    for value, idx_need_merge in value_mapping.items():
        if len(idx_need_merge) <= 1:
            continue

        target_idx = reindex_list[idx_need_merge[0]]
        for i in idx_need_merge:
            if reindex_list[i] == target_idx:
                continue

            src_idx = reindex_list[i]

            constraint[target_idx] = constraint[target_idx] | constraint[src_idx]

            for j in homology_idx[src_idx]:
                reindex_list[j] = target_idx
                homology_idx[target_idx].append(j)

            homology_idx[src_idx] = []
    valid_idx = sorted(list(set(reindex_list)))
    new_constraint = [constraint[i] for i in valid_idx]
    constraint[:] = new_constraint[:]
    return constraint


def calc_dim_constraint(tensor: torch.Tensor, dim_changes: typing.List):
    """Count all constraints under a dimension"""

    constraints = {}

    arr = tensor.detach().numpy()

    for dim in dim_changes:
        constraints[dim] = []

        for i in range(tensor.shape[dim]):
            arr_i = arr.take(i, axis=dim)

            constraint = np.unique(arr_i)
            constraint = set(constraint.tolist())

            constraints[dim].append(constraint)

    return constraints


def calc_dim_changes(node, tensors_i, tensors_o=None) -> typing.List[typing.Tuple[typing.List, torch.Tensor]]:
    """Calculate which dimensions of tensor have changed"""

    def vector_wrapper(t):
        if type(t) not in [tuple, list]:
            return (t,)
        else:
            return t

    # operator inference
    if isinstance(node, nn.Module):
        tensors = vector_wrapper(node(*tensors_i))
    else:
        with torch.no_grad():
            tensors = vector_wrapper(node(vector_wrapper(tensors_i)))

    if tensors_o is not None:
        for i in range(len(tensors_o)):
            tensors_o[i].data.copy_(tensors[i].clone().data)
    else:
        tensors_o = tensors

    dim_changes = []

    for tensor_o in tensors_o:
        dim_change = []

        for i in range(len(tensor_o.shape)):
            reduce_dim = [j for j in range(len(tensor_o.shape)) if j != i]
            value = set(torch.sum(tensor_o, dim=reduce_dim).detach().tolist())
            if len(value) > 1:
                if i not in dim_change:
                    dim_change.append(i)

        dim_changes.append((dim_change, tensor_o))
    return dim_changes


class DimensionChangeInfo(object):
    def __init__(self, modifier: 'Modifier'):
        self.modifier = modifier

        # Which dimensions of the input/output tensor will change
        self.dim_changes_i = OrderedDict()
        self.dim_changes_o = OrderedDict()

        # All center nodes（operators that change actively, such as conv2d, linear in the pruning process)
        self.centers = OrderedDict()
        self.tensor_changes = OrderedDict()
        self.tensor_keys = OrderedDict()

        # The dimension of the final pruning (when multiple dimensions may change, a dimension will be
        # selected through dependency analysis and conflict elimination)
        self.dim_choices = OrderedDict()
        self.tensor_choices = OrderedDict()
        self.dim_transform = None

        # The mapping relationship between the current node and the central node
        self.constraints_i = OrderedDict()
        self.constraints_o = OrderedDict()

        # In operators such as grouped convolution and bidirectional LSTM, each tensor needs to be
        # modified uniformly according to the group.
        self.groups_i = []
        self.groups_o = []

        # Pruning index for input and output ( for structured pruning, it may be a channel,
        # for unstructured pruning, it may be every point)
        self.pruned_idx_i = []
        self.pruned_idx_o = []

    def build_key(self, center, tensor):
        """Generate human-readable keys to reduce the debugging cost of complex computational graphs"""
        pre_tensor_idx = [id(t) for t in self.modifier.pre_tensors()]
        nxt_tensor_idx = [id(t) for t in self.modifier.next_tensors()]

        if id(tensor) in pre_tensor_idx:
            tensor_idx = pre_tensor_idx.index(id(tensor))
            return f'{center.unique_name()}:input_{tensor_idx}'
        elif id(tensor) in nxt_tensor_idx:
            tensor_idx = nxt_tensor_idx.index(id(tensor))
            return f'{center.unique_name()}:output_{tensor_idx}'
        else:
            assert False

    def build_choice_key(self, tensor):
        """Generate human-readable keys to reduce the debugging cost of complex computational graphs"""
        pre_tensor_idx = [id(t) for t in self.modifier.pre_tensors()]
        nxt_tensor_idx = [id(t) for t in self.modifier.next_tensors()]

        if id(tensor) in pre_tensor_idx:
            tensor_idx = pre_tensor_idx.index(id(tensor))
            return f'input_{tensor_idx}'
        elif id(tensor) in nxt_tensor_idx:
            tensor_idx = nxt_tensor_idx.index(id(tensor))
            return f'output_{tensor_idx}'
        else:
            assert False

    def is_multi_dim_changed(self):
        dim_change_i = self.merge_i()
        dim_change_o = self.merge_o()
        if len(dim_change_i) > 1 or len(dim_change_o) > 1:
            return True

        return False

    def is_changes_conflict(self, tensor_changes):
        if len(tensor_changes) == 0 and len(self.tensor_changes) > 0:
            return True

        for tensor_id, dim_choose in tensor_changes.items():

            dim_changes_flat = list()
            for dim_changes in self.tensor_changes[tensor_id]:
                dim_changes_flat += dim_changes

            if not set(dim_choose).issubset(set(dim_changes_flat)):
                return True

        return False

    def merge_t(self, tensor):
        """Merge the dimension change information of all tensors"""
        dim_change = set()
        for change in self.tensor_changes[id(tensor)]:
            dim_change.update(change)

        return sorted(list(dim_change))

    def merge_i(self) -> typing.List:
        """Merge the dimension change information of input tensors"""

        dim_change = set()
        for t in self.modifier.pre_tensors():
            if id(t) not in self.tensor_changes.keys():
                continue

            for change in self.tensor_changes[id(t)]:
                dim_change.update(change)

        return sorted(list(dim_change))

    def merge_o(self) -> typing.List:
        """Merge the dimension change information of output tensors"""

        dim_change = set()
        for t in self.modifier.next_tensors():
            # The tensor used internally such as hidden state in RNN is not included in the dependency analysis
            if id(t) in self.tensor_changes:
                for change in self.tensor_changes[id(t)]:
                    dim_change.update(change)

        return sorted(list(dim_change))

    def update_i(
        self,
        center: 'Modifier',
        tensor: torch.Tensor,
        dim_changes: typing.List,
        dim_transform=None,
        update_constraint=True,
        tensor_constraint=None,
    ):
        """Update the dimension change information of the input tensor"""

        if dim_transform is not None:
            self.dim_transform = dim_transform

        constraint_i = None

        if update_constraint:
            if tensor_constraint is not None:
                constraint_i = tensor_constraint
            else:
                constraint_i = calc_dim_constraint(tensor, dim_changes)

                # Redirect pruning constraints to central node
                if dim_transform:
                    for dim, constraint in constraint_i.items():
                        for i in range(len(constraint)):
                            new_constraint = set()

                            for c in constraint[i]:
                                if c in dim_transform.keys():
                                    transformed_idx = dim_transform[c]
                                    new_constraint.update(transformed_idx)

                            if len(new_constraint) > 0:
                                constraint[i] = new_constraint

            for dim, constraint in constraint_i.items():
                if dim not in self.constraints_i:
                    self.constraints_i[dim] = {}

                self.constraints_i[dim][center.unique_name()] = self.constraints_i[dim].get(center.unique_name(), [])
                self.constraints_i[dim][center.unique_name()].append(constraint)

        self.update_(self.dim_changes_i, center, tensor, dim_changes)
        return constraint_i

    def update_o(
        self,
        center: 'Modifier',
        tensor: torch.Tensor,
        dim_changes: typing.List,
        update_constraint=False,
        default_constraint=None,
    ):
        """Update the dimension change information of the output tensor"""
        constraint_o = None

        if update_constraint:
            if default_constraint is not None:
                constraint_o = default_constraint
            else:
                constraint_o = calc_dim_constraint(tensor, dim_changes)

            for dim, constraint in constraint_o.items():
                if dim not in self.constraints_o:
                    self.constraints_o[dim] = {}

                self.constraints_o[dim][center.unique_name()] = self.constraints_o[dim].get(center.unique_name(), [])
                self.constraints_o[dim][center.unique_name()].append(constraint)

        self.update_(self.dim_changes_o, center, tensor, dim_changes)
        return constraint_o

    def update_(self, dim_changes_dict: typing.Dict, center: 'Modifier', tensor, dim_changes: typing.List):
        key = self.build_key(center, tensor)
        if key not in dim_changes_dict.keys():
            dim_changes_dict[key] = dim_changes

        if id(tensor) not in self.tensor_changes:
            self.tensor_changes[id(tensor)] = []
            self.tensor_keys[id(tensor)] = []

        self.tensor_changes[id(tensor)].append(dim_changes)
        self.tensor_keys[id(tensor)].append(key)
        self.tensor_keys[id(tensor)] = list(set(self.tensor_keys[id(tensor)]))
        self.centers[center.unique_name()] = center
        return self

    def update_choice(self, tensor, choice):
        key = self.build_choice_key(tensor)
        self.dim_choices[key] = choice
        self.tensor_choices[id(tensor)] = choice

    def get_neighbor_changes(self, center, neighbor):
        if isinstance(center, TraceNode) and isinstance(neighbor, TraceNode):
            center = center.modifier
            neighbor = neighbor.modifier

        changes = []

        if neighbor in self.modifier.pre_modifiers():
            for t in self.modifier.pre_tensors(neighbor):
                changes.append(self.dim_changes_i.get(self.build_key(center, t), None))
        else:
            for t in self.modifier.next_tensors(neighbor):
                changes.append(self.dim_changes_o.get(self.build_key(center, t), None))

        if changes == [None]:
            return None

        return changes

    def get_neighbor_choices(self, neighbor):
        if isinstance(neighbor, TraceNode):
            neighbor = neighbor.modifier

        choices = []

        if neighbor in self.modifier.pre_modifiers():
            for t in self.modifier.pre_tensors(neighbor):
                choices.append(self.get_tensor_choices(t))
        else:
            for t in self.modifier.next_tensors(neighbor):
                choices.append(self.get_tensor_choices(t))

        if choices == [None]:
            return None

        return choices

    def get_tensor_choices(self, tensor) -> typing.List:
        return self.dim_choices.get(self.build_choice_key(tensor), None)

    def get_tensor_changes(self, tensor) -> typing.List:
        return self.tensor_changes[id(tensor)]

    def get_input_centers(self):
        center_names = []
        for key, value in self.dim_changes_i.items():
            center_names.append(key.split(":")[0])
        return center_names

    def rebuild(self):
        """Reconstruct the entire dimension change information according to dim_choice"""

        valid_changes = []

        for tensor_id, choice in self.tensor_choices.items():
            for key in self.tensor_keys[tensor_id]:
                if 'input' in key:
                    tensor_change = self.dim_changes_i[key]
                else:
                    tensor_change = self.dim_changes_o[key]

                if set(choice).issubset(set(tensor_change)):
                    center_name = key.split(":")[0]
                    center = self.centers[center_name]
                    all_tensors = self.modifier.pre_tensors() + self.modifier.next_tensors()
                    tensor = [t for t in all_tensors if id(t) == tensor_id][0]
                    valid_changes.append((center, tensor, tensor_change))

        self.dim_changes_i = OrderedDict()
        self.dim_changes_o = OrderedDict()
        self.centers = OrderedDict()
        self.tensor_changes = OrderedDict()
        self.tensor_keys = OrderedDict()
        constraint_i_new = OrderedDict()
        constraint_o_new = OrderedDict()

        for changes in valid_changes:
            center, tensor, tensor_change = changes
            if self.modifier.is_pre_tensor(tensor):
                self.update_i(center, tensor, tensor_change, update_constraint=False)
                constraint_old = self.constraints_i
                constraint_new = constraint_i_new
            else:
                self.update_o(center, tensor, tensor_change, update_constraint=False)
                constraint_old = self.constraints_o
                constraint_new = constraint_o_new

            choice = self.get_tensor_choices(tensor)
            for dim, constraints in constraint_old.items():
                if dim in choice:
                    if dim not in constraint_new:
                        constraint_new[dim] = {}

                    constraint_new[dim] = constraints

        for dim, dim_constraints in constraint_i_new.items():
            for center_name, constraints in dim_constraints.items():
                if len(constraints) == 1:
                    continue

                merge = [set() for i in constraints[0]]
                for constraint in constraints:
                    for i in range(len(constraint)):
                        if constraint[i] != {-1}:
                            merge[i].update(constraint[i])
                constraints[:] = [merge]

        self.constraints_i = constraint_i_new
        self.constraints_o = constraint_o_new

    def __str__(self):
        return (
            f"dim_changes_i:{str(self.dim_changes_i)}, dim_changes_o:{str(self.dim_changes_o)},"
            f" dim_choices:{str(self.dim_choices)}"
        )


class Modifier(object):
    graph_modifier: "GraphChannelModifier"
    node: TraceNode
    dim_changes_info: DimensionChangeInfo
    forward_dim_mapping: typing.Dict[int, typing.Dict[int, typing.Dict[int, typing.Set]]]
    backward_dim_mapping: typing.Dict[int, typing.Dict[int, typing.Dict[int, typing.Set]]]
    prunable: bool
    weight_mask: typing.Dict[str, torch.Tensor]
    bias_mask: typing.Dict[str, torch.Tensor]

    def __init__(self, node: TraceNode):
        self.graph_modifier = None
        self.node = node

        # Tensor change dependencies between this operator and other operators
        self.dim_changes_info = DimensionChangeInfo(self)

        # When the dimension of the input/output tensor changes, the dimension of the affected output/input tensor
        self.forward_dim_mapping = OrderedDict()
        self.backward_dim_mapping = OrderedDict()

        # Whether the operator allows pruning (for example, conv2d, linear, rnn, etc. are pruned,
        # add, mul, etc. are not pruned)
        self.prunable = False

        self.weight_mask = OrderedDict()
        self.bias_mask = OrderedDict()
        self.mask_applied = False

        self.tensor_id_to_str = {}
        for i in range(len(self.pre_tensors())):
            self.tensor_id_to_str[id(self.pre_tensors()[i])] = f"input_{i}"
        for i in range(len(self.next_tensors())):
            self.tensor_id_to_str[id(self.next_tensors()[i])] = f"output_{i}"

        self.constant_node = False
        if len(self.pre_tensors()) == 1:
            self.constant_node = True
            for t in self.next_tensors():
                if isinstance(t, torch.Tensor) and len(t.shape) > 0:
                    self.constant_node = False
                    break

    def __hash__(self):
        return hash(self.unique_name())

    def __eq__(self, other: 'Modifier'):
        return self.unique_name() == other.unique_name()

    def args_parsed(self):
        return self.node.module.args_parsed

    def masker(self) -> masker.ChannelMasker:
        return getattr(self.node.module, "masker", None)

    def enable_mask(self):
        if self.masker() is not None:
            self.masker().enable()

    def disable_mask(self):
        if self.masker() is not None:
            self.masker().disable()

    def reset_mask(self):
        self.weight_mask.clear()
        self.bias_mask.clear()
        if hasattr(self.module(), "weight"):
            self.weight_mask["weight"] = torch.ones_like(self.module().weight)
        if hasattr(self.module(), "bias"):
            self.bias_mask["bias"] = (
                torch.ones_like(self.module().bias) if type(self.module().bias) is torch.nn.Parameter else None
            )

    def register_mask(self, modifiers, importance, sparsity):
        if self.masker() is not None:
            self.masker().set_in_remove_idx(self.dim_changes_info.pruned_idx_i)
            self.masker().set_ot_remove_idx(self.dim_changes_info.pruned_idx_o)

    def apply_mask(self, modifiers):
        """Use mask to modify the channel of the operator"""
        if self.masker() is not None and self.masker().in_remove_idx is not None:
            self.modify_input(self.masker().in_remove_idx)

        if self.masker() is not None and self.masker().ot_remove_idx is not None:
            self.modify_output(self.masker().ot_remove_idx)

        self.mask_applied = True

    def modify_input(self, remove_idx):
        """Modify the input tensor of the operator"""
        pass

    def modify_output(self, remove_idx):
        """Modify the output tensor of the operator"""
        pass

    def module(self):
        return self.node.module

    def unique_name(self):
        return self.node.unique_name

    def is_pre_tensor(self, tensor):
        return id(tensor) in [id(x) for x in self.pre_tensors()]

    def is_nxt_tensor(self, tensor):
        return id(tensor) in [id(x) for x in self.next_tensors()]

    def pre_tensors(self, parent: 'Modifier' = None, non_constant=False):
        if parent is None:
            if non_constant:
                return [t for t in self.node.prev_tensors if len(t.shape) > 0]
            return self.node.prev_tensors

        tensors = []
        for t in parent.next_tensors():
            if self.is_pre_tensor(t):
                if non_constant and len(t.shape) == 0:
                    continue
                tensors.append(t)
        return tensors

    def next_tensors(self, child: 'Modifier' = None, non_constant=False):
        if child is None:
            if non_constant:
                return [t for t in self.node.next_tensors if len(t.shape) > 0]
            return self.node.next_tensors

        tensors = []
        for t in child.pre_tensors():
            if self.is_nxt_tensor(t):
                if non_constant and len(t.shape) == 0:
                    continue
                tensors.append(t)
        return tensors

    def pre_modifiers(self, edge: torch.Tensor = None) -> typing.List['Modifier']:
        if edge is None:
            return [n.modifier for n in self.node.prev_nodes]

        modifiers = []
        for m in self.pre_modifiers():
            for t in m.next_tensors():
                if t is edge:
                    modifiers.append(m)

        return modifiers

    def next_modifiers(self, edge: torch.Tensor = None) -> typing.List['Modifier']:
        if edge is None:
            return [n.modifier for n in self.node.next_nodes]

        modifiers = []
        for m in self.next_modifiers():
            for t in m.pre_tensors():
                if t is edge:
                    modifiers.append(m)

        return modifiers

    def get_pruned_idx(self, modifiers):
        """Obtain the input tensor pruning information from the pruning information of other operators"""

        pruned_idx = set()
        input_modify_dim = self.dim_changes_info.get_tensor_choices(self.pre_tensors()[0])[0]

        for center_name, _ in self.dim_changes_info.centers.items():
            center = modifiers[center_name]
            center_pruned_idx = set(center.dim_changes_info.pruned_idx_o)

            constraints_i = self.dim_changes_info.constraints_i[input_modify_dim][center_name]
            for constraint_i in constraints_i:
                for leaf_idx in range(len(constraint_i)):
                    center_idx = constraint_i[leaf_idx]
                    if len(center_idx & center_pruned_idx) > 0:
                        pruned_idx.add(leaf_idx)

        pruned_idx = list(pruned_idx)
        pruned_idx.sort()
        sparsity = len(pruned_idx) / self.pre_tensors()[0].shape[input_modify_dim]

        return pruned_idx, sparsity

    def calc_idx_group(self):
        return None

    def calc_dim_changes(self) -> typing.List[typing.Tuple[typing.List, torch.Tensor]]:
        return calc_dim_changes(self.module(), self.pre_tensors(), self.next_tensors())

    def change_dimension(self) -> bool:
        return False

    def dim_change_forward(self, center, tensor, dim_changes_i: typing.List, dim_transform, tensor_constraint):
        # Leaf nodes require no additional computation
        if len(self.next_tensors()) == 0:
            self.dim_changes_info.update_i(center, tensor, dim_changes_i, dim_transform)
            return True

        if self.node.kind() == "data":
            return True

        # Skip constant node
        if self.constant_node:
            return True

        # The default implementation is regarded as Identity()
        # Directly inheriting the dim_constraint of the previous layer, reducing the amount of calculation
        tensor_constraint = self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        for tensor_o in self.next_tensors():
            if id(tensor) == id(tensor_o):
                continue

            try:
                tensor_o.copy_(tensor.clone())
            except Exception as e:
                log.error(
                    f"error modifier = {self.unique_name()}, type = {type(self.module())}, kind = {self.node.kind()}"
                )
                raise e

        for tensor_o in self.next_tensors():
            self.dim_changes_info.update_o(center, tensor_o, dim_changes_i)

            for m in self.next_modifiers(tensor_o):
                # The identity() operator does not change the constraint, so it can directly pass its own
                # constraints to reduce the calculation of the next layer
                m.dim_change_forward(center, tensor_o, dim_changes_i, dim_transform, tensor_constraint)

    def calc_dim_mapping(self) -> bool:
        """Calculate the dimension change map between input and output tensor"""

        pre_tensors = self.pre_tensors(non_constant=True)

        # input 的维度数量必须相同，否则需要创建一个子类单独实现
        input_dim_num = len(pre_tensors[0].shape)
        for dim_change_i in range(input_dim_num):
            for tensor_i in self.pre_tensors(non_constant=True):
                fill_tensor_by_dim_changes(tensor_i, [dim_change_i])

            for dim_changes in self.calc_dim_changes():
                dim_changes_o, tensor_o = dim_changes
                for dim_change_o in dim_changes_o:
                    id_o = id(tensor_o)

                    for tensor_i in self.pre_tensors(non_constant=True):
                        id_i = id(tensor_i)
                        self.forward_dim_mapping[id_i][dim_change_i][id_o].add(dim_change_o)

                        if len(tensor_o.shape) > 0:
                            self.backward_dim_mapping[id_o][dim_change_o][id_i].add(dim_change_i)

        return True

    def init_dim_mapping(self) -> bool:
        # Init the dimension change map between input and output tensor
        # TODO：use cache to speed up
        if len(self.forward_dim_mapping) > 0:
            return True

        # this is a constant or I/O node
        if len(self.pre_tensors()) == 0 or len(self.next_tensors()) == 0:
            return False

        pre_tensors = self.pre_tensors(non_constant=True)
        nxt_tensors = self.next_tensors(non_constant=True)

        for tensor_i in pre_tensors:
            self.forward_dim_mapping[id(tensor_i)] = OrderedDict()
            for i in range(len(tensor_i.shape)):
                self.forward_dim_mapping[id(tensor_i)][i] = OrderedDict()
                for tensor_o in nxt_tensors:
                    self.forward_dim_mapping[id(tensor_i)][i][id(tensor_o)] = set()

        for tensor_o in nxt_tensors:
            self.backward_dim_mapping[id(tensor_o)] = OrderedDict()
            for i in range(len(tensor_o.shape)):
                self.backward_dim_mapping[id(tensor_o)][i] = OrderedDict()
                for tensor_i in pre_tensors:
                    self.backward_dim_mapping[id(tensor_o)][i][id(tensor_i)] = set()

        if not self.calc_dim_mapping():
            return False

        return True

    def print_dim_mapping(self):
        mapping_str = []

        mappings = OrderedDict({**self.forward_dim_mapping, **self.backward_dim_mapping})
        for id_1, v1 in mappings.items():
            name_1 = self.tensor_id_to_str[id_1]
            for dim_1, v2 in v1.items():
                for id_2, dim_2 in v2.items():
                    name_2 = self.tensor_id_to_str[id_2]
                    format_str = f"{name_1}:{dim_1}->{name_2}:{dim_2}"
                    mapping_str.append(format_str)

        log.debug("\n".join(mapping_str))
        return mapping_str

    def dim_choose(self, tensor_changes: typing.Dict[int, typing.List]):
        """When multiple dimensions are variable, select one of the dimensions.
        The selection method of each operator and pruning algorithm may be different"""

        return True

    def dim_choose_traversal(
        self,
        modifiers: typing.List,
        tensor_choices: typing.Dict[int, typing.List],
        tensor: torch.Tensor,
    ):
        """Propagate the selected dimension to all relevant operators"""

        if self in modifiers:
            return True

        if self.constant_node:
            return True

        dim_choice = tensor_choices[id(tensor)]

        changed = False
        dim_changes = self.dim_changes_info.get_tensor_changes(tensor)
        for dim_change in dim_changes:
            if dim_choice != dim_change:
                changed = True
                break

        # dim choose 和 dim_change 完全相同，说明此tensor完全未发生变化，无需传播
        if not changed:
            return True

        # 根据dim choose重新计算input、output的dim change
        tensor_choices_cur = self.calc_dim_choices(tensor, dim_choice)
        if len(tensor_choices_cur) == 0 and len(dim_choice) > 0:
            return True

        # 根据dim choose计算出的dim change存在冲突（例如input_0只支持裁剪dim=0，而input_1只支持裁剪dim=1)
        if self.dim_changes_info.is_changes_conflict(tensor_choices_cur):
            log.warning(
                f'[{self.unique_name()}][{self.tensor_id_to_str[id(tensor)]}][{dim_choice}] dim choose conflict'
            )
            return False

        modifiers.append(self)
        tensor_choices.update(tensor_choices_cur)

        valid = True
        for m in self.pre_modifiers():
            if not valid:
                break
            for t in self.pre_tensors(m):
                if id(t) not in tensor_choices:
                    continue

                if not m.dim_choose_traversal(modifiers, tensor_choices, t):
                    valid = False
                    break

        if valid:
            for m in self.next_modifiers():
                if not valid:
                    break
                for t in self.next_tensors(m):
                    if id(t) not in tensor_choices:
                        continue

                    if not m.dim_choose_traversal(modifiers, tensor_choices, t):
                        valid = False
                        break
        if not valid:
            modifiers.remove(self)
            for tensor_id in tensor_choices_cur.keys():
                if tensor_id in tensor_choices.keys():
                    del tensor_choices[tensor_id]

            return False

        return True

    def calc_dim_choices(self, tensor, dim_choose: typing.List) -> typing.Dict[int, typing.List]:
        """Select the dimension of the current operator according to the dimension selection of other operators"""

        # For the identity operator, the dim_choose of all tensors is the same
        tensor_choices = {}

        if not self.init_dim_mapping():
            return tensor_choices

        if self.is_pre_tensor(tensor):
            if len(dim_choose) > 0:
                for t in self.pre_tensors(non_constant=True):
                    tensor_choices[id(t)] = dim_choose

            dim_choices_o = OrderedDict()
            for i in dim_choose:
                choices = self.forward_dim_mapping[id(tensor)][i]
                for tid, choice in choices.items():
                    if tid not in dim_choices_o:
                        dim_choices_o[tid] = set()

                    dim_choices_o[tid].update(choice)

            for tid, choice in dim_choices_o.items():
                # Even the empty set must be passed forward, otherwise the selection between nodes may be inconsistent
                tensor_choices[tid] = list(choice)

        elif self.is_nxt_tensor(tensor):
            if len(dim_choose) > 0:
                for t in self.next_tensors(non_constant=True):
                    tensor_choices[id(t)] = dim_choose

            dim_choices_i = OrderedDict()
            for i in dim_choose:
                choices = self.backward_dim_mapping[id(tensor)][i]
                for tid, choice in choices.items():
                    if tid not in dim_choices_i:
                        dim_choices_i[tid] = set()

                    dim_choices_i[tid].update(choice)

            for tid, choice in dim_choices_i.items():
                tensor_choices[tid] = list(choice)
        else:
            assert False

        return tensor_choices


class PaddingModifier(Modifier):
    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        args_split = self.module().args_template.split(",")
        args_split[-1] = "-1"
        args_template = ",".join(args_split)
        self.module().args_template = args_template

        changes = self.calc_dim_changes()

        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        for change in changes:
            dim_change_o, tensor_o = change

            if dim_change_o:
                self.dim_changes_info.update_o(center, tensor_o, dim_change_o)
                for m in self.next_modifiers(tensor_o):
                    m.dim_change_forward(center, tensor_o, dim_change_o, dim_transform, None)


class PoolingModifier(Modifier):
    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        assert dim_changes_i == [1], "Pooling2D only support change channel dimension."

        if isinstance(self.node.module, nn.Module):
            output = self.node.module(self.pre_tensors()[0])
            tensor_o = self.next_tensors()[0]
            tensor_o.data.copy_(output.data)
        else:
            changes = self.calc_dim_changes()
            tensor_o = changes[0][1]

        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        self.dim_changes_info.update_o(center, tensor_o, dim_changes_i)

        # padding will change index info
        fill_tensor_by_dim_changes(self.next_tensors()[0], dim_changes_i)

        constraint_i = self.dim_changes_info.constraints_i[dim_changes_i[0]][center.unique_name()][0]
        transform = OrderedDict()
        for i in range(len(constraint_i)):
            transform[i] = constraint_i[i]

        for m in self.next_modifiers(tensor_o):
            m.dim_change_forward(center, tensor_o, dim_changes_i, transform, None)


class PReLUChannelModifier(Modifier):
    def register_mask(self, modifiers, importance, sparsity):
        pruned_idx, sparsity = self.get_pruned_idx(modifiers)
        self.dim_changes_info.pruned_idx_i = pruned_idx
        self.dim_changes_info.pruned_idx_o = pruned_idx

        if len(pruned_idx) > 0:
            remove_idx = self.dim_changes_info.pruned_idx_i
            self.masker().set_in_remove_idx(remove_idx)
            self.masker().set_ot_remove_idx(remove_idx)

            self.weight_mask["weight"][remove_idx] = 0

            self.masker().register_mask("weight", self.weight_mask["weight"])

    def modify_input(self, remove_idx):
        bn = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[0])], remove_idx)

        if bn.weight.shape[0] != len(preserve_idx):
            log.info(f'[PRELU] {self.unique_name()}: channel {bn.num_parameters} -> {len(preserve_idx)}')
            bn.weight = torch.nn.Parameter(bn.weight[preserve_idx])
            bn.num_parameters = len(preserve_idx)


class BatchNormChannelModifier(Modifier):
    def __init__(self, node: TraceNode):
        super(BatchNormChannelModifier, self).__init__(node)
        self.prunable = True

    def register_mask(self, modifiers, importance, sparsity):
        pruned_idx, sparsity = self.get_pruned_idx(modifiers)

        self.dim_changes_info.pruned_idx_i = pruned_idx
        self.dim_changes_info.pruned_idx_o = pruned_idx
        if len(pruned_idx) > 0:
            remove_idx = self.dim_changes_info.pruned_idx_i
            self.masker().set_in_remove_idx(remove_idx)
            self.masker().set_ot_remove_idx(remove_idx)

            self.weight_mask["weight"][remove_idx] = 0
            self.bias_mask["bias"] = self.weight_mask["weight"]

            self.masker().register_mask("weight", self.weight_mask["weight"])
            self.masker().register_mask("bias", self.bias_mask["bias"])

    def modify_input(self, remove_idx):
        bn = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[0])], remove_idx)

        if bn.weight.shape[0] != len(preserve_idx):
            while self.graph_modifier.bn_compensation:
                if len(self.next_modifiers()) != 1:
                    break

                if len(self.next_modifiers()[0].next_modifiers()) != 1:
                    break

                act = self.next_modifiers()[0].module()
                conv = self.next_modifiers()[0].next_modifiers()[0].module()

                if isinstance(act, nn.Module):
                    if type(act) not in [
                        nn.ReLU,
                        nn.ReLU6,
                        nn.LeakyReLU,
                        nn.Sigmoid,
                        nn.Tanh,
                        nn.Hardsigmoid,
                        nn.Hardtanh,
                        nn.Hardswish,
                        nn.LogSigmoid,
                    ]:
                        log.debug(f"unsupported activation for bn compensation: {type(act)}")
                        break

                if isinstance(act, TraceNode):
                    if self.next_modifiers()[0].node.kind() not in [
                        'relu',
                        'relu6',
                        'leaky_relu',
                        'sigmoid',
                        'tanh',
                        'hardsigmoid',
                        'hardtanh',
                        'hardswish',
                        'logsigmoid',
                    ]:
                        log.debug(f"unsupported activation for bn compensation: {self.next_modifiers()[0].node.kind()}")
                        break

                if type(conv) is not torch.nn.Conv2d:
                    break

                if conv.groups == 1:
                    with torch.no_grad():
                        bias = torch.tensor(bn.bias)
                        activation_bias = act(bias)
                        fuse_weight = torch.sum(conv.weight, dim=[2, 3])
                        bn_bias = fuse_weight * activation_bias
                        bn_bias = bn_bias[:, [True if i in remove_idx else False for i in range(bn_bias.shape[1])]]
                        bn_bias = torch.sum(bn_bias, dim=[1])
                        if conv.bias is None:
                            conv.bias = torch.nn.Parameter(bn_bias)
                        else:
                            conv.bias = torch.nn.Parameter(conv.bias + bn_bias)
                break

            log.info(f'[BN] {self.unique_name()}: channel {bn.num_features} -> {len(preserve_idx)}')
            bn.weight = torch.nn.Parameter(bn.weight[preserve_idx])
            bn.bias = torch.nn.Parameter(bn.bias[preserve_idx])
            bn.register_buffer('running_mean', bn.running_mean[preserve_idx])
            bn.register_buffer('running_var', bn.running_var[preserve_idx])
            bn.num_batches_tracked = bn.num_batches_tracked.zero_()
            bn.num_features = len(preserve_idx)


class ReIndexModifier(Modifier):
    """Only change the shape/layout of the tensor without changing the value in it，such as Reshape, Transpose,
    Permute, View, Expand, Flatten, Split and other operators"""

    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        changes = self.calc_dim_changes()

        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        for change in changes:
            dim_change_o, tensor_o = change

            if dim_change_o:
                self.dim_changes_info.update_o(center, tensor_o, dim_change_o)
                for m in self.next_modifiers(tensor_o):
                    m.dim_change_forward(center, tensor_o, dim_change_o, dim_transform, None)


class SplitModifier(ReIndexModifier):
    def __init__(self, node: TraceNode):
        super(SplitModifier, self).__init__(node)

        # TODO：Get pruning information from center node
        self.prunable = True

        self.split_dict = OrderedDict()
        self.split_dim = self.get_split_dim(self.node)

        start = end = 0
        for t in self.node.next_tensors:
            end += t.shape[self.split_dim]
            for n in self.node.next_nodes:
                for t_ in n.prev_tensors:
                    if torch.equal(t, t_):
                        self.split_dict[n.unique_name] = (start, end)
            start = end

        self.ot_channel = [t.shape[self.split_dim] for t in self.node.next_tensors]

    def get_split_dim(self, node):
        args_parsed = node.module.args_parsed
        if len(args_parsed) > 2:
            split_dim = args_parsed[-1]
            split_dim = split_dim[split_dim.find('=') + 1 :]
            split_dim = int(split_dim)
            return split_dim

        return 0

    def apply_mask(self, modifiers):
        # Input parameters also need to be regenerated after pruning
        remove_idx = self.dim_changes_info.pruned_idx_i
        args_parsed = self.node.module.args_parsed_origin

        if len(args_parsed) > 1:
            if type(args_parsed[1]) == list:
                ch = [int(i) for i in args_parsed[1]]
                ch_new = []

                for k, v in self.split_dict.items():
                    origin_ch = [i for i in range(v[0], v[1])]

                    for idx in remove_idx:
                        if idx in origin_ch:
                            origin_ch.remove(idx)

                    ch_new.append(len(origin_ch))

                for i in range(len(ch)):
                    # Each subgraph only deletes its corresponding channel
                    ch[i] = str(min(ch[i], ch_new[i]))

                self.args_parsed()[1] = ch
            elif args_parsed[1].isdigit():
                ch = self.ot_channel[0] - len(remove_idx) // len(self.ot_channel)
                self.args_parsed()[1] = str(ch)
            elif args_parsed[1] == '{}':
                return True
            else:
                assert False

            self.node.module.args_to_string(deepcopy(self.args_parsed()))


class ReshapeModifier(ReIndexModifier):
    def __init__(self, node: TraceNode):
        super(ReshapeModifier, self).__init__(node)
        self.input_tensor = self.pre_tensors()[0]
        self.output_tensor = self.next_tensors()[0]
        self.original_shape = list(self.pre_tensors()[0].shape)
        self.changed_shape = list(self.next_tensors()[0].shape)
        self.input_modify_dim = -1
        self.output_modify_dim = -1

        # Multiple reshapes may be eliminated, so the constraints of reshapes are uncertain and cannot be used as leaves
        # self.prunable = True

        # Unify the input parameters of reshape and view into the same format
        if self.node.kind() in ['reshape', 'view']:
            args_parsed = self.args_parsed()
            args = args_parsed[1:] if type(args_parsed[1]) is not list else args_parsed[1]
            self.node.module.args_parsed = [args_parsed[0], args]

            args_parsed_origin = self.node.module.args_parsed_origin
            args = args_parsed_origin[1:] if type(args_parsed_origin[1]) is not list else args_parsed_origin[1]
            self.node.module.args_parsed_origin = [args_parsed_origin[0], args]

    def dim_choose(self, tensor_changes: typing.Dict[int, typing.List]) -> bool:
        dim_changes_i = tensor_changes.get(id(self.pre_tensors()[0]), None)
        if dim_changes_i is None:
            dim_changes_i = self.dim_changes_info.merge_i()

        if len(dim_changes_i) > 1:
            # Prefer the rear axis to reduce the range of dependent transfer
            for dim_choose_i in reversed(sorted(dim_changes_i)):
                modifiers = [self]

                self.init_dim_mapping()

                if self.original_shape[dim_choose_i] <= 2:
                    continue

                tensor_changes[id(self.input_tensor)] = [dim_choose_i]
                tensor_changes[id(self.output_tensor)] = list(
                    self.forward_dim_mapping[id(self.input_tensor)][dim_choose_i][id(self.output_tensor)]
                )

                valid = True
                for m in self.pre_modifiers(self.pre_tensors()[0]):
                    valid = m.dim_choose_traversal(modifiers, tensor_changes, self.pre_tensors()[0])
                    if not valid:
                        break

                if valid:
                    for m in self.next_modifiers(self.next_tensors()[0]):
                        valid = m.dim_choose_traversal(modifiers, tensor_changes, self.next_tensors()[0])
                        if not valid:
                            break

                if valid:
                    return True
                else:
                    if id(self.input_tensor) in tensor_changes:
                        del tensor_changes[id(self.input_tensor)]
                    if id(self.output_tensor) in tensor_changes:
                        del tensor_changes[id(self.output_tensor)]
            return False
        else:
            return True

    def apply_mask(self, modifiers):
        # reshape does not need to register the mask, but if the parameters are constant, you need to modify the
        # parameters, otherwise it will cause abnormal inference after pruning
        choice = self.dim_changes_info.get_tensor_choices(self.next_tensors()[0])
        if choice is None:
            return

        output_modify_dim = choice[0]

        args_origin = self.node.module.args_parsed_origin[1]

        if self.node.type() == 'reshape':
            if len(args_origin) > output_modify_dim and args_origin[output_modify_dim].isdigit():
                if int(args_origin[output_modify_dim]) == -1:
                    return

                pruned_idx, sparsity = self.get_pruned_idx(modifiers)

                output_shape = deepcopy(self.changed_shape)
                output_shape[output_modify_dim] = int(output_shape[output_modify_dim] * sparsity)
                self.args_parsed()[1] = [str(i) for i in output_shape]
                self.node.module.args_to_string(deepcopy(self.args_parsed()))

                self.mask_applied = True
            else:
                return

        elif self.node.type() == 'view':
            if args_origin[output_modify_dim].isdigit():
                if int(args_origin[output_modify_dim]) == -1:
                    return
                pruned_idx, sparsity = self.get_pruned_idx(modifiers)

                output_shape = deepcopy(self.changed_shape)
                output_shape[output_modify_dim] = output_shape[output_modify_dim] - int(
                    output_shape[output_modify_dim] * sparsity
                )
                if self.args_parsed()[1][output_modify_dim].isdigit():
                    self.args_parsed()[1][output_modify_dim] = str(output_shape[output_modify_dim])
                self.node.module.args_to_string(deepcopy(self.args_parsed()))

                self.mask_applied = True
            else:
                return


class CatModifier(Modifier):
    def __init__(self, node: TraceNode):
        super(CatModifier, self).__init__(node)

        if self.node.module.kwargs.get('dim', None) is not None:
            self.dim = self.node.module.kwargs['dim']
        else:
            if len(self.args_parsed()) > 1:
                self.dim = int(self.args_parsed()[1])
            else:
                self.dim = 0

    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        offset = 0

        for t in self.pre_tensors():
            if id(t) != id(tensor):
                offset += t.shape[dim_changes_i[0]]
            else:
                break

        # In the case of batch cat, there will be dependencies between multiple input tensors
        if self.dim not in dim_changes_i:
            for t in self.pre_tensors():
                if id(t) != id(tensor):
                    if list(t.shape) == list(tensor.shape):
                        t.data.copy_(tensor.clone().data)
                    else:
                        # Different shapes between tensors
                        if tensor.shape[self.dim] >= t.shape[self.dim]:
                            idx_tensor = torch.tensor([idx for idx in range(t.shape[self.dim])])
                            select_tensor = torch.index_select(tensor, self.dim, idx_tensor)
                            t.data.copy_(select_tensor.data)
                        else:
                            idx_tensor = torch.tensor([idx for idx in range(tensor.shape[self.dim])])
                            t.data.index_copy_(self.dim, idx_tensor, tensor)

        dim_changes_o, tensor_o = self.calc_dim_changes()[0]

        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )
        self.dim_changes_info.update_o(center, tensor_o, dim_changes_o)

        if offset == 0 or dim_transform is None:
            new_dim_transform = dim_transform
        else:
            new_dim_transform = {}
            for key, value in dim_transform.items():
                new_dim_transform[key + offset] = value

        for m in self.next_modifiers(tensor_o):
            m.dim_change_forward(center, tensor_o, dim_changes_o, new_dim_transform, None)


class MatMulModifier(Modifier):
    def __init__(self, node: TraceNode):
        super(MatMulModifier, self).__init__(node)
        self.prunable = True

    def dim_choose(self, tensor_changes: typing.Dict[int, typing.List]) -> bool:
        for i in [0, 1]:
            input_tensor = self.pre_tensors()[i]
            output_tensor = self.next_tensors()[0]

            dim_changes_i = tensor_changes.get(id(input_tensor), None)
            if dim_changes_i is None:
                dim_changes_i = self.dim_changes_info.merge_t(input_tensor)

            if len(dim_changes_i) > 1:
                valid = True

                for dim_choose_i in sorted(dim_changes_i):
                    modifiers = [self]

                    tensor_changes[id(input_tensor)] = [dim_choose_i]

                    for m in self.pre_modifiers(self.pre_tensors()[0]):
                        valid = m.dim_choose_traversal(modifiers, tensor_changes, self.pre_tensors()[0])
                        if not valid:
                            break

                    if len(self.dim_changes_info.merge_o()) > 1:
                        self.init_dim_mapping()
                        tensor_changes[id(output_tensor)] = list(
                            self.forward_dim_mapping[id(input_tensor)][dim_choose_i][id(output_tensor)]
                        )
                        if valid:
                            for m in self.next_modifiers(self.next_tensors()[0]):
                                valid = m.dim_choose_traversal(modifiers, tensor_changes, self.next_tensors()[0])
                                if not valid:
                                    break

                    if valid:
                        break
                    else:
                        del tensor_changes[id(input_tensor)]
                        del tensor_changes[id(output_tensor)]

                if not valid:
                    return False

            else:
                continue

        return True

    def calc_dim_mapping(self) -> bool:
        (input_0, input_1) = self.pre_tensors()
        output = self.next_tensors()[0]

        id_i0 = id(input_0)
        id_i1 = id(input_1)
        id_o = id(output)

        input_dim0 = len(input_0.shape)
        input_dim1 = len(input_1.shape)

        for i in range(input_dim0 - 1):
            self.forward_dim_mapping[id_i0][i][id_o].add(i)
            self.backward_dim_mapping[id_o][i][id_i0].add(i)

        if input_dim1 >= 2:
            dim_mapping = [i for i in range(input_dim0)][-input_dim1:]
            for i in range(input_dim1):
                if i != input_dim1 - 2:
                    dim = dim_mapping[i]
                    self.forward_dim_mapping[id_i1][i][id_o].add(dim)
                    self.backward_dim_mapping[id_o][dim][id_i1].add(i)

        return True

    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        if id(tensor) == id(self.pre_tensors()[0]):
            other_tensor = self.pre_tensors()[1]
            other_tensor[:] = 0
            if len(other_tensor.shape) >= 2:
                idx = [
                    0 if i == len(other_tensor.shape) - 2 else slice(None, None, None)
                    for i in range(len(other_tensor.shape))
                ]
            else:
                idx = [
                    0 if i == len(other_tensor.shape) - 2 else slice(None, None, None)
                    for i in range(len(other_tensor.shape))
                ]
            torch.Tensor.__setitem__(other_tensor, tuple(idx), 1)
        else:
            other_tensor = self.pre_tensors()[0]
            other_tensor[:] = 0
            idx = [
                0 if i == len(other_tensor.shape) - 1 else slice(None, None, None)
                for i in range(len(other_tensor.shape))
            ]
            torch.Tensor.__setitem__(other_tensor, idx, 1)

        changes = self.calc_dim_changes()

        for change in changes:
            dim_change_o, tensor_o = change

            if dim_change_o:
                self.dim_changes_info.update_o(center, tensor_o, dim_change_o)
                for m in self.next_modifiers(tensor_o):
                    m.dim_change_forward(center, tensor_o, dim_change_o, dim_transform, None)


class LinearChannelModifier(Modifier):
    def __init__(self, node: TraceNode):
        super().__init__(node)
        self.input_tensor = self.pre_tensors()[0]
        self.output_tensor = self.next_tensors()[0]
        self.dim_c = len(self.input_tensor.shape) - 1
        self.prunable = True

    def calc_idx_group(self):
        dim_choice_i = self.dim_changes_info.get_tensor_choices(self.input_tensor)
        if dim_choice_i is None:
            dim = self.dim_c
        elif len(dim_choice_i) == 1:
            dim = dim_choice_i[0]
        else:
            assert False

        self.dim_changes_info.groups_i = [set([i for i in range(self.input_tensor.shape[dim])])]
        self.dim_changes_info.groups_o = [set([i for i in range(self.output_tensor.shape[dim])])]

    def calc_dim_mapping(self) -> bool:
        for i in range(len(list(self.input_tensor.shape))):
            self.forward_dim_mapping[id(self.input_tensor)][i][id(self.output_tensor)] = {i}
            self.backward_dim_mapping[id(self.output_tensor)][i][id(self.input_tensor)] = {i}

        return True

    def register_mask(self, modifiers, importance, sparsity):
        if self.dim_changes_info.pruned_idx_i:
            remove_idx = self.dim_changes_info.pruned_idx_i
            self.weight_mask["weight"][:, remove_idx] = 0
            self.masker().set_in_remove_idx(remove_idx)

        if self.dim_changes_info.pruned_idx_o:
            remove_idx = self.dim_changes_info.pruned_idx_o
            self.weight_mask["weight"][remove_idx, :] = 0
            self.masker().set_ot_remove_idx(remove_idx)

            bias_mask = self.bias_mask.get("bias", None)
            if bias_mask is not None:
                bias_mask[remove_idx] = 0
                self.masker().register_mask("bias", bias_mask)

        self.masker().register_mask("weight", self.weight_mask["weight"])

    def modify_input(self, remove_idx):
        if self.dim_changes_info.get_tensor_choices(self.input_tensor) != [self.dim_c]:
            return

        linear = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[1])], remove_idx)

        if linear.weight.shape[1] != len(preserve_idx):
            log.info(f'[FC] {self.unique_name()}: input {linear.in_features} -> {len(preserve_idx)}')
            linear.weight = torch.nn.Parameter(linear.weight[:, preserve_idx])
            linear.in_features = len(preserve_idx)

    def modify_output(self, remove_idx):
        if self.dim_changes_info.get_tensor_choices(self.output_tensor) != [self.dim_c]:
            return

        log.debug(f'[FC] {self.unique_name()}: remove_idx = {remove_idx}')

        linear = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[0])], remove_idx)

        if linear.weight.shape[0] != len(preserve_idx):
            log.info(f'[FC] {self.unique_name()}: output {linear.out_features} -> {len(preserve_idx)}')
            linear.weight = torch.nn.Parameter(linear.weight[preserve_idx, :])
            linear.out_features = len(preserve_idx)

            if linear.bias is not None:
                linear.bias = torch.nn.Parameter(linear.bias[preserve_idx])

    def dim_choose(self, tensor_changes: typing.Dict[int, typing.List]) -> bool:
        id_i = id(self.input_tensor)
        id_o = id(self.output_tensor)

        dim_changes_i = tensor_changes.get(id_i, None)
        dim_choice_o = tensor_changes.get(id_o, None)

        if dim_changes_i is None:
            dim_changes_i = self.dim_changes_info.merge_i()

        if len(dim_changes_i) > 1:
            for dim_choose_i in reversed(sorted(dim_changes_i)):
                modifiers = [self]

                self.init_dim_mapping()

                tensor_changes[id_i] = [dim_choose_i]

                if dim_choice_o is not None:
                    dim_choose_i_mapping = list(self.backward_dim_mapping[id_o][dim_choice_o[0]][id_i])
                    if dim_choose_i_mapping and dim_choose_i_mapping != tensor_changes[id_i]:
                        continue

                tensor_changes[id_o] = [dim_choose_i]

                valid = True
                for m in self.pre_modifiers(self.pre_tensors()[0]):
                    valid = m.dim_choose_traversal(modifiers, tensor_changes, self.pre_tensors()[0])
                    if not valid:
                        break

                if valid > 0:
                    for m in self.next_modifiers(self.next_tensors()[0]):
                        valid = m.dim_choose_traversal(modifiers, tensor_changes, self.next_tensors()[0])
                        if not valid:
                            break

                if valid:
                    return True
                else:
                    if id_i in tensor_changes.keys():
                        del tensor_changes[id_i]
                    if id_o in tensor_changes.keys():
                        del tensor_changes[id_o]
            return False
        else:
            return True

    def change_dimension(self) -> bool:
        dim_changes_o = [self.dim_c]

        fill_tensor_by_dim_changes(self.output_tensor, dim_changes_o)

        tensor_constraint = self.dim_changes_info.update_o(
            self, self.next_tensors()[0], dim_changes_o, update_constraint=True
        )

        for m in self.next_modifiers():
            m.dim_change_forward(self, self.next_tensors()[0], dim_changes_o, None, tensor_constraint)

        return True

    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        # Full connection can isolate changes in dim_c dimension
        dim_changes_o = deepcopy(dim_changes_i)
        if self.dim_c in dim_changes_o:
            dim_changes_o.remove(self.dim_c)

        # Dimension changes other than dim_c need to be passed to downstream nodes
        if len(dim_changes_o) > 0:
            self.dim_changes_info.update_o(center, self.next_tensors()[0], dim_changes_o)

            fill_tensor_by_dim_changes(self.output_tensor, dim_changes_o)

            constraint_i = self.dim_changes_info.constraints_i[dim_changes_o[0]][center.unique_name()][0]
            transform = OrderedDict()
            for i in range(len(constraint_i)):
                transform[i] = constraint_i[i]

            for m in self.next_modifiers():
                m.dim_change_forward(center, self.next_tensors()[0], dim_changes_o, transform, None)


class RNNChannelModifier(Modifier):
    def __init__(self, node: TraceNode):
        super().__init__(node)
        self.input_tensor = self.pre_tensors()[0]
        self.output_tensor = self.next_tensors()[0]
        self.dim_c = len(self.input_tensor.shape) - 1
        self.bidirectional = self.module().bidirectional
        self.prunable = True

    def reset_mask(self):
        self.weight_mask.clear()
        self.bias_mask.clear()

        for n, p in self.module().named_parameters():
            if n.startswith('weight'):
                self.weight_mask[n] = torch.ones_like(p)
            elif n.startswith('bias'):
                self.bias_mask[n] = torch.ones_like(p)

    def calc_idx_group(self):
        dim_choice_i = self.dim_changes_info.get_tensor_choices(self.input_tensor)
        if dim_choice_i is None:
            dim = self.dim_c
        elif len(dim_choice_i) == 1:
            dim = dim_choice_i[0]
        else:
            assert False

        self.dim_changes_info.groups_i = [set([i for i in range(self.input_tensor.shape[dim])])]
        if self.bidirectional:
            output_idx = [i for i in range(self.output_tensor.shape[self.dim_c])]
            output_chunk = len(output_idx) // 2
            self.dim_changes_info.groups_o = [
                set(output_idx[i : i + output_chunk]) for i in range(0, len(output_idx), output_chunk)
            ]
        else:
            self.dim_changes_info.groups_o = [set([i for i in range(self.output_tensor.shape[dim])])]

    def calc_dim_mapping(self) -> bool:
        for i in range(len(list(self.input_tensor.shape))):
            self.forward_dim_mapping[id(self.input_tensor)][i][id(self.output_tensor)] = {i}
            self.backward_dim_mapping[id(self.output_tensor)][i][id(self.input_tensor)] = {i}

        return True

    def tile_indices_with_gate_size(self, indices, gate_size, offset):
        broadcasted = [indices] * gate_size
        return [offset * idx + i for idx, x in enumerate(broadcasted) for i in x]

    def split_indices_with_directions(self, indices, offset, num_directions):
        split_pos = len(indices) // num_directions
        idx_bwd = [i - offset for i in indices[split_pos:]]
        idx_fwd = indices[:split_pos]
        return idx_fwd, idx_bwd

    def register_mask(self, modifiers, importance, sparsity):
        gs = rnn_gate_size(self.module())
        num_directions = 2 if self.module().bidirectional else 1
        has_proj = hasattr(self.module(), 'proj_size') and self.module().proj_size > 0

        if self.dim_changes_info.pruned_idx_i:
            remove_idx = self.dim_changes_info.pruned_idx_i
            self.weight_mask['weight_ih_l0'][:, remove_idx] = 0
            self.masker().set_in_remove_idx(remove_idx)

        if self.dim_changes_info.pruned_idx_o:
            if has_proj:
                u_name = self.unique_name()
                hu_name = f'{u_name}:h'

                remove_idx = []
                idx_num = len(importance[hu_name])
                remove_num = int(sparsity[u_name] * len(importance[hu_name]))
                if self.bidirectional:
                    idx_num //= 2
                    remove_num //= 2
                    remove_idx += get_smallest_k(importance[hu_name][:idx_num], remove_num)
                    remove_idx += get_smallest_k(importance[hu_name][idx_num:], remove_num, offset=idx_num)
                else:
                    remove_idx += get_smallest_k(importance[hu_name], remove_num)
                remove_idx_proj = self.dim_changes_info.pruned_idx_o
            else:
                remove_idx = self.dim_changes_info.pruned_idx_o
                remove_idx_proj = None

            remove_idx_bwd = None
            remove_idx_fwd = None
            remove_idx_proj_bwd = None
            remove_idx_proj_fwd = None
            if num_directions > 1:
                offset = self.module().hidden_size
                remove_idx_fwd, remove_idx_bwd = self.split_indices_with_directions(remove_idx, offset, num_directions)
                if remove_idx_proj is not None:
                    offset = self.module().proj_size
                    remove_idx_proj_fwd, remove_idx_proj_bwd = self.split_indices_with_directions(
                        remove_idx_proj, offset, num_directions
                    )
                    assert len(remove_idx_proj_fwd) == len(remove_idx_proj_bwd)

            if gs > 1:
                offset = self.module().hidden_size
                if num_directions > 1:
                    remove_idx_bwd_gs = self.tile_indices_with_gate_size(remove_idx_bwd, gs, offset)
                    remove_idx_fwd_gs = self.tile_indices_with_gate_size(remove_idx_fwd, gs, offset)
                else:
                    remove_idx_gs = self.tile_indices_with_gate_size(remove_idx, gs, offset)

            for n in self.weight_mask:
                remove_idx_r = remove_idx
                remove_idx_c = remove_idx
                remove_idx_pc = None
                if num_directions > 1:
                    if n.endswith('_reverse'):
                        if gs > 1:
                            remove_idx_r = remove_idx_bwd_gs
                        else:
                            remove_idx_r = remove_idx_bwd
                        remove_idx_c = remove_idx_bwd
                        if has_proj:
                            remove_idx_pc = remove_idx_proj_bwd
                    else:
                        if gs > 1:
                            remove_idx_r = remove_idx_fwd_gs
                        else:
                            remove_idx_r = remove_idx_fwd
                        remove_idx_c = remove_idx_fwd
                        if has_proj:
                            remove_idx_pc = remove_idx_proj_fwd
                elif gs > 1:
                    remove_idx_r = remove_idx_gs
                    remove_idx_pc = remove_idx_proj

                if n.startswith('weight_ih_l0'):
                    self.weight_mask[n][remove_idx_r, :] = 0
                elif n.startswith('weight_ih'):
                    self.weight_mask[n][remove_idx_r, :] = 0
                    if remove_idx_proj is None:
                        self.weight_mask[n][:, remove_idx] = 0
                    else:
                        self.weight_mask[n][:, remove_idx_proj] = 0
                    self.masker().register_mask(n, self.weight_mask[n])
                elif n.startswith('weight_hh'):
                    self.weight_mask[n][remove_idx_r, :] = 0
                    if remove_idx_pc is None:
                        self.weight_mask[n][:, remove_idx_c] = 0
                    else:
                        self.weight_mask[n][:, remove_idx_pc] = 0
                    self.masker().register_mask(n, self.weight_mask[n])
                elif n.startswith('weight_hr'):
                    if remove_idx_pc is not None:
                        self.weight_mask[n][remove_idx_pc, :] = 0
                    self.weight_mask[n][:, remove_idx_c] = 0
                    self.masker().register_mask(n, self.weight_mask[n])

            for n in self.bias_mask:
                if self.bias_mask[n] is None:
                    continue
                remove_idx_ = remove_idx
                if num_directions > 1:
                    if n.endswith('_reverse'):
                        if gs > 1:
                            remove_idx_ = remove_idx_bwd_gs
                        else:
                            remove_idx_ = remove_idx_bwd
                    else:
                        if gs > 1:
                            remove_idx_ = remove_idx_fwd_gs
                        else:
                            remove_idx_ = remove_idx_fwd
                elif gs > 1:
                    remove_idx_ = remove_idx_gs
                self.bias_mask[n][remove_idx_] = 0
                self.masker().register_mask(n, self.bias_mask[n])
            self.masker().set_ot_remove_idx(remove_idx)

            if remove_idx_proj is not None:
                self.masker().set_custom_remove_idx(remove_idx_proj)

        self.masker().register_mask('weight_ih_l0', self.weight_mask['weight_ih_l0'])

    def modify_input(self, remove_idx):
        rnn = self.node.module
        assert len(self.node.prev_tensors) == 1, 'RNNs with hidden state inputs are not supported'
        preserve_idx = complementary_list([i for i in range(self.weight_mask['weight_ih_l0'].shape[1])], remove_idx)

        if rnn.weight_ih_l0.shape[1] != len(preserve_idx):
            log.info(f'[RNN] {self.unique_name()}: input {rnn.input_size} -> {len(preserve_idx)}')

            rnn.weight_ih_l0 = torch.nn.Parameter(rnn.weight_ih_l0[:, preserve_idx])

            if rnn.bidirectional:
                rnn.weight_ih_l0_reverse = torch.nn.Parameter(rnn.weight_ih_l0_reverse[:, preserve_idx])

            rnn.input_size = len(preserve_idx)

    def modify_output(self, remove_idx):
        rnn = self.node.module

        log.debug(f'[RNN] {self.unique_name()}: remove_idx = {remove_idx}')

        num_directions = 2 if rnn.bidirectional else 1
        has_proj = hasattr(self.module(), 'proj_size') and self.module().proj_size > 0
        gs = rnn_gate_size(rnn)
        if num_directions > 1:
            offset = rnn.hidden_size
            remove_idx_fwd, remove_idx_bwd = self.split_indices_with_directions(remove_idx, offset, num_directions)

        if gs > 1:
            offset = rnn.hidden_size
            if num_directions > 1:
                remove_idx_bwd_gs = self.tile_indices_with_gate_size(remove_idx_bwd, gs, offset)
                remove_idx_fwd_gs = self.tile_indices_with_gate_size(remove_idx_fwd, gs, offset)
            else:
                remove_idx_gs = self.tile_indices_with_gate_size(remove_idx, gs, offset)

        remove_idx_proj = None
        if has_proj:
            remove_idx_proj = self.masker().custom_remove_idx
            if remove_idx_proj is not None:
                offset = rnn.proj_size
                remove_idx_proj_fwd, remove_idx_proj_bwd = self.split_indices_with_directions(
                    remove_idx_proj, offset, num_directions
                )

        for i in range(rnn.num_layers):
            for j in range(num_directions):
                suffix = '_reverse' if j > 0 else ''
                desc = f'layer{suffix} hidden #{i}'

                weight_ih = getattr(rnn, f'weight_ih_l{i}{suffix}')
                weight_hh = getattr(rnn, f'weight_hh_l{i}{suffix}')
                weight_hr = getattr(rnn, f'weight_hr_l{i}{suffix}', None)

                bias_ih = getattr(rnn, f'bias_ih_l{i}{suffix}', None)
                bias_hh = getattr(rnn, f'bias_hh_l{i}{suffix}', None)

                remove_idx_r = remove_idx
                remove_idx_c = remove_idx
                remove_idx_pc = None
                if num_directions > 1:
                    if j > 0:
                        if gs > 1:
                            remove_idx_r = remove_idx_bwd_gs
                        else:
                            remove_idx_r = remove_idx_bwd
                        remove_idx_c = remove_idx_bwd
                        if has_proj:
                            remove_idx_pc = remove_idx_proj_bwd
                    else:
                        if gs > 1:
                            remove_idx_r = remove_idx_fwd_gs
                        else:
                            remove_idx_r = remove_idx_fwd
                        remove_idx_c = remove_idx_fwd
                        if has_proj:
                            remove_idx_pc = remove_idx_proj_fwd
                elif gs > 1:
                    remove_idx_r = remove_idx_gs
                    remove_idx_pc = remove_idx_proj

                preserve_idx_ih_r = complementary_list(
                    [j for j in range(self.weight_mask[f'weight_ih_l{i}{suffix}'].shape[0])], remove_idx_r
                )
                preserve_idx_hh_r = complementary_list(
                    [j for j in range(self.weight_mask[f'weight_hh_l{i}{suffix}'].shape[0])], remove_idx_r
                )

                if weight_hr is None:
                    preserve_idx_hh_c = complementary_list(
                        [j for j in range(self.weight_mask[f'weight_hh_l{i}{suffix}'].shape[1])], remove_idx_c
                    )
                else:
                    preserve_idx_hh_c = complementary_list(
                        [j for j in range(self.weight_mask[f'weight_hh_l{i}{suffix}'].shape[1])], remove_idx_pc
                    )
                    preserve_idx_hr_c = complementary_list(
                        [j for j in range(self.weight_mask[f'weight_hr_l{i}{suffix}'].shape[1])], remove_idx_c
                    )

                preserve_idx_ih_c = None
                if i != 0 and preserve_idx_ih_c is None:
                    if weight_hr is not None:
                        preserve_idx_ih_c = complementary_list(
                            [j for j in range(self.weight_mask[f'weight_ih_l{i}{suffix}'].shape[1])], remove_idx_proj
                        )
                    else:
                        preserve_idx_ih_c = preserve_idx_ih_r
                        if num_directions > 1 or gs > 1:
                            preserve_idx_ih_c = complementary_list(
                                [j for j in range(self.weight_mask[f'weight_ih_l{i}{suffix}'].shape[1])], remove_idx
                            )

                if weight_ih.shape[0] != len(preserve_idx_ih_r):
                    if i != 0 and weight_ih.shape[1] != len(preserve_idx_ih_c):
                        desc_i = f'layer{suffix} input #{i}'
                        log.info(
                            f'[RNN] {self.unique_name()}: {desc_i} {weight_ih.shape[1]} -> {len(preserve_idx_ih_c)}'
                        )

                    log.info(f'[RNN] {self.unique_name()}: {desc} {rnn.hidden_size * gs} -> {len(preserve_idx_ih_r)}')

                    if i != 0:
                        new_w = weight_ih[preserve_idx_ih_r, :][:, preserve_idx_ih_c]
                        setattr(rnn, f'weight_ih_l{i}{suffix}', torch.nn.Parameter(new_w))
                    else:
                        setattr(rnn, f'weight_ih_l{i}{suffix}', torch.nn.Parameter(weight_ih[preserve_idx_ih_r, :]))

                    if bias_ih is not None:
                        setattr(rnn, f'bias_ih_l{i}{suffix}', torch.nn.Parameter(bias_ih[preserve_idx_ih_r]))

                desc = f'layer{suffix} output #{i}'
                if weight_hh.shape[0] != len(preserve_idx_hh_r) or weight_hh.shape[1] != len(preserve_idx_hh_c):
                    log.info(f'[RNN] {self.unique_name()}: {desc} {rnn.hidden_size * gs} -> {len(preserve_idx_hh_r)}')

                    if weight_hr is None:
                        setattr(
                            rnn,
                            f'weight_hh_l{i}{suffix}',
                            torch.nn.Parameter(weight_hh[preserve_idx_hh_r, :][:, preserve_idx_hh_c]),
                        )
                    else:
                        setattr(
                            rnn,
                            f'weight_hh_l{i}{suffix}',
                            torch.nn.Parameter(weight_hh[preserve_idx_hh_r, :][:, preserve_idx_hh_c]),
                        )
                        setattr(
                            rnn,
                            f'weight_hr_l{i}{suffix}',
                            torch.nn.Parameter(weight_hr[preserve_idx_hh_c, :][:, preserve_idx_hr_c]),
                        )

                    if bias_hh is not None:
                        setattr(rnn, f'bias_hh_l{i}{suffix}', torch.nn.Parameter(bias_hh[preserve_idx_hh_r]))

        if weight_hr is None:
            rnn.hidden_size = len(preserve_idx_hh_c)
        else:
            rnn.proj_size = len(preserve_idx_hh_c)
            rnn.hidden_size = len(preserve_idx_hr_c)

    def dim_choose(self, tensor_changes: typing.Dict[int, typing.List]) -> bool:
        id_i = id(self.input_tensor)
        id_o = id(self.output_tensor)

        dim_changes_i = tensor_changes.get(id_i, None)
        dim_choice_o = tensor_changes.get(id_o, None)

        if dim_changes_i is None:
            dim_changes_i = self.dim_changes_info.merge_i()

        if len(dim_changes_i) > 1:
            if self.dim_c not in dim_changes_i:
                return False

            dim_choose_i = self.dim_c
            modifiers = [self]

            self.init_dim_mapping()

            tensor_changes[id_i] = [dim_choose_i]

            if dim_choice_o is not None:
                dim_choose_i_mapping = list(self.backward_dim_mapping[id_o][dim_choice_o[0]][id_i])
                if dim_choose_i_mapping and dim_choose_i_mapping != tensor_changes[id_i]:
                    return False

            tensor_changes[id_o] = [dim_choose_i]

            valid = True
            for m in self.pre_modifiers(self.pre_tensors()[0]):
                valid = m.dim_choose_traversal(modifiers, tensor_changes, self.pre_tensors()[0])
                if not valid:
                    break

            if valid > 0:
                for m in self.next_modifiers(self.next_tensors()[0]):
                    valid = m.dim_choose_traversal(modifiers, tensor_changes, self.next_tensors()[0])
                    if not valid:
                        break

            if valid:
                return True
            else:
                if id_i in tensor_changes.keys():
                    del tensor_changes[id_i]
                if id_o in tensor_changes.keys():
                    del tensor_changes[id_o]
            return False
        else:
            return True

    def change_dimension(self) -> bool:
        dim_changes_o = [self.dim_c]

        fill_tensor_by_dim_changes(self.output_tensor, dim_changes_o)

        tensor_constraint = self.dim_changes_info.update_o(
            self, self.next_tensors()[0], dim_changes_o, update_constraint=True
        )

        for m in self.next_modifiers():
            m.dim_change_forward(self, self.next_tensors()[0], dim_changes_o, None, tensor_constraint)

        return True

    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        dim_changes_o = deepcopy(dim_changes_i)
        if self.dim_c in dim_changes_o:
            dim_changes_o.remove(self.dim_c)

        if len(dim_changes_o) > 0:
            log.error(f"[{self.unique_name()}] Modifying dimensions other than dim_c is temporarily not supported")
            assert False


class Conv2dChannelModifier(Modifier):
    def __init__(self, node: TraceNode):
        super().__init__(node)
        self.dim_n = 0
        self.dim_c = 1
        self.dim_h = 2
        self.dim_w = 3
        self.input_tensor = self.pre_tensors()[0]
        self.output_tensor = self.next_tensors()[0]
        self.prunable = True
        self.group = self.module().groups

    def calc_idx_group(self):
        if not is_dw_conv(self.module()):
            if self.group > 1:
                input_idx = [i for i in range(self.input_tensor.shape[self.dim_c])]
                output_idx = [i for i in range(self.output_tensor.shape[self.dim_c])]
                input_chunk = len(input_idx) // self.group
                output_chunk = len(output_idx) // self.group
                self.dim_changes_info.groups_i = [
                    set(input_idx[i : i + input_chunk]) for i in range(0, len(input_idx), input_chunk)
                ]
                self.dim_changes_info.groups_o = [
                    set(output_idx[i : i + output_chunk]) for i in range(0, len(output_idx), output_chunk)
                ]
            else:
                self.dim_changes_info.groups_i = [set([i for i in range(self.input_tensor.shape[self.dim_c])])]
                self.dim_changes_info.groups_o = [set([i for i in range(self.output_tensor.shape[self.dim_c])])]

    def init_dim_mapping(self) -> bool:
        if len(self.forward_dim_mapping) > 0:
            return True

        # Convolutional pruning is not allowed to modify dim_w, dim_h
        self.forward_dim_mapping[id(self.input_tensor)][self.dim_n][id(self.output_tensor)] = {self.dim_n}
        self.backward_dim_mapping[id(self.output_tensor)][self.dim_n][id(self.input_tensor)] = {self.dim_n}
        self.forward_dim_mapping[id(self.input_tensor)][self.dim_c][id(self.output_tensor)] = set()
        self.backward_dim_mapping[id(self.output_tensor)][self.dim_c][id(self.input_tensor)] = set()

        return True

    def register_mask(self, modifiers, importance, sparsity):

        if is_dw_conv(self.module()):
            remove_idx = self.dim_changes_info.pruned_idx_i
            self.weight_mask["weight"][remove_idx, :] = 0
            self.masker().set_in_remove_idx(remove_idx)
            self.masker().set_ot_remove_idx(remove_idx)

            bias_mask = self.bias_mask.get("bias", None)
            if bias_mask is not None:
                bias_mask[remove_idx] = 0
                self.masker().register_mask("bias", bias_mask)
        else:
            if self.dim_changes_info.pruned_idx_i:
                remove_idx = self.dim_changes_info.pruned_idx_i
                group = self.group
                remove_idx.sort()
                if group != 1:
                    num_g_out = self.weight_mask["weight"].shape[0] // group
                    weight_2 = self.weight_mask["weight"].shape[1]
                    start_in = end_in = 0
                    for i in range(group):
                        end_in += weight_2
                        g_remove_idx = []
                        for idx in remove_idx:
                            if start_in <= idx < end_in:
                                g_remove_idx.append(idx)
                        g_remove_idx = [(idx - weight_2 * i) for idx in g_remove_idx]
                        self.weight_mask["weight"][num_g_out * i : num_g_out * (i + 1), g_remove_idx] = 0
                        start_in = end_in
                else:
                    self.weight_mask["weight"][:, remove_idx] = 0
                self.masker().set_in_remove_idx(remove_idx)
            if self.dim_changes_info.pruned_idx_o:
                remove_idx = self.dim_changes_info.pruned_idx_o
                self.register_out_mask(remove_idx)

        self.masker().register_mask("weight", self.weight_mask["weight"])

    def register_out_mask(self, remove_idx):
        self.weight_mask["weight"][remove_idx, :] = 0
        self.masker().set_ot_remove_idx(remove_idx)

        bias_mask = self.bias_mask.get("bias", None)
        if bias_mask is not None:
            bias_mask[remove_idx] = 0
            self.masker().register_mask("bias", bias_mask)

    def modify_input(self, remove_idx):
        conv = self.node.module

        log.debug(f'[CONV] {self.unique_name()}: remove_idx = {remove_idx}')

        if is_dw_conv(self.module()):
            preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[0])], remove_idx)

            if conv.groups != len(preserve_idx):
                log.info(f'[DW_CONV] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
                conv.groups = len(preserve_idx)
                conv.in_channels = len(preserve_idx)
                conv.out_channels = len(preserve_idx)
                conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :])
                if conv.bias is not None:
                    log.info(f'[DW_CONV] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                    conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])

        else:
            group = self.group
            if group != 1:
                if conv.in_channels == (self.weight_mask["weight"].shape[1]) * group - len(remove_idx):
                    return
                num_g_remove_idx = len(remove_idx) // group
                num_g_out = self.weight_mask["weight"].shape[0] // group
                weight_2 = self.weight_mask["weight"].shape[1]
                conv_weight = None
                for i in range(group):
                    g_remove_idx = remove_idx[num_g_remove_idx * i : num_g_remove_idx * (i + 1)]
                    g_remove_idx = [idx - weight_2 * i for idx in g_remove_idx]
                    preserve_idx = complementary_list(
                        [j for j in range(self.weight_mask["weight"].shape[1])], g_remove_idx
                    )
                    weight = conv.weight[num_g_out * i : num_g_out * (i + 1), preserve_idx]
                    if conv_weight is None:
                        conv_weight = weight
                    else:
                        conv_weight = torch.cat([conv_weight, weight], dim=0)
                remove_channel = conv.in_channels - len(remove_idx)
                log.info(f'[CONV-group] {self.unique_name()}: input {conv.in_channels} -> {remove_channel}')
                conv.weight = torch.nn.Parameter(conv_weight)
                conv.in_channels = remove_channel

            else:
                preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[1])], remove_idx)
                if conv.in_channels != len(preserve_idx):
                    log.info(f'[CONV] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
                    conv.weight = torch.nn.Parameter(
                        conv.weight[
                            :,
                            preserve_idx,
                        ]
                    )
                    conv.in_channels = len(preserve_idx)

    def modify_output(self, remove_idx):
        conv = self.node.module

        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[0])], remove_idx)
        log.debug(f'[CONV] {self.unique_name()}: remove_idx = {remove_idx}')

        if is_dw_conv(self.module()):
            if conv.groups != len(preserve_idx):
                log.info(f'[DW_CONV] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
                conv.groups = len(preserve_idx)
                conv.in_channels = len(preserve_idx)
                conv.out_channels = len(preserve_idx)
                conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :])

                if conv.bias is not None:
                    log.info(f'[DW_CONV] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                    conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])

        else:
            if conv.out_channels != len(preserve_idx):
                log.info(f'[CONV] {self.unique_name()}: output {conv.out_channels} -> {len(preserve_idx)}')
                conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :])
                conv.out_channels = len(preserve_idx)

                if conv.bias is not None:
                    log.info(f'[CONV] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                    conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])

    def change_dimension(self) -> bool:
        if is_dw_conv(self.module()):
            return True

        dim_changes_o = [self.dim_c]
        fill_tensor_by_dim_changes(self.output_tensor, dim_changes_o)

        tensor_constraint = self.dim_changes_info.update_o(
            self, self.next_tensors()[0], dim_changes_o, update_constraint=True
        )

        for m in self.next_modifiers():
            m.dim_change_forward(self, self.next_tensors()[0], dim_changes_o, None, tensor_constraint)

        return True

    def dim_change_forward(self, center, tensor, dim_changes_i, dim_transform, tensor_constraint):
        tensor_constraint = self.dim_changes_info.update_i(
            center, tensor, dim_changes_i, dim_transform, tensor_constraint=tensor_constraint
        )

        dw_conv = is_dw_conv(self.module())

        if dw_conv:
            dim_changes_o = dim_changes_i
        elif self.dim_n in dim_changes_i:
            dim_changes_o = [self.dim_n]
        else:
            dim_changes_o = None

        if dim_changes_o:
            self.dim_changes_info.update_o(center, self.next_tensors()[0], dim_changes_o)

            fill_tensor_by_dim_changes(self.output_tensor, dim_changes_o)

            constraint_i = self.dim_changes_info.constraints_i[dim_changes_o[0]][center.unique_name()][0]
            transform = OrderedDict()
            for i in range(len(constraint_i)):
                transform[i] = constraint_i[i]

            for m in self.next_modifiers():
                if dw_conv:
                    m.dim_change_forward(center, self.next_tensors()[0], dim_changes_o, transform, tensor_constraint)
                else:
                    m.dim_change_forward(center, self.next_tensors()[0], dim_changes_o, transform, None)


class TransConvChannelModifier(Conv2dChannelModifier):
    def register_mask(self, modifiers, importance, sparsity):
        if self.dim_changes_info.pruned_idx_i:
            remove_idx = self.dim_changes_info.pruned_idx_i
            self.weight_mask["weight"][
                remove_idx,
                :,
            ] = 0
            self.masker().set_in_remove_idx(remove_idx)
        if self.dim_changes_info.pruned_idx_o:
            remove_idx = self.dim_changes_info.pruned_idx_o
            self.weight_mask["weight"][
                :,
                remove_idx,
            ] = 0
            self.masker().set_ot_remove_idx(remove_idx)

            # 普通conv中bias仅在output改变时改变
            bias_mask = self.bias_mask.get("bias", None)
            if bias_mask is not None:
                bias_mask[remove_idx] = 0
                self.masker().register_mask("bias", bias_mask)

        self.masker().register_mask("weight", self.weight_mask["weight"])

    def modify_input(self, remove_idx):
        conv = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[0])], remove_idx)

        if conv.in_channels != len(preserve_idx):
            log.info(f'[TRANS_CONV2D] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
            conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :])
            conv.in_channels = len(preserve_idx)

    def modify_output(self, remove_idx):
        conv = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask["weight"].shape[1])], remove_idx)

        if conv.out_channels != len(preserve_idx):
            log.info(f'[TRANS_CONV2D] {self.unique_name()}: output {conv.out_channels} -> {len(preserve_idx)}')
            conv.weight = torch.nn.Parameter(conv.weight[:, preserve_idx])
            conv.out_channels = len(preserve_idx)

            if conv.bias is not None:
                log.info(f'[TRANS_CONV2D] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])


CHANNEL_MODIFIERS = {
    nn.Conv2d: Conv2dChannelModifier,
    nn.Linear: LinearChannelModifier,
    nn.ConvTranspose2d: TransConvChannelModifier,
    nn.ConvTranspose1d: TransConvChannelModifier,
    nn.AvgPool2d: PoolingModifier,
    nn.AdaptiveAvgPool2d: PoolingModifier,
    nn.MaxPool2d: PoolingModifier,
    'adaptive_avg_pool2d': PoolingModifier,
    'max_pool2d': PoolingModifier,
    'pad': PaddingModifier,
    nn.Upsample: PoolingModifier,
    nn.UpsamplingBilinear2d: PoolingModifier,
    nn.UpsamplingNearest2d: PoolingModifier,
    "interpolate": PoolingModifier,
    nn.PReLU: PReLUChannelModifier,
    nn.BatchNorm2d: BatchNormChannelModifier,
    nn.BatchNorm1d: BatchNormChannelModifier,
    'matmul': MatMulModifier,
    'cat': CatModifier,
    'view': ReshapeModifier,
    "flatten": ReIndexModifier,
    nn.Flatten: ReIndexModifier,
    'reshape': ReshapeModifier,
    'transpose': ReIndexModifier,
    'permute': ReIndexModifier,
    'split': SplitModifier,
    'chunk': ReIndexModifier,
    'mean': ReIndexModifier,
    'sum': ReIndexModifier,
    'getitem': ReIndexModifier,
    nn.RNN: RNNChannelModifier,
    nn.GRU: RNNChannelModifier,
    nn.LSTM: RNNChannelModifier,
}


def create_channel_modifier(n):
    for key in CHANNEL_MODIFIERS.keys():
        if type(key) == str:
            if n.kind() == key:
                return CHANNEL_MODIFIERS[key](n)
        elif isinstance(n.module, key):
            return CHANNEL_MODIFIERS[key](n)

    # ChannelModifier is used by default
    return Modifier(n)


class SubGraph(object):
    modifiers: typing.List[Modifier]
    leaf: typing.List[Modifier]

    def __init__(self, center: Modifier, modifiers=None):
        self.center = center
        self.modifiers = modifiers if modifiers is not None else []
        self.modifiers_dict = {m.unique_name(): m for m in self.modifiers}
        self.leaf = list()
        self.dependent_centers = set()
        self.center_constraint = OrderedDict()
        self.center_group = OrderedDict()
        self.leaf_group = OrderedDict()
        self.skip = False

    def add_modifier(self, modifier: Modifier):
        self.modifiers.append(modifier)
        self.modifiers_dict[modifier.unique_name()] = modifier

    def constraint_mapping(self, constraint, mapping):
        result = set()
        for i in constraint:
            result.update(mapping[i])

        return result

    def calc_prune_idx_by_bn_variance(
        self, center_list, center_to_leaf_all, leaf_to_center_all, importance, center_to_center_all, sparsity
    ):
        pruned_leaf_constraint_all = {}
        pruned_center_constraint_all = {}

        invalid_bn_idxes = {}

        invalid_center_idxes = {}
        ignored_bn = set()

        for leaf in self.leaf:
            if type(leaf.module()) != nn.BatchNorm2d:
                continue

            while True:
                if len(leaf.pre_modifiers()) == 1:
                    leaf = leaf.pre_modifiers()[0]
                else:
                    break

                if leaf in self.leaf:
                    if type(leaf.module()) != nn.BatchNorm2d:
                        continue

                    ignored_bn.add(leaf)

        for leaf in self.leaf:
            if leaf in ignored_bn:
                continue
            if type(leaf.module()) != nn.BatchNorm2d:
                continue

            is_real_leaf = True
            for leaf_center_name in leaf.dim_changes_info.centers.keys():
                # All centers of a valid leaf must be in this subgraph
                if leaf_center_name not in self.modifiers_dict.keys():
                    is_real_leaf = False

            if not is_real_leaf:
                continue

            center_to_leaf = center_to_leaf_all[leaf.unique_name()]
            leaf_to_center = leaf_to_center_all[leaf.unique_name()]

            invalid_bn = (leaf.module().running_var < 1e-8).tolist()
            invalid_bn_dict = {}
            for i in range(len(invalid_bn)):
                if invalid_bn[i]:
                    invalid_bn_dict[i] = True
                else:
                    invalid_bn_dict[i] = False

            invalid_bn_idxes[leaf.unique_name()] = invalid_bn_dict

            for idx, state in invalid_bn_dict.items():
                for center_name, idx_mapping in leaf_to_center.items():
                    center_idxes = list(idx_mapping[idx])
                    if set(center_idxes) == {-1}:
                        continue

                    for center_idx in center_idxes:
                        if center_name not in invalid_center_idxes:
                            invalid_center_idxes[center_name] = {leaf.unique_name(): {}}

                        if leaf.unique_name() not in invalid_center_idxes[center_name]:
                            invalid_center_idxes[center_name][leaf.unique_name()] = {}

                        invalid_center_idxes[center_name][leaf.unique_name()][center_idx] = state

                        center_to_center = center_to_center_all[center_name]
                        for depend_center_name in center_to_center[center_idx].keys():
                            if depend_center_name not in invalid_center_idxes:
                                invalid_center_idxes[depend_center_name] = {leaf.unique_name(): {}}

                            if leaf.unique_name() not in invalid_center_idxes[depend_center_name]:
                                invalid_center_idxes[depend_center_name][leaf.unique_name()] = {}

                            depend_center_idxes = list(center_to_center[center_idx][depend_center_name])
                            for depend_center_idx in depend_center_idxes:
                                invalid_center_idxes[depend_center_name][leaf.unique_name()][depend_center_idx] = state

        for center_name, leaf_info in invalid_center_idxes.items():
            invalid_center_idx_dict = {}

            for leaf_name, invalid_center_idx in leaf_info.items():
                for idx, state in invalid_center_idx.items():
                    if idx not in invalid_center_idx_dict:
                        invalid_center_idx_dict[idx] = state
                    else:
                        invalid_center_idx_dict[idx] &= state

            invalid_center_idx_set = set()
            for idx, state in invalid_center_idx_dict.items():
                if state:
                    invalid_center_idx_set.add(idx)

            pruned_center_constraint_all[center_name] = invalid_center_idx_set

            for leaf in self.leaf:
                center_to_leaf = center_to_leaf_all[leaf.unique_name()]
                leaf_to_center = leaf_to_center_all[leaf.unique_name()]

                if center_name not in center_to_leaf.keys():
                    continue

                is_real_leaf = True
                for leaf_center_name in leaf.dim_changes_info.centers.keys():
                    # All centers of a valid leaf must be in this subgraph
                    if leaf_center_name not in self.modifiers_dict.keys():
                        is_real_leaf = False

                if not is_real_leaf:
                    continue

                for idx in invalid_center_idx_set:
                    leaf_idx = self.constraint_mapping([idx], center_to_leaf[center_name])

                    if leaf_idx != {-1}:
                        if leaf.unique_name() not in pruned_leaf_constraint_all:
                            pruned_leaf_constraint_all[leaf.unique_name()] = []
                        pruned_leaf_constraint_all[leaf.unique_name()].append(leaf_idx)

        return pruned_center_constraint_all, pruned_leaf_constraint_all

    def calc_prune_idx_by_center_importance(
        self, center_list, center_to_leaf_all, leaf_to_center_all, importance, center_to_center_all, sparsity
    ):
        leaf_delta_idx = {}
        pruned_leaf_constraint_all = {}
        pruned_center_constraint_all = {}
        calculated_center_constraint_all = {}

        for center in center_list:
            calculated_constraint = calculated_center_constraint_all.get(center.unique_name(), set())
            constraint_need_prune = []

            for i in self.center_constraint[center.unique_name()]:
                if not i.issubset(calculated_constraint):
                    constraint_need_prune.append(i)

            for leaf in self.leaf:
                center_to_leaf = center_to_leaf_all[leaf.unique_name()]
                leaf_to_center = leaf_to_center_all[leaf.unique_name()]

                if center.unique_name() not in center_to_leaf.keys():
                    continue

                is_real_leaf = True
                for leaf_center_name in leaf.dim_changes_info.centers.keys():
                    # All centers of a valid leaf must be in this subgraph
                    if leaf_center_name not in self.modifiers_dict.keys():
                        is_real_leaf = False

                if not is_real_leaf:
                    continue

                log.debug(f"calc leaf prune idx: {center.unique_name()}, {leaf.unique_name()}")

                leaf_constraint_need_prune = []
                for constraint in constraint_need_prune:
                    constraint_set = set()
                    for center_idxes in constraint:
                        if center_idxes in center_to_leaf[center.unique_name()]:
                            constraint_set.update(center_to_leaf[center.unique_name()][center_idxes])
                    leaf_constraint_need_prune.append(constraint_set)

                leaf_constraint_all = []
                calculated_leaf_constraint_all = []
                pruned_leaf_constraint = []
                for center_name, constraints in self.center_constraint.items():
                    if center_name not in center_to_leaf.keys():
                        continue

                    calculated_constraint = calculated_center_constraint_all.get(center_name, set())
                    pruned_constraint = pruned_center_constraint_all.get(center_name, set())

                    for constraint in constraints:
                        leaf_idx_constraint = set()
                        for center_idxes in constraint:
                            if center_idxes in center_to_leaf[center_name]:
                                leaf_idx_constraint.update(center_to_leaf[center_name][center_idxes])

                        if constraint.issubset(calculated_constraint):
                            calculated_leaf_constraint_all.append(leaf_idx_constraint)
                            if len(leaf_idx_constraint) > 0 and constraint.issubset(pruned_constraint):
                                # When a center has been pruned, the corresponding leaf directly
                                # reuses the pruning result of the center
                                pruned_leaf_constraint.append(leaf_idx_constraint)
                        else:
                            if len(leaf_idx_constraint) > 0:
                                leaf_constraint_all.append(leaf_idx_constraint)

                # All center idx corresponding to leaf have been calculated
                if len(leaf_constraint_all) == 0:
                    pruned_leaf_constraint_all[leaf.unique_name()] = pruned_leaf_constraint
                    continue

                merge_constraint(leaf_constraint_all)
                merge_constraint(calculated_leaf_constraint_all)

                constraint = []
                for i in leaf_constraint_all:
                    if i in leaf_constraint_need_prune and i not in calculated_leaf_constraint_all:
                        constraint.append(i)

                leaf_constraint_all = constraint

                leaf_importance = []
                for constraint in leaf_constraint_all:
                    importance_ = 0
                    for leaf_idxes in constraint:
                        for center_name, idx_mapping in leaf_to_center.items():
                            center_idxes = list(idx_mapping[leaf_idxes])
                            if set(center_idxes) == {-1}:
                                continue

                            if max(center_idxes) >= len(importance[center_name]):
                                assert False

                            importance_ += float(sum(importance[center_name][center_idxes]))
                            center_to_center = center_to_center_all[center_name]
                            for center_idx in center_idxes:
                                for depend_center_name in center_to_center[center_idx].keys():
                                    depend_center_idxes = list(center_to_center[center_idx][depend_center_name])
                                    importance_ += float(sum(importance[depend_center_name][depend_center_idxes]))

                    leaf_importance.append((constraint, importance_))

                for group in self.leaf_group[leaf.unique_name()]:
                    valid_importance = []
                    for i in leaf_importance:
                        constraint = i[0]
                        if len(constraint & group) > 0:
                            assert constraint.issubset(group)
                            valid_importance.append(i)

                    if len(valid_importance) == 0:
                        continue

                    valid_importance = sorted(valid_importance, key=lambda x: x[1])
                    current_sparsity = 0
                    total_idx = sum([len(i[0]) for i in valid_importance])
                    target_idx = total_idx * sparsity[center.unique_name()]

                    pruned_leaf_idx = set()
                    while current_sparsity < sparsity[center.unique_name()]:
                        if len(valid_importance) == 0:
                            break
                        unimportance_idx = valid_importance.pop(0)
                        constraint = unimportance_idx[0]

                        if center.unique_name() in pruned_center_constraint_all:
                            center_constraint_len = len(self.center_constraint[center.unique_name()])
                            center_pruned_constraint_len = len(pruned_center_constraint_all[center.unique_name()])
                            center_constraint = self.constraint_mapping(
                                constraint, leaf_to_center[center.unique_name()]
                            )

                            global_center_sparsity = (
                                center_pruned_constraint_len + len(pruned_leaf_idx) + len(center_constraint)
                            ) / center_constraint_len
                            if global_center_sparsity > sparsity[center.unique_name()]:
                                break

                        current_sparsity = (
                            len(pruned_leaf_idx) + len(constraint) + leaf_delta_idx.get(leaf.unique_name(), 0)
                        ) / total_idx

                        pruned_leaf_idx.update(constraint)
                        pruned_leaf_constraint.append(constraint)

                        calculated_leaf_constraint_all.append(constraint)

                    delta_idx = len(pruned_leaf_idx) - target_idx
                    leaf_delta_idx[leaf.unique_name()] = leaf_delta_idx.get(leaf.unique_name(), 0)
                    leaf_delta_idx[leaf.unique_name()] = leaf_delta_idx[leaf.unique_name()] + delta_idx

                pruned_leaf_constraint_all[leaf.unique_name()] = pruned_leaf_constraint

                for center_name in leaf_to_center.keys():
                    calculated_center_constraint_all[center_name] = calculated_center_constraint_all.get(
                        center_name, set()
                    )
                    pruned_center_constraint_all[center_name] = pruned_center_constraint_all.get(center_name, set())

                    # Sync leaf's pruning idx to all centers
                    for constraint in calculated_leaf_constraint_all + leaf_constraint_all:
                        center_constraint = self.constraint_mapping(constraint, leaf_to_center[center_name])
                        if center_constraint != -1:
                            calculated_center_constraint_all[center_name].update(center_constraint)

                    for constraint in pruned_leaf_constraint:
                        center_constraint = self.constraint_mapping(constraint, leaf_to_center[center_name])
                        if center_constraint != {-1}:
                            pruned_center_constraint_all[center_name].update(center_constraint)

        return pruned_center_constraint_all, pruned_leaf_constraint_all

    def calc_prune_idx(self, importance, sparsity):
        """
        Calculate the dependence of index in the process of pruning. For convolutional pruning, it is the dependence
        between channels. For more complex unstructured/semi-structured pruning, it may have a finer granularity.
        """
        if self.center not in sparsity or sparsity[self.center] == 0.0:
            return

        center_constraint = {}
        leaf_prune_dim = {}
        leaf_constraint = {}
        leaf_constraint_len = {}

        for m in self.modifiers:
            log.debug(f"modifier {m.unique_name()} prune dim = {dict(m.dim_changes_info.dim_choices)}")

        for leaf in self.leaf:
            # After all subgraph dependencies are resolved, each operator will only be pruned in a single dimension
            assert len(leaf.dim_changes_info.constraints_i) == 1, leaf.unique_name()
            leaf_prune_dim[leaf.unique_name()] = list(leaf.dim_changes_info.constraints_i.keys())[0]
            leaf_constraint[leaf.unique_name()] = list(leaf.dim_changes_info.constraints_i.values())[0]

            for center_name, constraints in leaf_constraint[leaf.unique_name()].items():
                if center_name not in self.modifiers_dict.keys():
                    continue

                leaf_constraint_len[leaf.unique_name()] = len(constraints[0])
                for constraint in constraints:
                    center_constraint[center_name] = center_constraint.get(center_name, [])
                    center_constraint[center_name] += constraint

        for center_name, constraints in center_constraint.items():
            merge_constraint(constraints)

        self.center_constraint = center_constraint

        for center in self.dependent_centers:
            if len(center.dim_changes_info.groups_o) > 0:
                self.center_group[center.unique_name()] = center.dim_changes_info.groups_o

        # Build constraint mapping between center and leaf
        center_to_leaf_all = {}
        leaf_to_center_all = {}
        for leaf in self.leaf:
            center_to_leaf = {}
            leaf_to_center = {}

            center_to_leaf_all[leaf.unique_name()] = center_to_leaf
            leaf_to_center_all[leaf.unique_name()] = leaf_to_center

            # center_constraint loses the constraint mapping information between center
            # and leaf, so use the original leaf_constraint
            for center_name, constraints in leaf_constraint[leaf.unique_name()].items():
                if center_name not in self.modifiers_dict:
                    continue

                if center_name not in center_to_leaf:
                    center_to_leaf[center_name] = {}
                    leaf_to_center[center_name] = {}

                for constraint in constraints:
                    for leaf_idxes in range(len(constraint)):
                        leaf_to_center[center_name][leaf_idxes] = leaf_to_center[center_name].get(leaf_idxes, set())
                        leaf_to_center[center_name][leaf_idxes].update(constraint[leaf_idxes])

                        for center_idxes in constraint[leaf_idxes]:
                            center_to_leaf[center_name][center_idxes] = center_to_leaf[center_name].get(
                                center_idxes, set()
                            )
                            center_to_leaf[center_name][center_idxes].add(leaf_idxes)

                if -1.0 in center_to_leaf[center_name]:
                    del center_to_leaf[center_name][-1.0]

        # Aggregate all leaf constraints into a global center constraint
        for leaf in self.leaf:
            center_to_leaf = center_to_leaf_all[leaf.unique_name()]
            leaf_to_center = leaf_to_center_all[leaf.unique_name()]

            # Obtain the constraint of leaf through the constraint of center
            leaf_constraint_all = []
            for center_name, constraints in self.center_constraint.items():
                if center_name not in center_to_leaf.keys():
                    continue
                for constraint in constraints:
                    leaf_idx_constraint = set()
                    for center_idxes in constraint:
                        if center_idxes in center_to_leaf[center_name]:
                            leaf_idx_constraint.update(center_to_leaf[center_name][center_idxes])

                    if leaf_idx_constraint not in leaf_constraint_all:
                        leaf_constraint_all.append(leaf_idx_constraint)
            merge_constraint(leaf_constraint_all)

            # Pass the leaf constraint back to the center, so that the center nodes
            # can get dependencies between each other
            leaf_center_constraints = {}
            for center_name in leaf_to_center.keys():
                if center_name not in self.modifiers_dict:
                    continue
                leaf_center_constraints[center_name] = []
                for leaf_idx_constraint in leaf_constraint_all:
                    index_constraint = set()
                    for leaf_idxes in leaf_idx_constraint:
                        index_constraint.update(leaf_to_center[center_name][leaf_idxes])
                    if -1.0 in index_constraint:
                        index_constraint.remove(-1.0)
                    if index_constraint not in leaf_center_constraints[center_name]:
                        leaf_center_constraints[center_name].append(index_constraint)
            for center_name in leaf_center_constraints.keys():
                self.center_constraint[center_name] += leaf_center_constraints[center_name]
                merge_constraint(self.center_constraint[center_name])

            log.debug(f"leaf {leaf.unique_name()} constraint merge over")

        # Aggregate all leaf group into a global center group
        for leaf in self.leaf:
            center_to_leaf = center_to_leaf_all[leaf.unique_name()]
            leaf_to_center = leaf_to_center_all[leaf.unique_name()]

            leaf_group_all = []
            # TODO: Is it possible to skip when center_group has only one element?
            for center_name, center_group in self.center_group.items():
                if center_name not in center_to_leaf.keys():
                    continue

                for group in center_group:
                    leaf_idx_group = set()
                    for center_idxes in group:
                        # Nodes such as split may cause the number of idx in leaf and center to be inconsistent
                        if center_idxes in center_to_leaf[center_name]:
                            leaf_idx_group.update(center_to_leaf[center_name][center_idxes])

                    leaf_group_all.append(leaf_idx_group)

            if len(leaf.dim_changes_info.groups_i) > 0:
                leaf_group_all += leaf.dim_changes_info.groups_i

            merge_group(leaf_group_all)

            leaf_center_groups = {}
            for center_name in leaf_to_center.keys():
                leaf_center_groups[center_name] = []
                for leaf_idx_group in leaf_group_all:
                    index_group = set()
                    # Nodes such as split may cause the number of idx in leaf and center to be inconsistent
                    for leaf_idxes in leaf_idx_group:
                        if leaf_idxes in leaf_to_center[center_name]:
                            index_group.update(leaf_to_center[center_name][leaf_idxes])
                    if -1.0 in index_group:
                        index_group.remove(-1.0)
                    if len(index_group) > 0:
                        leaf_center_groups[center_name].append(index_group)

            for center_name in leaf_center_groups.keys():
                self.center_group[center_name] = self.center_group.get(center_name, [])
                self.center_group[center_name] += leaf_center_groups[center_name]
                merge_group(self.center_group[center_name])

        for leaf in self.leaf:
            center_to_leaf = center_to_leaf_all[leaf.unique_name()]

            leaf_group = []
            self.leaf_group[leaf.unique_name()] = leaf_group
            for center_name, center_group in self.center_group.items():
                if center_name not in center_to_leaf.keys():
                    continue

                for group in center_group:
                    leaf_idx_group = set()
                    for center_idxes in group:
                        # split等节点可能导致leaf和center中的idx数量不一致，需要判断存在合法性
                        if center_idxes in center_to_leaf[center_name]:
                            leaf_idx_group.update(center_to_leaf[center_name][center_idxes])

                    leaf_group.append(leaf_idx_group)

            if len(leaf.dim_changes_info.groups_i) > 0:
                leaf_group += leaf.dim_changes_info.groups_i

            merge_group(leaf_group)

        log.debug(f"subgraph {self.center} group merge over")

        # 1) Select a center
        # 2) Map the center to all leaves, and then complete leaf pruning
        # 3) Update the global center_pruned_constraint after leaf pruning
        # 4) Prune the next center, and then exclude the pruned idx in center_pruned_constraint
        # 5) Repeat the above steps until all centers are pruned
        center_list = []
        for center_name, constraint in self.center_constraint.items():
            constraint_all = set()
            for i in constraint:
                constraint_all.update(i)
            center_list.append((len(constraint_all), self.modifiers_dict[center_name]))

        # Prioritize the center with the shortest constraint. If the center with the longest constraint
        # is processed first, the short one may have an incorrect sparsity rate
        center_list = sorted(center_list, key=lambda x: x[0])
        center_list = [i[1] for i in center_list]

        center_to_center_all = {}

        for center in center_list:
            center_name = center.unique_name()
            center_to_center = {}
            center_to_center_all[center.unique_name()] = center_to_center

            for center_idxes in self.center_constraint[center.unique_name()]:
                for center_idxes in center_idxes:
                    if center_idxes not in center_to_center:
                        center_to_center[center_idxes] = {}

                    for leaf in self.leaf:
                        leaf_name = leaf.unique_name()
                        center_to_leaf = center_to_leaf_all[leaf_name]
                        leaf_to_center = leaf_to_center_all[leaf_name]

                        if center.unique_name() not in center_to_leaf.keys():
                            continue

                        if center_idxes not in center_to_leaf[center_name]:
                            continue

                        leaf_idxes = center_to_leaf[center_name][center_idxes]
                        for leaf_idx in leaf_idxes:
                            for depend_center_name in leaf_to_center.keys():
                                if depend_center_name == center_name:
                                    continue

                                depend_center_idxes = leaf_to_center[depend_center_name][leaf_idx]
                                if depend_center_idxes == {-1}:
                                    continue
                                if depend_center_name not in center_to_center[center_idxes]:
                                    center_to_center[center_idxes][depend_center_name] = set()
                                center_to_center[center_idxes][depend_center_name].update(depend_center_idxes)

        if importance is not None:
            pruned_center_constraint_all, pruned_leaf_constraint_all = self.calc_prune_idx_by_center_importance(
                center_list, center_to_leaf_all, leaf_to_center_all, importance, center_to_center_all, sparsity
            )
        else:
            pruned_center_constraint_all, pruned_leaf_constraint_all = self.calc_prune_idx_by_bn_variance(
                center_list, center_to_leaf_all, leaf_to_center_all, importance, center_to_center_all, sparsity
            )

        for center_name, constraint in pruned_center_constraint_all.items():
            calculated_constraint = constraint

            if -1 in calculated_constraint:
                calculated_constraint.remove(-1)
            calculated_constraint = list(calculated_constraint)
            calculated_constraint.sort()

            self.modifiers_dict[center_name].dim_changes_info.pruned_idx_o = calculated_constraint

        for leaf_name, constraint in pruned_leaf_constraint_all.items():
            calculated_constraint = set()
            for i in constraint:
                calculated_constraint.update(i)

            if -1 in calculated_constraint:
                calculated_constraint.remove(-1)
            calculated_constraint = list(calculated_constraint)
            calculated_constraint.sort()

            self.modifiers_dict[leaf_name].dim_changes_info.pruned_idx_i = calculated_constraint

        log.debug(f"subgraph {self.center} prune idx compute over")

    def eliminate_conflict(self):
        tensor_choices = OrderedDict()

        # Make sure each tensor only prunes at most one dimension
        for m in reversed(self.modifiers):
            if m.dim_changes_info.is_multi_dim_changed():
                log.debug(f"[{m.unique_name()}] multi dim changed")
                if not m.dim_choose(tensor_choices):
                    log.error(f"[{m.unique_name()}] conflict can't be eliminated")
                    raise Exception("conflict can't be eliminated")

        for m in self.modifiers:
            for t in m.pre_tensors() + m.next_tensors():
                if id(t) in tensor_choices:
                    m.dim_changes_info.update_choice(t, tensor_choices[id(t)])
                elif id(t) in m.dim_changes_info.tensor_changes:
                    m.dim_changes_info.update_choice(t, m.dim_changes_info.merge_t(t))
                else:
                    # The tensor has not changed
                    pass

        for m in self.modifiers:
            m.dim_changes_info.rebuild()
            m.calc_idx_group()

    def calc_importance(self):
        pass

    def build(self):
        self.modifiers = list(set(self.modifiers))
        self.modifiers = sorted(self.modifiers, key=lambda m: m.node.forward_order)

        leaf = set()

        for m in self.modifiers:

            is_leaf = True
            for next_m in m.next_modifiers():
                if next_m in self.modifiers:
                    is_leaf = False
                    break

            # In some ring subgraphs, a operator may be both center and leaf at the same time
            for center in m.dim_changes_info.centers.values():
                if m.prunable and center in self.modifiers and center is not m:
                    is_leaf = True

            if not is_leaf:
                continue

            leaf.add(m)

            for center in m.dim_changes_info.centers.values():
                if center is not m:
                    self.dependent_centers.add(center)

        self.leaf = list(leaf)
        self.leaf = sorted(self.leaf, key=lambda m: m.node.forward_order)

        return self

    def __eq__(self, other):
        if type(other) in [list, tuple]:
            if self.modifiers == other:
                return True
        return False


class SubGraphDivider(object):
    def __init__(self, graph: TraceGraph, modifiers: typing.Dict[str, Modifier]):
        self.graph = graph
        self.modifiers = OrderedDict(sorted(modifiers.items(), key=lambda x: x[1].node.forward_order))
        self.tensors = graph.all_tensors()
        self.sub_graphs = OrderedDict()

    def reset_tensors(self):
        for t in self.tensors:
            # Do not modify the constants, otherwise the branches of operators such as shape need to be recalculated
            if len(t.shape) >= 1:
                # Cannot assign a value of 0, because 0 cannot distinguish between index 0 and an uninitialized index
                t.data.fill_(-1)

    def change_dimension(self):
        self.reset_tensors()

        dim_changed = False
        for m in self.modifiers.values():
            if dim_changed:
                # Use the tensor generated in the trace process to save memory
                self.reset_tensors()

            dim_changed = m.change_dimension()
            if dim_changed:
                log.debug(f"operator [{m.unique_name()}] tracking over")

    def divide_subgraph(self):
        self.sub_graphs.clear()

        sub_graphs = {}

        for modifier in self.modifiers.values():
            for center_name, center in modifier.dim_changes_info.centers.items():
                if center_name not in sub_graphs:
                    sub_graphs[center_name] = set()

                sub_graphs[center_name].add(modifier)

        center_mapping = {}
        for modifier in self.modifiers.values():
            center_names = modifier.dim_changes_info.get_input_centers()

            # TODO: It is more reasonable to arrange according to the forward order
            center_names = sorted(list(set(center_names)))
            if len(center_names) <= 1:
                continue

            main_center = center_names[0]
            while main_center not in sub_graphs and main_center in center_mapping:
                main_center = center_mapping[main_center]

            for redundant_center in center_names[1:]:
                if redundant_center == main_center:
                    continue

                if redundant_center in sub_graphs:
                    sub_graphs[main_center].update(sub_graphs[redundant_center])
                    del sub_graphs[redundant_center]
                    center_mapping[redundant_center] = main_center
                else:
                    redirect_center = redundant_center

                    while redirect_center not in sub_graphs:
                        redirect_center = center_mapping[redirect_center]

                    if redirect_center in sub_graphs and redirect_center != main_center:
                        sub_graphs[main_center].update(sub_graphs[redirect_center])
                        del sub_graphs[redirect_center]
                        center_mapping[redirect_center] = main_center

        for center_name, modifiers in sub_graphs.items():
            self.sub_graphs[center_name] = SubGraph(center_name, modifiers).build()

    def divide(self) -> typing.Dict[str, SubGraph]:
        log.info("Start tracking tensor dimension changes...")
        self.change_dimension()

        log.info("Start dividing subgraphs according to tensor dependencies...")
        self.divide_subgraph()

        log.info("Start to eliminate dimension change conflicts...")
        for sub_graph in self.sub_graphs.values():
            sub_graph.eliminate_conflict()

        log.info("Start generating new subgraphs without conflicts...")
        self.divide_subgraph()
        return self.sub_graphs


class GraphChannelModifier(object):
    graph: TraceGraph
    center_nodes: typing.List[TraceNode]
    sub_graphs: typing.Dict[str, SubGraph]

    def __init__(self, graph: TraceGraph, center_nodes, bn_compensation=False):
        """Initialize a channel modifier for a calculation graph

        Args:
            graph: Compute graph generated by tracer
            center_nodes: Operators that actively modify the channel

        """

        self.graph = graph
        self.center_nodes = center_nodes
        self.bn_compensation = bn_compensation
        if self.bn_compensation:
            log.info("open bn compensation")
        self.modifiers = self.register_modifier()
        with torch.no_grad():
            self.sub_graphs = SubGraphDivider(self.graph, self.modifiers).divide()
        self.reset_masker()

    def reset_masker(self):
        self.unregister_masker()
        for n in self.graph.forward_nodes:
            masker.ChannelMasker(n.module, n.unique_name)

    def unregister_masker(self):
        mask_applied = False
        for sub_graph in self.sub_graphs.values():
            for m in sub_graph.modifiers:
                m.reset_mask()
                mask_applied = m.mask_applied or mask_applied

        for n in self.graph.forward_nodes:
            if hasattr(n.module, "masker"):
                n.module.masker.unregister_all()
                delattr(n.module, "masker")

        if mask_applied:
            self.graph.inited = False

    def register_modifier(self) -> typing.Dict[str, Modifier]:

        modifiers = OrderedDict()
        for n in self.graph.all_nodes():
            modifier = create_channel_modifier(n)
            modifier.graph_modifier = self
            modifiers[n.unique_name] = modifier
            setattr(n, "modifier", modifier)
        return modifiers

    def unregister_modifier(self):
        for n in self.graph.all_nodes():
            delattr(n, "modifier")

        self.unregister_masker()

    def get_modifier(self, module=None, unique_name=None) -> Modifier:
        if module is not None:
            unique_name = self.graph.module_original_name_dict[id(module)]
        return self.graph.nodes_map[unique_name].modifier


def fill_tensor_by_dim_changes(tensor, dim_changes, values=None):
    tensor[:] = 0

    for dim in reversed(dim_changes):
        shape = tensor.shape[dim]
        if values is not None:
            assert len(values) == shape
        for i in range(shape):
            value_shape = list(tensor.shape)
            value_shape[dim] = 1

            if values:
                value = torch.ones(value_shape, dtype=tensor.dtype) * values[i]
            else:
                value = torch.ones(value_shape, dtype=tensor.dtype) * i

            tensor.index_add_(dim, torch.tensor(i), value)

    return tensor
