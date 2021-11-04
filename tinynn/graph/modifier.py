import typing
import math
import numpy
from copy import deepcopy
from math import gcd  # Python versions 3.5 and above
from functools import reduce  # Python version 3.x

import torch
import torch.nn as nn

from tinynn.graph import masker
from tinynn.util.util import get_logger
from tinynn.graph.tracer import TraceGraph, TraceNode

log = get_logger(__name__)


def random(tensor, module):
    if type(module) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        return torch.randperm(tensor.shape[0])
    if type(module) in [nn.ConvTranspose2d, nn.ConvTranspose1d]:
        return torch.randperm(tensor.shape[1])


def l1_norm(tensor, module):
    """ Calculate the L1-normalization of each channel """
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


def l2_norm(tensor, module):
    """ Calculate the L2-normalization of each channel """
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


def fpgm(tensor, module):
    """ Calculate the geometric median (Filter Pruning via Geometric Median for Deep Convolutional Neural
    Networks Acceleration, https://arxiv.org/abs/1811.00250) """
    assert type(module) in [nn.Linear, nn.Conv2d]
    num_channels = tensor.shape[0]
    batched_weight = tensor.view(num_channels, -1)
    return torch.cdist(batched_weight, batched_weight, p=2).abs().sum(0)


def is_dw_conv(module):
    """ Check whether the model is depth-wise convolution """
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d, nn.ConvTranspose1d)):
        if module.in_channels == module.groups == module.out_channels:
            return True
    return False


def lcm(denominators):
    """ least common multiple """
    return reduce(lambda a, b: a * b // math.gcd(a, b), denominators)


def complementary_list(a, b):
    return list(set(a).difference(set(b)))


def split_idx(nums, start, end):
    """
    对group list进行切分，获得其start到end之间的子group list
    """
    res = []
    for idx in nums:
        if start >= len(idx):
            start -= len(idx)
            end -= len(idx)
        else:
            if end > len(idx):
                res.append(idx[start:])
                start = 0
                end -= len(idx)
            else:
                res.append(idx[start:end])
                break
    return res


def deduplicate_range(pos_list):
    """删除pos对中重叠的范围，"""
    poses = list(pos_list)
    poses.sort()
    res = set()
    start = end = -1
    for pos in poses:
        if start == -1:
            start, end = pos
            res.add(pos)
        elif pos[0] == end and pos[1] > end:
            res.add(pos)
            end = pos[1]
        elif pos[0] < end < pos[1]:
            res.add((pos[0], end))
            res.add((end, pos[1]))
            end = pos[1]
        elif pos[0] > end:
            res.add(pos)
            end = pos[1]
    return res


def list_flatten(pos_list):
    """将[[],[],[]]->[[]]并得到每个group的起始结束pos对"""
    pos_list_ = [[]]
    offset_list = []
    start = end = 0
    for idx in pos_list:
        end += len(idx)
        pos_list_[0] += idx
        offset_list.append((start, end))
        start = end
    return pos_list_, offset_list


def list_group(flatten_list, offset_list):
    """list_flatten的逆操作"""
    pos_lists = []
    for offset in offset_list:
        pos_lists.append(flatten_list[offset[0]:offset[1]])
    return pos_lists


def justify_group(leaf_dict, idx_map):
    """调整group格式"""
    for k, v in leaf_dict.items():
        if k in idx_map.map_dict.keys() and len(v) > len(idx_map.map_dict[k]):
            idx_map.map_dict[k] = v
        if k not in idx_map.map_dict.keys():
            idx_map.map_dict[k] = v


class Modifier(object):
    node: TraceNode
    weight_mask: torch.Tensor
    bias_mask: torch.Tensor
    input_modify_: bool
    output_modify_: bool

    def __init__(self, node: TraceNode):
        self.node = node
        self.input_modify_ = False
        self.output_modify_ = False
        self.reset_mask()

    def enable_mask(self):
        if self.masker() is not None:
            self.masker().enable()

    def disable_mask(self):
        if self.masker() is not None:
            self.masker().disable()

    def reset_mask(self):
        if hasattr(self.module(), "weight"):
            self.weight_mask = torch.ones_like(self.module().weight)
        if hasattr(self.module(), "bias"):
            self.bias_mask = torch.ones_like(self.module().bias) if type(
                self.module().bias) is torch.nn.Parameter else None

    def traversal(self, input_modify: bool, output_modify: bool, sub_graph):
        pass

    def module(self):
        return self.node.module

    def masker(self) -> masker.Masker:
        return self.node.module.masker

    def register_mask(self, importance, graph_sparsity):
        pass

    def unique_name(self):
        return self.node.unique_name


# TODO: 添加mask，对premute等操作进行完整的内存依赖trace
class ChannelTracer(object):
    def __init__(self, t):
        self.t = t
        self.depend_shape = t.shape


class IdxMap(object):
    """
    定义一个IdxMap数据结构表示当前节点与中心节点对应通道的映射关系；
    保存每个中心节点对应当前节点的通道的映射，例如节点n的idxmap中有一项conv1:[[8,9,10,11,12,13,14,15]]
    表示conv1的8号通道对应节点n的第1个通道。由于存在分组的情况(group convolution, split等算子），
    我们需要将通道进行分组再计算删除的通道位置，所以IdxMap的格式为[[...],[...],[...],[...]]表示多个group
    """
    map_dict: typing.Dict[str, typing.List]

    def __init__(self):
        self.map_dict = {}

    def set_idx(self, unique_name, idxs):
        self.map_dict[unique_name] = idxs

    def get_grouped_idx(self, group):
        """Group the index (only use for leaf node) """

        new_dict = {}
        for k, v in self.map_dict.items():
            v_, _ = list_flatten(v)
            group_channel = len(v_[0]) // group
            start_pos = end_pos = 0
            new_v = []
            for i in range(group):
                end_pos += group_channel
                group_split = split_idx(v, start_pos, end_pos)
                for tmp in group_split:
                    new_v.append(tmp)
                start_pos = end_pos
            if len(new_v) > len(v):
                new_dict[k] = new_v
            else:
                new_dict[k] = v
        return new_dict

    def get_channel_number(self):
        lens = set()
        channel = 0

        for k, v in self.map_dict.items():
            channel = 0
            for sub_idx in v:
                channel += len(sub_idx)
            lens.add(channel)
        if len(lens) != 1:
            log.error(f"The number of channels of center nodes is not aligned ({self.map_dict}).")
            assert False

        return channel

    def set_idx_map(self, idx_map):
        self.map_dict = {}
        for k, v in idx_map.map_dict.items():
            self.map_dict[k] = v


class ChannelModifier(Modifier):
    """ Automatically handle the dependency of the operator and modify the number of channels """

    def __init__(self, node: TraceNode = None):
        super().__init__(node=node)
        self.mask_applied = False

        # input channel info
        self.in_idx_map = IdxMap()

        # output channel info
        self.ot_idx_map = IdxMap()

    def masker(self) -> masker.ChannelMasker:
        return self.node.module.masker if hasattr(self.node.module, "masker") else None

    def apply_mask(self):
        """ Use mask to modify the channel of the operator """

        if self.masker() is not None and self.masker().in_remove_idx is not None:
            self.modify_input(self.masker().in_remove_idx)

        if self.masker() is not None and self.masker().ot_remove_idx is not None:
            self.modify_output(self.masker().ot_remove_idx)

        self.mask_applied = True

    def modify_input(self, remove_idx):
        """ Modify the input channel of the operator """
        pass

    def modify_output(self, remove_idx):
        """ Modify the output channel of the operator """
        pass

    def in_channel(self):
        if len(self.node.prev_tensors) > 0:
            # Use NCHW layout by default
            return self.node.prev_tensors[0].shape[1]

    def ot_channel(self):
        if len(self.node.next_tensors) > 0:
            # Use NCHW layout by default
            return self.node.next_tensors[0].shape[1]

    def group(self):
        return 1

    def register_mask(self, importance, graph_sparsity):
        if self.masker() is not None:
            remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
            self.masker().set_in_remove_idx(remove_idx)
            self.masker().set_ot_remove_idx(remove_idx)

    def traversal(self, input_modify, output_modify, sub_graph):
        """ Traverse the entire subgraph that depends on each other """

        self.input_modify_ = True
        self.output_modify_ = True
        if self not in sub_graph:
            sub_graph.append(self)
        else:
            self.input_modify_ |= input_modify
            self.output_modify_ |= output_modify
            return self

        for n in self.node.prev_nodes:
            n.modifier.traversal(False, True, sub_graph)

        for n in self.node.next_nodes:
            n.modifier.traversal(True, False, sub_graph)

        return self

    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):
        """ Starting from the center node, pass the channel index to the all downstream nodes """

        if self.input_modify_:
            self.in_idx_map.set_idx(center_name, idxes)
        if self.output_modify_ and self.input_modify_:
            self.ot_idx_map.set_idx(center_name, idxes)
        if self.unique_name() in leaf_names:
            return
        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_forward(self.unique_name(),
                                                          center_name,
                                                          idxes,
                                                          sub_graph_dict,
                                                          leaf_names)

    def idx_back(self, pre_name, leaf_names, center_names, leaf_map_dict, sub_graph_dict):
        """ Starting from the leaf node, pass the channel index to all neighboring nodes """

        if self.unique_name() == pre_name:  # 一次反向传播的起点
            justify_group(leaf_map_dict, self.in_idx_map)
        else:
            if self.output_modify_:
                justify_group(leaf_map_dict, self.ot_idx_map)
            if self.input_modify_ and self.output_modify_ and self.node.unique_name not in center_names:
                self.in_idx_map.set_idx_map(self.ot_idx_map)
            for n in self.node.next_nodes:
                if n.unique_name in sub_graph_dict.keys() and sub_graph_dict[n.unique_name].input_modify_:
                    sub_graph_dict[n.unique_name].idx_back_forward(leaf_names,
                                                                   leaf_map_dict,
                                                                   sub_graph_dict,
                                                                   self.unique_name())

        if self.unique_name() in center_names and pre_name != self.node.unique_name:
            return
        for n in self.node.prev_nodes:
            if n.unique_name in sub_graph_dict.keys() and sub_graph_dict[n.unique_name].output_modify_:
                sub_graph_dict[n.unique_name].idx_back(self.unique_name(),
                                                       leaf_names,
                                                       center_names,
                                                       leaf_map_dict,
                                                       sub_graph_dict)

    def idx_back_forward(self, leaf_names, leaf_map_dict, sub_graph_dict, pre_name):
        """ Broadcast the information of the leaf node to all downstream nodes """

        if self.input_modify_:
            justify_group(leaf_map_dict, self.in_idx_map)
        if self.unique_name() in leaf_names:
            return
        if self.output_modify_ and self.input_modify_:
            self.ot_idx_map.set_idx_map(self.in_idx_map)
        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys() and sub_graph_dict[n.unique_name].input_modify_:
                sub_graph_dict[n.unique_name].idx_back_forward(leaf_names,
                                                               leaf_map_dict,
                                                               sub_graph_dict,
                                                               self.unique_name())


class ConvChannelModifier(ChannelModifier):
    def group(self):
        if is_dw_conv(self.module()):
            return 1

        return self.module().groups

    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):

        if self.unique_name() == center_name and self.output_modify_:
            self.ot_idx_map.set_idx(center_name, idxes)
        elif is_dw_conv(self.module()):
            self.in_idx_map.set_idx(center_name, idxes)
            self.ot_idx_map.set_idx(center_name, idxes)
        else:
            self.in_idx_map.set_idx(center_name, idxes)
            return

        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_forward(self.unique_name(),
                                                          center_name,
                                                          idxes,
                                                          sub_graph_dict,
                                                          leaf_names)

    def register_mask(self, importance, graph_sparsity):

        if is_dw_conv(self.module()):
            remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
            self.weight_mask[remove_idx, :, ] = 0
            self.masker().set_in_remove_idx(remove_idx)
            self.masker().set_ot_remove_idx(remove_idx)

            # dw conv中bias一定改变
            if self.bias_mask is not None:
                self.bias_mask[remove_idx] = 0
                self.masker().register_mask("bias", self.bias_mask)
        else:
            if self.input_modify_:
                remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
                group = self.group()
                remove_idx.sort()
                if group != 1:
                    num_g_out = self.weight_mask.shape[0] // group
                    weight_2 = self.weight_mask.shape[1]
                    start_in = end_in = 0
                    for i in range(group):
                        end_in += weight_2
                        g_remove_idx = []
                        for idx in remove_idx:
                            if start_in <= idx < end_in:
                                g_remove_idx.append(idx)
                        g_remove_idx = [(idx - weight_2 * i) for idx in g_remove_idx]
                        self.weight_mask[num_g_out * i:num_g_out * (i + 1), g_remove_idx, ] = 0
                        start_in = end_in
                else:
                    self.weight_mask[:, remove_idx, ] = 0
                self.masker().set_in_remove_idx(remove_idx)
            if self.output_modify_:
                remove_idx = calc_remove_idx(self.ot_idx_map, importance, graph_sparsity, self.unique_name())
                self.register_out_mask(remove_idx)

        self.masker().register_mask("weight", self.weight_mask)

    def register_out_mask(self, remove_idx):
        self.weight_mask[remove_idx, :, ] = 0
        self.masker().set_ot_remove_idx(remove_idx)

        if self.bias_mask is not None:
            self.bias_mask[remove_idx] = 0
            self.masker().register_mask("bias", self.bias_mask)

    def modify_input(self, remove_idx):
        conv = self.node.module

        if is_dw_conv(self.module()):
            preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

            if conv.groups != len(preserve_idx):
                log.info(f'[DW_CONV] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
                conv.groups = len(preserve_idx)
                conv.in_channels = len(preserve_idx)
                conv.out_channels = len(preserve_idx)
                conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :, ])
                if conv.bias is not None:
                    log.info(f'[DW_CONV] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                    conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])

        else:
            group = self.group()
            if group != 1:
                if conv.in_channels == (self.weight_mask.shape[1]) * group - len(remove_idx):
                    return
                num_g_remove_idx = len(remove_idx) // group
                num_g_out = self.weight_mask.shape[0] // group
                weight_2 = self.weight_mask.shape[1]
                conv_weight = None
                for i in range(group):
                    g_remove_idx = remove_idx[num_g_remove_idx * i:num_g_remove_idx * (i + 1)]
                    g_remove_idx = [idx - weight_2 * i for idx in g_remove_idx]
                    preserve_idx = complementary_list([j for j in range(self.weight_mask.shape[1])], g_remove_idx)
                    weight = conv.weight[num_g_out * i:num_g_out * (i + 1), preserve_idx, ]
                    if conv_weight is None:
                        conv_weight = weight
                    else:
                        conv_weight = torch.cat([conv_weight, weight], dim=0)
                remove_channel = conv.in_channels - len(remove_idx)
                log.info(f'[CONV-group] {self.unique_name()}: input {conv.in_channels} -> {remove_channel}')
                conv.weight = torch.nn.Parameter(conv_weight)
                conv.in_channels = remove_channel

            else:
                preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[1])], remove_idx)
                if conv.in_channels != len(preserve_idx):
                    log.info(f'[CONV] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
                    conv.weight = torch.nn.Parameter(conv.weight[:, preserve_idx, ])
                    conv.in_channels = len(preserve_idx)

    def modify_output(self, remove_idx):
        conv = self.node.module

        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

        if is_dw_conv(self.module()):
            if conv.groups != len(preserve_idx):
                log.info(f'[DW_CONV] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
                conv.groups = len(preserve_idx)
                conv.in_channels = len(preserve_idx)
                conv.out_channels = len(preserve_idx)
                conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :, ])

                if conv.bias is not None:
                    log.info(f'[DW_CONV] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                    conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])

        else:
            if conv.out_channels != len(preserve_idx):
                log.info(f'[CONV] {self.unique_name()}: output {conv.out_channels} -> {len(preserve_idx)}')
                conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :, ])
                conv.out_channels = len(preserve_idx)

                if conv.bias is not None:
                    log.info(f'[CONV] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                    conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])

    def traversal(self, input_modify, output_modify, sub_graph):
        if self not in sub_graph:
            sub_graph.append(self)
        else:
            self.input_modify_ |= input_modify
            self.output_modify_ |= output_modify
            return self

        assert (((input_modify and output_modify) is False) and ((input_modify or output_modify) is True))

        if is_dw_conv(self.module()):
            if input_modify:
                output_modify = input_modify
            elif output_modify:
                input_modify = output_modify

        self.input_modify_ = input_modify
        self.output_modify_ = output_modify

        if input_modify:
            for n in self.node.prev_nodes:
                n.modifier.traversal(False, True, sub_graph)

        if output_modify:
            for n in self.node.next_nodes:
                n.modifier.traversal(True, False, sub_graph)

        return self


class ConvTransChannelModifier(ConvChannelModifier):
    def register_mask(self, importance, graph_sparsity):
        if self.input_modify_:
            remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
            self.weight_mask[remove_idx, :, ] = 0
            self.masker().set_in_remove_idx(remove_idx)
        if self.output_modify_:
            remove_idx = calc_remove_idx(self.ot_idx_map, importance, graph_sparsity, self.unique_name())
            self.weight_mask[:, remove_idx, ] = 0
            self.masker().set_ot_remove_idx(remove_idx)

            # 普通conv中bias仅在output改变时改变
            if self.bias_mask is not None:
                self.bias_mask[remove_idx] = 0
                self.masker().register_mask("bias", self.bias_mask)

        self.masker().register_mask("weight", self.weight_mask)

    def modify_input(self, remove_idx):
        conv = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

        if conv.in_channels != len(preserve_idx):
            log.info(f'[TRANS_CONV2D] {self.unique_name()}: input {conv.in_channels} -> {len(preserve_idx)}')
            conv.weight = torch.nn.Parameter(conv.weight[preserve_idx, :, ])
            conv.in_channels = len(preserve_idx)

    def modify_output(self, remove_idx):
        conv = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[1])], remove_idx)

        if conv.out_channels != len(preserve_idx):
            log.info(f'[TRANS_CONV2D] {self.unique_name()}: output {conv.out_channels} -> {len(preserve_idx)}')
            conv.weight = torch.nn.Parameter(conv.weight[:, preserve_idx, ])
            conv.out_channels = len(preserve_idx)

            if conv.bias is not None:
                log.info(f'[TRANS_CONV2D] {self.unique_name()}: bias {conv.bias.shape[0]} -> {len(preserve_idx)}')
                conv.bias = torch.nn.Parameter(conv.bias[preserve_idx])


class LinearChannelModifier(ChannelModifier):

    def modify_input(self, remove_idx):
        linear = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[1])], remove_idx)

        if linear.weight.shape[1] != len(preserve_idx):
            log.info(f'[FC] {self.unique_name()}: input {linear.in_features} -> {len(preserve_idx)}')
            linear.weight = torch.nn.Parameter(linear.weight[:, preserve_idx])
            linear.in_features = len(preserve_idx)

    def modify_output(self, remove_idx):
        linear = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

        if linear.weight.shape[0] != len(preserve_idx):
            log.info(f'[FC] {self.unique_name()}: output {linear.out_features} -> {len(preserve_idx)}')
            linear.weight = torch.nn.Parameter(linear.weight[preserve_idx, :])
            linear.out_features = len(preserve_idx)

            if linear.bias is not None:
                linear.bias = torch.nn.Parameter(linear.bias[preserve_idx])

    def register_mask(self, importance, graph_sparsity):
        if self.input_modify_:
            remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
            self.weight_mask[:, remove_idx] = 0
            self.masker().set_in_remove_idx(remove_idx)

        if self.output_modify_:
            remove_idx = calc_remove_idx(self.ot_idx_map, importance, graph_sparsity, self.unique_name())
            self.weight_mask[remove_idx, :] = 0
            self.masker().set_ot_remove_idx(remove_idx)
            if self.bias_mask is not None and self.output_modify_ is not None:
                self.bias_mask[remove_idx] = 0
                self.masker().register_mask("bias", self.bias_mask)

        self.masker().register_mask("weight", self.weight_mask)

    def traversal(self, input_modify, output_modify, sub_graph):
        if self not in sub_graph:
            sub_graph.append(self)
        else:
            self.input_modify_ |= input_modify
            self.output_modify_ |= output_modify
            return self

        self.input_modify_ = input_modify
        self.output_modify_ = output_modify

        assert (((input_modify and output_modify) is False) and ((input_modify or output_modify) is True))

        if output_modify:
            for n in self.node.next_nodes:
                n.modifier.traversal(True, False, sub_graph)

        return self

    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):
        if self.input_modify_:
            self.in_idx_map.set_idx(center_name, idxes)

        if self.output_modify_:
            self.ot_idx_map.set_idx(center_name, idxes)

            if self.unique_name() in leaf_names:
                return

            for n in self.node.next_nodes:
                if n.unique_name in sub_graph_dict.keys():
                    sub_graph_dict[n.unique_name].idx_forward(self.unique_name(),
                                                              center_name,
                                                              idxes,
                                                              sub_graph_dict,
                                                              leaf_names)


class PReLUChannelModifier(ChannelModifier):
    def register_mask(self, importance, graph_sparsity):
        remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
        self.masker().set_in_remove_idx(remove_idx)
        self.masker().set_ot_remove_idx(remove_idx)

        self.weight_mask[remove_idx] = 0

        self.masker().register_mask("weight", self.weight_mask)

    def modify_input(self, remove_idx):
        bn = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

        if bn.weight.shape[0] != len(preserve_idx):
            log.info(f'[PRELU] {self.unique_name()}: channel {bn.num_parameters} -> {len(preserve_idx)}')
            bn.weight = torch.nn.Parameter(bn.weight[preserve_idx])
            bn.num_parameters = len(preserve_idx)


class BatchNormChannelModifier(ChannelModifier):
    def register_mask(self, importance, graph_sparsity):
        remove_idx = calc_remove_idx(self.in_idx_map, importance, graph_sparsity, self.unique_name())
        self.masker().set_in_remove_idx(remove_idx)
        self.masker().set_ot_remove_idx(remove_idx)

        self.weight_mask[remove_idx] = 0

        self.masker().register_mask("weight", self.weight_mask)
        self.masker().register_mask("bias", self.weight_mask)

    def modify_input(self, remove_idx):
        bn = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

        if bn.weight.shape[0] != len(preserve_idx):
            log.info(f'[BN] {self.unique_name()}: channel {bn.num_features} -> {len(preserve_idx)}')
            bn.weight = torch.nn.Parameter(bn.weight[preserve_idx])
            bn.bias = torch.nn.Parameter(bn.bias[preserve_idx])
            bn.register_buffer('running_mean', bn.running_mean[preserve_idx])
            bn.register_buffer('running_var', bn.running_var[preserve_idx])
            bn.num_batches_tracked = bn.num_batches_tracked.zero_()
            bn.num_features = len(preserve_idx)


class LayerNormChannelModifier(BatchNormChannelModifier):
    def modify_input(self, remove_idx):
        ln = self.node.module
        preserve_idx = complementary_list([i for i in range(self.weight_mask.shape[0])], remove_idx)

        if ln.weight.shape[0] != len(preserve_idx):
            log.info(f'[LN] {self.unique_name()}: channel {ln.normalized_shape[0]} -> {len(preserve_idx)}')
            ln.weight = torch.nn.Parameter(ln.weight[preserve_idx])
            ln.bias = torch.nn.Parameter(ln.bias[preserve_idx])

            # Processing normalized_shape here is simply to treat it as (x1, x2, x3)
            ln.normalized_shape = (len(preserve_idx), ln.normalized_shape[1], ln.normalized_shape[2])


class ElementWiseChannelModifier(ChannelModifier):
    def traversal(self, input_modify, output_modify, sub_graph):
        self.input_modify_ = True
        self.output_modify_ = True
        if self not in sub_graph:
            sub_graph.append(self)
        else:
            return self
        assert (((input_modify and output_modify) is False) and ((input_modify or output_modify) is True))

        input_modifiers = []

        if input_modify:
            for n in self.node.prev_nodes:
                input_modifiers.append(n.modifier.traversal(False, True, sub_graph))

        elif output_modify:
            for n in self.node.prev_nodes:
                n.modifier.traversal(False, True, sub_graph)

        # 无论input还是output变换，都需要出发所有的下游节点变化
        for n in self.node.next_nodes:
            n.modifier.traversal(True, False, sub_graph)

        return self

    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):
        if self.input_modify_:
            self.in_idx_map.set_idx(center_name, idxes)
        if self.output_modify_ and self.input_modify_:
            self.ot_idx_map.set_idx(center_name, idxes)
        if self.unique_name() in leaf_names:
            return

        center_changed = [center_name]

        if len(self.in_idx_map.map_dict) > 1:
            max_group = 0

            for k, v in self.in_idx_map.map_dict.items():
                max_group = len(v) if len(v) > max_group else max_group

            for k, v in self.in_idx_map.map_dict.items():
                if len(v) != max_group:
                    center_changed.append(k)

            justify_group(self.in_idx_map.get_grouped_idx(max_group), self.in_idx_map)
            justify_group(self.ot_idx_map.get_grouped_idx(max_group), self.ot_idx_map)

        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                for center in center_changed:
                    sub_graph_dict[n.unique_name].idx_forward(self.unique_name(),
                                                              center,
                                                              self.ot_idx_map.map_dict[center],
                                                              sub_graph_dict,
                                                              leaf_names)


class CatChannelModifier(ChannelModifier):
    def group(self):
        return len(self.node.prev_nodes)

    def in_channel(self):
        ch = []
        for t in self.node.prev_tensors:
            ch.append(t.shape[1])

        return sum(ch)

    def traversal(self, input_modify, output_modify, sub_graph):
        if self not in sub_graph:
            sub_graph.append(self)
        else:
            return self
        self.input_modify_ = True
        self.output_modify_ = True
        assert (((input_modify and output_modify) is False) and ((input_modify or output_modify) is True))

        # Channel changes of different inputs are isolated
        if output_modify:
            for n in self.node.prev_nodes:
                n.modifier.traversal(False, True, sub_graph)
            for n in self.node.next_nodes:
                n.modifier.traversal(True, False, sub_graph)

        # 无论input还是output变换，都需要出发所有的下游节点变化
        for n in self.node.next_nodes:
            n.modifier.traversal(True, False, sub_graph)

        return self

    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):
        idxes_ = []
        start_idx = end_idx = 0
        cnt = 0

        for n in self.node.prev_nodes:
            if n.unique_name in sub_graph_dict.keys() \
                    and center_name in sub_graph_dict[n.unique_name].ot_idx_map.map_dict.keys():
                ot_ch = sub_graph_dict[n.unique_name].ot_channel()
                if isinstance(ot_ch, list):
                    ot_ch = ot_ch[self.node.prev_indices[cnt]]
                end_idx += ot_ch
                for idx_group in idxes:
                    idxes_.append(idx_group)
            else:
                ot_ch = create_modifier(n).ot_channel()
                if isinstance(ot_ch, list):
                    ot_ch = ot_ch[self.node.prev_indices[cnt]]
                end_idx += ot_ch

                # Expand the index of different center nodes to the same dimension
                idx_tmp = [-1 for _ in range(start_idx, end_idx)]
                idxes_.append(idx_tmp)
            cnt += 1
            start_idx = end_idx
        self.in_idx_map.set_idx(center_name, idxes_)
        self.ot_idx_map.set_idx(center_name, idxes_)

        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_forward(self.unique_name(), center_name, idxes_, sub_graph_dict,
                                                          leaf_names)

    def idx_back(self, pre_name, leaf_names, center_names, leaf_map_dict, sub_graph_dict):
        justify_group(leaf_map_dict, self.ot_idx_map)
        start_idx = end_idx = 0
        cnt = 0
        for n in self.node.prev_nodes:
            if n.unique_name in sub_graph_dict.keys() and sub_graph_dict[n.unique_name].output_modify_:
                sub_leaf_map_dict = {}
                ot_ch = sub_graph_dict[n.unique_name].ot_channel()
                if isinstance(ot_ch, list):
                    ot_ch = ot_ch[self.node.prev_indices[cnt]]
                end_idx += ot_ch

                # Split the index into the dimension corresponding to the input
                for k, v in leaf_map_dict.items():
                    sub_leaf_map_dict[k] = split_idx(v, start_idx, end_idx)

                sub_graph_dict[n.unique_name].idx_back(self.node.unique_name,
                                                       leaf_names,
                                                       center_names,
                                                       sub_leaf_map_dict,
                                                       sub_graph_dict)
            else:
                ot_ch = create_modifier(n).ot_channel()
                if isinstance(ot_ch, list):
                    ot_ch = ot_ch[self.node.prev_indices[cnt]]
                end_idx += ot_ch
            cnt += 1
            start_idx = end_idx

    def idx_back_forward(self, leaf_names, leaf_map_dict, sub_graph_dict, pre_name):
        tmp_leaf_dict = {}
        for k, v in leaf_map_dict.items():
            start_idx = end_idx = 0
            cnt = 0
            tmp_leaf_dict[k] = []
            for n in self.node.prev_nodes:
                if n.unique_name in sub_graph_dict.keys() and n.unique_name == pre_name:
                    ot_ch = sub_graph_dict[n.unique_name].ot_channel()
                    if isinstance(ot_ch, list):
                        ot_ch = ot_ch[self.node.prev_indices[cnt]]
                    end_idx += ot_ch
                    for idx_group in v:
                        tmp_leaf_dict[k].append(idx_group)

                else:
                    ot_ch = create_modifier(n).ot_channel()
                    if isinstance(ot_ch, list):
                        ot_ch = ot_ch[self.node.prev_indices[cnt]]
                    end_idx += ot_ch
                    if k in self.ot_idx_map.map_dict.keys():
                        idx_tmp = split_idx(self.ot_idx_map.map_dict[k], start_idx, end_idx)
                    else:
                        idx_tmp = [[-1 for _ in range(start_idx, end_idx)]]
                    for idx_t in idx_tmp:
                        tmp_leaf_dict[k].append(idx_t)

                cnt += 1
                start_idx = end_idx
        justify_group(tmp_leaf_dict, self.in_idx_map)
        self.ot_idx_map.set_idx_map(self.in_idx_map)

        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys() and sub_graph_dict[n.unique_name].input_modify_:
                sub_graph_dict[n.unique_name].idx_back_forward(
                    leaf_names, self.ot_idx_map.map_dict, sub_graph_dict, self.unique_name())


class SplitChannelModifier(ChannelModifier):
    def __init__(self, node):
        super(SplitChannelModifier, self).__init__(node=node)
        self.split_dict = {}
        start = end = 0
        for t in self.node.next_tensors:
            end += t.shape[1]
            for n in self.node.next_nodes:
                for t_ in n.prev_tensors:
                    if torch.equal(t, t_):
                        self.split_dict[n.unique_name] = (start, end)
            start = end

    def ot_channel(self):
        ch = []
        for t in self.node.next_tensors:
            ch.append(t.shape[1])

        return ch

    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):
        self.in_idx_map.set_idx(center_name, idxes)
        idxes_ = []

        # Split the index into the dimension corresponding to the output
        for n in self.node.next_nodes:
            start, end = self.split_dict[n.unique_name]
            split_idxes = split_idx(idxes, start, end)
            if split_idxes[0] not in idxes_:
                idxes_.append(split_idxes[0])
        self.ot_idx_map.set_idx(center_name, idxes_)

        cnt = 0
        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_forward(self.unique_name(), center_name, [idxes_[cnt]],
                                                          sub_graph_dict,
                                                          leaf_names)
            cnt += 1

    def idx_back(self, pre_name, leaf_names, center_names, leaf_map_dict, sub_graph_dict):
        # Expand the index of different center nodes to the same dimension
        for k, v in leaf_map_dict.items():
            if k in self.ot_idx_map.map_dict.keys():
                tmp = self.ot_idx_map.map_dict[k]
                orig_idxes = split_idx(tmp, self.split_dict[pre_name][0], self.split_dict[pre_name][1])
                if len(orig_idxes) > len(leaf_map_dict[k]):
                    leaf_map_dict[k] = orig_idxes

            if k not in self.ot_idx_map.map_dict.keys():
                tmp = [[-1 for _ in range(self.in_channel())]]

            self.ot_idx_map.map_dict[k] = []
            start, end = self.split_dict[pre_name]

            if start != 0:
                self.ot_idx_map.map_dict[k] += split_idx(tmp, 0, start)
            self.ot_idx_map.map_dict[k] += leaf_map_dict[k]

            if end != self.in_channel():
                self.ot_idx_map.map_dict[k] += split_idx(tmp, end, self.in_channel())

        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys() and self.split_dict[n.unique_name] == self.split_dict[pre_name]:
                sub_graph_dict[n.unique_name].idx_back_forward(leaf_names,
                                                               leaf_map_dict,
                                                               sub_graph_dict,
                                                               self.unique_name())

        if self.unique_name() in center_names:
            return

        for n in self.node.prev_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_back(pre_name,
                                                       leaf_names,
                                                       center_names,
                                                       self.ot_idx_map.map_dict,
                                                       sub_graph_dict)

    def idx_back_forward(self, leaf_names, leaf_map_dict, sub_graph_dict, pre_name):
        justify_group(leaf_map_dict, self.in_idx_map)
        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                tmp_sub_leaf_map = {}
                start, end = self.split_dict[n.unique_name]
                for k, v in leaf_map_dict.items():
                    tmp_sub_leaf_map[k] = split_idx(v, start, end)
            sub_graph_dict[n.unique_name].idx_back_forward(leaf_names, tmp_sub_leaf_map, sub_graph_dict,
                                                           self.unique_name())

    def register_mask(self, importance, graph_sparsity):
        # arg case：torch.split(a, 2)，torch.split(a, b), torch.split(a, [2,4])

        arg_list = self.node.module.args_parsed
        if len(arg_list) > 1:
            if arg_list[1].isdigit():
                ch = int(int(arg_list[1]) * (1 - graph_sparsity[0]))
                self.node.module.args_parsed[1] = str(ch)
            if arg_list[1][0] == '[':
                ch = eval(arg_list[1])
                ch = [int(i * (1 - graph_sparsity[0])) for i in ch]
                self.node.module.args_parsed[1] = str(ch)
            self.node.module.args_string = ''
            for tmp in self.node.module.args_parsed:
                if self.node.module.args_string != '':
                    tmp = ', ' + tmp
                self.node.module.args_string += tmp


class ReshapeChannelModifier(ChannelModifier):
    def idx_forward(self, pre_name, center_name, idxes, sub_graph_dict, leaf_names):
        self.in_idx_map.set_idx(center_name, idxes)
        before_tensor = self.node.prev_tensors[0]
        b_shape = before_tensor.shape
        after_tensor = self.node.next_tensors[0]
        zoom = after_tensor.shape[1] // b_shape[1]
        tmp = torch.tensor([-1 for i in range(before_tensor.numpy().size)]).reshape(b_shape)
        ct = 0

        for idxes_ in idxes:
            for i in idxes_:
                tmp[:, ct] = i
                ct += 1

        tmp = tmp.reshape(after_tensor.shape)
        after_idx = []
        for i in range(tmp.shape[1]):
            z = tmp[:, i, ]
            unique_z = z.unique()
            if len(unique_z) == 1:
                after_idx.append(int(unique_z[0]))
            else:
                log.error("Currently only supports one channel mapping to multiple channels.")
                assert False

        after_idxes = []
        s = e = 0
        for idxes_ in idxes:
            idxes_len = len(idxes_)
            e += idxes_len * zoom
            after_idxes.append(after_idx[s:e])
            s = e

        self.ot_idx_map.set_idx(center_name, after_idxes)
        if self.unique_name() in leaf_names:
            return
        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_forward(self.unique_name(), center_name, after_idxes, sub_graph_dict,
                                                          leaf_names)

    def idx_back(self, pre_name, leaf_names, center_names, leaf_map_dict, sub_graph_dict):
        justify_group(leaf_map_dict, self.ot_idx_map)

        for k, v in leaf_map_dict.items():
            before_idxes = []
            for idx in v:
                tmp = []
                for i in idx:
                    if i == -1 or i not in tmp:
                        tmp.append(i)
                before_idxes.append(tmp)
            leaf_map_dict[k] = before_idxes

        justify_group(leaf_map_dict, self.in_idx_map)
        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_back_forward(leaf_names, leaf_map_dict, sub_graph_dict, pre_name)

        if self.unique_name() in center_names:
            return
        for n in self.node.prev_nodes:
            if n.unique_name in sub_graph_dict.keys():
                sub_graph_dict[n.unique_name].idx_back(pre_name, leaf_names, center_names, leaf_map_dict,
                                                       sub_graph_dict)

    def idx_back_forward(self, leaf_names, leaf_map_dict, sub_graph_dict, pre_name):
        justify_group(leaf_map_dict, self.in_idx_map)
        before_tensor = self.node.prev_tensors[0]
        b_shape = before_tensor.shape
        after_tensor = self.node.next_tensors[0]
        zoom = after_tensor.shape[1] // b_shape[1]

        tmp_map_dict = {}
        for k, v in leaf_map_dict.items():
            tmp_map_dict[k] = []
            for group_idxes in v:
                tmp_ = []
                for i in group_idxes:
                    tmp_ += [i] * zoom
                tmp_map_dict[k].append(tmp_)

        for n in self.node.next_nodes:
            if n.unique_name in sub_graph_dict.keys() and sub_graph_dict[n.unique_name].input_modify_:
                sub_graph_dict[n.unique_name].idx_back_forward(leaf_names,
                                                               tmp_map_dict,
                                                               sub_graph_dict,
                                                               self.unique_name())


MODIFIERS = {
    nn.Conv2d: ConvChannelModifier,
    nn.ConvTranspose2d: ConvTransChannelModifier,
    nn.Conv1d: ConvChannelModifier,
    nn.ConvTranspose1d: ConvTransChannelModifier,
    nn.Linear: LinearChannelModifier,
    nn.BatchNorm2d: BatchNormChannelModifier,
    nn.BatchNorm1d: BatchNormChannelModifier,
    "add": ElementWiseChannelModifier,
    "mul": ElementWiseChannelModifier,
    "truediv": ElementWiseChannelModifier,
    "sub": ElementWiseChannelModifier,
    "cat": CatChannelModifier,
    'view': ReshapeChannelModifier,
    "flatten": ReshapeChannelModifier,
    nn.Flatten: ReshapeChannelModifier,
    'reshape': ReshapeChannelModifier,
    nn.PReLU: PReLUChannelModifier,
    nn.LayerNorm: LayerNormChannelModifier,
    "split": SplitChannelModifier,
}


def create_modifier(n):
    for key in MODIFIERS.keys():
        if type(key) == str:
            if n.kind() == key:
                return MODIFIERS[key](n)
        elif isinstance(n.module, key):
            return MODIFIERS[key](n)

    # ChannelModifier is used by default
    return ChannelModifier(n)


def get_subgraph(graph: TraceGraph, node: TraceNode):
    for n in graph.forward_nodes + graph.output_nodes + graph.constant_nodes:
        setattr(n, "modifier", create_modifier(n))

    sub_graph = []
    node.modifier.traversal(False, True, sub_graph)

    for n in graph.forward_nodes + graph.output_nodes + graph.constant_nodes:
        delattr(n, "modifier")

    sub_graph = sorted(sub_graph, key=lambda i: i.node.forward_order)

    return sub_graph


def get_subgraphs(graph: TraceGraph, center_nodes, remove_redundancy=True):
    sub_graphs = []
    for n in center_nodes:
        sub_graphs.append(get_subgraph(graph, n))

    if remove_redundancy:
        unique_sub_graphs = []

        while len(sub_graphs) > 0:
            sub_graph = sub_graphs.pop(0)
            subgraph_key = [x.node.unique_name for x in sub_graph]

            is_subset = False

            for tmp_sub_graph in sub_graphs + unique_sub_graphs:
                tmp_subgraph_key = [x.node.unique_name for x in tmp_sub_graph]
                if set(subgraph_key).issubset(set(tmp_subgraph_key)):
                    is_subset = is_subgraph(sub_graph, tmp_sub_graph)

            if is_subset:
                continue

            unique_sub_graphs.append(sub_graph)

        return unique_sub_graphs

    return sub_graphs


def is_subgraph(subgraph, tmp_subgraph):
    for m1 in subgraph:
        for m2 in tmp_subgraph:
            if m1.node.unique_name == m2.node.unique_name:
                if m1.input_modify_ != m2.input_modify_ or m1.output_modify_ != m2.output_modify_:
                    return False
    return True


def register_sub_masker(sub_graph, importance, sparsity):
    sorted_sub_graph = sorted(sub_graph, key=lambda m: m.node.forward_order)
    sub_graph_dict = {m.unique_name(): m for m in sorted_sub_graph}
    graph_sparsity = []
    center_names = []
    leaf_names = []
    for m in sorted_sub_graph:
        # The dependency analysis of the following operators is temporarily not supported, skip the subgraph directly
        if m.node.type() in ['permute', 'unsqueeze', 'transpose']:
            log.warning(f"Skip Subgraph of {m.node.unique_name}")
            return
        if m.output_modify_ and importance.get(m.unique_name(), None) is not None:
            graph_sparsity.append(sparsity[m.unique_name()])
            center_names.append(m.unique_name())

        if not is_dw_conv(m.node.module) and m.node.type() in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear,
                                                               nn.Conv1d, nn.ConvTranspose1d, 'output'] \
                and m.input_modify_:
            leaf_names.append(m.unique_name())

    if len(set(graph_sparsity)) > 1:
        log.error(f"All node's sparsity in one subgraph must be the same:{[(n, sparsity[n]) for n in center_names]}")
        assert False

    if graph_sparsity[0] == 0.0:
        log.debug(f"Skip Subgraph {[(n, sparsity[n]) for n in center_names]}")
        return

    center_idx_forward(sorted_sub_graph, center_names, leaf_names, sub_graph_dict)

    leaf_idx_back(sorted_sub_graph, leaf_names, center_names, sub_graph_dict)

    remove_idx_calc(sorted_sub_graph, importance, graph_sparsity)


def center_idx_forward(sub_graph, center_names, leaf_names, sub_graph_dict):
    for m in sub_graph:
        if m.node.unique_name in center_names:
            u_name = m.node.unique_name
            idx = list(range(0, m.ot_channel()))
            m.ot_idx_map.set_idx(u_name, [idx])
            leaf_dict = m.ot_idx_map.get_grouped_idx(m.group())
            justify_group(leaf_dict, m.ot_idx_map)
            m.idx_forward(u_name, u_name, m.ot_idx_map.map_dict[u_name], sub_graph_dict, leaf_names)


def leaf_idx_back(sub_graph, leaf_names, center_names, sub_graph_dict):
    for m in sub_graph:
        if m.node.unique_name in leaf_names:
            leaf_dict = m.in_idx_map.get_grouped_idx(m.group())
            m.idx_back(m.unique_name(), leaf_names, center_names, leaf_dict, sub_graph_dict)


def remove_idx_calc(sub_graph, importance, graph_sparsity):
    for m in sub_graph:
        m.register_mask(importance, graph_sparsity)
        m.enable_mask()


def calc_remove_idx(idx_map, importance, graph_sparsity, unique_name):
    pos_list = set()
    channel = idx_map.get_channel_number()
    remove_idx = []
    importance_sum = torch.zeros((channel,), device=next(iter(importance.values())).device)

    # Accumulate importance by channel
    for k, _v in idx_map.map_dict.items():
        v, _ = list_flatten(_v)
        start_pos = end_pos = 0
        for group_idx in _v:
            end_pos += len(group_idx)
            tmp_end = end_pos
            while start_pos < end_pos and v[0][start_pos] == -1:
                start_pos += 1
            while start_pos < end_pos and v[0][end_pos - 1] == -1:
                end_pos -= 1
            if start_pos < end_pos:
                pos_list.add((start_pos, end_pos))
            start_pos = end_pos = tmp_end

        pos_list = deduplicate_range(pos_list)

        pos_i = []
        pos_idx = []
        for i, idx in enumerate(v[0]):
            if idx == -1:
                continue
            pos_i.append(i)
            pos_idx.append(idx)

        importance_sum[pos_i] += importance[k][pos_idx]

    for pos in pos_list:
        start, end = pos
        _, idx = torch.topk(importance_sum[start:end], int(graph_sparsity[0] * (end - start)), largest=False)
        idx += start
        remove_idx.extend(idx.tolist())
    remove_idx.sort()

    if graph_sparsity[0] > 0 and len(remove_idx) == 0:
        log.warning(f"Sparsity is too small to prune ({unique_name})({graph_sparsity[0]})({idx_map.map_dict})")

    return remove_idx


class ChannelModifierGraph(object):
    graph: TraceGraph
    center_nodes: typing.List[TraceNode]
    sub_graphs: typing.List[typing.List[ChannelModifier]]

    def __init__(self, graph: TraceGraph, center_nodes):
        """ Initialize a channel modifier for a calculation graph

        Args:
            graph: Compute graph generated by tracer
            center_nodes: Operators that actively modify the channel

        """

        self.graph = graph
        self.center_nodes = center_nodes
        self.sub_graphs = get_subgraphs(graph, center_nodes)
        self.reset_masker()

    def reset_masker(self):
        self.unregister_masker()
        for n in self.graph.forward_nodes:
            masker.ChannelMasker(n.module, n.unique_name)

    def unregister_masker(self):
        mask_applied = False
        for sub_graph in self.sub_graphs:
            for m in sub_graph:
                m.reset_mask()
                mask_applied = m.mask_applied or mask_applied

        for n in self.graph.forward_nodes:
            if hasattr(n.module, "masker"):
                n.module.masker.unregister_all()
                delattr(n.module, "masker")

        if mask_applied:
            self.graph.inited = False

    def get_all_modifier(self):
        result = []
        for i in self.sub_graphs:
            result.extend(i)
        return result
