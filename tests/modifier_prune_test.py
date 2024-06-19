import os
import random
import unittest

import torch
import torch.nn as nn

from distutils.version import LooseVersion
from operator import add, mul, sub, truediv

# from unittest.case import SkipTest

# import numpy as np

# import torch.nn.functional as F

# from common_utils import IS_CI

from interval import Interval

from tinynn.converter import TFLiteConverter

# from tinynn.graph import modifier
# from tinynn.graph.modifier import l2_norm, update_weight_metric
from tinynn.graph.modifier import l2_norm

# from tinynn.graph.tracer import model_tracer, trace
from tinynn.prune.oneshot_pruner import OneShotChannelPruner

# from tinynn.util.util import import_from_path


CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
MODULE_INIT = True

print(LooseVersion)
print(add, mul, sub, truediv)


def model_generate(model, dummy_input, name='test.tflite'):
    converter = TFLiteConverter(model, dummy_input, os.path.join(CURRENT_PATH, 'out', name))
    converter.convert()


def removed_idx_group_check(removed_idx, total_idx_len, removed_idx_len, group, offset=0):
    for i in range(group):
        remove_group_len = removed_idx_len // group
        for j in range(i * remove_group_len, i * remove_group_len + remove_group_len):
            idx_group_len = total_idx_len // group
            assert removed_idx[j] in Interval(
                offset + i * idx_group_len, offset + i * idx_group_len + idx_group_len, upper_closed=False
            )


def get_rd_lst(length):
    rd_lst = random.sample(range(0, 100000), length)
    random.shuffle(rd_lst)

    print(rd_lst)
    return rd_lst


def get_topk(lst, k, offset=0):
    idx_lst = [(i, lst[i]) for i in range(len(lst))]
    sorted_lst = sorted(idx_lst, key=lambda x: x[1])
    sorted_lst_k = sorted_lst[:k]
    idx = [sorted_lst_k[i][0] + offset for i in range(len(sorted_lst_k))]

    return sorted(idx)


def init_conv_by_list(conv, ch_value):
    if not MODULE_INIT:
        return

    assert conv.weight.shape[0] == len(ch_value)

    for i in range(len(ch_value)):
        conv.weight.data[i, :] = ch_value[i]


def init_rnn_by_list(rnn, ch_value):
    if not MODULE_INIT:
        return

    for i in range(len(ch_value)):
        rnn.weight_ih_l0.data[i, :] = ch_value[i]
        rnn.weight_hh_l0.data[i, :] = ch_value[i]

    if rnn.bidirectional:
        for i in range(len(ch_value)):
            rnn.weight_ih_l0_reverse.data[i, :] = ch_value[i]
            rnn.weight_hh_l0_reverse.data[i, :] = ch_value[i]


class ModifierTester(unittest.TestCase):
    def test_pixel_shuffle_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
                self.pixelshuffle = nn.PixelShuffle(2)
                self.conv1 = nn.Conv2d(4, 4, (3, 3), padding=(1, 1))

            def forward(self, x):
                x = self.conv0(x)
                x = self.pixelshuffle(x)
                x = self.conv1(x)

                return x

        def test_func():
            model = TestModel()

            ch_4 = get_rd_lst(4)
            ch_16 = get_rd_lst(16)

            init_conv_by_list(model.conv0, ch_16)
            init_conv_by_list(model.conv1, ch_4)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
            importance_conv0_merge = [sum(importance_conv0[i : i + 4]) for i in range(0, len(importance_conv0), 4)]

            conv0_idxes = get_topk(importance_conv0_merge, 2)

            prune_idxes = []
            for i in conv0_idxes:
                prune_idxes += [j for j in range(i * 4, i * 4 + 4)]

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)

            assert m_conv0.dim_changes_info.pruned_idx_o == prune_idxes

        for i in range(20):
            test_func()


if __name__ == '__main__':
    unittest.main()
