import os
import random
import unittest

from distutils.version import LooseVersion
from operator import add, mul, sub, truediv
from unittest.case import SkipTest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from interval import Interval

from tinynn.converter import TFLiteConverter
from tinynn.graph import modifier
from tinynn.graph.modifier import l2_norm, update_weight_metric
from tinynn.graph.tracer import model_tracer, trace
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.prune.oneshot_pruner import OneShotChannelPruner as OneShotChannelPrunerOld
from tinynn.util.util import import_from_path, set_global_log_level

from common_utils import IS_CI

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))
MODULE_INIT = True


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
    def test_cat_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 8, (3, 3))
                self.conv1 = nn.Conv2d(3, 8, (3, 3))
                self.conv2 = nn.Conv2d(16, 32, (3, 3))
                self.linear = nn.Linear(800, 100)

            def forward(self, x):
                x0 = self.conv0(x)
                x1 = self.conv1(x)

                cat0 = torch.cat([x0, x1], dim=1)
                conv2 = self.conv2(cat0)
                view0 = conv2.view(1, -1)
                linear0 = self.linear(view0)
                return linear0

        def test_func():
            model = TestModel()

            rd_lst_8 = get_rd_lst(8)
            rd_lst_32 = get_rd_lst(32)
            init_conv_by_list(model.conv0, rd_lst_8)
            init_conv_by_list(model.conv1, rd_lst_8)
            init_conv_by_list(model.conv2, rd_lst_32)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.register_mask()

            conv0_idxes = get_topk(importance_conv0, 4)
            conv1_idxes = get_topk(importance_conv1, 4)
            conv2_idxes = get_topk(importance_conv2, 16)

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_linear0 = pruner.graph_modifier.get_modifier(model.linear)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_i == conv0_idxes + [i + 8 for i in conv0_idxes]
            assert m_conv2.dim_changes_info.pruned_idx_o == conv2_idxes
            assert m_linear0.dim_changes_info.pruned_idx_i == [
                i for j in conv2_idxes for i in range(j * 25, j * 25 + 25)
            ]

            pruner.apply_mask()

            assert model.conv1.out_channels == 4
            assert model.conv0.out_channels == 4
            assert model.conv2.in_channels == 8
            assert model.conv2.out_channels == 16
            assert model.linear.in_features == 400
            assert model.linear.out_features == 100

            model(torch.ones(1, 3, 9, 9))

            print("test over")

        for i in range(20):
            test_func()

    def test_cat_add_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 8, (3, 3), padding=(1, 1))
                self.conv1 = nn.Conv2d(3, 8, (3, 3), padding=(1, 1))
                self.conv2 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
                self.conv3 = nn.Conv2d(16, 32, (3, 3), padding=(1, 1))

            def forward(self, x):
                x0 = self.conv0(x)
                x1 = self.conv1(x)
                x2 = self.conv2(x)

                cat0 = torch.cat([x0, x1], dim=1)
                add0 = torch.add(cat0, x2)
                return self.conv3(add0)

        def test_func():
            model = TestModel()

            while True:
                ch_8 = get_rd_lst(8)
                ch_16 = get_rd_lst(16)
                ch_32 = get_rd_lst(32)

                init_conv_by_list(model.conv0, ch_8)
                init_conv_by_list(model.conv1, ch_8)
                init_conv_by_list(model.conv2, ch_16)
                init_conv_by_list(model.conv3, ch_32)

                importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
                importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
                importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()
                importance_conv3 = l2_norm(model.conv3.weight, model.conv3).tolist()

                importance_add0 = list(map(add, importance_conv0 + importance_conv1, importance_conv2))

                # Duplicate values may lead to multiple possibilities for remove idx
                if len(set(importance_add0)) == len(importance_add0):
                    break

            conv0_idxes = get_topk(importance_add0[:8], 4)
            conv1_idxes = get_topk(importance_add0[8:], 4)
            conv3_idxes = get_topk(importance_conv3, 16)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv3 = pruner.graph_modifier.get_modifier(model.conv3)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_o == conv0_idxes + [8 + i for i in conv1_idxes]
            assert m_conv3.dim_changes_info.pruned_idx_o == conv3_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 4
            assert model.conv1.out_channels == 4
            assert model.conv2.out_channels == 8
            assert model.conv3.in_channels == 8
            assert model.conv3.out_channels == 16

        for i in range(20):
            test_func()

    def test_flatten_graph(self):
        class TestFlattenModel(nn.Module):
            def __init__(self):
                super(TestFlattenModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3))
                self.conv1 = nn.Conv2d(16, 32, (3, 3))
                self.dropout = nn.Dropout()
                self.linear1 = nn.Linear(800, 100)
                self.linear2 = nn.Linear(100, 10)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                flatten0 = torch.flatten(conv1, 1)
                dropout0 = self.dropout(flatten0)
                linear1 = self.linear1(dropout0)
                linear2 = self.linear2(linear1)
                return linear2

        def test_func():
            model = TestFlattenModel()

            ch_32 = get_rd_lst(32)
            init_conv_by_list(model.conv1, ch_32)

            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
            conv1_idxes = get_topk(importance_conv1, 24)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.75, "metrics": "l2_norm"})

            pruner.register_mask()

            linear1_idxes = np.array([i for i in range(800)])
            linear1_idxes = linear1_idxes.reshape([32, 25])
            linear1_idxes = linear1_idxes[conv1_idxes, :]
            linear1_idxes = linear1_idxes.reshape([600]).tolist()

            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_linear1 = pruner.graph_modifier.get_modifier(model.linear1)

            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_linear1.dim_changes_info.pruned_idx_i == linear1_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 4
            assert model.conv1.out_channels == 8
            assert model.linear1.in_features == 200
            assert model.linear1.out_features == 25
            assert model.linear2.in_features == 25
            assert model.linear2.out_features == 10

        for i in range(20):
            test_func()

    def test_loop_cat_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, (3, 3))
                self.conv2 = nn.Conv2d(64, 128, (3, 3))
                self.relu1 = torch.nn.modules.activation.ReLU(inplace=True)
                self.relu2 = torch.nn.modules.activation.ReLU(inplace=True)
                self.relu3 = torch.nn.modules.activation.ReLU(inplace=True)
                self.relu4 = torch.nn.modules.activation.ReLU(inplace=True)

            def forward(self, x):
                conv1 = self.conv1(x)
                relu1 = self.relu1(conv1)
                relu2 = self.relu2(conv1)
                relu3 = self.relu3(conv1)
                relu4 = self.relu4(conv1)
                z = torch.cat([relu1, relu2, relu3, relu4], dim=1)
                return self.conv2(z)

        def test_func():
            model = TestModel()

            ch_16 = get_rd_lst(16)
            init_conv_by_list(model.conv1, ch_16)

            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
            conv1_idxes = get_topk(importance_conv1, 8)

            pruner = OneShotChannelPruner(model, torch.randn((1, 3, 9, 9)), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.register_mask()

            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_i == [j + i * 16 for i in range(4) for j in conv1_idxes]

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))
            assert model.conv1.out_channels == 8
            assert model.conv2.in_channels == 32
            assert model.conv2.out_channels == 64

        for i in range(20):
            test_func()

    def test_group_cat_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 16, (3, 3))
                self.conv2 = nn.Conv2d(3, 32, (3, 3))
                self.conv3 = nn.Conv2d(48, 64, (3, 3), groups=4)

            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(x)
                cat0 = torch.cat([conv1, conv2], dim=1)
                return self.conv3(cat0)

        def test_func():
            model = TestModel()

            ch_16 = get_rd_lst(16)
            ch_32 = get_rd_lst(32)

            init_conv_by_list(model.conv1, ch_16)
            init_conv_by_list(model.conv2, ch_32)

            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()

            conv1_idxes_g1 = get_topk(importance_conv1[:12], 6)
            conv1_idxes_g2 = get_topk(importance_conv1[12:], 2, offset=12)
            conv1_idxes = conv1_idxes_g1 + conv1_idxes_g2

            conv2_idxes_g1 = get_topk(importance_conv2[:8], 4)
            conv2_idxes_g2 = get_topk(importance_conv2[8:20], 6, offset=8)
            conv2_idxes_g3 = get_topk(importance_conv2[20:], 6, offset=20)
            conv2_idxes = conv2_idxes_g1 + conv2_idxes_g2 + conv2_idxes_g3

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv3 = pruner.graph_modifier.get_modifier(model.conv3)

            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_o == conv2_idxes
            assert m_conv3.dim_changes_info.pruned_idx_i == conv1_idxes + [i + 16 for i in conv2_idxes]

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv1.out_channels == 8
            assert model.conv2.out_channels == 16
            assert model.conv3.in_channels == 24
            assert model.conv3.out_channels == 32

        for i in range(20):
            test_func()

    # def test_nonaligned_cat_graph(self):
    #     class TestModel(nn.Module):
    #         def __init__(self):
    #             super(TestModel, self).__init__()
    #             self.conv1 = nn.Conv2d(3, 8, (3, 3))
    #             self.conv2 = nn.Conv2d(3, 4, (3, 3))
    #             self.conv3 = nn.Conv2d(3, 4, (3, 3))
    #             self.conv4 = nn.Conv2d(16, 64, (3, 3))
    #
    #         def forward(self, x):
    #             conv1 = self.conv1(x)
    #             conv2 = self.conv2(x)
    #             conv3 = self.conv3(x)
    #             cat0 = torch.cat([conv1, conv2, conv3], dim=1)
    #             return self.conv4(cat0)
    #
    #     model = TestModel()
    #     pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.25, "metrics": "l2_norm"})
    #     pruner.prune()
    #
    #     model(torch.ones(1, 3, 9, 9))
    #
    #     assert model.conv1.out_channels == 6
    #     assert model.conv2.out_channels == 3
    #     assert model.conv3.out_channels == 3
    #     assert model.conv4.in_channels == 12
    #     assert model.conv4.out_channels == 48

    def test_group_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3))
                self.conv1 = nn.Conv2d(16, 32, (3, 3), groups=8)
                self.conv2 = nn.Conv2d(32, 32, (3, 3))

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                return self.conv2(conv1)

        def test_func():

            model = TestModel()

            ch_16 = get_rd_lst(16)
            ch_32 = get_rd_lst(32)
            init_conv_by_list(model.conv0, ch_16)
            init_conv_by_list(model.conv1, ch_32)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()

            conv0_idxes = []
            conv1_idxes = []

            for i in range(8):
                conv0_idxes += get_topk(importance_conv0[i * 2 : (i + 1) * 2], 1, offset=i * 2)
                conv1_idxes += get_topk(importance_conv1[i * 4 : (i + 1) * 4], 2, offset=i * 4)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_i == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_i == conv1_idxes

            removed_idx_group_check(m_conv0.dim_changes_info.pruned_idx_o, 16, 8, 8)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_i, 16, 8, 8)

            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o, 32, 16, 8)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_i, 32, 16, 8)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 8
            assert model.conv1.in_channels == 8
            assert model.conv1.out_channels == 16
            assert model.conv2.out_channels == 16

        for i in range(20):
            test_func()

    def test_multi_group_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3))
                self.conv1 = nn.Conv2d(16, 32, (3, 3), groups=4)
                self.conv2 = nn.Conv2d(16, 32, (3, 3), groups=8)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                conv2 = self.conv2(conv0)
                return conv1, conv2

        def test_func():
            model = TestModel()

            ch_16 = get_rd_lst(16)
            ch_32 = get_rd_lst(32)

            ch_16 = [88, 115, 930, 832, 723, 89, 45, 861, 715, 607, 813, 359, 792, 147, 262, 2]
            ch_32 = [
                82,
                846,
                54,
                510,
                886,
                191,
                914,
                529,
                55,
                4,
                666,
                468,
                372,
                535,
                699,
                273,
                310,
                162,
                543,
                360,
                348,
                569,
                99,
                692,
                194,
                991,
                559,
                612,
                109,
                727,
                199,
                113,
            ]
            init_conv_by_list(model.conv0, ch_16)
            init_conv_by_list(model.conv1, ch_32)
            init_conv_by_list(model.conv2, ch_32)

            conv0_idxes = []
            conv1_idxes = []
            conv2_idxes = []

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()

            for i in range(4):
                conv1_idxes += get_topk(importance_conv1[i * 8 : (i + 1) * 8], 4, offset=i * 8)

            for i in range(8):
                conv0_idxes += get_topk(importance_conv0[i * 2 : (i + 1) * 2], 1, offset=i * 2)
                conv2_idxes += get_topk(importance_conv2[i * 4 : (i + 1) * 4], 2, offset=i * 4)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_o == conv2_idxes

            removed_idx_group_check(m_conv0.dim_changes_info.pruned_idx_o, 16, 8, 8)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_i, 16, 8, 8)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_i, 16, 8, 8)

            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o, 32, 16, 4)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_o, 32, 16, 8)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 8
            assert model.conv1.in_channels == 8
            assert model.conv1.out_channels == 16
            assert model.conv2.out_channels == 16

        for i in range(20):
            test_func()

    def test_add_cat_group_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3))
                self.conv1 = nn.Conv2d(16, 32, (3, 3), groups=2)
                self.conv2 = nn.Conv2d(16, 32, (3, 3), groups=4)
                self.conv3 = nn.Conv2d(3, 16, (3, 3))
                self.conv4 = nn.Conv2d(16, 32, (3, 3), groups=8)
                self.conv5 = nn.Conv2d(64, 64, (3, 3))

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                conv2 = self.conv2(conv0)
                add1 = conv1.__add__(conv2)
                conv3 = self.conv3(x)
                conv4 = self.conv4(conv3)
                cat0 = torch.cat([add1, conv4], dim=1)
                return self.conv5(cat0)

        def test_func():
            while True:
                model = TestModel()

                ch_conv0 = get_rd_lst(16)
                ch_conv1 = get_rd_lst(32)
                ch_conv2 = get_rd_lst(32)
                ch_conv4 = get_rd_lst(32)

                init_conv_by_list(model.conv0, ch_conv0)
                init_conv_by_list(model.conv1, ch_conv1)
                init_conv_by_list(model.conv2, ch_conv2)
                init_conv_by_list(model.conv4, ch_conv4)

                importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
                importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
                importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()
                importance_conv4 = l2_norm(model.conv4.weight, model.conv4).tolist()
                importance_conv12 = list(map(add, importance_conv1, importance_conv2))

                if len(importance_conv12) == len(set(importance_conv12)):
                    break

            conv0_idxes = []
            conv4_idxes = []
            conv12_idxes = []

            for i in range(4):
                conv0_idxes += get_topk(importance_conv0[i * 4 : (i + 1) * 4], 2, i * 4)
                conv12_idxes += get_topk(importance_conv12[i * 8 : (i + 1) * 8], 4, i * 8)

            for i in range(8):
                conv4_idxes += get_topk(importance_conv4[i * 4 : (i + 1) * 4], 2, i * 4)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv4 = pruner.graph_modifier.get_modifier(model.conv4)
            m_conv5 = pruner.graph_modifier.get_modifier(model.conv5)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv12_idxes
            assert m_conv2.dim_changes_info.pruned_idx_o == conv12_idxes
            assert m_conv5.dim_changes_info.pruned_idx_i == conv12_idxes + [i + 32 for i in conv4_idxes]

            removed_idx_group_check(m_conv0.dim_changes_info.pruned_idx_o, 16, 8, 4)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_i, 16, 8, 4)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_i, 16, 8, 4)

            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o, 32, 16, 4)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_o, 32, 16, 4)

            removed_idx_group_check(m_conv4.dim_changes_info.pruned_idx_i, 16, 8, 8)
            removed_idx_group_check(m_conv4.dim_changes_info.pruned_idx_o, 32, 16, 8)

            removed_idx_group_check(m_conv5.dim_changes_info.pruned_idx_i[:16], 32, 16, 4)
            removed_idx_group_check(m_conv5.dim_changes_info.pruned_idx_i[16:], 32, 16, 8, offset=32)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(20):
            test_func()

    def test_multi_cat_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 8, (3, 3))
                self.conv1 = nn.Conv2d(8, 16, (3, 3))
                self.conv2 = nn.Conv2d(8, 16, (3, 3), groups=4)
                self.conv3 = nn.Conv2d(8, 16, (3, 3))
                self.conv4 = nn.Conv2d(32, 64, (3, 3))
                self.conv5 = nn.Conv2d(32, 64, (3, 3))

            def forward(self, x):
                conv0 = self.conv0(x)
                relu0 = F.relu(conv0)
                x1 = self.conv1(relu0)
                x2 = self.conv2(relu0)
                x3 = self.conv3(relu0)
                cat0 = torch.cat([x1, x2], dim=1)
                cat1 = torch.cat([x2, x3], dim=1)
                cat0 = self.conv4(cat0)
                cat1 = self.conv5(cat1)
                return cat0, cat1

        def test_func():

            model = TestModel()

            conv0_ch = get_rd_lst(8)
            conv1_ch = get_rd_lst(16)
            conv2_ch = get_rd_lst(16)
            conv3_ch = get_rd_lst(16)

            init_conv_by_list(model.conv0, conv0_ch)
            init_conv_by_list(model.conv1, conv1_ch)
            init_conv_by_list(model.conv2, conv2_ch)
            init_conv_by_list(model.conv3, conv3_ch)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0)
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1)
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2)
            importance_conv3 = l2_norm(model.conv3.weight, model.conv3)

            conv0_idxes = []
            conv2_idxes = []

            conv1_idxes = get_topk(importance_conv1, 8)
            conv3_idxes = get_topk(importance_conv3, 8)

            for i in range(4):
                conv0_idxes += get_topk(importance_conv0[i * 2 : (i + 1) * 2], 1, offset=i * 2)
                conv2_idxes += get_topk(importance_conv2[i * 4 : (i + 1) * 4], 2, offset=i * 4)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv3 = pruner.graph_modifier.get_modifier(model.conv3)
            m_conv4 = pruner.graph_modifier.get_modifier(model.conv4)
            m_conv5 = pruner.graph_modifier.get_modifier(model.conv5)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_o == conv2_idxes
            assert m_conv3.dim_changes_info.pruned_idx_o == conv3_idxes

            removed_idx_group_check(m_conv0.dim_changes_info.pruned_idx_o, 8, 4, 4)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_i, 8, 4, 4)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_i, 8, 4, 4)
            removed_idx_group_check(m_conv3.dim_changes_info.pruned_idx_i, 8, 4, 4)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_o, 16, 8, 4)
            removed_idx_group_check(m_conv4.dim_changes_info.pruned_idx_i[8:], 16, 8, 4, offset=16)
            removed_idx_group_check(m_conv5.dim_changes_info.pruned_idx_i[:8], 16, 8, 4)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(20):
            test_func()

    def test_split_group_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, (3, 3))
                self.conv2 = nn.Conv2d(3, 16, (3, 3))
                self.conv3 = nn.Conv2d(3, 16, (3, 3))
                self.conv4 = nn.Conv2d(16, 32, (3, 3), groups=2)
                self.conv5 = nn.Conv2d(16, 32, (3, 3), groups=4)

            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(x)
                conv3 = self.conv3(x)
                size = conv1.shape[1] // 2
                sp1, sp2 = torch.split(conv1, size, 1)
                add0 = conv2 + sp1
                add1 = sp2 + conv3
                return self.conv4(add0), self.conv5(add1)

        def test_func():
            model = TestModel()

            while True:
                ch_conv1 = get_rd_lst(32)
                ch_conv2 = get_rd_lst(16)
                ch_conv3 = get_rd_lst(16)

                init_conv_by_list(model.conv1, ch_conv1)
                init_conv_by_list(model.conv2, ch_conv2)
                init_conv_by_list(model.conv3, ch_conv3)

                importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
                importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()
                importance_conv3 = l2_norm(model.conv3.weight, model.conv3).tolist()

                importance_conv12 = list(map(add, importance_conv1[:16], importance_conv2))
                importance_conv13 = list(map(add, importance_conv1[16:], importance_conv3))

                if len(set(importance_conv12)) == len(importance_conv12) and len(set(importance_conv13)) == len(
                    importance_conv13
                ):
                    break

            conv2_idxes = []
            conv3_idxes = []

            for i in range(2):
                conv2_idxes += get_topk(importance_conv12[i * 8 : (i + 1) * 8], 4, offset=i * 8)

            for i in range(4):
                conv3_idxes += get_topk(importance_conv13[i * 4 : (i + 1) * 4], 2, offset=i * 4)

            conv1_idxes = conv2_idxes + [i + 16 for i in conv3_idxes]

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv3 = pruner.graph_modifier.get_modifier(model.conv3)
            m_conv4 = pruner.graph_modifier.get_modifier(model.conv4)
            m_conv5 = pruner.graph_modifier.get_modifier(model.conv5)

            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_o == conv2_idxes
            assert m_conv3.dim_changes_info.pruned_idx_o == conv3_idxes
            assert m_conv4.dim_changes_info.pruned_idx_i == conv2_idxes
            assert m_conv5.dim_changes_info.pruned_idx_i == conv3_idxes

            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o[:8], 16, 8, 2)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o[8:], 16, 8, 4, offset=16)

            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_o, 16, 8, 2)
            removed_idx_group_check(m_conv3.dim_changes_info.pruned_idx_o, 16, 8, 4)

            removed_idx_group_check(m_conv4.dim_changes_info.pruned_idx_i, 16, 8, 2)
            removed_idx_group_check(m_conv4.dim_changes_info.pruned_idx_o, 32, 16, 2)

            removed_idx_group_check(m_conv5.dim_changes_info.pruned_idx_i, 16, 8, 4)
            removed_idx_group_check(m_conv5.dim_changes_info.pruned_idx_o, 32, 16, 4)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(200):
            test_func()

    def group_element_wise_graph(self, op):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(16, 16, (1, 1), groups=8)
                self.conv1 = nn.Conv2d(16, 16, (1, 1))
                self.conv2 = nn.Conv2d(16, 16, (1, 1), groups=4)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(x)
                add0 = op(conv0, conv1)
                conv2 = self.conv2(add0)
                return conv2

        def test_func():
            model = TestModel()

            ch_conv0 = get_rd_lst(16)
            ch_conv1 = get_rd_lst(16)
            ch_conv2 = get_rd_lst(16)

            init_conv_by_list(model.conv0, ch_conv0)
            init_conv_by_list(model.conv1, ch_conv1)
            init_conv_by_list(model.conv2, ch_conv2)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0)
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1)
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2)
            importance_conv01 = list(map(add, importance_conv0, importance_conv1))

            idxes_conv0 = []
            idxes_conv2 = []

            for i in range(8):
                idxes_conv0 += get_topk(importance_conv01[i * 2 : (i + 1) * 2], 1, offset=i * 2)
            for i in range(4):
                idxes_conv2 += get_topk(importance_conv2[i * 4 : (i + 1) * 4], 2, offset=i * 4)

            pruner = OneShotChannelPruner(model, torch.ones(16, 16, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)

            assert m_conv0.dim_changes_info.pruned_idx_o == idxes_conv0
            assert m_conv1.dim_changes_info.pruned_idx_o == idxes_conv0
            assert m_conv2.dim_changes_info.pruned_idx_o == idxes_conv2

            removed_idx_group_check(m_conv0.dim_changes_info.pruned_idx_o, 16, 8, 8)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o, 16, 8, 8)
            removed_idx_group_check(m_conv2.dim_changes_info.pruned_idx_o, 16, 8, 4)

            pruner.apply_mask()

            model(torch.ones(16, 16, 9, 9))

        for i in range(20):
            test_func()

    def test_group_element_wise_graph(self):
        self.group_element_wise_graph(add)
        self.group_element_wise_graph(mul)
        self.group_element_wise_graph(sub)
        self.group_element_wise_graph(truediv)

    def group_element_wise_split_graph(self, op):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 32, (1, 1))
                self.conv1 = nn.Conv2d(16, 16, (1, 1), groups=8)
                self.conv2 = nn.Conv2d(16, 16, (1, 1), groups=2)

            def forward(self, x):
                conv0 = self.conv0(x)
                sp0, sp1 = torch.split(conv0, conv0.shape[1] // 2, 1)
                conv1 = self.conv1(sp0)
                add0 = op(conv1, sp1)
                conv2 = self.conv2(add0)
                return conv2

        def test_func():
            model = TestModel()

            ch_conv0 = get_rd_lst(32)
            ch_conv1 = get_rd_lst(16)

            init_conv_by_list(model.conv0, ch_conv0)
            init_conv_by_list(model.conv1, ch_conv1)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0)
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1)
            importance_conv01 = list(map(add, importance_conv0[16:], importance_conv1))

            idxes_conv0 = []
            idxes_conv1 = []

            for i in range(8):
                idxes_conv0 += get_topk(importance_conv0[i * 2 : (i + 1) * 2], 1, offset=i * 2)
                idxes_conv1 += get_topk(importance_conv01[i * 2 : (i + 1) * 2], 1, offset=i * 2)

            idxes_conv0 += [i + 16 for i in idxes_conv1]

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)

            assert m_conv0.dim_changes_info.pruned_idx_o == idxes_conv0
            assert m_conv1.dim_changes_info.pruned_idx_o == idxes_conv1

            removed_idx_group_check(m_conv0.dim_changes_info.pruned_idx_o, 32, 16, 8)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o, 16, 8, 8)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(20):
            test_func()

    def test_group_element_wise_split_graph(self):
        self.group_element_wise_split_graph(add)
        self.group_element_wise_split_graph(sub)
        self.group_element_wise_split_graph(mul)
        self.group_element_wise_split_graph(truediv)

    def test_mbv2_block(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 8, (1, 1))
                self.conv2 = nn.Conv2d(8, 16, (1, 1))
                self.conv3 = nn.Conv2d(16, 16, (3, 3), padding=(1, 1), groups=16)
                self.conv4 = nn.Conv2d(16, 8, (1, 1))
                self.bn4 = nn.BatchNorm2d(8)

            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(conv1)
                conv3 = self.conv3(conv2)
                conv4 = self.conv4(conv3)
                bn4 = self.bn4(conv4)
                add0 = conv1 + bn4
                return add0

        def test_func():
            model = TestModel()

            ch_8_0 = get_rd_lst(8)
            ch_8_1 = get_rd_lst(8)
            ch_16 = get_rd_lst(16)

            init_conv_by_list(model.conv1, ch_8_0)
            init_conv_by_list(model.conv4, ch_8_1)
            init_conv_by_list(model.conv2, ch_16)
            init_conv_by_list(model.conv3, ch_16)

            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()
            importance_conv4 = l2_norm(model.conv4.weight, model.conv4).tolist()
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2).tolist()

            importance_add0 = list(map(add, importance_conv1, importance_conv4))

            add0_idx = get_topk(importance_add0, 4)
            conv2_idxes = get_topk(importance_conv2, 8)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv3 = pruner.graph_modifier.get_modifier(model.conv3)
            m_conv4 = pruner.graph_modifier.get_modifier(model.conv4)

            assert m_conv1.dim_changes_info.pruned_idx_o == add0_idx
            assert m_conv4.dim_changes_info.pruned_idx_o == add0_idx

            assert m_conv2.dim_changes_info.pruned_idx_o == conv2_idxes
            assert m_conv3.dim_changes_info.pruned_idx_i == conv2_idxes
            assert m_conv3.dim_changes_info.pruned_idx_i == conv2_idxes
            assert m_conv4.dim_changes_info.pruned_idx_i == conv2_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            print("test over")

        for i in range(20):
            test_func()

    def test_res_2_net_block(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, (1, 1))
                self.conv2 = nn.Conv2d(16, 16, (1, 1), groups=8)
                self.conv3 = nn.Conv2d(16, 16, (1, 1), groups=2)
                self.conv4 = nn.Conv2d(16, 16, (1, 1))
                self.conv5 = nn.Conv2d(64, 64, (1, 1))

            def forward(self, x):
                conv1 = self.conv1(x)
                size0 = conv1.shape[1] // 4
                split0 = torch.split(conv1, size0, 1)
                conv2 = self.conv2(split0[0])
                add0 = conv2 + split0[1]
                conv3 = self.conv3(add0)
                add3 = conv3 + split0[2]
                conv4 = self.conv4(add3)
                cat0 = torch.cat([conv2, conv3, conv4, split0[3]], 1)
                return self.conv5(cat0)

        def test_func():
            model = TestModel()
            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)

            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o[:16], 32, 16, 8)
            removed_idx_group_check(m_conv1.dim_changes_info.pruned_idx_o[16:24], 16, 8, 2, offset=32)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            print("test over")

        for i in range(20):
            test_func()

    #
    # def test_conv1d_block(self):
    #     class TestModel(nn.Module):
    #         def __init__(self):
    #             super(TestModel, self).__init__()
    #             self.conv1 = nn.Conv1d(3, 16, (3,))
    #             self.conv2 = nn.Conv1d(16, 32, (3,))
    #
    #         def forward(self, x):
    #             conv1 = self.conv1(x)
    #             return self.conv2(conv1)
    #
    #     model = TestModel()
    #     pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9), {"sparsity": 0.25, "metrics": "l2_norm"})
    #     pruner.prune()
    #
    #     model(torch.ones(1, 3, 9))
    #
    #     assert model.conv1.out_channels == 12
    #     assert model.conv2.out_channels == 24
    #
    def test_loop_conv_block(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3))
                self.conv1 = nn.Conv2d(16, 32, (3, 3))

            def forward(self, x):
                conv0 = self.conv0(x)
                return self.conv1(conv0 + conv0 + conv0)

        def test_func():
            model = TestModel()

            ch_conv0 = get_rd_lst(16)
            init_conv_by_list(model.conv0, ch_conv0)
            importance_conv0 = l2_norm(model.conv0.weight, model.conv0)
            idxes_conv0 = get_topk(importance_conv0, 8)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            assert m_conv0.dim_changes_info.pruned_idx_o == idxes_conv0

            pruner.apply_mask()
            model(torch.ones(1, 3, 9, 9))

        for i in range(20):
            test_func()

    def test_shuffle_net_block(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3))
                self.conv1 = nn.Conv2d(16, 32, (3, 3))

            def forward(self, x):
                conv0 = self.conv0(x)
                reshape0 = conv0.view(conv0.shape[0], 2, 8, conv0.shape[2], conv0.shape[3])
                transpose0 = torch.transpose(reshape0, 1, 2)
                reshape1 = transpose0.reshape([conv0.shape[0], -1, conv0.shape[2], conv0.shape[3]])
                conv1 = self.conv1(reshape1)
                return conv1

        def test_func():
            model = TestModel()

            ch_conv0 = get_rd_lst(16)

            init_conv_by_list(model.conv0, ch_conv0)

            # shufflenet 内部的channel shuffle并不会导致channel分组，只是generate mask的时候reshape0算子的dim 1不能剪枝
            importance_conv0 = l2_norm(model.conv0.weight, model.conv0)

            idxes_conv0 = get_topk(importance_conv0, 8)

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)

            assert m_conv0.dim_changes_info.pruned_idx_o == idxes_conv0

            pruner.apply_mask()

            pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')

            model = import_from_path('out.test', "out/test.py", "test")()

            model(torch.ones(1, 3, 9, 9))

        for i in range(20):
            test_func()

    def test_multi_dim_fc(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc0 = nn.Linear(8, 32)
                self.fc1 = nn.Linear(32, 32)

            def forward(self, x):
                fc0 = self.fc0(x)
                fc1 = self.fc1(fc0)
                return fc1

        model = TestModel()

        pruner = OneShotChannelPruner(model, torch.rand((16, 16, 8)), {"sparsity": 0.5, "metrics": "l2_norm"})
        pruner.prune()
        model(torch.rand((16, 8)))

        assert model.fc0.out_features == 16
        assert model.fc1.in_features == 16
        assert model.fc1.out_features == 32

    def test_issue_65(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.fc1 = torch.nn.Linear(2049, 1024, bias=False)
                self.bn1 = torch.nn.BatchNorm1d(1024)
                self.lstm = torch.nn.LSTM(
                    input_size=1024, hidden_size=512, num_layers=3, bidirectional=False, batch_first=False, dropout=0.4
                )

            def forward(self, input_1):
                shape_1 = input_1.shape
                permute_1 = input_1.permute(3, 0, 1, 2)
                reshape_1 = permute_1.reshape(-1, 2049)
                fc1 = self.fc1(reshape_1)
                bn1 = self.bn1(fc1)
                reshape_2 = bn1.reshape(shape_1[3], shape_1[0], 1024)
                lstm = self.lstm(reshape_2)
                return lstm[0]

        dummy_input_0 = torch.ones((16, 1, 2049, 47), dtype=torch.float32)
        model = LSTMModel()
        pruner = OneShotChannelPruner(model, dummy_input_0, {"sparsity": 0.5, "metrics": "l2_norm"})
        pruner.prune()

    @unittest.skipIf(IS_CI, "This test may sometimes fail on CI")
    def test_basic_rnn(self):
        rnn_in_size = 28
        rnn_hidden_size = 128
        fc_out_channel = 10

        class TestModel(nn.Module):
            def __init__(self, module, bidirectional, proj_size, num_layers, bias):
                super(TestModel, self).__init__()

                fc_in_channel = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size

                if proj_size != 0:
                    fc_in_channel = proj_size * 2 if bidirectional else proj_size

                if module is nn.LSTM and proj_size > 0:
                    self.rnn = module(
                        rnn_in_size,
                        rnn_hidden_size,
                        num_layers=num_layers,
                        proj_size=proj_size,
                        bidirectional=bidirectional,
                        batch_first=False,
                        bias=bias,
                    )
                else:
                    self.rnn = module(
                        rnn_in_size,
                        rnn_hidden_size,
                        num_layers=num_layers,
                        bidirectional=bidirectional,
                        batch_first=False,
                        bias=bias,
                    )
                self.fc = nn.Linear(fc_in_channel, fc_out_channel)

            def forward(self, x):
                rnn, _ = self.rnn(x)
                fc = self.fc(rnn)
                return fc

        def test_func(module, bidirectional=False, proj_size=0, num_layers=1, bias=True):
            if module != nn.LSTM or LooseVersion(torch.__version__) < LooseVersion('1.8.0'):
                proj_size = 0

            model = TestModel(module, bidirectional, proj_size, num_layers, bias)

            cnt = 0

            while True:
                cnt += 1
                if cnt > 100:
                    assert False

                h_lst = get_rd_lst(rnn_hidden_size)
                init_rnn_by_list(model.rnn, h_lst)

                importance = {}
                update_weight_metric(importance, l2_norm, model.rnn, "rnn")
                importance = importance['rnn'].tolist()
                if bidirectional:
                    if len(set(importance[:rnn_hidden_size])) == len(importance[:rnn_hidden_size]):
                        if len(set(importance[rnn_hidden_size:])) == len(importance[rnn_hidden_size:]):
                            break
                else:
                    if len(set(importance)) == len(importance):
                        break

            prune_idx_o = []

            if proj_size != 0:
                idx_num_o = proj_size
                prune_num_o = proj_size // 2
            else:
                idx_num_o = rnn_hidden_size
                prune_num_o = rnn_hidden_size // 2

            if bidirectional:
                prune_idx_o += get_topk(importance[:idx_num_o], prune_num_o)
                prune_idx_o += get_topk(importance[idx_num_o:], prune_num_o, offset=idx_num_o)
            else:
                prune_idx_o = get_topk(importance, prune_num_o)

            pruner = OneShotChannelPruner(
                model, torch.rand((3, 3, rnn_in_size)), {"sparsity": 0.5, "metrics": "l2_norm"}
            )

            pruner.register_mask()

            m_rnn0 = pruner.graph_modifier.get_modifier(model.rnn)

            assert m_rnn0.dim_changes_info.pruned_idx_o == prune_idx_o

            pruner.apply_mask()
            model(torch.rand((3, 3, rnn_in_size)))

            assert model.rnn.hidden_size == 64

        for cell_type in (nn.RNN, nn.GRU, nn.LSTM):
            for num_layers in (1, 2):
                for bidirectional in (False, True):
                    for proj_size in (0, 64):
                        print(cell_type, num_layers, bidirectional, proj_size)
                        test_func(cell_type, num_layers=num_layers, bidirectional=bidirectional, proj_size=proj_size)

    def test_rnn(self):
        rnn_in_size = 28
        rnn_hidden_size = 128
        fc_out_channel = 10

        class TestModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super(TestModel, self).__init__()

                assert 'cell_type' in kwargs

                cell_type = kwargs.pop('cell_type')
                assert cell_type in (nn.RNN, nn.GRU, nn.LSTM)

                bidirectional = kwargs.get('bidirectional', False)
                num_directions = 2 if bidirectional else 1
                fc_in_channel = rnn_hidden_size * num_directions

                if 'proj_size' in kwargs:
                    fc_in_channel = kwargs['proj_size'] * num_directions

                self.rnn = cell_type(rnn_in_size, rnn_hidden_size, *args, **kwargs)
                self.fc = nn.Linear(fc_in_channel, fc_out_channel)

            def forward(self, x):
                rnn, _ = self.rnn(x)
                fc = self.fc(rnn)
                return fc

        for cell_type in (nn.RNN, nn.GRU, nn.LSTM):
            for num_layers in (1, 2):
                for bidirectional in (False, True):
                    for batch_first in (False, True):
                        for proj_size in (0, 120):

                            if cell_type != nn.LSTM and proj_size > 0:
                                continue

                            kwargs = {
                                'num_layers': num_layers,
                                'bidirectional': bidirectional,
                                'batch_first': batch_first,
                                'cell_type': cell_type,
                            }

                            if proj_size > 0:
                                if LooseVersion(torch.__version__) >= LooseVersion('1.8.0'):
                                    kwargs.update({'proj_size': proj_size})
                                else:
                                    continue

                            filtered_args = {k: v for k, v in kwargs.items() if k != 'cell_type'}
                            print(f'\nTesting {cell_type.__name__} with {filtered_args}')

                            model = TestModel(**kwargs)

                            pruner = OneShotChannelPruner(
                                model, torch.rand((3, 3, rnn_in_size)), {"sparsity": 0.5, "metrics": "l2_norm"}
                            )
                            pruner.prune()
                            model(torch.rand((3, 3, rnn_in_size)))

                            assert model.rnn.hidden_size == 64

    def test_lstm_proj_add_fc(self):
        if LooseVersion(torch.__version__) < LooseVersion('1.8.0'):
            raise SkipTest("LSTM with projection is not supported in PyTorch < 1.8")

        rnn_in_size = 28
        rnn_hidden_size = 128
        fc_out_channel = 10
        proj_size = 120

        class TestModel(nn.Module):
            def __init__(self, *args, **kwargs):
                super(TestModel, self).__init__()

                bidirectional = kwargs.get('bidirectional', False)
                num_directions = 2 if bidirectional else 1
                fc_in_channel = proj_size * num_directions

                self.rnn = nn.LSTM(rnn_in_size, rnn_hidden_size, proj_size=proj_size, *args, **kwargs)
                self.fc0 = nn.Linear(rnn_in_size, fc_in_channel)
                self.fc1 = nn.Linear(fc_in_channel, fc_out_channel)

            def forward(self, x):
                rnn, _ = self.rnn(x)
                fc0 = self.fc0(x) + rnn
                fc1 = self.fc1(fc0)
                return fc1

        for num_layers in (1, 2):
            for bidirectional in (False, True):

                kwargs = {
                    'num_layers': num_layers,
                    'bidirectional': bidirectional,
                }

                print(f'\nTesting with {kwargs}')

                model = TestModel(**kwargs)

                model(torch.rand((3, 3, rnn_in_size)))

                pruner = OneShotChannelPruner(
                    model, torch.rand((3, 3, rnn_in_size)), {"sparsity": 0.5, "metrics": "l2_norm"}
                )
                pruner.prune()

                model(torch.rand((3, 3, rnn_in_size)))

    def test_cat_split_group_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 8, (3, 3))
                self.conv1 = nn.Conv2d(3, 8, (3, 3))
                self.conv2 = nn.Conv2d(8, 32, (3, 3), groups=2)
                self.conv3 = nn.Conv2d(8, 32, (3, 3), groups=4)

            def forward(self, x):
                x0 = self.conv0(x)
                x1 = self.conv1(x)

                cat0 = torch.cat([x0, x1], dim=1)
                sp0, sp1 = torch.split(cat0, 8, dim=1)
                conv2 = self.conv2(sp0)
                conv3 = self.conv3(sp1)
                return conv2, conv3

        def test_func():
            model = TestModel()

            rd_lst_8 = get_rd_lst(8)
            rd_lst_32 = get_rd_lst(32)
            init_conv_by_list(model.conv0, rd_lst_8)
            init_conv_by_list(model.conv1, rd_lst_8)
            init_conv_by_list(model.conv2, rd_lst_32)
            init_conv_by_list(model.conv3, rd_lst_32)

            importance_conv0 = l2_norm(model.conv0.weight, model.conv0).tolist()
            importance_conv1 = l2_norm(model.conv1.weight, model.conv1).tolist()

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.register_mask()

            conv0_idxes = get_topk(importance_conv0[:4], 2) + get_topk(importance_conv0[4:], 2, offset=4)
            conv1_idxes = (
                get_topk(importance_conv1[:2], 1)
                + get_topk(importance_conv1[2:4], 1, offset=2)
                + get_topk(importance_conv1[4:6], 1, offset=4)
                + get_topk(importance_conv1[6:], 1, offset=6)
            )

            m_conv0 = pruner.graph_modifier.get_modifier(model.conv0)
            m_conv1 = pruner.graph_modifier.get_modifier(model.conv1)
            m_conv2 = pruner.graph_modifier.get_modifier(model.conv2)
            m_conv3 = pruner.graph_modifier.get_modifier(model.conv3)

            assert m_conv0.dim_changes_info.pruned_idx_o == conv0_idxes
            assert m_conv1.dim_changes_info.pruned_idx_o == conv1_idxes
            assert m_conv2.dim_changes_info.pruned_idx_i == conv0_idxes
            assert m_conv3.dim_changes_info.pruned_idx_i == conv1_idxes

            pruner.apply_mask()

            assert model.conv1.out_channels == 4
            assert model.conv0.out_channels == 4
            assert model.conv2.in_channels == 4
            assert model.conv2.out_channels == 16
            assert model.conv3.in_channels == 4
            assert model.conv3.out_channels == 16

            pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
            model = import_from_path('out.test', "out/test.py", "test")()

            model(torch.ones(1, 3, 9, 9))

        for i in range(20):
            test_func()

    def test_split_linear_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=8, out_features=8)
                self.linear1 = nn.Linear(in_features=2, out_features=32)
                self.linear2 = nn.Linear(in_features=6, out_features=32)

            def forward(self, x):
                x = self.linear0(x)
                sp0, sp1 = torch.split(x, (2, 6), dim=2)
                linear1 = self.linear1(sp0)
                linear2 = self.linear2(sp1)
                return linear1, linear2

        def test_func():
            model = TestModel()

            dummy_input = torch.ones(1, 4, 8)
            pruner = OneShotChannelPruner(
                model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": False}
            )
            pruner.prune()

            pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
            model = import_from_path('out.test', "out/test.py", "test")()

            print([i.shape for i in model(dummy_input)])

        for i in range(20):
            test_func()

    def test_split_cat(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc0 = nn.Linear(8, 16)
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(32, 32)

            def forward(self, x):
                fc0 = self.fc0(x)
                fc1 = self.fc1(x)
                cat0 = torch.cat([fc0, fc1], dim=1)
                fc2 = self.fc2(cat0)
                return fc2

        def test_func():
            model = TestModel()

            pruner = OneShotChannelPruner(model, torch.rand((2, 8)), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.prune()
            model(torch.rand((16, 8)))

            assert model.fc0.out_features == 8
            assert model.fc1.out_features == 8
            assert model.fc2.in_features == 16
            assert model.fc2.out_features == 32

        for i in range(20):
            test_func()

    def test_batch_cat(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc0 = nn.Linear(8, 16)
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 32)

            def forward(self, x):
                fc0 = self.fc0(x)
                fc1 = self.fc1(x)
                cat0 = torch.cat([fc0, fc1], dim=0)
                fc2 = self.fc2(cat0)
                return fc2

        def test_func():
            model = TestModel()

            pruner = OneShotChannelPruner(model, torch.rand((2, 8)), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.prune()
            model(torch.rand((16, 8)))

            assert model.fc0.out_features == 8
            assert model.fc1.out_features == 8
            assert model.fc2.in_features == 8
            assert model.fc2.out_features == 32

        for i in range(20):
            test_func()

    def test_batch_group_cat(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 1))
                self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(1, 1))
                self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), groups=4)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(x)
                cat0 = torch.cat([conv0, conv1], dim=0)
                conv2 = self.conv2(cat0)
                return conv2

        def test_func():
            model = TestModel()

            pruner = OneShotChannelPruner(model, torch.rand((1, 3, 3, 3)), {"sparsity": 0.5, "metrics": "l2_norm"})
            pruner.prune()
            model(torch.rand((1, 3, 3, 3)))

            assert model.conv0.out_channels == 8
            assert model.conv1.out_channels == 8
            assert model.conv2.in_channels == 8
            assert model.conv2.out_channels == 16

        for i in range(20):
            test_func()

    def test_cat_with_default_dim(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc0 = nn.Linear(8, 16)
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 32)

            def forward(self, x):
                fc0 = self.fc0(x)
                fc1 = self.fc1(x)
                cat0 = torch.cat([fc0, fc1])
                fc2 = self.fc2(cat0)
                return fc2

        model = TestModel()

        pruner = OneShotChannelPruner(model, torch.rand((2, 8)), {"sparsity": 0.5, "metrics": "l2_norm"})
        pruner.prune()
        model(torch.rand((16, 8)))

        assert model.fc0.out_features == 8
        assert model.fc1.out_features == 8
        assert model.fc2.in_features == 8
        assert model.fc2.out_features == 32

    def test_transformer_reshape(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc0 = nn.Linear(768, 768)
                self.fc1 = nn.Linear(64, 64)

            def forward(self, x):
                fc0 = self.fc0(x)
                reshape0 = torch.reshape(fc0, [1, fc0.shape[0], fc0.shape[1] // 64, 64])
                fc1 = self.fc1(reshape0)
                return fc1

        model = TestModel()

        pruner = OneShotChannelPruner(
            model, torch.rand((197, 768)), {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": True}
        )
        pruner.prune()

        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
        model = import_from_path('out.test', "out/test.py", "test")()
        model(torch.rand((197, 768)))

    def test_transformer_matmul(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=(16, 16), stride=(16, 16))
                self.fc0 = nn.Linear(768, 768)
                self.fc1 = nn.Linear(768, 768)
                self.fc2 = nn.Linear(197, 197)

            def forward(self, x):
                x = self.conv0(x)
                x = x.flatten(2)
                x = x.transpose(-1, -2)
                x = torch.cat((torch.ones(1, 1, x.shape[-1]), x), dim=1)
                fc0 = self.fc0(x)
                fc1 = self.fc1(x)
                reshape0 = torch.reshape(fc0, [1, fc0.shape[1], fc0.shape[2] // 64, 64])
                reshape1 = torch.reshape(fc1, [1, fc1.shape[1], fc1.shape[2] // 64, 64])
                q = reshape0.permute(0, 2, 1, 3)
                k = reshape1.permute(0, 2, 1, 3).transpose(-1, -2)

                qk = torch.matmul(q, k)
                fc4 = self.fc2(qk)
                return fc4

        dummy_input = torch.rand((1, 3, 224, 224))
        model = TestModel()
        model_generate(model, dummy_input=dummy_input, name="transformer.tflite")

        pruner = OneShotChannelPruner(
            model, torch.rand((1, 3, 224, 224)), {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": False}
        )
        pruner.prune()
        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
        model = import_from_path('out.test', "out/test.py", "test")()
        model(dummy_input)

    def test_matmul_dim_mapping(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()

            def forward(self, q, k):
                return torch.matmul(q, k)

        def test_func(matrix_a, matrix_b, mapping_str):
            model = TestModel()
            dummy_input = (matrix_a, matrix_b)
            model = TestModel()
            with model_tracer():
                graph = trace(model, dummy_input)
                graph_modifier = modifier.GraphChannelModifier(graph, [])

            matmul0 = graph_modifier.get_modifier(unique_name="matmul_0_f")
            matmul0.init_dim_mapping()
            mapping = matmul0.print_dim_mapping()
            assert mapping == mapping_str

        test_func(
            torch.rand((1, 12, 197, 64)),
            torch.rand((1, 12, 64, 197)),
            [
                'input_0:0->output_0:{0}',
                'input_0:1->output_0:{1}',
                'input_0:2->output_0:{2}',
                'input_0:3->output_0:set()',
                'input_1:0->output_0:{0}',
                'input_1:1->output_0:{1}',
                'input_1:2->output_0:set()',
                'input_1:3->output_0:{3}',
                'output_0:0->input_0:{0}',
                'output_0:0->input_1:{0}',
                'output_0:1->input_0:{1}',
                'output_0:1->input_1:{1}',
                'output_0:2->input_0:{2}',
                'output_0:2->input_1:set()',
                'output_0:3->input_0:set()',
                'output_0:3->input_1:{3}',
            ],
        )
        print("test over")

        test_func(
            torch.rand((1, 12, 197, 64)),
            torch.rand((64, 197)),
            [
                'input_0:0->output_0:{0}',
                'input_0:1->output_0:{1}',
                'input_0:2->output_0:{2}',
                'input_0:3->output_0:set()',
                'input_1:0->output_0:set()',
                'input_1:1->output_0:{3}',
                'output_0:0->input_0:{0}',
                'output_0:0->input_1:set()',
                'output_0:1->input_0:{1}',
                'output_0:1->input_1:set()',
                'output_0:2->input_0:{2}',
                'output_0:2->input_1:set()',
                'output_0:3->input_0:set()',
                'output_0:3->input_1:{1}',
            ],
        )

        test_func(
            torch.rand((1, 12, 197, 64)),
            torch.rand((64,)),
            [
                'input_0:0->output_0:{0}',
                'input_0:1->output_0:{1}',
                'input_0:2->output_0:{2}',
                'input_0:3->output_0:set()',
                'input_1:0->output_0:set()',
                'output_0:0->input_0:{0}',
                'output_0:0->input_1:set()',
                'output_0:1->input_0:{1}',
                'output_0:1->input_1:set()',
                'output_0:2->input_0:{2}',
                'output_0:2->input_1:set()',
            ],
        )

        print("test over")

    def test_transformer_reshape_transpose(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=(16, 16), stride=(16, 16))
                self.fc0 = nn.Linear(768, 768)
                self.fc1 = nn.Linear(768, 768)
                self.fc2 = nn.Linear(768, 768)
                self.fc3 = nn.Linear(768, 768)

            def forward(self, x):
                fc0 = self.fc0(x)
                fc1 = self.fc1(x)
                fc2 = self.fc2(x)
                reshape0 = torch.reshape(fc0, [1, fc0.shape[1], fc0.shape[2] // 64, 64])
                reshape1 = torch.reshape(fc1, [1, fc1.shape[1], fc1.shape[2] // 64, 64])
                reshape2 = torch.reshape(fc2, [1, fc2.shape[1], fc2.shape[2] // 64, 64])
                q = reshape0.permute(0, 2, 1, 3)
                k = reshape1.permute(0, 2, 1, 3).transpose(-1, -2)
                v = reshape2.permute(0, 2, 1, 3)

                qk = torch.matmul(q, k)
                qk = torch.softmax(qk, dim=-1)

                qkv = torch.matmul(qk, v)
                qkv = qkv.permute(0, 2, 1, 3)
                qkv = qkv.reshape([qkv.shape[1], qkv.shape[2] * qkv.shape[3]])
                fc3 = self.fc3(qkv)
                fc3 = fc3.reshape([1, fc3.shape[0], fc3.shape[1]])
                add0 = torch.add(x, fc3)

                return add0

        dummy_input = torch.rand((1, 197, 768))
        model = TestModel()
        model_generate(model, dummy_input=dummy_input, name="transformer.tflite")

        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": True})
        pruner.prune()

        model(dummy_input)

    def test_conv_transformer_reshape_transpose(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(in_channels=3, out_channels=768, kernel_size=(16, 16), stride=(16, 16))
                self.fc0 = nn.Linear(768, 768)
                self.fc1 = nn.Linear(768, 768)
                self.fc2 = nn.Linear(768, 768)
                self.fc3 = nn.Linear(768, 768)

            def forward(self, x):
                x = self.conv0(x)
                x = x.flatten(2)
                x = x.transpose(-1, -2)
                x = torch.cat((torch.ones(1, 1, x.shape[-1]), x), dim=1)
                fc0 = self.fc0(x)
                fc1 = self.fc1(x)
                fc2 = self.fc2(x)
                reshape0 = torch.reshape(fc0, [1, fc0.shape[1], fc0.shape[2] // 64, 64])
                reshape1 = torch.reshape(fc1, [1, fc1.shape[1], fc1.shape[2] // 64, 64])
                reshape2 = torch.reshape(fc2, [1, fc2.shape[1], fc2.shape[2] // 64, 64])
                q = reshape0.permute(0, 2, 1, 3)
                k = reshape1.permute(0, 2, 1, 3).transpose(-1, -2)
                v = reshape2.permute(0, 2, 1, 3)

                qk = torch.matmul(q, k)
                qk = torch.softmax(qk, dim=-1)

                qkv = torch.matmul(qk, v)
                qkv = qkv.permute(0, 2, 1, 3)
                qkv = qkv.reshape([qkv.shape[1], qkv.shape[2] * qkv.shape[3]])
                fc3 = self.fc3(qkv)
                fc3 = fc3.reshape([1, fc3.shape[0], fc3.shape[1]])
                add0 = torch.add(x, fc3)

                return add0

        dummy_input = torch.rand((1, 3, 224, 224))
        model = TestModel()
        model_generate(model, dummy_input=dummy_input, name="transformer.tflite")

        pruner = OneShotChannelPruner(
            model, torch.rand((1, 3, 224, 224)), {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": False}
        )
        pruner.prune()

        model(dummy_input)

    def test_reshape_prune_node(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc0 = nn.Linear(16, 32)
                self.fc1 = nn.Linear(2, 64)

            def forward(self, input_1):
                fc0 = self.fc0(input_1)
                reshape0 = fc0.reshape(1, 2, 16)
                transpose = torch.transpose(reshape0, 1, 2)
                fc1 = self.fc1(transpose)
                return fc1

        model = TestModel()

        pruner = OneShotChannelPruner(
            model, torch.rand((1, 16)), {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": True}
        )

        pruner.prune()

        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
        new_module = import_from_path('out.test', "out/test.py", "test")()

        new_module(torch.rand((1, 16)))

    def test_channel_shuffle(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3), (1, 1))
                self.conv1 = nn.Conv2d(16, 32, (3, 3), (1, 1))

            def forward(self, x):
                conv_0 = self.conv0(x)
                size_0 = conv_0.size()
                view_0 = conv_0.reshape([size_0[0], 2, size_0[1] // 2, size_0[2], size_0[3]])
                transpose_0 = torch.transpose(view_0, 1, 2)
                contiguous_1 = transpose_0.contiguous()
                view_1 = contiguous_1.view(size_0[0], -1, size_0[2], size_0[3])
                conv_1 = self.conv1(view_1)
                return conv_1

        model = TestModel()

        pruner = OneShotChannelPruner(
            model, torch.rand((1, 3, 6, 6)), {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": True}
        )

        pruner.prune()

        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
        new_module = import_from_path('out.test', "out/test.py", "test")()

        new_module(torch.rand((1, 3, 6, 6)))

    def test_shufflenetv2_chunk(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 16, (1, 1), (1, 1))
                self.conv1 = nn.Conv2d(8, 8, (1, 1), (1, 1))
                self.conv2 = nn.Conv2d(8, 8, (1, 1), (1, 1), groups=8)
                self.conv3 = nn.Conv2d(8, 8, (1, 1), (1, 1))
                self.conv4 = nn.Conv2d(16, 32, (1, 1), (1, 1))

            def forward(self, x):
                conv_0 = self.conv0(x)
                chunk_0 = conv_0.chunk(2, dim=1)
                conv1 = self.conv1(chunk_0[1])
                conv2 = self.conv2(conv1)
                conv3 = self.conv3(conv2)
                cat_1 = torch.cat([chunk_0[0], conv3], dim=1)

                conv4 = self.conv4(cat_1)
                return conv4

        model = TestModel()

        pruner = OneShotChannelPruner(
            model, torch.rand((1, 3, 9, 9)), {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": True}
        )

        pruner.prune()

        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
        new_module = import_from_path('out.test', "out/test.py", "test")()

        new_module(torch.rand((1, 3, 6, 6)))

    def test_prelu(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 16, (1, 1), (1, 1))
                self.prelu0 = nn.PReLU(16)
                self.conv1 = nn.Conv2d(16, 32, (1, 1), (1, 1))

            def forward(self, x):
                conv_0 = self.conv0(x)
                prelu_0 = self.prelu0(conv_0)
                conv_1 = self.conv1(prelu_0)
                return conv_1

        model = TestModel()

        pruner = OneShotChannelPruner(model, torch.rand((1, 3, 9, 9)), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.prune()
        model(torch.rand((1, 3, 9, 9)))

    def test_bn_compensation(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm2d(3)
                self.bn.register_buffer('running_mean', torch.Tensor([0, 0, 0]))
                self.bn.register_buffer('running_var', torch.Tensor([0, 0, 0]))
                self.bn.weight = torch.nn.Parameter(self.bn.weight * 0)
                self.bn.bias = torch.nn.Parameter(torch.Tensor([1, 1, 1]))
                self.conv0 = nn.Conv2d(3, 6, (1, 1), (1, 1), bias=True)
                self.conv0.bias = torch.nn.Parameter(torch.zeros_like(self.conv0.bias))

            def forward(self, x):
                bn = self.bn(x)
                conv0 = self.conv0(bn)
                return conv0

        model = TestModel()

        dummy_input = torch.zeros((1, 3, 5, 5))

        output1 = model(dummy_input)
        print(output1)

        remove_idx = [1, 2]
        bn_bias = model.conv0.weight * model.bn.bias
        bn_bias = bn_bias[:, [True if i in remove_idx else False for i in range(bn_bias.shape[1])]]
        bn_bias = torch.sum(bn_bias, dim=[1, 2, 3])
        model.conv0.bias = torch.nn.Parameter(model.conv0.bias + bn_bias)
        model.bn.bias = torch.nn.Parameter(torch.Tensor([1, 0, 0]))

        output2 = model(dummy_input)
        print(output2)

        print(torch.allclose(output1, output2))


if __name__ == '__main__':
    unittest.main()
