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
from tinynn.graph.modifier import l2_norm
from tinynn.prune.oneshot_pruner import OneShotChannelPruner


def removed_idx_group_check(removed_idx, total_idx_len, removed_idx_len, group, offset=0):
    for i in range(group):
        remove_group_len = removed_idx_len // group
        for j in range(i * remove_group_len, i * remove_group_len + remove_group_len):
            idx_group_len = total_idx_len // group
            assert removed_idx[j] in Interval(
                offset + i * idx_group_len, offset + i * idx_group_len + idx_group_len, upper_closed=False
            )


def get_rd_lst(length):
    rd_lst = random.sample(range(0, 1000), length)
    random.shuffle(rd_lst)

    print(rd_lst)
    return rd_lst


def get_topk(lst, k, offset=0):
    _, idx = torch.topk(torch.tensor(lst), k, largest=False)

    return sorted([i + offset for i in idx.tolist()])


def init_conv_by_list(conv, ch_value):
    assert conv.weight.shape[0] == len(ch_value)

    for i in range(len(ch_value)):
        conv.weight.data[i, :] = ch_value[i]


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
                view0 = conv2.view((1, -1))
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

            assert model.conv0.masker.ot_remove_idx == conv0_idxes
            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.in_remove_idx == conv0_idxes + [i + 8 for i in conv0_idxes]
            assert model.conv2.masker.ot_remove_idx == conv2_idxes

            pruner.apply_mask()

            assert model.conv1.out_channels == 4
            assert model.conv0.out_channels == 4
            assert model.conv2.in_channels == 8
            assert model.conv2.out_channels == 16
            assert model.linear.in_features == 400
            assert model.linear.out_features == 100

            model(torch.ones(1, 3, 9, 9))

        for i in range(100):
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

            assert model.conv0.masker.ot_remove_idx == conv0_idxes
            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.ot_remove_idx == conv0_idxes + [8 + i for i in conv1_idxes]
            assert model.conv3.masker.ot_remove_idx == conv3_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 4
            assert model.conv1.out_channels == 4
            assert model.conv2.out_channels == 8
            assert model.conv3.in_channels == 8
            assert model.conv3.out_channels == 16

        for i in range(100):
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

            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.linear1.masker.in_remove_idx == linear1_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 4
            assert model.conv1.out_channels == 8
            assert model.linear1.in_features == 200
            assert model.linear1.out_features == 25
            assert model.linear2.in_features == 25
            assert model.linear2.out_features == 10

        for i in range(10):
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

            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.in_remove_idx == [j + i * 16 for i in range(4) for j in conv1_idxes]

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))
            assert model.conv1.out_channels == 8
            assert model.conv2.in_channels == 32
            assert model.conv2.out_channels == 64

        for i in range(10):
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

            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.ot_remove_idx == conv2_idxes
            assert model.conv3.masker.in_remove_idx == conv1_idxes + [i + 16 for i in conv2_idxes]

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv1.out_channels == 8
            assert model.conv2.out_channels == 16
            assert model.conv3.in_channels == 24
            assert model.conv3.out_channels == 32

        for i in range(10):
            test_func()

    def test_nonaligned_cat_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv2d(3, 8, (3, 3))
                self.conv2 = nn.Conv2d(3, 4, (3, 3))
                self.conv3 = nn.Conv2d(3, 4, (3, 3))
                self.conv4 = nn.Conv2d(16, 64, (3, 3))

            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = self.conv2(x)
                conv3 = self.conv3(x)
                cat0 = torch.cat([conv1, conv2, conv3], dim=1)
                return self.conv4(cat0)

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.25, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv1.out_channels == 6
        assert model.conv2.out_channels == 3
        assert model.conv3.out_channels == 3
        assert model.conv4.in_channels == 12
        assert model.conv4.out_channels == 48

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

            removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)

            removed_idx_group_check(model.conv1.masker.ot_remove_idx, 32, 16, 8)
            removed_idx_group_check(model.conv2.masker.in_remove_idx, 32, 16, 8)

            assert model.conv0.masker.ot_remove_idx == conv0_idxes
            assert model.conv1.masker.ot_remove_idx == conv1_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 8
            assert model.conv1.in_channels == 8
            assert model.conv1.out_channels == 16
            assert model.conv2.out_channels == 16

        for i in range(10):
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

            removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)

            removed_idx_group_check(model.conv1.masker.ot_remove_idx, 32, 16, 4)
            removed_idx_group_check(model.conv2.masker.ot_remove_idx, 32, 16, 8)

            assert model.conv0.masker.ot_remove_idx == conv0_idxes
            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.ot_remove_idx == conv2_idxes

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

            assert model.conv0.out_channels == 8
            assert model.conv1.in_channels == 8
            assert model.conv1.out_channels == 16
            assert model.conv2.out_channels == 16

        for i in range(10):
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

            assert model.conv0.masker.ot_remove_idx == conv0_idxes
            assert model.conv1.masker.ot_remove_idx == conv12_idxes
            assert model.conv2.masker.ot_remove_idx == conv12_idxes
            assert model.conv5.masker.in_remove_idx == conv12_idxes + [i + 32 for i in conv4_idxes]

            removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 4)
            removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 4)
            removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 4)

            removed_idx_group_check(model.conv1.masker.ot_remove_idx, 32, 16, 4)
            removed_idx_group_check(model.conv2.masker.ot_remove_idx, 32, 16, 4)

            removed_idx_group_check(model.conv4.masker.in_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv4.masker.ot_remove_idx, 32, 16, 8)

            removed_idx_group_check(model.conv5.masker.in_remove_idx[:16], 32, 16, 4)
            removed_idx_group_check(model.conv5.masker.in_remove_idx[16:], 32, 16, 8, offset=32)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(50):
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

            assert model.conv0.masker.ot_remove_idx == conv0_idxes
            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.ot_remove_idx == conv2_idxes
            assert model.conv3.masker.ot_remove_idx == conv3_idxes

            removed_idx_group_check(model.conv0.masker.ot_remove_idx, 8, 4, 4)
            removed_idx_group_check(model.conv1.masker.in_remove_idx, 8, 4, 4)
            removed_idx_group_check(model.conv2.masker.in_remove_idx, 8, 4, 4)
            removed_idx_group_check(model.conv3.masker.in_remove_idx, 8, 4, 4)
            removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 4)
            removed_idx_group_check(model.conv4.masker.in_remove_idx[8:], 16, 8, 4, offset=16)
            removed_idx_group_check(model.conv5.masker.in_remove_idx[:8], 16, 8, 4)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(50):
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

            ch_conv1 = get_rd_lst(32)
            ch_conv2 = get_rd_lst(16)
            ch_conv3 = get_rd_lst(16)

            init_conv_by_list(model.conv1, ch_conv1)
            init_conv_by_list(model.conv2, ch_conv2)
            init_conv_by_list(model.conv3, ch_conv3)

            importance_conv1 = l2_norm(model.conv1.weight, model.conv1)
            importance_conv2 = l2_norm(model.conv2.weight, model.conv2)
            importance_conv3 = l2_norm(model.conv3.weight, model.conv3)

            importance_conv12 = list(map(add, importance_conv1[:16], importance_conv2))
            importance_conv13 = list(map(add, importance_conv1[16:], importance_conv3))

            conv2_idxes = []
            conv3_idxes = []

            for i in range(2):
                conv2_idxes += get_topk(importance_conv12[i * 8 : (i + 1) * 8], 4, offset=i * 8)

            for i in range(4):
                conv3_idxes += get_topk(importance_conv13[i * 4 : (i + 1) * 4], 2, offset=i * 4)

            conv1_idxes = conv2_idxes + [i + 16 for i in conv3_idxes]

            pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

            pruner.register_mask()

            assert model.conv1.masker.ot_remove_idx == conv1_idxes
            assert model.conv2.masker.ot_remove_idx == conv2_idxes
            assert model.conv3.masker.ot_remove_idx == conv3_idxes
            assert model.conv4.masker.in_remove_idx == conv2_idxes
            assert model.conv5.masker.in_remove_idx == conv3_idxes

            removed_idx_group_check(model.conv1.masker.ot_remove_idx[:8], 16, 8, 2)
            removed_idx_group_check(model.conv1.masker.ot_remove_idx[8:], 16, 8, 4, offset=16)

            removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 2)
            removed_idx_group_check(model.conv3.masker.ot_remove_idx, 16, 8, 4)

            removed_idx_group_check(model.conv4.masker.in_remove_idx, 16, 8, 2)
            removed_idx_group_check(model.conv4.masker.ot_remove_idx, 32, 16, 2)

            removed_idx_group_check(model.conv5.masker.in_remove_idx, 16, 8, 4)
            removed_idx_group_check(model.conv5.masker.ot_remove_idx, 32, 16, 4)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(50):
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

            assert model.conv0.masker.ot_remove_idx == idxes_conv0
            assert model.conv1.masker.ot_remove_idx == idxes_conv0
            assert model.conv2.masker.ot_remove_idx == idxes_conv2

            removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 4)

            pruner.apply_mask()

            model(torch.ones(16, 16, 9, 9))

        for i in range(50):
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

            assert model.conv0.masker.ot_remove_idx == idxes_conv0
            assert model.conv1.masker.ot_remove_idx == idxes_conv1

            removed_idx_group_check(model.conv0.masker.ot_remove_idx, 32, 16, 8)
            removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)
            removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 2)

            pruner.apply_mask()

            model(torch.ones(1, 3, 9, 9))

        for i in range(50):
            test_func()

    def test_group_element_wise_split_graph(self):
        self.group_element_wise_split_graph(add)
        self.group_element_wise_split_graph(sub)
        self.group_element_wise_split_graph(mul)
        self.group_element_wise_split_graph(truediv)

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()
        removed_idx_group_check(model.conv1.masker.ot_remove_idx[:16], 32, 16, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx[16:24], 16, 8, 2, offset=32)
        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

    def test_conv1d_block(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv1 = nn.Conv1d(3, 16, (3,))
                self.conv2 = nn.Conv1d(16, 32, (3,))

            def forward(self, x):
                conv1 = self.conv1(x)
                return self.conv2(conv1)

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9), {"sparsity": 0.25, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9))

        assert model.conv1.out_channels == 12
        assert model.conv2.out_channels == 24

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

            assert model.conv0.masker.ot_remove_idx == idxes_conv0

            pruner.apply_mask()
            model(torch.ones(1, 3, 9, 9))

        for i in range(10):
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


if __name__ == '__main__':
    unittest.main()
