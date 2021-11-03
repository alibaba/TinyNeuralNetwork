import unittest

import torch
import torch.nn as nn
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from interval import Interval
import torch.nn.functional as F


def removed_idx_group_check(removed_idx, total_idx_len, removed_idx_len, group, offset=0):
    for i in range(group):
        remove_group_len = removed_idx_len // group
        for j in range(i * remove_group_len, i * remove_group_len + remove_group_len):
            idx_group_len = total_idx_len // group
            assert removed_idx[j] in Interval(offset + i * idx_group_len,
                                              offset + i * idx_group_len + idx_group_len,
                                              upper_closed=False)


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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv1.out_channels == 2
        assert model.conv0.out_channels == 2
        assert model.conv2.in_channels == 4
        assert model.conv2.out_channels == 8
        assert model.linear.in_features == 200
        assert model.linear.out_features == 100

    def test_tail_linear_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 16, (3, 3), padding=(1, 1))
                self.linear = nn.Linear(16 * 9 * 9, 100)

            def forward(self, x):
                conv0 = self.conv0(x)
                view0 = conv0.view((1, -1))
                linear0 = self.linear(view0)
                return linear0

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv0.out_channels == 4
        assert model.linear.in_features == 324
        assert model.linear.out_features == 100

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv0.out_channels == 2
        assert model.conv1.out_channels == 2
        assert model.conv2.out_channels == 4
        assert model.conv3.in_channels == 4
        assert model.conv3.out_channels == 8

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

        model = TestFlattenModel()
        model(torch.ones(1, 3, 9, 9))
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv0.out_channels == 4
        assert model.conv1.out_channels == 8
        assert model.linear1.in_features == 200
        assert model.linear1.out_features == 25
        assert model.linear2.in_features == 25
        assert model.linear2.out_features == 10

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.randn((1, 3, 9, 9)), {"sparsity": 0.25, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))
        assert model.conv1.out_channels == 12
        assert model.conv2.in_channels == 48
        assert model.conv2.out_channels == 96

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.25, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv1.out_channels == 12
        assert model.conv2.out_channels == 24
        assert model.conv3.in_channels == 36
        assert model.conv3.out_channels == 48

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)

        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 32, 16, 8)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 32, 16, 8)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv0.out_channels == 8
        assert model.conv1.in_channels == 8
        assert model.conv1.out_channels == 16
        assert model.conv2.out_channels == 16

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)

        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 32, 16, 4)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 32, 16, 8)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

        assert model.conv0.out_channels == 8
        assert model.conv1.in_channels == 8
        assert model.conv1.out_channels == 16
        assert model.conv2.out_channels == 16

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

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
                z1 = torch.cat([x1, x2], dim=1)
                z2 = torch.cat([x2, x3], dim=1)
                z1 = self.conv4(z1)
                z2 = self.conv5(z2)
                return z1, z2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 8, 4, 4)

        removed_idx_group_check(model.conv1.masker.in_remove_idx, 8, 4, 4)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 8, 4, 4)
        removed_idx_group_check(model.conv3.masker.in_remove_idx, 8, 4, 4)

        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 4)

        removed_idx_group_check(model.conv4.masker.in_remove_idx[8:], 16, 8, 4, offset=16)
        removed_idx_group_check(model.conv5.masker.in_remove_idx[:8], 16, 8, 4)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

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

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

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

    def test_group_mul_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(16, 16, (1, 1), groups=8)
                self.conv1 = nn.Conv2d(16, 16, (1, 1))
                self.conv2 = nn.Conv2d(16, 16, (1, 1), groups=4)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(x)
                add0 = conv0 * conv1
                conv2 = self.conv2(add0)
                return conv2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(16, 16, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 4)

        pruner.apply_mask()

        model(torch.ones(16, 16, 9, 9))

    def test_group_add_graph(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.conv0 = nn.Conv2d(16, 16, (1, 1), groups=8)
                self.conv1 = nn.Conv2d(16, 16, (1, 1))
                self.conv2 = nn.Conv2d(16, 16, (1, 1), groups=4)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(x)
                add0 = conv0 + conv1
                conv2 = self.conv2(add0)
                return conv2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(16, 16, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 4)

        pruner.apply_mask()

        model(torch.ones(16, 16, 9, 9))

    def test_center_add_group_graph(self):
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
                add0 = conv1 + sp1
                conv2 = self.conv2(add0)
                return conv2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 32, 16, 8)
        removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 2)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

    def test_center_sub_group_graph(self):
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
                add0 = conv1 - sp1
                conv2 = self.conv2(add0)
                return conv2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 32, 16, 8)
        removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 2)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

    def test_center_div_group_graph(self):
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
                add0 = conv1 / sp1
                conv2 = self.conv2(add0)
                return conv2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 32, 16, 8)
        removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 2)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

    def test_center_mul_group_graph(self):
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
                add0 = conv1 * sp1
                conv2 = self.conv2(add0)
                return conv2

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        removed_idx_group_check(model.conv0.masker.ot_remove_idx, 32, 16, 8)
        removed_idx_group_check(model.conv1.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv1.masker.ot_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.in_remove_idx, 16, 8, 8)
        removed_idx_group_check(model.conv2.masker.ot_remove_idx, 16, 8, 2)

        pruner.apply_mask()

        model(torch.ones(1, 3, 9, 9))

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
                self.conv1 = nn.Conv2d(3, 16, (3, 3))
                self.conv2 = nn.Conv2d(16, 32, (3, 3))
                self.conv3 = nn.Conv2d(16, 32, (3, 3))
                self.conv4 = nn.Conv2d(16, 32, (3, 3))
                self.conv5 = nn.Conv2d(32, 32, (3, 3))

            def forward(self, x):
                conv1 = self.conv1(x)
                conv2 = F.relu(self.conv2(conv1))
                conv3 = F.relu(self.conv3(conv1))
                conv4 = F.relu(self.conv4(conv1))

                return self.conv5(conv2 + conv3 + conv4)

        model = TestModel()
        pruner = OneShotChannelPruner(model, torch.ones(1, 3, 9, 9), {"sparsity": 0.5, "metrics": "l2_norm"})
        pruner.prune()

        model(torch.ones(1, 3, 9, 9))


if __name__ == '__main__':
    unittest.main()
