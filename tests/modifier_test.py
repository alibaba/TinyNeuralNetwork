import os
import unittest
from copy import deepcopy

import torch
import torch.nn as nn

from tinynn.converter import TFLiteConverter
from tinynn.graph.modifier import is_dw_conv
from tinynn.graph.modifier import (
    GraphChannelModifier,
    fill_tensor_by_dim_changes,
    calc_dim_changes,
    calc_dim_constraint,
    merge_group,
    merge_constraint,
)
from tinynn.graph.tracer import model_tracer, trace
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.util.util import import_from_path

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))


def model_generate(model, dummy_input, name='test.tflite'):
    converter = TFLiteConverter(model, dummy_input, os.path.join(CURRENT_PATH, 'out', name))
    converter.convert()


def graph_generate(model, dummy_input, center_types):
    with model_tracer():
        graph = trace(model, dummy_input)
        center_nodes = []
        for n in graph.forward_nodes:
            if n.type() in center_types and not is_dw_conv(n.module):
                center_nodes.append(n)
    return graph, center_nodes


class ModifierForwardTester(unittest.TestCase):
    def test_case_0(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 8, (3, 3))
                self.conv1 = nn.Conv2d(8, 16, (3, 3))
                self.linear = nn.Linear(400, 100)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                view0 = conv1.view((1, -1))
                linear0 = self.linear(view0)
                return linear0

        model = TestModel()

        dummy_input = torch.ones((1, 3, 9, 9))

        model(dummy_input)

        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        pruner.apply_mask()

        model(dummy_input)

    def test_case_1(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(4, 8)
                self.linear1 = nn.Linear(4, 16)
                self.linear2 = nn.Linear(8, 4)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape0 = linear0.reshape((2, 4))

                linear1 = self.linear1(x)
                reshape1 = linear1.reshape((8, 2))
                transpose0 = reshape1.transpose(0, 1)
                linear2 = self.linear2(transpose0)
                add0 = torch.add(reshape0, linear2)

                return add0

        def test_func():
            model = TestModel()

            dummy_input = torch.ones((1, 4))

            model_generate(model, dummy_input)

            model(dummy_input)

            pruner = OneShotChannelPruner(
                model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": False}
            )

            pruner.register_mask()

            graph_modifier = pruner.graph_modifier

            m_linear0 = graph_modifier.get_modifier(model.linear0)
            m_linear1 = graph_modifier.get_modifier(model.linear1)
            m_linear2 = graph_modifier.get_modifier(model.linear2)
            m_add0 = graph_modifier.get_modifier(unique_name='add_0_f')
            m_reshape0 = graph_modifier.get_modifier(unique_name="reshape_0_f")

            assert m_add0.dim_changes_info.get_neighbor_changes(m_linear0, m_reshape0) == [[0, 1]]
            assert m_add0.dim_changes_info.get_neighbor_changes(m_linear1, m_linear2) is None
            assert m_add0.dim_changes_info.get_neighbor_changes(m_linear2, m_linear2) == [[1]]

            assert m_add0.dim_changes_info.constraints_i == {
                1: {
                    'linear0': [[{0.0, 4.0}, {1.0, 5.0}, {2.0, 6.0}, {3.0, 7.0}]],
                    'linear2': [[{0.0}, {1.0}, {2.0}, {3.0}]],
                }
            }

            pruner.apply_mask()

            pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')

            model = import_from_path('out.test', "out/test.py", "test")()

            model(dummy_input)

        for i in range(20):
            test_func()

    def test_case_2(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(4, 4)
                self.linear1 = nn.Linear(4, 4)
                self.linear2 = nn.Linear(4, 4)
                self.linear3 = nn.Linear(4, 4)

            def forward(self, x):
                linear0 = self.linear0(x)
                linear1 = self.linear1(x)

                cat0 = torch.cat([linear0, linear1], dim=1)
                sp0, sp1 = torch.split(cat0, 4, dim=1)
                linear2 = self.linear2(sp0)
                linear3 = self.linear3(sp1)

                return linear2, linear3

        model = TestModel()

        dummy_input = torch.ones((1, 4))

        model_generate(model, dummy_input)

        model(dummy_input)

        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        graph_modifier = pruner.graph_modifier

        m_linear0 = graph_modifier.get_modifier(model.linear0)
        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_split_0 = graph_modifier.get_modifier(unique_name="split_0_f")
        m_linear2 = graph_modifier.get_modifier(model.linear2)
        m_linear3 = graph_modifier.get_modifier(model.linear3)

        assert m_linear2.dim_changes_info.get_neighbor_changes(m_linear0, m_split_0) == [[1]]
        assert m_linear2.dim_changes_info.get_neighbor_changes(m_linear1, m_split_0) is None

        assert m_linear3.dim_changes_info.get_neighbor_changes(m_linear0, m_split_0) is None
        assert m_linear3.dim_changes_info.get_neighbor_changes(m_linear1, m_split_0) == [[1]]

        pruner.apply_mask()

        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')

        model = import_from_path('out.test', "out/test.py", "test")()

        model(dummy_input)

    def test_case_3(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(4, 4)
                self.linear1 = nn.Linear(4, 4)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape0 = linear0.reshape((4, 2))
                transpose0 = reshape0.transpose(0, 1)

                cat0 = torch.cat([linear0, transpose0], dim=0)
                linear1 = self.linear1(cat0)
                return linear1

        model = TestModel()

        dummy_input = torch.ones((2, 4))

        model_generate(model, dummy_input)

        model(dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        try:
            GraphChannelModifier(graph, center_nodes)
            assert False
        except Exception as e:
            assert str(e) == "conflict can't be eliminated"

    def test_case_4(self):
        """
        测试最基本的冲突消除

        Returns:

        """

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(4, 32)
                self.linear1 = nn.Linear(8, 16)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape0 = linear0.reshape((4, 8))
                linear1 = self.linear1(reshape0)
                return linear1

        model = TestModel()

        dummy_input = torch.ones((1, 4))

        model_generate(model, dummy_input)

        model(dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        graph_modifier = GraphChannelModifier(graph, center_nodes)

        m_linear0 = graph_modifier.get_modifier(model.linear0)
        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_reshape0 = graph_modifier.get_modifier(unique_name='reshape_0_f')
        m_output0 = graph_modifier.get_modifier(unique_name='output_0_f')

        assert m_reshape0.dim_changes_info.get_neighbor_choices(m_linear0) == [[1]]
        assert m_linear1.dim_changes_info.get_neighbor_choices(m_reshape0) == [[1]]
        assert m_linear1.dim_changes_info.get_neighbor_choices(m_output0) == [[1]]

    def test_case_5(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), padding=(0, 0))
                self.linear0 = nn.Linear(in_features=2, out_features=2)
                self.linear1 = nn.Linear(in_features=2, out_features=8)
                self.linear2 = nn.Linear(in_features=2, out_features=8)

            def forward(self, x1, x2):
                conv0 = self.conv0(x1)
                reshape0 = conv0.reshape((2, 4))
                transpose0 = reshape0.transpose(0, 1)
                linear0 = self.linear0(transpose0)

                linear1 = self.linear1(x2)
                reshape1 = linear1.reshape((4, 2))

                add0 = linear0 + reshape1
                linear2 = self.linear2(add0)
                return linear2

        model = TestModel()

        dummy_input = (torch.ones((1, 2, 2, 2)), torch.ones((1, 2)))

        model(*dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        graph_modifier = GraphChannelModifier(graph, center_nodes)

        m_reshape1 = graph_modifier.get_modifier(unique_name='reshape_1_f')

        assert (
            str(m_reshape1.dim_changes_info)
            == 'dim_changes_i:OrderedDict([(\'linear1:input_0\', [1])]),'
            ' dim_changes_o:OrderedDict([(\'linear1:output_0\', [0, 1])]), dim_choices:OrderedDict([(\'input_0\','
            ' [1]), (\'output_0\', [1])])'
        )

    def test_case_6(self):
        """
        根据dim choice缩小子图

        Returns:

        """

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=8)
                self.linear1 = nn.Linear(in_features=4, out_features=8)
                self.linear2 = nn.Linear(in_features=8, out_features=16)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape_0 = linear0.reshape((2, 4))
                linear1 = self.linear1(reshape_0)
                linear2 = self.linear2(linear1)
                return linear2

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        graph_modifier = GraphChannelModifier(graph, center_nodes)

        m_linear0 = graph_modifier.get_modifier(model.linear0)
        m_reshape0 = graph_modifier.get_modifier(unique_name='reshape_0_f')
        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_linear2 = graph_modifier.get_modifier(model.linear2)
        m_output0 = graph_modifier.get_modifier(unique_name='output_0_f')

        assert graph_modifier.sub_graphs[m_linear0.unique_name()] == [m_linear0, m_reshape0, m_linear1]
        assert graph_modifier.sub_graphs[m_linear1.unique_name()] == [m_linear1, m_linear2]
        assert graph_modifier.sub_graphs[m_linear2.unique_name()] == [m_linear2, m_output0]

    def test_case_7(self):
        """
        根据dim choice缩小子图

        Returns:

        """

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(1, 1), padding=(0, 0))
                self.linear0 = nn.Linear(in_features=2, out_features=2)
                self.linear1 = nn.Linear(in_features=4, out_features=4)
                self.linear2 = nn.Linear(in_features=4, out_features=8)

            def forward(self, x1, x2):
                conv0 = self.conv0(x1)
                reshape0 = conv0.reshape((2, 4))

                linear0 = self.linear0(x2)
                transpose0 = linear0.transpose(0, 1)
                linear1 = self.linear1(transpose0)

                add0 = reshape0 + linear1
                linear2 = self.linear2(add0)

                return linear2

        model = TestModel()

        dummy_input = (torch.ones((1, 2, 2, 2)), torch.ones((4, 2)))

        model(*dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        graph_modifier = GraphChannelModifier(graph, center_nodes)

        m_conv0 = graph_modifier.get_modifier(model.conv0)
        m_reshape0 = graph_modifier.get_modifier(unique_name='reshape_0_f')
        m_linear0 = graph_modifier.get_modifier(model.linear0)
        m_transpose0 = graph_modifier.get_modifier(unique_name='transpose_0_f')
        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_add0 = graph_modifier.get_modifier(unique_name='add_0_f')
        m_linear2 = graph_modifier.get_modifier(model.linear2)
        m_output0 = graph_modifier.get_modifier(unique_name='output_0_f')

        assert len(graph_modifier.sub_graphs) == 1
        assert graph_modifier.sub_graphs[m_conv0.unique_name()] == [
            m_conv0,
            m_reshape0,
            m_linear0,
            m_transpose0,
            m_linear1,
            m_add0,
            m_linear2,
            m_output0,
        ]

        assert m_linear2.dim_changes_info.dim_changes_i == {'conv0:input_0': [0], 'linear0:input_0': [0]}
        assert m_linear2.dim_changes_info.dim_changes_o == {'linear0:output_0': [0], 'conv0:output_0': [0]}
        assert m_linear2.dim_changes_info.dim_choices == {'input_0': [0], 'output_0': [0]}

    def test_case_8(self):
        """
        测试两种不同constraint的收束

        Returns:

        """

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=8)
                self.linear1 = nn.Linear(in_features=4, out_features=8)
                self.linear2 = nn.Linear(in_features=2, out_features=8)
                self.linear3 = nn.Linear(in_features=8, out_features=8)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape0 = linear0.reshape((2, 4))
                reshape1 = linear0.reshape((4, 2))
                linear1 = self.linear1(reshape0)
                linear2 = self.linear2(reshape1)
                linear3 = self.linear3(linear0)

                return linear1, linear2, linear3

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        graph_modifier = GraphChannelModifier(graph, center_nodes)

        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_linear2 = graph_modifier.get_modifier(model.linear2)
        m_linear3 = graph_modifier.get_modifier(model.linear3)

        assert m_linear1.dim_changes_info.constraints_i == {
            1: {'linear0': [[{0.0, 4.0}, {1.0, 5.0}, {2.0, 6.0}, {3.0, 7.0}]]}
        }
        assert m_linear2.dim_changes_info.constraints_i == {
            1: {'linear0': [[{0.0, 2.0, 4.0, 6.0}, {1.0, 3.0, 5.0, 7.0}]]}
        }
        assert m_linear3.dim_changes_info.constraints_i == {
            1: {'linear0': [[{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}]]}
        }

    def test_case_9(self):
        """
        constraint冲突，无法剪枝

        Returns:

        """

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=6)
                self.linear1 = nn.Linear(in_features=3, out_features=6)
                self.linear2 = nn.Linear(in_features=2, out_features=4)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape0 = linear0.reshape((2, 3))
                reshape1 = linear0.reshape((3, 2))
                linear1 = self.linear1(reshape0)
                linear2 = self.linear2(reshape1)

                return linear1, linear2

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        pruner = OneShotChannelPruner(
            model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm", "skip_last_fc": False}
        )
        graph_modifier = pruner.graph_modifier
        pruner.register_mask()

        m_linear0 = graph_modifier.get_modifier(model.linear0)
        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_linear2 = graph_modifier.get_modifier(model.linear2)

        assert graph_modifier.sub_graphs[m_linear0.unique_name()].center_constraint == {
            'linear0': [{0.0, 1.0, 2.0, 3.0, 4.0, 5.0}]
        }

        assert graph_modifier.sub_graphs[m_linear1.unique_name()].center_constraint == {
            'linear1': [{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}]
        }

        assert graph_modifier.sub_graphs[m_linear2.unique_name()].center_constraint == {
            'linear2': [{0.0}, {1.0}, {2.0}, {3.0}]
        }

    def test_case_10(self):
        """
        constraint冲突，无法剪枝

        Returns:

        """

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=4)
                self.linear1 = nn.Linear(in_features=2, out_features=8)
                self.linear2 = nn.Linear(in_features=12, out_features=12)

            def forward(self, x):
                linear0 = self.linear0(x)
                linear1 = self.linear1(x)
                cat0 = torch.cat([linear0, linear1], dim=1)
                linear2 = self.linear2(cat0)

                return linear2

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        m_linear0 = pruner.graph_modifier.get_modifier(model.linear0)

        assert pruner.graph_modifier.sub_graphs[m_linear0.unique_name()].center_constraint == {
            'linear0': [{0.0}, {1.0}, {2.0}, {3.0}],
            'linear1': [{0.0}, {1.0}, {2.0}, {3.0}, {4.0}, {5.0}, {6.0}, {7.0}],
        }

    def test_case_11(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=12)
                self.linear1 = nn.Linear(in_features=2, out_features=4)
                self.linear2 = nn.Linear(in_features=2, out_features=8)
                self.linear3 = nn.Linear(in_features=12, out_features=12)
                self.linear4 = nn.Linear(in_features=2, out_features=4)
                self.linear5 = nn.Linear(in_features=2, out_features=4)
                self.linear6 = nn.Linear(in_features=8, out_features=8)

            def forward(self, x):
                linear0 = self.linear0(x)
                linear1 = self.linear1(x)
                linear2 = self.linear2(x)
                linear4 = self.linear4(x)
                linear5 = self.linear5(x)

                cat0 = torch.cat([linear1, linear2], dim=1)
                add0 = linear0 + cat0
                linear3 = self.linear3(add0)

                cat1 = torch.cat([linear4, linear5], dim=1)
                add1 = linear2 + cat1
                linear6 = self.linear6(add1)

                return linear3, linear6

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        GraphChannelModifier(graph, center_nodes)

    def test_case_12(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=8)
                self.linear1 = nn.Linear(in_features=8, out_features=8)

            def forward(self, x):
                linear0 = self.linear0(x)

                reshape0 = linear0.reshape((8, 1))
                reshape1 = reshape0.reshape((1, 8))
                add0 = linear0 + reshape1
                linear1 = self.linear1(add0)

                return linear1

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        GraphChannelModifier(graph, center_nodes)

    def test_case_13(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=16)
                self.linear1 = nn.Linear(in_features=8, out_features=8)

            def forward(self, x):
                linear0 = self.linear0(x)
                sp0, sp1 = torch.split(linear0, 8, dim=1)

                reshape0 = sp1.reshape((8, 1))
                reshape1 = reshape0.reshape((1, 8))
                add0 = sp0 + reshape1
                linear1 = self.linear1(add0)

                return linear1

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        graph_modifier = GraphChannelModifier(graph, center_nodes)

        m_linear1 = graph_modifier.get_modifier(model.linear1)

        assert m_linear1.dim_changes_info.constraints_i == {
            1: {
                'linear0': [
                    [
                        {0.0, 8.0},
                        {1.0, 9.0},
                        {2.0, 10.0},
                        {11.0, 3.0},
                        {4.0, 12.0},
                        {13.0, 5.0},
                        {6.0, 14.0},
                        {15.0, 7.0},
                    ]
                ]
            }
        }

    def test_case_14(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(in_features=2, out_features=12)
                self.linear1 = nn.Linear(in_features=2, out_features=4)
                self.linear2 = nn.Linear(in_features=2, out_features=8)
                self.linear3 = nn.Linear(in_features=4, out_features=4)
                self.linear4 = nn.Linear(in_features=12, out_features=12)

            def forward(self, x):
                linear0 = self.linear0(x)
                linear1 = self.linear1(x)
                linear2 = self.linear2(x)

                reshape0 = linear0.reshape((3, 4))
                linear3 = self.linear3(reshape0)

                cat0 = torch.cat([linear1, linear2], dim=1)
                add0 = linear0 + cat0
                linear4 = self.linear4(add0)

                return linear3, linear4

        model = TestModel()

        dummy_input = torch.ones((1, 2))

        model(dummy_input)

        model_generate(model, dummy_input)

        graph, center_nodes = graph_generate(model, dummy_input, [nn.Linear])

        GraphChannelModifier(graph, center_nodes)

    def test_case_15(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear0 = nn.Linear(4, 8)
                self.linear1 = nn.Linear(4, 8)
                self.linear2 = nn.Linear(2, 4)
                self.linear3 = nn.Linear(4, 8)

            def forward(self, x):
                linear0 = self.linear0(x)
                reshape0 = linear0.reshape((2, 4))
                linear1 = self.linear1(reshape0)
                permute0 = linear1.transpose(0, 1)
                linear2 = self.linear2(permute0)
                linear3 = self.linear3(linear2)

                return linear3

        model = TestModel()

        dummy_input = torch.ones((1, 4))

        model(dummy_input)

        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm"})

        pruner.register_mask()

        graph_modifier = pruner.graph_modifier
        m_linear0 = graph_modifier.get_modifier(model.linear0)
        m_reshape0 = graph_modifier.get_modifier(unique_name='reshape_0_f')
        m_transpose0 = graph_modifier.get_modifier(unique_name='transpose_0_f')
        m_linear1 = graph_modifier.get_modifier(model.linear1)
        m_linear2 = graph_modifier.get_modifier(model.linear2)
        m_linear3 = graph_modifier.get_modifier(model.linear3)
        m_output0 = graph_modifier.get_modifier(unique_name='output_0_f')

        assert graph_modifier.sub_graphs[m_linear0.unique_name()] == [
            m_linear0,
            m_reshape0,
            m_linear1,
            m_transpose0,
            m_linear2,
        ]
        assert graph_modifier.sub_graphs[m_linear2.unique_name()] == [m_linear2, m_linear3]
        assert graph_modifier.sub_graphs[m_linear3.unique_name()] == [m_linear3, m_output0]

        pruner.apply_mask()

        pruner.graph.generate_code('out/test.py', 'out/test.pth', 'test')
        new_module = import_from_path('out.test', "out/test.py", "test")()

        new_module(dummy_input)

    def test_calc_dim_changes(self):
        def test_func(shape, dims, node):
            tensor = fill_tensor_by_dim_changes(torch.zeros(shape, dtype=torch.int32), dims)

            dim_changes = None

            if isinstance(node, list):
                for n in node:
                    dim_changes, tensor = calc_dim_changes(n, tensor)[0]
            else:
                dim_changes, tensor = calc_dim_changes(node, tensor)[0]

            return dim_changes

        assert test_func((2, 2, 2), [1], lambda x: x[0].reshape((2, 4))) == [1]
        assert test_func((2, 4), [1], [lambda x: x[0].reshape((1, 8)), lambda x: x[0].reshape((2, 4))]) == [1]
        assert test_func((2, 4, 3), [0, 2], lambda x: x[0].reshape((2, 3, 4))) == [0, 1]
        assert test_func((4, 24, 24), [0], lambda x: x[0].reshape((2, 48, 24))) == [0, 1]
        assert test_func((2, 4, 3), [1], lambda x: x[0].reshape((2, 2, 6))) == [1, 2]
        assert test_func((2, 4, 2, 4, 2), [1, 3], lambda x: x[0].reshape((2, 2, 2, 2, 2, 4))) == [1, 2, 4, 5]
        assert test_func((2, 2, 2), [1, 2], lambda x: x[0].reshape((2, 4))) == [1]
        assert test_func((2, 2, 2, 3), [1, 2], lambda x: x[0].reshape((2, 4, 3))) == [1]
        assert test_func((2, 2, 2, 3), [1, 3], lambda x: x[0].reshape((4, 6))) == [0, 1]
        assert test_func((2, 2, 2), [1], lambda x: x[0].transpose(1, 2)) == [2]
        assert test_func((2, 2, 2), [1], lambda x: x[0].permute((0, 2, 1))) == [2]

    def test_merge_group(self):
        def test_func(group):
            return merge_group(group)

        assert merge_group([{0, 1, 2, 3}, {2, 3}]) == [{0, 1}, {2, 3}]
        assert merge_group([{0, 1, 2, 3}, {2, 3, 4, 5}]) == [{0, 1}, {4, 5}, {2, 3}]
        assert merge_group([{0, 1, 2, 3, 4, 5, 6, 7, 8}, {0, 1, 2, 3}, {2, 3, 4, 5}]) == [
            {6, 7, 8},
            {4, 5},
            {0, 1},
            {2, 3},
        ]

    def test_merge_constraint(self):
        def test_func(constraint, result):
            constraint.sort()
            result.sort()
            sorted_constraint = merge_constraint(constraint)
            assert sorted_constraint == result

        test_func([{0, 1, 2, 3}, {2, 3}], [{0, 1, 2, 3}])
        test_func([{0, 1}, {1, 2}, {2, 3}], [{0, 1, 2, 3}])
        test_func([{0, 1}, {2, 3}, {4, 5}, {0, 6}], [{0, 1, 6}, {2, 3}, {4, 5}])
        test_func([{0}, {0, 1}, {2, 3, 4}, {4, 5}], [{0, 1}, {2, 3, 4, 5}])

    def test_calc_index_constraint(self):
        def test_func(shape, dims, node):
            tensor = fill_tensor_by_dim_changes(torch.zeros(shape, dtype=torch.int32), dims)

            dim_changes = None

            if isinstance(node, list):
                for n in node:
                    dim_changes, tensor = calc_dim_changes(n, tensor)[0]
            else:
                dim_changes, tensor = calc_dim_changes(node, tensor)[0]

            constraint = calc_dim_constraint(tensor, dim_changes)

            return constraint

        assert test_func((1, 8), [1], lambda x: x[0].reshape((2, 4))) == {
            0: [{0, 1, 2, 3}, {4, 5, 6, 7}],
            1: [{0, 4}, {1, 5}, {2, 6}, {3, 7}],
        }
        assert test_func((2, 8), [0], lambda x: x[0].transpose(0, 1)) == {1: [{0}, {1}]}


if __name__ == '__main__':
    unittest.main()
