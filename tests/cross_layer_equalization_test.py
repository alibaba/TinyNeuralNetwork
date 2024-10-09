import unittest

import torch
import torch.nn as nn
from tinynn.graph.quantization.algorithm.cross_layer_equalization import cross_layer_equalize


class TestCrossLayerEqualization(unittest.TestCase):
    def test_cle_conv(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 8, 2)
                self.conv1 = nn.Conv2d(8, 16, 2)
                self.conv2 = nn.Conv2d(16, 64, 2)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                conv2 = self.conv2(conv1)
                return conv2

        torch.manual_seed(10)

        dummy_input = torch.randn(1, 3, 224, 224)
        model = TestModel()
        model.eval()

        origin_output = model(dummy_input)
        cle_model = cross_layer_equalize(model, dummy_input, torch.device('cpu'), hba_flag=False)
        cle_output = cle_model(dummy_input)
        torch.testing.assert_allclose(origin_output, cle_output)

    def test_cle_group_conv(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv0 = nn.Conv2d(3, 8, 2)
                self.conv1 = nn.Conv2d(8, 8, 2, groups=4)
                self.conv1 = nn.Conv2d(8, 8, 2, groups=8)
                self.conv2 = nn.Conv2d(8, 16, 2)

            def forward(self, x):
                conv0 = self.conv0(x)
                conv1 = self.conv1(conv0)
                conv2 = self.conv2(conv1)
                return conv2

        torch.manual_seed(10)

        dummy_input = torch.randn(1, 3, 224, 224)
        model = TestModel()
        model.eval()

        origin_output = model(dummy_input)
        cle_model = cross_layer_equalize(model, dummy_input, torch.device('cpu'), hba_flag=False)
        cle_output = cle_model(dummy_input)
        torch.testing.assert_allclose(origin_output, cle_output)

    def test_cle_linear(self):
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc0 = nn.Linear(3, 8)
                self.fc1 = nn.Linear(8, 16)
                self.fc2 = nn.Linear(16, 32)

            def forward(self, x):
                fc0 = self.fc0(x)
                fc1 = self.fc1(fc0)
                fc2 = self.fc2(fc1)
                return fc2

        torch.manual_seed(10)

        dummy_input = torch.randn(1, 3)
        model = TestModel()
        model.eval()

        origin_output = model(dummy_input)
        cle_model = cross_layer_equalize(model, dummy_input, torch.device('cpu'), hba_flag=False)
        cle_output = cle_model(dummy_input)
        torch.testing.assert_allclose(origin_output, cle_output)
