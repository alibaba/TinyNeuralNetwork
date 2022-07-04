import unittest

import torch
import torchvision
import torch.nn as nn
from tinynn.prune.oneshot_pruner import OneShotChannelPruner
import inspect

from models.hrnet125 import hrnet125
from models.efficientnet_v2_s import efficientnet_v2_s
from models.efficientnet_v2_xl import efficientnet_v2_xl


class ModifierTester(unittest.TestCase):
    def test_hrnet(self):
        model = hrnet125()
        dummy_input = torch.ones((1, 3, 224, 224))
        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()
        model(dummy_input)
        print("test hrnet over!\n")

    def test_efficientnet_v2_xl(self):
        model = efficientnet_v2_xl()
        dummy_input = torch.ones((1, 3, 224, 224))
        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()
        model(dummy_input)
        print("test effnetv2_xl over!\n")

    def test_efficientnet_v2_s(self):
        model = efficientnet_v2_s()
        dummy_input = torch.ones((1, 3, 224, 224))
        pruner = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.75, "metrics": "l2_norm"})
        pruner.prune()
        model(dummy_input)
        print("test effnetv2_s over!\n")

    def test_timm(self):
        try:
            import timm
        except ImportError:
            print('Timm can not find!')
            return
        model_list = [
            'gluon_xception65',
            'resnest14d',
            'legacy_seresnet18',
            'inception_v4',
            'mnasnet_050',
        ]
        for model_name in model_list:
            print(f"prune {model_name} ing!\n")
            model = timm.create_model(model_name, pretrained=False)
            model.eval()
            dummy_input = torch.ones((1, 3, 224, 224))
            pruner = OneShotChannelPruner(
                model, dummy_input, {"sparsity": 0.75, "metrics": "l2_norm", "skip_last_fc": True}
            )
            pruner.prune()
            model(dummy_input)
            print(f"test {model_name} over!\n")


if __name__ == '__main__':
    unittest.main()
