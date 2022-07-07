import os
import time
import unittest

import torch
import torch.nn as nn
import torchvision
import random
import torch.nn.functional

from tinynn.prune.oneshot_pruner import OneShotChannelPruner
from tinynn.util.util import import_from_path, get_logger

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

log = get_logger(__name__)


def get_topk(lst, k, offset=0):
    idx_lst = [(i, lst[i]) for i in range(len(lst))]
    sorted_lst = sorted(idx_lst, key=lambda x: x[1])
    sorted_lst_k = sorted_lst[:k]
    idx = [sorted_lst_k[i][0] + offset for i in range(len(sorted_lst_k))]

    return sorted(idx)


def get_rd_lst(length):
    rd_lst = random.sample(range(0, 10000), length)
    random.shuffle(rd_lst)

    print(rd_lst)
    return rd_lst


def init_conv_by_list(conv, ch_value):
    assert conv.weight.shape[0] == len(ch_value)

    for i in range(len(ch_value)):
        conv.weight.data[i, :] = ch_value[i]


def module_init(model: nn.Module, init_dict=None):
    init_value = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            channel = module.out_features
        elif isinstance(module, nn.Conv2d):
            channel = module.out_channels
        else:
            continue

        if init_dict and name in init_dict:
            ch_value = init_dict[name]
        else:
            ch_value = get_rd_lst(channel)

        init_conv_by_list(module, ch_value)
        init_value[name] = ch_value

    print(init_value)
    return init_value


def speed_test(model, dummy_input):
    with torch.no_grad():
        model.eval()
        st = time.time()
        pruner_new = OneShotChannelPruner(model, dummy_input, {"sparsity": 0.5, "metrics": "l2_norm"})
        log.info(f"[SPEED TEST][Pruner Init] {time.time() - st}")
        st = time.time()

        pruner_new.register_mask()
        log.info(f"[SPEED TEST][Register Mask] {time.time() - st}")
        st = time.time()
        pruner_new.apply_mask()

        log.info(f"[SPEED TEST][Apply Mask] {time.time() - st}")

        pruner_new.graph.generate_code('out/new_model.py', 'out/new_model.pth', 'new_model')
        new_model_pruned = import_from_path('out.new_model', "out/new_model.py", "new_model")()
        new_model_pruned(dummy_input)


class ModifierForwardTester(unittest.TestCase):
    def test_mbv2(self):
        model = torchvision.models.mobilenet_v2(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    @unittest.skipIf(not hasattr(torchvision.models, 'mobilenet_v3_small'), 'mobilenet_v3_small is not available')
    def test_mbv3(self):
        model = torchvision.models.mobilenet_v3_small(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    @unittest.skipIf(not hasattr(torchvision.models, 'mobilenet_v3_large'), 'mobilenet_v3_large is not available')
    def test_mbv3_large(self):
        model = torchvision.models.mobilenet_v3_large(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    def test_vgg16(self):
        model = torchvision.models.vgg16(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    def test_googlenet(self):
        model = torchvision.models.googlenet(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    def test_shufflenet(self):
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    def test_resnet18(self):
        model = torchvision.models.resnet18(pretrained=False)
        module_init(model)
        speed_test(model, torch.randn((1, 3, 224, 224)))

    def test_densenet121(self):
        model = torchvision.models.densenet121(pretrained=False)
        speed_test(model, torch.randn((1, 3, 224, 224)))


if __name__ == '__main__':
    unittest.main()
