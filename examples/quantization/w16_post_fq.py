import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import numpy as np

from tinynn.graph.tracer import model_tracer
from tinynn.util.train_util import DLContext
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.util.quantization_analysis_util import graph_error_analysis, layer_error_analysis

device = torch.device('cuda', 0)


class TensorDataset:
    def __init__(self, path):
        assert os.path.exists(path), "%sincorrect path" % path
        self.path = path
        self.data_list = [fname for fname in os.listdir(self.path) if fname.lower().endswith('.npy')]

    def __getitem__(self, index):
        input_path = os.path.join(self.path, self.data_list[index])
        input_npy = np.load(input_path)
        input_npy = input_npy.reshape((1, 58))
        return torch.from_numpy(input_npy)

    def __len__(self):
        return len(self.data_list)


def main_worker(args):
    # !change to your calibrate data
    dataloader = TensorDataset('/data/zhouye/0603/TinyNeuralNetwork/examples/pr_solve/sun_0930/new_data')

    test_data = dataloader[2]

    with model_tracer():
        dummy_input = torch.rand(1, 58)

        # load qmodel
        from examples.pr_solve.sun_0930.graphmodule_q_v1 import QGraphModule

        model = QGraphModule()
        model.load_state_dict(
            torch.load("/data/zhouye/0603/TinyNeuralNetwork/examples/pr_solve/sun_0930/graphmodule_v1.pth")
        )

        model.eval()
        output_float = model(test_data)

        quantizer = QATQuantizer(
            model,
            dummy_input,
            work_dir='out',
            config={
                'extra_tracer_opts': {'patch_torch_size': True},
                'force_overwrite': False,
                'rewrite_graph': False,
                'asymmetric': False,
                'per_tensor': False,
                'override_qconfig_func': set_ptq_fake_quantize,
            },
        )
        qat_model = quantizer.quantize()
        # modify rescale_activations_with_quant_min_max to set weight int16 quant range
        quantizer.rescale_activations_with_quant_min_max(0, 65535)

    # Move model to the appropriate device
    qat_model.to(device=device)
    qat_model.eval()
    context = DLContext()
    context.device = device

    qat_model.apply(torch.quantization.disable_fake_quant)
    qat_model.apply(torch.quantization.enable_observer)
    for i in range(100):
        qat_model(dataloader[i].to(device=device))
    # Disable observer and enable fake quantization to validate model with quantization error
    qat_model.apply(torch.quantization.disable_observer)
    qat_model.apply(torch.quantization.enable_fake_quant)
    qat_model(dummy_input.to(device=device))

    print(qat_model)
    dummy_input_real = test_data
    output = qat_model(test_data.to(device=device))
    print("quant: ", output)
    print("fp: ", output_float)
    print("diff: ", output.detach().cpu() - output_float.detach().cpu())
    graph_error_analysis(qat_model, dummy_input_real, metric='cosine')
    layer_error_analysis(qat_model, dummy_input_real, metric='cosine')

    for n, m in qat_model.named_children():
        print(n)
        if hasattr(m, 'weight_fake_quant'):
            print(
                f"|weight q_param: scale:{float(m.weight_fake_quant.scale)}, zp:{int(m.weight_fake_quant.zero_point)}"
            )
        if hasattr(m, 'activation_post_process'):
            print(
                f"|activation q_param: scale:{float(m.activation_post_process.scale)},"
                f" zp:{int(m.activation_post_process.zero_point)}"
            )
    # exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cle', type=bool, default=False)

    args = parser.parse_args()
    main_worker(args)
