import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../'))

import torch
import numpy as np

from tinynn.converter import TFLiteConverter
from tinynn.graph.tracer import model_tracer
from tinynn.util.train_util import DLContext
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize

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

    with model_tracer():
        dummy_input = torch.rand(1, 58)

        # from graphmodule_q import QGraphModule
        from examples.pr_solve.sun_0930.graphmodule_q_v1 import QGraphModule

        model = QGraphModule()
        model.load_state_dict(
            torch.load("/data/zhouye/0603/TinyNeuralNetwork/examples/pr_solve/sun_0930/graphmodule_v1.pth")
        )

        model.eval()

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
    quantizer.rescale_activations_with_quant_min_max(0, 255)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_converted_model = quantizer.convert(qat_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = quantizer.backend

        fp_weight_dict = {}
        for n, m in qat_model.named_children():
            if hasattr(m, 'weight'):
                fp_weight_dict[n] = m.weight.data
        converter = TFLiteConverter(
            qat_converted_model,
            dummy_input,
            strict_symmetric_check=True,
            quantize_target_type='int16',
            tflite_path='out/qat_model.tflite',
            output_transpose=False,
            fp_weight_dict=fp_weight_dict,
        )
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--cle', type=bool, default=False)

    args = parser.parse_args()
    main_worker(args)
