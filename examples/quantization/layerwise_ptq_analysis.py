import argparse
import os

import torch
import torch.nn as nn

from tinynn.graph.tracer import model_tracer
from tinynn.util.cifar10 import get_dataloader, calibrate, validate
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.util.train_util import DLContext, get_device

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml
import os

yaml_ = yaml.YAML()

# import your model
from examples.models.cifar10.mobilenet import DEFAULT_STATE_DICT, Mobilenet


def main_worker(args):
    with model_tracer():
        model = Mobilenet()
        model.load_state_dict(torch.load(DEFAULT_STATE_DICT))

        # Provide a viable input for the model
        dummy_input = torch.rand((1, 3, 224, 224))

        # Initialize the model running context, which include train/test dataloader
        context = DLContext()
        device = get_device()
        context.device = device
        context.train_loader, context.val_loader = get_dataloader(args.data_path, 224, args.batch_size, args.workers)
        # Set calibrate iter nums
        context.max_iteration = 20

        # You need to provide a validate-function that return metric of the model.
        acc_origin = validate(model, context)

        # The result will be saved at "out/acc_{YOUR_MODEL_NAME}.yml".
        # Result is like: {'layer name 0': 0.935(-0.005), 'layer name 1': 0.940(0.000)}
        acc_result_pth = f'out/acc_{type(model).__name__}.yml'
        if os.path.exists(acc_result_pth):
            with open(acc_result_pth, 'r') as f:
                acc_per_layer = yaml_.load(f)
                if acc_per_layer is None:
                    acc_per_layer = {}
        else:
            acc_per_layer = {}

        # Use `PostQuantizer::effective_layers` to get the list of layer which may produce quantization error.
        # If you only focus on specific layers, feel free to modify `layer_name_list`.
        # quantizer = PostQuantizer(model, dummy_input, work_dir='out', config ={'ignore_layerwise_config': True})
        quantizer = PostQuantizer(model, dummy_input, work_dir='out', config={"layerwise_config": {'default': True}})
        model_q = quantizer.quantize()
        layerwise_config = quantizer.layerwise_config
        layer_name_list = quantizer.effective_layers

        for layer_name in layer_name_list:
            if layer_name in acc_per_layer:
                continue

            print(f"[TinyNeralNetwork] starting ptq {layer_name}({type(getattr(model_q, layer_name)).__name__}):")
            for k in layerwise_config:
                if k == layer_name:
                    layerwise_config[k] = True
                else:
                    layerwise_config[k] = False

            with model_tracer():
                # Use `layerwise_config` to do mixed-precision quantization only at `layer_name`.
                quantizer_cur = PostQuantizer(
                    model, dummy_input, work_dir='out', config={"layerwise_config": layerwise_config}
                )
                q_model_cur = quantizer_cur.quantize()

            # Use DataParallel to speed up calibrating when possible
            if torch.cuda.device_count() > 1:
                q_model_cur = nn.DataParallel(q_model_cur)

            q_model_cur.to(device=device)
            context.device = device
            # Post quantization calibration, use your calibrate function to replace the calibrate function.
            calibrate(q_model_cur, context)

            with torch.no_grad():
                q_model_cur.eval()
                q_model_cur.cpu()

                # The step below converts the model to an actual quantized model, which uses the quantized kernels.
                q_model_cur = torch.quantization.convert(q_model_cur)

                # Now, 'q_model_cur' is a quantized model which only quantize 'layer_name'.
                # In PyTorch backend, you need move the quantized model to cpu to test its metric.
                context.device = torch.device("cpu")
                if torch.cuda.device_count() > 1:
                    q_model_cur = q_model_cur.module
                # Use your validate function(return acc) to replace the validate function.
                acc = validate(q_model_cur, context)
                acc_per_layer[layer_name] = f"{acc:.4f}({(acc - acc_origin):4f})"
                print(f'[TinyNeralNetwork] Only quantize {layer_name}:{acc:.4f}({(acc - acc_origin):4f})')
                with open(acc_result_pth, 'w') as f:
                    yaml_.dump(acc_per_layer, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/cifar10", help='path to dataset')
    parser.add_argument('--config', type=str, default=os.path.join(CURRENT_PATH, 'config.yml'))
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=128)

    args = parser.parse_args()
    main_worker(args)
