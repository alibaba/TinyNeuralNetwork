import argparse
import os
import sys

CURRENT_PATH = os.path.abspath(os.path.dirname(__file__))

sys.path.insert(1, os.path.join(CURRENT_PATH, '../../../../'))

import torch
import torch.nn as nn
import torch.quantization as torch_q

from tinynn.graph.tracer import model_tracer, import_patcher
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.util.train_util import DLContext, get_device
from tinynn.graph.quantization.fake_quantize import set_ptq_fake_quantize
from tinynn.util.quantization_analysis_util import graph_error_analysis, layer_error_analysis
from tinynn.converter import TFLiteConverter

from examples.quantization.specific.util.imagenet_util import get_dataloader, validate, calibrate

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml

yaml_ = yaml.YAML()

with import_patcher():
    from transformers import ViTForImageClassification


class ViTWrapper(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        return self.vit(x).logits


def ptq_vit(args):
    device = get_device()
    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = get_dataloader(
        args.data_path, 224, args.batch_size, args.workers, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )

    dummy_input = torch.rand((1, 3, 224, 224))
    # We use huggingface to get pretrained vit model
    model_url = 'google/vit-base-patch16-224'
    with import_patcher():
        model = ViTForImageClassification.from_pretrained(model_url)
        model = ViTWrapper(model)

    model.to(device)
    context.device = device
    model.eval()
    validate(model, context)

    # Use quantizer to get layerwise config to do mixed quantization
    with model_tracer():
        # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
        # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
        quantizer = QATQuantizer(
            model,
            dummy_input,
            work_dir='out',
            config={
                'override_qconfig_func': set_ptq_fake_quantize,
                'extra_tracer_opts': {'eliminate_dead_graph': True},
            },
        )
        quantizer.quantize()

    layerwise_config_cur = quantizer.layerwise_config
    # Keep all residual add and gelu layer FP calculation,
    # and keep two quantization sensitive fc layer FP calculation additionally.
    for name in quantizer.layerwise_config:
        if (
            name.startswith('add_')
            or name.startswith('gelu')
            or name
            in (
                'vit_vit_encoder_layer_5_output_dense',
                'vit_vit_encoder_layer_4_output_dense',
                'vit_vit_encoder_layer_4_output_dropout',
                'vit_vit_encoder_layer_5_output_dropout',
            )
        ):
            layerwise_config_cur[name] = False
        else:
            layerwise_config_cur[name] = True

    with model_tracer():
        # More information for QATQuantizer initialization, see `examples/quantization/qat.py`.
        # We set 'override_qconfig_func' when initializing QATQuantizer to use fake-quantize to do post quantization.
        quantizer = QATQuantizer(
            model,
            dummy_input,
            work_dir='out',
            config={
                "layerwise_config": layerwise_config_cur,
                'override_qconfig_func': set_ptq_fake_quantize,
                "force_overwrite": True,
                'set_quantizable_op_stats': True,
                'extra_tracer_opts': {'eliminate_dead_graph': True},
            },
        )
        ptq_model = quantizer.quantize()

    if torch.cuda.device_count() > 1:
        ptq_model = nn.DataParallel(ptq_model)

    ptq_model.to(device=device)
    # Set number of iteration for calibration
    context.max_iteration = 20

    # Post quantization calibration
    ptq_model.apply(torch_q.disable_fake_quant)
    ptq_model.apply(torch_q.enable_observer)
    calibrate(ptq_model, context)

    # Disable observer and enable fake quantization to validate model with quantization error
    ptq_model.apply(torch_q.disable_observer)
    ptq_model.apply(torch_q.enable_fake_quant)
    dummy_input = dummy_input.to(device)
    ptq_model(dummy_input)
    print(ptq_model)

    # Perform quantization error analysis with real dummy input
    dummy_input_real = next(iter(context.train_loader))[0][:1]

    # The error is accumulated by directly giving the difference in layer output
    # between the quantized model and the floating model. If you want a quantized model with high accuracy,
    # the layers closest to the final output should be less than 10%, which means the final
    # layer's cosine similarity should be greater than 90%.
    graph_error_analysis(ptq_model, dummy_input_real, metric='cosine')

    # We quantize each layer separately and compare the difference
    # between the original output and the output with quantization error for each layer,
    # which is used to calculate the quantization sensitivity of each layer.
    layer_error_analysis(ptq_model, dummy_input_real, metric='cosine')

    # Validate the model with quantization error via fake quantization
    validate(ptq_model, context)

    with torch.no_grad():
        ptq_model.eval()
        ptq_model.cpu()

        if isinstance(ptq_model, nn.DataParallel):
            ptq_model = ptq_model.module

        ptq_model = torch.quantization.convert(ptq_model)
        ptq_model(dummy_input.cpu())

        # validate the quantized mode, the acc results are almost identical to the previous fake-quantized results
        context.device = torch.device('cpu')
        validate(ptq_model, context)

        torch.backends.quantized.engine = quantizer.backend

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        converter = TFLiteConverter(
            ptq_model,
            dummy_input,
            tflite_path='out/vit_ptq.tflite',
            quantize_target_type='int8',
            rewrite_quantizable=True,
        )
        converter.convert()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', metavar='DIR', default="/data/datasets/imagenet", help='path to dataset')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--model-pth-dir', type=str, default='out')

    args = parser.parse_args()
    ptq_vit(args)
