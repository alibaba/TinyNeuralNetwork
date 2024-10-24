import os
from matplotlib import pyplot as plt
from typing import List

import torch
import torch.nn as nn

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from .train_util import get_logger, get_module_device

log = get_logger(__name__, 'INFO')


def sqnr(x: torch.Tensor, y: torch.Tensor):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return (20 * torch.log10(Ps / Pn)).item()


def cosine(x: torch.Tensor, y: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    """calulate the cosine similarity between x and y"""
    if x.shape != y.shape:
        raise ValueError(f'Can not compute loss for tensors with different shape. ({x.shape} and {y.shape})')
    reduction = str(reduction).lower()

    if x.ndim == 1:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    x = x.flatten(start_dim=1).float()
    y = y.flatten(start_dim=1).float()

    cosine_sim = torch.cosine_similarity(x, y, dim=-1)

    if reduction == 'mean':
        return torch.mean(cosine_sim)
    elif reduction == 'sum':
        return torch.sum(cosine_sim)
    elif reduction == 'none':
        return cosine_sim
    else:
        raise ValueError(f'Cosine similarity do not supported {reduction} method.')


METRIC_DICT = {
    'cosine': cosine,
    'sqnr': sqnr,
}


def error_print(metric, q_errors_activ, q_errors_weight, sort_num):
    logs = []
    if len(q_errors_weight) > 0:
        logs.append('')
        logs.append(f'Weights ({metric} sorted {sort_num}):')
        for n, m, e in q_errors_weight:
            logs.append(f'{n:40} {metric}: {e:.4f}, scale: {m.scale.item():.4f}, zero_point: {m.zero_point.item()}')
    if len(q_errors_activ) > 0:
        logs.append('')
        logs.append(f'Activations ({metric} sorted {sort_num}):')
        for n, m, e in q_errors_activ:
            logs.append(f'{n:50} {metric}: {e:.4f}, scale: {m.scale.item():.4f}, zero_point: {m.zero_point.item()}')
    if len(q_errors_weight) == 0 and len(q_errors_activ) == 0:
        logs.append('')
        logs.append('All good!')
    if len(logs) > 0:
        logs.insert(0, 'Quantization error report:')
        logs.append('')

        full_log = '\n'.join(logs)
        log.warning(full_log)


def layer_error_analysis(q_model: nn.Module, dummy_input, metric: str = 'cosine', sort_num: float = 20):
    """Generates the layerwise quant error report using the given metric, the q_model need to be qat_prepared.

    Args:
        q_model: The quant prepared model
        dummy_input: A viable input to the model
        metric: Metrics for measuring the error of floating point tensor and quantized tensor.
                Default to be 'cosine', optional 'sqnr'.
        sort_num : The smallest sort_num layer0 on given metric. Defaults to 20
    """

    if isinstance(q_model, DataParallel) or isinstance(q_model, DistributedDataParallel):
        model = q_model.module
    else:
        model = q_model
    metric_fn = METRIC_DICT[metric]

    train_flag = model.training
    model.eval()
    with torch.no_grad():
        modules_list = {}
        names_list = {}
        float_results = {}
        hooks = []

        def forward_hook(module, input, output):
            name = names_list[module]
            float_results[name] = input

        fake_quant_enabled_dict = {}
        observer_enabled_dict = {}
        for n, m in model.named_modules():
            if isinstance(m, torch.quantization.FakeQuantize):
                names_list[m] = n
                modules_list[n] = m

                fake_quant_enabled_dict[m] = m.fake_quant_enabled.clone()
                observer_enabled_dict[m] = m.observer_enabled.clone()

                hooks.append(m.register_forward_hook(forward_hook))

        if len(modules_list) == 0:
            log.warning('No FakeQuantize modules found. Are you sure you had prepared your model?')

        model.apply(torch.quantization.disable_fake_quant)
        model.apply(torch.quantization.disable_observer)

        device = get_module_device(model)

        if type(dummy_input) is torch.Tensor:
            actual_input = [dummy_input]
        elif isinstance(dummy_input, (tuple, list)):
            actual_input = list(dummy_input)
        else:
            log.error(f'Unsupported type {type(dummy_input)} for dummy input')
            assert False

        for i in range(len(actual_input)):
            dummy_input = actual_input[i]
            if type(dummy_input) is torch.Tensor:
                if dummy_input.device != device:
                    actual_input[i] = dummy_input.to(device)

        with torch.no_grad():
            model(*actual_input)

        for h in hooks:
            h.remove()
        hooks.clear()

        for m, v in fake_quant_enabled_dict.items():
            m.fake_quant_enabled = v

        q_errors_weight = []
        q_errors_activ = []
        while len(float_results) > 0:
            n, f = float_results.popitem()
            mod = modules_list[n]
            with torch.no_grad():
                q = mod(*f)
                loss = metric_fn(f[0], q)
            actual_n = '.'.join(n.split('.')[:-1])
            if n.endswith('.weight_fake_quant'):
                q_errors_weight.append((actual_n, mod, loss))
            else:
                q_errors_activ.append((actual_n, mod, loss))

        q_errors_weight = sorted(q_errors_weight, key=lambda x: x[2])
        q_errors_activ = sorted(q_errors_activ, key=lambda x: x[2])

        q_errors_weight = q_errors_weight[:sort_num]
        q_errors_activ = q_errors_activ[:sort_num]

        error_print(metric, q_errors_activ, q_errors_weight, sort_num)

        for m, v in observer_enabled_dict.items():
            m.observer_enabled = v

    if train_flag:
        model.train()


def graph_error_analysis(q_model: nn.Module, dummy_input, metric: str = 'cosine'):
    """Generates the cumulative quant error report using the given metric, the q_model need to be qat_prepared.

    Args:
        q_model: The quant prepared model.
        dummy_input: A viable input to the model
        metric: Metrics for measuring the error of floating point tensor and quantized tensor.
                Default to be 'cosine', optional 'sqnr'.
    """
    if isinstance(q_model, DataParallel) or isinstance(q_model, DistributedDataParallel):
        model = q_model.module
    else:
        model = q_model
    metric_fn = METRIC_DICT[metric]

    train_flag = model.training
    model.eval()

    with torch.no_grad():
        modules_list = {}
        names_list = {}
        results = {}
        hooks = []

        def forward_hook(module, input, output):
            name = names_list[module]
            results[name] = input

        fake_quant_enabled_dict = {}
        observer_enabled_dict = {}
        for n, m in model.named_modules():
            if isinstance(m, torch.quantization.FakeQuantize):
                names_list[m] = n
                modules_list[n] = m

                fake_quant_enabled_dict[m] = m.fake_quant_enabled.clone()
                observer_enabled_dict[m] = m.observer_enabled.clone()

                hooks.append(m.register_forward_hook(forward_hook))

        model.apply(torch.quantization.disable_fake_quant)
        model.apply(torch.quantization.disable_observer)

        if len(modules_list) == 0:
            log.warning('No FakeQuantize modules found. Are you sure you had prepared your model?')

        device = get_module_device(model)

        if type(dummy_input) is torch.Tensor:
            actual_input = [dummy_input]
        elif isinstance(dummy_input, (tuple, list)):
            actual_input = list(dummy_input)
        else:
            log.error(f'Unsupported type {type(dummy_input)} for dummy input')
            assert False

        for i in range(len(actual_input)):
            dummy_input = actual_input[i]
            if type(dummy_input) is torch.Tensor:
                if dummy_input.device != device:
                    actual_input[i] = dummy_input.to(device)

        model(*actual_input)

        # Restore fake-quantize and record activation with quantization error.
        for m, v in fake_quant_enabled_dict.items():
            m.fake_quant_enabled = v
        float_results = results
        results = {}

        model(*actual_input)

        for h in hooks:
            h.remove()
        hooks.clear()

        q_errors_activ = []
        for name, f_tensor in float_results.items():
            assert name in results, f'{name} not in results'
            actual_n = '.'.join(name.split('.')[:-1])
            loss = metric_fn(f_tensor[0], results[name][0])
            if not name.endswith('.weight_fake_quant'):
                q_errors_activ.append((actual_n, modules_list[name], loss))

        error_print(metric, q_errors_activ, [], '')

        for m, v in observer_enabled_dict.items():
            m.observer_enabled = v

    if train_flag:
        model.train()


def get_weight_dis(
    model: nn.Module,
    unique_name_list: List[str] = None,
    nbins=256,
    save_path: str = 'out',
    threshold=20,
    fig_size=(7, 7),
):
    """Draw the weight distribution of model

    Args:
        model: We recommend use ptq-prepared model to draw fused weight distribution
        unique_name_list: You can set the layer which you want to get distribution, default to all layer of model
        nbins: Bins of distribution, default to be 256
        save_path: Weight distribution fig weill saved at "[save_path]/weight_distribution"
        threshold: The threshold of weight range to used to prompt anomalies
        fig_size: Set fig size
    """
    with torch.no_grad():
        save_dir = os.path.join(save_path, 'weight_distribution')
        log.info(f"jpgs will saved at {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        warning_layer = dict()
        for name, mod in model.named_modules():
            if (not hasattr(mod, 'weight')) or isinstance(mod, nn.BatchNorm2d):
                continue
            if unique_name_list is None or name in unique_name_list:
                op_type = type(mod).__name__
                x = mod.weight.cpu()
                if op_type in dir(torch.nn.intrinsic.qat) and hasattr(mod, 'bn'):
                    # Use torch.nn.util.fusion.fuse_conv_bn_weights to caculate bn_fused conv's weight.
                    bn_var_rsqrt = torch.rsqrt(mod.bn.running_var + mod.bn.eps)
                    x = mod.weight * (mod.bn.weight * bn_var_rsqrt).reshape([-1] + [1] * (len(mod.weight.shape) - 1))
                    x = x.cpu()
                y = torch.histc(x, nbins)
                x_min = torch.min(x)
                x_max = torch.max(x)
                if x_max - x_min > threshold:
                    warning_layer[name] = (op_type, float(x_min), float(x_max))
                bin_width = (x_max - x_min) / nbins
                x_s = [x_min + (idx + 0.5) * bin_width for idx in range(nbins)]

                fig, ax = plt.subplots(figsize=fig_size)
                ax.set_yscale('log')
                ax.plot(x_s, y.detach().numpy())
                ax.set_title(f'Op_uname:  {name}[{op_type}]')
                ax.set_xlabel(f'Range:[{x_min:.4f},{x_max:.4f}]')
                ax.set_ylabel('Count')
                save_path = os.path.join(save_dir, f'{name}.jpg')
                plt.savefig(save_path)
                plt.cla()
        if warning_layer:
            log_str = f'\n---------the layer weight range length greater than {threshold}---------\n'
            for k, v in warning_layer.items():
                log_str += f'{k}, {v}\n'
            log_str += '---------------------------------------------------------------'
            log.warning(log_str)
