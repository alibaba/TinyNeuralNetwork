"""Cross Layer Equalization
Cross-Layer-Equalization can scale weights equivalently to reduce weight outliers in per_tensor mode.
You can use CLE to adjust the model weight, then do next ptq/qat(rep-model qat need to restore BN).
"""
import copy
import os
import functools
from typing import Tuple

import torch
import torch.nn as nn

import torch.quantization as torch_q
from tinynn.graph.tracer import model_tracer, trace, TraceNode
from tinynn.graph.quantization.quantizer import PostQuantizer, load_processed_ptq_rules
from tinynn.util.util import get_logger
from tinynn.util.bn_restore import model_restore_bn, ConvBnTrain
from tinynn.util.util import import_from_path

log = get_logger(__name__)

cls_support_type = (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)
cls_scalable_type = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU, torch.nn.Identity)


def is_group_supported(current_group):
    """Currently Supported layer combinations for CLS are:
    1. [conv-conv]
    """
    current_group_ = [mod for n, mod in current_group]
    if (
        len(current_group_) == 2
        and isinstance(current_group_[0], cls_support_type)
        and isinstance(current_group_[1], cls_support_type)
    ):
        return True
    else:
        # Todo: more general CLE.
        return False


def graph_traverse(node: TraceNode, layer_groups, current_group=None, visited_nodes=None):
    """Recursively traverse the computational graph and find all conv-groups that can be weight-equal."""
    if visited_nodes is None:
        visited_nodes = []
    if node in visited_nodes:
        return
    if current_group is None:
        current_group = []

    # add cc or cdc to layer_group
    if is_group_supported(current_group) and current_group not in layer_groups:
        layer_groups.append(current_group)
        current_group = [current_group[-1]]

    visited_nodes.append(node)

    if isinstance(node.module, cls_support_type):
        current_group.append((node.unique_name, node.module))

    if len(node.next_nodes) > 1 or not isinstance(node.module, (cls_scalable_type, cls_support_type)):
        if is_group_supported(current_group) and current_group not in layer_groups:
            layer_groups.append(current_group)
        current_group = []

    for n in node.next_nodes:
        graph_traverse(n, layer_groups, current_group, visited_nodes)
        current_group = []


def get_cls_set(cur_graph):
    layer_groups = []
    visited_nodes = []
    for node in cur_graph.forward_nodes:
        graph_traverse(node, layer_groups, visited_nodes=visited_nodes)
    return layer_groups


def equalize(weight_1, weight_2, group=1, threshold=0.5, s_min=1e-6, s_max=1e6):
    """calculate scale for two layer according to their weights"""
    shape_2 = weight_2.shape
    # for group conv
    weight_1_re = torch.reshape(weight_1, (weight_1.shape[0], -1))
    weight_2_re = torch.reshape(
        weight_2,
        (
            group,
            shape_2[0] // group,
        )
        + shape_2[1:],
    )
    num_dims = weight_2_re.dim()
    assert num_dims >= 3, f"weight_2_re shape dim={num_dims}, <3"
    new_order = [2, 0, 1] + list(range(3, num_dims))
    weight_2_re = weight_2_re.permute(new_order)
    weight_2_re = torch.reshape(weight_2_re, (weight_2_re.shape[0] * weight_2_re.shape[1], -1))
    r1 = weight_1_re.abs().max(1).values.double()
    r2 = weight_2_re.abs().max(1).values.double()
    s = r1 / torch.sqrt(r1 * r2)

    # ignore too small scale
    s = torch.clamp(s, s_min, s_max)
    # refuse to scale unnecessary layers pair
    s = torch.where((r1 + r2) < threshold, torch.ones_like(s), s)

    return s


def _weight_equal_helper(cls, threshold=0.5):
    layer_pair = [m for n, m in cls]
    if len(layer_pair) == 2:
        conv_0, conv_1 = layer_pair
        weight1, bias1, weight2, groups = (
            conv_0.weight,
            conv_0.bias,
            conv_1.weight,
            conv_1.groups if hasattr(conv_1, 'groups') else 1,
        )
        s = equalize(weight1, weight2, group=groups, threshold=threshold)
        weight_1 = weight1 / s.reshape([-1] + ([1] * (weight1.ndim - 1)))
        weight_2 = torch.reshape(weight2, (groups, weight2.shape[0] // groups) + weight2.shape[1:])
        weight_2 *= torch.reshape(s, [groups, 1, -1] + [1] * (weight_2.ndim - 3))
        weight_2 = torch.reshape(weight_2, (weight_2.shape[1] * groups,) + weight_2.shape[2:])
        conv_0.weight.data.copy_(weight_1)
        if conv_0.bias is not None:
            conv_0.bias.data.copy_(bias1 / s)
        conv_1.weight.data.copy_(weight_2)
    else:
        log.warning(f'layer_pair nums != 2, do not support, current layer:{cls}.')


def equalize_model(model: nn.Module, dummy_input, threshold=0.5, iters=2) -> Tuple[list, nn.Module]:
    """perform Cross-Layer Equalization(CLE) on the given model iters times.

    Args:
        model: The bn of model should be fused into conv.
        dummy_input (torch.tensor): A viable input for the model.
        threshold: Default to be 1000, used to prevent unquantifiable anomalies in the output of inter conv.
    Returns:
        typing.Tuple[List, nn.Module], layers groups and model after CLE.
    """
    with torch.no_grad():
        with model_tracer():
            cur_graph = trace(model, dummy_input)
            param = {}
            for k, v in model.state_dict().items():
                p, _ = cur_graph.get_submodule_with_parent_from_name(k)
                if k.endswith('.weight'):
                    param[k] = p.abs().max()
                elif k.endswith('.bias'):
                    param[k] = p.max()

            layer_groups = get_cls_set(cur_graph)
            for i in range(iters):
                for cls in layer_groups:
                    _weight_equal_helper(cls, threshold)

            stat_we = model.state_dict()
            for k, v in stat_we.items():
                p, mod = cur_graph.get_submodule_with_parent_from_name(k)
                if isinstance(mod, cls_support_type):
                    if k.endswith('.weight'):
                        after_max = p.abs().max()
                    elif k.endswith('.bias'):
                        after_max = p.max()
                    if after_max.data.item() != param[k].data.item():
                        # Print the weight and bias change when applying CLE
                        log.info(f'{k}: {param[k].data.item():.5f} -> {after_max.data.item():.5f}')

    return layer_groups, model


def cross_layer_equalize(
    model: nn.Module, dummy_input, device, threshold=0.5, work_dir="out", cle_iters=2, hba_flag=False
) -> nn.Module:
    """Higher-level API to perform Cross-Layer Equalization(CLE) and High Bias Abosrb (HBA) on the given model.

    Args:
        model: The bn of model should be fused into conv.
        dummy_input (torch.tensor): A viable input for the model.
        device (torch.device): Specifies the device of the model.
        threshold: Default to be 1000, used to prevent unquantifiable anomalies in the output of inter conv.
        work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be
                generated. Defaults to None, in which case "out" will be used.
        cle_iters: The iteration nums of cle.
        hba_flag: Whether to do HBA, default to be True.
    Returns:
        The model which has been done cle.
    """
    model = model_rewrite(model, dummy_input, work_dir=work_dir)
    model = model_fuse_bn(model, dummy_input)

    log.info("start to do Cross Layer Equalization. the range change of weight/bias after CLE:")
    layers_groups, model = equalize_model(model, dummy_input, threshold, iters=cle_iters)

    if hba_flag:
        log.info("start to do High Bias Absorbing. the range change of bias after HBA:")
        model = high_bias_absorb(model, device, layers_groups)
    clear_model_fused_bn(model)

    return model


def bias_absorb_helper_(layer1, layer2, model, origin_model):
    if not hasattr(layer1[1], 'bias') or not hasattr(layer2[1], 'bias'):
        return
    pre_layer = getattr(model, layer1[0])
    cur_layer = getattr(model, layer2[0])

    if isinstance(pre_layer, ConvBnTrain) and isinstance(cur_layer, ConvBnTrain):
        # when use bn_restore to do HBA after CLE
        pre_bn = pre_layer.bn
        cur_conv = cur_layer.conv
    elif isinstance(pre_layer, nn.Conv2d) and isinstance(pre_layer, nn.Conv2d):
        if hasattr(pre_layer, 'fused_bn_') and hasattr(cur_layer, 'fused_bn_'):
            pre_bn = pre_layer.fused_bn_
            cur_conv = cur_layer
        else:
            log.info("High Bias Absorbing is not supported for conv without BatchNorm.")
            return

    # AIMET use BN's weight and bias to get 3sigma.
    c = pre_bn.bias - 3 * torch.abs(pre_bn.weight)
    zero = torch.zeros_like(c)
    c = torch.where(c < 0, zero, c).to(torch.float)
    cur_weight = cur_conv.weight.data
    # sum along 3rd and 4rd aixs
    reduced_weight = cur_weight.sum(dim=[2, 3])
    if reduced_weight.shape[1] == 1:
        # for dw conv
        reduced_weight = reduced_weight.reshape(-1)
        bias_correct = reduced_weight * c
    else:
        bias_correct = torch.matmul(reduced_weight, c)
    cur_bias = cur_conv.bias + bias_correct

    origin_pre_conv = getattr(origin_model, layer1[0])
    origin_cur_conv = getattr(origin_model, layer2[0])
    max_before = origin_pre_conv.bias.data.max()
    origin_pre_conv.bias.data = origin_pre_conv.bias.data - c
    origin_cur_conv.bias.data = cur_bias
    if max_before != origin_pre_conv.bias.data.max():
        log.info(f'{layer1[0]} bias: {max_before} -> {origin_pre_conv.bias.data.max()}')


def bias_absorb_(model, layers_groups, origin_model):
    with torch.no_grad():
        for layer_group in layers_groups:
            if len(layer_group) == 2:
                bias_absorb_helper_(layer_group[0], layer_group[1], model, origin_model)
            else:
                log.warning('Unsupported layer group')


def high_bias_absorb_empirical(
    cle_model, device, layer_groups, cali_func=None, *cali_func_args, use_origin_bn=True, layers_fused_bn=None
):
    """Absorb bias value greater than 3 * sigma to next layer's bias, which use real data to get pre-bias
    distribution."""
    cle_model.to(device)
    origin_model = copy.deepcopy(cle_model)
    if use_origin_bn:
        bias_absorb_(cle_model, layer_groups, origin_model)
    else:
        if cali_func is None:
            log.warning(
                "High Bias Absorbing can not run, you can setting args as below:\n"
                "1. If your origin model has bn, please set `bn_fuse=True` at `cross_layer_equalize` \n"
                "2. if your origin model do not have bn(e.g. RepVGG_deploy), please set the right "
                "`cali_func`, `cali_func_arg` and `layers_fused_bn."
            )
        else:
            if layers_fused_bn is None:
                layers_fused_bn = [name for name, mod in cle_model.named_modules() if isinstance(mod, torch.nn.Conv2d)]
            cle_bn_model = model_restore_bn(
                cle_model, device, cali_func, *cali_func_args, layers_fused_bn=layers_fused_bn
            )
            bias_absorb_(cle_bn_model, layer_groups, origin_model)
    clear_model_fused_bn(origin_model)
    return origin_model


def high_bias_absorb(cle_model, device, layer_groups):
    """Absorb bias value greater than 3 * sigma to next layer's bias, which use origin BN to get pre-bias distribution.

    Args:
        cle_model: The model which has been done cle.
        device: The appropriate device, e.g. torch.device("cuda").
        layer_groups: The Layer groups which returned by CLE.
    Return:
        The model after HBA.
    """
    cle_model.to(device)
    origin_model = copy.deepcopy(cle_model)
    bias_absorb_(cle_model, layer_groups, origin_model)
    return origin_model


def model_fuse_bn(model: nn.Module, dummy_input):
    """Fuse bn to conv inplace, and attach the origin bn to fused conv with attr:`fused_bn_`"""
    with model_tracer():
        with torch.no_grad():
            model.eval()
            quantizer = PostQuantizer(
                model,
                dummy_input,
                work_dir='out',
                config={'rewrite_graph': False, 'force_overwrite': False, 'fuse_only': True},
            )
            graph = trace(quantizer.model, quantizer.dummy_input)
            graph.quantized = True
            for node in graph.forward_nodes:
                node.quantized = True
            custom_data = ([], set())
            processed_rules = load_processed_ptq_rules()
            processed_rules = {nn.BatchNorm2d: processed_rules[nn.BatchNorm2d]}
            is_fusable = functools.partial(quantizer.is_fusable, current_rules=processed_rules, graph=graph)
            graph.filter_forward_nodes(is_fusable, custom_data, reverse=True)
            quant_list = custom_data[0]
            for quant_nodes in quant_list:
                if isinstance(getattr(graph.module, quant_nodes[1]), nn.BatchNorm2d):
                    bn_cur = getattr(graph.module, quant_nodes[1])
                torch_q.fuse_modules(graph.module, quant_nodes, inplace=True)
                if hasattr(getattr(graph.module, quant_nodes[0]), 'fused_bn_'):
                    log.warning("conv have attr fused_bn_, HBA can not apply on this conv")
                else:
                    setattr(getattr(graph.module, quant_nodes[0]), 'fused_bn_', bn_cur)
            fused_model = graph.module
            return fused_model


def model_rewrite(model, dummy_input, work_dir='out') -> nn.Module:
    """rewrite model to non-block style"""
    with model_tracer():
        graph = trace(model, dummy_input)
        model_name = type(model).__name__
        model_rewrite = f'{model_name}_cle_Rewrite'
        model_name_rewrite_lower = model_rewrite.lower()
        model_ns = f'out.{model_name_rewrite_lower}'
        model_code_path = os.path.join(work_dir, f'{model_name_rewrite_lower}.py')
        model_weights_path = os.path.join(work_dir, f'{model_name_rewrite_lower}.pth')
        graph.eliminate_dead_graph_pass()
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)
        graph.generate_code(model_code_path, model_weights_path, model_rewrite)

        # Import the new model
        rewritten_model = import_from_path(model_ns, model_code_path, model_rewrite)()
        rewritten_model.load_state_dict(torch.load(model_weights_path))
        os.unlink(model_weights_path)
        return rewritten_model


def clear_model_fused_bn(model: nn.Module):
    """remove the attached bn from fused conv"""
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)) and hasattr(mod, 'fused_bn_'):
            delattr(mod, 'fused_bn_')
