"""Cross Layer Equalization
Cross-Layer-Equalization can scale weights equivalently to reduce weight range in per_tensor mode.
This tool is useful in two cases：
1. When you use PTQ to quantize your model, you should fuse bn to the conv, then use CLE to improve acc .
2. When you try to quantify Re-parameterized model, e.g.Repvgg, MobileOne, the inference model often already fused BN,
you can use CLE to adjust the model weight, then do next ptq/qat(rep-model qat need to rebuild BN).
"""
import torch
import torch.nn as nn
from tinynn.graph.tracer import model_tracer, trace, TraceGraph, TraceNode
from tinynn.util.util import get_logger

log = get_logger(__name__)

cls_support_type = (torch.nn.Conv2d, torch.nn.Conv1d)
cls_scalable_type = (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.PReLU, torch.nn.Identity)


def is_dw_conv(layer: nn.Module) -> bool:
    if isinstance(layer, cls_support_type) and layer.groups == layer.in_channels and layer.groups == layer.out_channels:
        return True
    else:
        return False


def is_normal_conv(layer: nn.Module) -> bool:
    return isinstance(layer, cls_support_type) and layer.groups == 1


def is_group_supported(current_group):
    """Currently Supported layer combinations for CLS are:
    1. [conv-conv]
    2. [dw-conv]
    3. [conv-dw-conv]
    """
    current_group_ = [mod for n, mod in current_group]
    if (
        len(current_group_) == 2
        and isinstance(current_group_[0], cls_support_type)
        and is_normal_conv(current_group_[1])
    ):
        return True
    elif (
        len(current_group_) == 3
        and is_normal_conv(current_group_[0])
        and is_dw_conv(current_group_[1])
        and is_normal_conv(current_group_[2])
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


def equalize(weight1, bias1, weight2):
    """Use the CLE algorithm mentioned in https://arxiv.org/abs/1906.04721"""
    # Rearrange the second conv weight.
    weight2 = weight2.permute(1, 0, 2, 3)
    out_channel = weight1.shape[0]
    for i in range(out_channel):
        r1 = weight1[i].abs().max()
        r2 = weight2[i].abs().max()
        s = r1 / torch.sqrt(r1 * r2)
        weight1[i] = weight1[i] * (1.0 / s)
        weight2[i] = weight2[i] * s
        bias1[i] = bias1[i] * (1.0 / s)
    # Rearrange to origin weight shape.
    weight2 = weight2.permute(1, 0, 2, 3)
    return weight1, bias1, weight2


def equalize_cdc(weight1, bias1, weight2, bias2, weight3, threshold):
    """Use the CLE algorithm mentioned in https://arxiv.org/abs/1906.04721"""
    weight3 = weight3.permute(1, 0, 2, 3)
    out_channel = weight1.shape[0]
    S1, S2 = [], []
    # compute scale for each weight kernel.
    for i in range(out_channel):
        r1 = weight1[i].abs().max()
        r2 = weight2[i].abs().max()
        r3 = weight3[i].abs().max()
        s = r1 / pow(r1.double() * r3.double() * r2.double(), 1.0 / 3)
        S1.append(s)
        s = pow(r1.double() * r3.double() * r2.double(), 1.0 / 3) / r3
        S2.append(s)
        # In order to avoid the activation value is too large,
        # we made some restrictions on the weight amplification of the DW convolution channel.
        # In the future, more refined constraints will be designed.
        if S1[-1] / S2[-1] > threshold:
            S1[-1] = torch.tensor(1)
            S2[-1] = torch.tensor(1)

    # Do CLE for the first and second conv weight.
    for i in range(out_channel):
        weight1[i] = weight1[i] * (1.0 / S1[i])
        bias1[i] = bias1[i] * (1.0 / S1[i])
        weight2[i] = weight2[i] * S1[i]

    # Do CLE for the second and third conv weight.
    for i in range(out_channel):
        weight2[i] = weight2[i] * (1.0 / S2[i])
        bias2[i] = bias2[i] * (1.0 / S2[i])
        weight3[i] = weight3[i] * S2[i]

    weight3 = weight3.permute(1, 0, 2, 3)
    return weight1, bias1, weight2, bias2, weight3


def _weight_equal_helper(cls, threshold):
    layer_pair = [m for n, m in cls]
    if len(layer_pair) == 3:
        conv_0, conv_1, conv_2 = layer_pair
        assert is_normal_conv(conv_0) and is_dw_conv(conv_1) and is_normal_conv(conv_2), 'not conv-dw-conv'
        weight1, bias1, weight2, bias2, weight3 = conv_0.weight, conv_0.bias, conv_1.weight, conv_1.bias, conv_2.weight
        e_weight1, e_bias1, e_weight2, e_bias2, e_weight3 = equalize_cdc(
            weight1, bias1, weight2, bias2, weight3, threshold
        )
        conv_0.weight.data.copy_(e_weight1)
        conv_0.bias.data.copy_(e_bias1)
        conv_1.weight.data.copy_(e_weight2)
        conv_1.bias.data.copy_(e_bias2)
        conv_2.weight.data.copy_(e_weight3)
    elif len(layer_pair) == 2:
        conv_0, conv_1 = layer_pair
        weight1, bias1, weight2 = conv_0.weight, conv_0.bias, conv_1.weight
        e_weight1, e_bias1, e_weight2 = equalize(weight1, bias1, weight2)
        conv_0.weight.data.copy_(e_weight1)
        conv_0.bias.data.copy_(e_bias1)
        conv_1.weight.data.copy_(e_weight2)
    else:
        log.warning(f'layer_pair nums != 2,3, do not support, current layer:{cls}.')


def cross_layer_equalize(model: nn.Module, dummy_input, threshold=1000):
    """Higher-level API to perform Cross-Layer Equalization(CLE) on the given model in place.
    This function has two usage case:
    1. Per_tensor PTQ after bn_fused.
    2. Rep-model(e.g. MobileOne) which be reparameterized.

    Args:
        model: The bn of model should be fused into conv.
        dummy_input (torch.tensor): A viable input for the model.
        threshold: (Optional) Default to be 1000, used to prevent unquantifiable anomalies in the output of inter conv.
    """
    with torch.no_grad():
        with model_tracer():
            cur_graph = trace(model, dummy_input)
            param = {}
            for k, v in model.state_dict().items():
                weight, _ = cur_graph.get_submodule_with_parent_from_name(k)
                param[k] = weight.abs().max()
            layer_groups = get_cls_set(cur_graph)
            for cls in layer_groups:
                _weight_equal_helper(cls, threshold)
            stat_we = model.state_dict()
            for k, v in stat_we.items():
                weight, mod = cur_graph.get_submodule_with_parent_from_name(k)
                if isinstance(mod, torch.nn.Conv2d):
                    after_max = weight.abs().max()
                    log.info(f'{k}: {param[k].data.item():.5f} -> {after_max.data.item():.5f}')
