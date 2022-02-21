import copy
import logging
import os
import queue
import typing

from distutils.version import LooseVersion, StrictVersion

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.qat as nniqat
import torch.nn.intrinsic.quantized as nniq
import torch.nn.qat as nnqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.quantization as torch_q

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from tinynn.graph.quantization.fake_quantize import FakeQuantizeBFloat16
from tinynn.graph.quantization.modules import QPReLU
from tinynn.graph.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from tinynn.graph.tracer import (ConstantNode, TraceFunction, TraceGraph,
                                 TraceNode, load_creation_funcs,
                                 module_constructor_lines,
                                 override_current_trace_graph, qualified_name,
                                 trace)
from tinynn.util.train_util import get_module_device
from tinynn.util.util import import_from_path

# Fusable OPs for Quantize Aware Training
FUSE_RULE_LIST = {
    (torch.nn.Conv1d, torch.nn.BatchNorm1d),
    (torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.ReLU),
    (torch.nn.Conv2d, torch.nn.BatchNorm2d),
    (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU),
    (torch.nn.Conv3d, torch.nn.BatchNorm3d),
    (torch.nn.Conv3d, torch.nn.BatchNorm3d, torch.nn.ReLU),
    (torch.nn.Conv1d, torch.nn.ReLU),
    (torch.nn.Conv2d, torch.nn.ReLU),
    (torch.nn.Conv3d, torch.nn.ReLU),
    (torch.nn.Linear, torch.nn.ReLU),
    (torch.nn.BatchNorm2d, torch.nn.ReLU),
    (torch.nn.BatchNorm3d, torch.nn.ReLU),
}

# Processed QAT fuse rules
processed_qat_rules = {}

# Constant func names
creation_func_names = []

log = logging.getLogger(__name__)


class QATQuantizer(object):
    rewrite_graph: bool
    force_overwrite: bool
    is_input_quantized: typing.Optional[typing.Tuple[bool]]
    backend: str
    remove_weights_after_load: bool
    asymmetric: bool
    per_tensor: bool

    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new QATQuantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be generated. \
                Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__()

        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            self.model = model.module
        else:
            self.model = model

        self.dummy_input = dummy_input
        self.work_dir = 'out' if work_dir is None else work_dir

        self.parse_config(config)

        if self.backend != 'qnnpack':
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

    def parse_config(self, config: typing.Optional[dict]):
        default_values = {'rewrite_graph': True, 'force_overwrite': True,
                          'is_input_quantized': None, 'backend': 'qnnpack',
                          'remove_weights_after_load': False,
                          'asymmetric': True, 'per_tensor': True}

        if config is None:
            config = dict()

        for k, v in default_values.items():
            actual_v = config.get(k, v)
            setattr(self, k, actual_v)

    def quantize(self) -> nn.Module:
        """ Performs QAT rewrite and preparation

        Returns:
            nn.Module: The QAT-ready model
        """

        # We need a model in training mode so that QAT could take place
        self.model.train()

        # After tracing the model, we will get a TraceGraph object
        graph = trace(self.model, self.dummy_input)

        if self.rewrite_graph:
            # Retrives the name of the model from type info
            model_name = type(self.model).__name__
            model_name_qat = f'{model_name}_qat'
            model_name_qat_lower = model_name_qat.lower()
            model_ns = f'tinynn_rewritten_models.{model_name_qat}'

            model_code_path = os.path.join(self.work_dir, f'{model_name_qat_lower}.py')
            model_weights_path = os.path.join(self.work_dir, f'{model_name_qat_lower}.pth')

            # Generate the code for the modified model
            # We will try to do some op fusion and rewriting in the `rewrite_quantize_graph` function.
            # By default, we will try to insert QuantStubs before every input and DeQuantStubs after every output in the generated graph.
            # If this doesn't suit your needs, e.g. you have intergal/quantized inputs or want to skip the quantization rewrite for some ops,
            # then you may modify the code generated freely and remember to skip this step so it won't be overwritten.
            if self.force_overwrite or not os.path.exists(model_code_path) or not os.path.exists(model_weights_path):
                self.rewrite_quantize_graph(graph)
                graph.generate_code(model_code_path, model_weights_path, model_name_qat)

            # Import the new model
            rewritten_model = import_from_path(model_ns, model_code_path, model_name_qat)()
            rewritten_model.load_state_dict(torch.load(model_weights_path))

            device = get_module_device(self.model)
            if device is not None:
                rewritten_model.to(device=device)

            # Remove the weights file to save space
            if self.remove_weights_after_load:
                os.unlink(model_weights_path)

            # Set the model to training mode before tracing again, since we need to perform QAT
            rewritten_model.train()
            rewritten_graph = trace(rewritten_model, self.dummy_input)
        else:
            rewritten_graph = graph

        # Fuse the modules (e.g conv, bn, relu) in the computation graph according to the fuse rules.
        # By default, we assume all input tensors are of floating type.
        # If you want to use quantized/integral inputs, then you may need to pass in `is_input_quantized`.
        qat_model = self.prepare_qat(rewritten_graph, self.is_input_quantized, self.backend)

        return qat_model

    def prepare_qat_prep(self, graph: TraceGraph, is_input_quantized: typing.Optional[typing.Tuple[bool]] = None, backend: str = 'qnnpack'):
        """ Some common logic before calling torch.quantization.prepare[_qat]

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.

        """

        qat_analysis_queue = queue.Queue()
        visited = set()

        def _qat_analysis(node: TraceNode, quantized: bool):
            # Find quantized subgraphs in the whole computation graph

            if node.unique_name in visited:
                return

            visited.add(node.unique_name)

            if node in graph.output_nodes:
                return
            if type(node.module) is torch_q.QuantStub:
                quantized = True
                node.quantized = quantized
            elif type(node.module) is torch_q.DeQuantStub:
                node.quantized = True
                quantized = False
            else:
                node.quantized = quantized
                log.debug(f"[QUANTIZED]{node.unique_name}:{quantized}")

            for n in node.next_nodes:
                qat_analysis_queue.put((n, quantized))

        if is_input_quantized is not None:
            assert len(is_input_quantized) == len(graph.input_nodes)

            for n, q in zip(graph.input_nodes, is_input_quantized):
                qat_analysis_queue.put((n, q))
        else:
            for n in graph.input_nodes:
                qat_analysis_queue.put((n, False))

        creation_func_names = load_creation_func_names()

        def _is_extra_constant_nodes(node, custom_data):
            return node.full_name() in creation_func_names

        extra_constant_nodes = graph.filter_forward_nodes(_is_extra_constant_nodes)
        for n in graph.constant_nodes + extra_constant_nodes:
            qat_analysis_queue.put((n, not n.next_tensors[0].dtype == torch.float32))

        while not qat_analysis_queue.empty():
            node, quantized = qat_analysis_queue.get()
            _qat_analysis(node, quantized)

        log.debug("qat analysis over")

        processed_qat_rules = load_processed_qat_rules()

        def _find_quantized_prelu_nodes(node: TraceNode, custom_node):
            # Find quantized PReLU nodes
            return node.type() == nn.PReLU and node.quantized

        # Replace PReLU nodes with our custom variants
        quantized_prelu_nodes = graph.filter_forward_nodes(_find_quantized_prelu_nodes)
        graph.update_submodule_in_nodes_from_predicate(quantized_prelu_nodes, QPReLU)

        def _is_fusable(node, custom_data):
            # Tell whether a TraceNode is fusable with some nearby nodes

            # Skip nodes that is not in a quantized computation graph
            if not node.quantized:
                return False

            cur_node = node
            names = []
            final_names = []
            current_rules = processed_qat_rules
            current_state = False
            while True:
                cur_module = cur_node.module
                cur_class = type(cur_module)
                prev_nodes = cur_node.prev_nodes
                log.debug('cur: ', cur_class)
                if cur_class in current_rules:
                    cur_name = graph.module_original_name_dict[id(cur_module)]
                    if cur_name in custom_data[1]:
                        log.debug('found existing nodes, skipping')
                        break
                    current_state, current_rules = current_rules[cur_class]
                    log.debug('dict: ', current_rules, current_state)
                    if len(prev_nodes) == 0:
                        break
                    if current_state and len(cur_node.prev_nodes) != 1:
                        current_state = False
                    names.append(cur_name)
                    cur_node = cur_node.prev_nodes[0]
                    if current_state is True:
                        log.debug('update best: ', names)
                        final_names.clear()
                        final_names.extend(names)
                else:
                    break

            if len(final_names) > 0:
                final_names.reverse()
                log.debug('final:', final_names)
                custom_data[0].append(final_names)
                for name in final_names:
                    custom_data[1].add(name)
                return True

            return False

        custom_data = ([], set())
        graph.filter_forward_nodes(_is_fusable, custom_data, reverse=True)
        quant_list = custom_data[0]
        log.info(f'found nodes to fuse: {quant_list}')

        for quant_nodes in quant_list:
            torch_q.fuse_modules(graph.module, quant_nodes, inplace=True)

        self.prepare_qconfig(graph, backend)

    def prepare_qconfig(self, graph: TraceGraph, backend: str):
        """ Prepare qconfig for various configurations.

        Args:
            graph (TraceGraph): The computation graph of the model
            backend (str, optional): The backend of quantization
        """

        log.info('setting qat backend and call prepare_qat')
        qconfig = torch_q.get_default_qat_qconfig(backend)
        qconfig_c = None
        if self.backend == 'qnnpack':
            if not self.asymmetric:
                sym_fq = qconfig.activation.with_args(observer=torch_q.MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                                      dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = qconfig.weight.with_args(observer=torch_q.MovingAveragePerChannelMinMaxObserver.with_args(quant_min=-127, quant_max=127),
                                                  quant_min=-127, quant_max=127, dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
                                                  reduce_range=False, ch_axis=0)
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        else:
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

        torch.backends.quantized.engine = backend
        graph.module.qconfig = qconfig
        if qconfig_c is not None:
            q = queue.Queue()
            q.put(graph.module)

            while not q.empty():
                m = q.get()
                if type(m).__name__ in ('Conv2d', 'ConvBnReLU2d', 'ConvBn2d', 'ConvReLU2d'):
                    m.qconfig = qconfig_c
                else:
                    for c in m.children():
                        q.put(c)

    def prepare_qat(self, graph: TraceGraph, is_input_quantized: typing.Optional[typing.Tuple[bool]] = None, backend: str = 'qnnpack') -> torch.nn.Module:
        """ Prepare model for QAT training

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.

        Returns:
            torch.nn.Module: The QAT-ready model
        """

        graph.module.train()

        self.prepare_qat_prep(graph, is_input_quantized, backend)

        # Unfornately, the suggested way below will try to fuse all the modules
        # even if some of the nodes are not in a quantized computation graph.
        # So we wrote some alternatives for the function.
        #   torch.quantization.prepare_qat(graph.module, inplace=True)

        if hasattr(torch_q, 'get_default_qat_module_mappings'):
            mapping = torch_q.get_default_qat_module_mappings()
        elif hasattr(torch_q, 'get_qat_module_mappings'):
            mapping = torch_q.get_qat_module_mappings()
        else:
            mapping = torch_q.DEFAULT_QAT_MODULE_MAPPING

        if LooseVersion(torch.__version__) < LooseVersion("1.7.0"):
            model = torch_q.prepare(graph.module, inplace=True)
            for n in graph.forward_nodes:
                if not n.quantized:
                    if hasattr(n.module, "_forward_hooks"):
                        if len(n.module._forward_hooks) > 0:
                            n.module._forward_hooks.popitem()
                    if hasattr(n.module, "qconfig"):
                        delattr(n.module, "qconfig")
                    if hasattr(n.module, "activation_post_process"):
                        delattr(n.module, "activation_post_process")
            torch_q.convert(model, mapping, inplace=True)
        else:
            torch_q.propagate_qconfig_(graph.module, qconfig_dict=None)
            for n in graph.forward_nodes:
                if not n.quantized:
                    if hasattr(n.module, "qconfig"):
                        delattr(n.module, "qconfig")
            model = torch_q.convert(graph.module, mapping=mapping, inplace=True, remove_qconfig=False)
            torch_q.prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
            for n in graph.forward_nodes:
                if not n.quantized:
                    if hasattr(n.module, "qconfig"):
                        delattr(n.module, "qconfig")
                    if hasattr(n.module, "_forward_hooks"):
                        if len(n.module._forward_hooks) > 0:
                            n.module._forward_hooks.popitem()
                    if hasattr(n.module, "activation_post_process"):
                        delattr(n.module, "activation_post_process")

        if not self.per_tensor:
            for n, m in graph.module.named_modules():
                if n.endswith('.weight_fake_quant'):
                    observer = getattr(m, 'activation_post_process', None)
                    if observer is not None:
                        m.quant_min = -127
                        m.quant_max = 127
                        observer.quant_min = -127
                        observer.quant_max = 127

            self.per_channel_qconfig_post_process(graph)

        return graph.module

    def per_channel_qconfig_post_process(self, graph):
        connected_types = ['cat', 'chunk', 'split']

        def _find_quantized_cat_nodes(node: TraceNode, custom_node):
            # Find quantized cat nodes
            return node.type() == 'cat' and node.quantized

        # For cat nodes, the `activation_post_process` around it needs to be unified
        quantized_cat_nodes = graph.filter_forward_nodes(_find_quantized_cat_nodes)

        q = queue.Queue()
        visited_center = set()
        for n in quantized_cat_nodes:
            q.put((n, 'both', 0))
            parents = []
            names = []
            props = []
            visited_other = set()
            while not q.empty():
                n, mode, fq_count = q.get()
                if n.kind() in ('shape', 'size') or n.unique_name in visited_center or n.unique_name in visited_other:
                    continue

                if n.type() in connected_types:
                    visited_center.add(n.unique_name)
                else:
                    visited_other.add(n.unique_name)

                if isinstance(n.module, nn.Module):
                    orig_name = graph.module_original_name_dict.get(id(n.module))
                    new_mod, parent = graph.get_submodule_with_parent_from_name(orig_name)
                    prop = orig_name.split('.')[-1]
                    if isinstance(new_mod, torch_q.FakeQuantize):
                        if fq_count == 0:
                            parents.append(parent)
                            names.append(orig_name)
                            props.append(prop)
                        fq_count += 1
                    elif hasattr(new_mod, 'activation_post_process'):
                        if fq_count == 0:
                            parents.append(new_mod)
                            names.append(f'{orig_name}.activation_post_process')
                            props.append('activation_post_process')
                        fq_count += 1
                    if isinstance(new_mod, (torch_q.DeQuantStub, torch_q.QuantStub)):
                        fq_count = 2
                elif n.type() in connected_types:
                    mode = 'both'
                    fq_count = 0

                if fq_count < 2:
                    if mode in ('both', 'up'):
                        for node in n.prev_nodes:
                            q.put((node, 'up', fq_count))
                    if mode in ('both', 'down'):
                        for node in n.next_nodes:
                            q.put((node, 'down', fq_count))

            if len(names) > 1:
                log.debug(f'Unifying the following nodes into one: {", ".join(names)}')
                unified = getattr(parents[0], props[0])
                for parent, prop in zip(parents[1:], props[1:]):
                    setattr(parent, prop, unified)

    def rewrite_quantize_graph(self, graph: TraceGraph) -> None:
        """ Rewrites the computation graph for quantization """
        if graph.quantized:
            return

        creation_func_names = load_creation_func_names()

        def _is_extra_constant_nodes(node, custom_data):
            return node.full_name() in creation_func_names

        extra_constant_nodes = graph.filter_forward_nodes(_is_extra_constant_nodes)

        # First, we insert the QuantStub nodes for every input/constant node
        for idx, node in reversed(list(enumerate(graph.input_nodes + graph.constant_nodes + extra_constant_nodes))):
            fake_quant = torch_q.QuantStub()

            graph.module_unique_name_dict[id(fake_quant)] = f'fake_quant_{idx}'
            graph.module_original_name_dict[id(fake_quant)] = f'fake_quant_{idx}'

            fake_quant_cls = type(fake_quant)
            module_constructor_lines[id(fake_quant)] = f'{qualified_name(fake_quant_cls)}()'

            graph.insert_after(node, fake_quant)

        # Second, we insert the DeQuantStub nodes for every output node
        for idx, node in enumerate(graph.output_nodes):
            fake_dequant_cls = torch_q.DeQuantStub
            if node.rev_index:
                modules = []
                for rev_idx in range(len(node.prev_nodes)):
                    fake_dequant = fake_dequant_cls()

                    graph.module_unique_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}_{rev_idx}'
                    graph.module_original_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}_{rev_idx}'

                    module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'
                    modules.append(fake_dequant)

                graph.insert_before(node, modules)
            else:
                fake_dequant = fake_dequant_cls()

                graph.module_unique_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}'
                graph.module_original_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}'

                module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'

                graph.insert_before(node, fake_dequant)

        # Third, we rewrite neg/sub/div using supported functions(e.g add, mul)
        def _is_neg_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'neg' and cur_module.prev_tensors[0].dtype == torch.float32

        neg_nodes = graph.filter_forward_nodes(_is_neg_node)
        log.info(f'rewriting neg for {[node.unique_name for node in neg_nodes]}')
        for idx, node in enumerate(neg_nodes):
            node.module.func_type = '__mul__'
            node.module.kind = 'mul'

            full_name_parts = node.module.full_name.split('.')
            full_name_parts[-1] = node.module.func_type

            node.module.full_name = '.'.join(full_name_parts)

            with override_current_trace_graph(graph):
                node.module.parse_args(node.module.prev_tensors[0], -1.0)

        def _is_div_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            # Current, only the following condition could be handled.
            #   a / constant => a * (1 / constant)
            if cur_class == TraceFunction:
                return cur_module.kind == 'truediv' and \
                    len(cur_module.prev_tensors) == 1 and \
                    cur_module.prev_tensors[0].dtype == torch.float32 and \
                    cur_module.func_type != '__rtruediv__' and \
                    node.next_tensors[0].dtype == torch.float32

        div_nodes = graph.filter_forward_nodes(_is_div_node)
        log.info(f'rewriting div for {[node.unique_name for node in div_nodes]}')
        for idx, node in enumerate(div_nodes):
            op_type = node.module.func_type
            node.module.func_type = '__mul__'
            node.module.kind = 'mul'

            full_name_parts = node.module.full_name.split('.')
            full_name_parts[-1] = node.module.func_type

            node.module.full_name = '.'.join(full_name_parts)

            # Here we make a simple guess, if dot is in the string, then it's a floating number.
            # Otherwise, it is an integral number.
            if '.' in node.module.args_string_no_self:
                other_arg = float(node.module.args_string_no_self)
            else:
                other_arg = int(node.module.args_string_no_self)

            with override_current_trace_graph(graph):
                node.module.parse_args(node.prev_tensors[0], 1.0 / other_arg)

        def _is_sub_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'sub' and cur_module.prev_tensors[0].dtype == torch.float32 and \
                    node.next_tensors[0].dtype == torch.float32

        sub_nodes = graph.filter_forward_nodes(_is_sub_node)
        log.info(f'rewriting sub for {[node.unique_name for node in sub_nodes]}')
        for idx, node in enumerate(sub_nodes):
            op_type = node.module.func_type

            full_name_parts = node.module.full_name.split('.')
            full_name_parts[-1] = node.module.func_type

            if len(node.module.prev_tensors) == 1 and node.module.func_type != '__rsub__':
                node.module.func_type = '__add__'
                node.module.kind = 'add'
                node.module.full_name = '.'.join(full_name_parts)

                # Here we make a simple guess, if dot is in the string, then it's a floating number.
                # Otherwise, it is an integral number.
                if '.' in node.module.args_string_no_self or 'e' in node.module.args_string_no_self:
                    other_arg = float(node.module.args_string_no_self)
                else:
                    other_arg = int(node.module.args_string_no_self)

                with override_current_trace_graph(graph):
                    node.module.parse_args(node.prev_tensors[0], -other_arg)
            elif len(node.module.prev_tensors) == 2 and len(node.prev_nodes) == 2:
                new_fullname_parts = copy.deepcopy(full_name_parts)
                new_fullname_parts[-1] = '__mul__'
                new_fullname = '.'.join(new_fullname_parts)
                current_tensor = node.prev_tensors[0]
                input_node = node.prev_nodes[1]
                input_tensor = node.prev_tensors[1]
                output_tensor = input_tensor * -1

                with override_current_trace_graph(graph):
                    trace_func = TraceFunction(new_fullname, True, prefix='fuse_').parse_args(input_tensor, -1)
                    graph.insert_between(input_node, node, trace_func, [output_tensor])

                    node.module.func_type = '__add__'
                    node.module.kind = 'add'

                    full_name_parts[-1] = node.module.func_type

                    node.module.full_name = '.'.join(full_name_parts)

                    node.module.parse_args(current_tensor, output_tensor)

            elif node.module.func_type == '__rsub__' and len(node.module.prev_tensors) == 1:
                new_fullname_parts = copy.deepcopy(full_name_parts)
                new_fullname_parts[-1] = '__mul__'
                new_fullname = '.'.join(new_fullname_parts)
                input_node = node.prev_nodes[0]
                input_tensor = node.prev_tensors[0]
                output_tensor = input_tensor * -1

                if '.' in node.module.args_string_no_self:
                    other_arg = float(node.module.args_string_no_self)
                else:
                    other_arg = int(node.module.args_string_no_self)

                with override_current_trace_graph(graph):
                    trace_func = TraceFunction(new_fullname, True, prefix='fuse_').parse_args(input_tensor, -1)
                    graph.insert_between(input_node, node, trace_func, [output_tensor])

                    node.module.func_type = '__radd__'
                    node.module.kind = 'add'

                    full_name_parts[-1] = node.module.func_type

                    node.module.full_name = '.'.join(full_name_parts)

        # Then, we write torch.stack nodes to torch.unsqueeze + torch.cat
        def _is_stack_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'stack' and node.next_tensors[0].dtype == torch.float32

        stack_nodes = graph.filter_forward_nodes(_is_stack_node)

        for idx, node in enumerate(stack_nodes):
            args = node.module.args_string_no_self
            if ',' in args:
                log.error('rewrite doesn\'t support multiple args for torch.stack')
                assert False
            if len(args) > 0:
                dim = int(args)
            else:
                dim = 0
            for n in node.prev_nodes:
                shared_tensors = list(set(node.prev_tensors).intersection(set(n.next_tensors)))
                if len(shared_tensors) == 0:
                    log.debug('tensor rewrite already done, skipping')
                    continue
                if len(shared_tensors) > 1:
                    log.error('rewrite supports torch.stack with nodes with exact one input')
                    assert False
                with override_current_trace_graph(graph):
                    trace_func = TraceFunction('torch.unsqueeze', prefix='fuse_').parse_args(shared_tensors[0], dim)
                next_tensors = [torch.unsqueeze(x, dim) for x in shared_tensors]
                graph.insert_between(n, node, trace_func, next_tensors)

            node.module.func_type = 'cat'
            node.module.kind = 'cat'

            full_name_parts = node.module.full_name.split('.')
            full_name_parts[-1] = node.module.func_type

            node.module.full_name = '.'.join(full_name_parts)

        # Next, we rewrite add/mul/cat with one float32 output using torch.nn.quantized.FloatFunctional
        def _is_convertible_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind in ('add', 'mul', 'cat') and node.next_tensors[0].dtype == torch.float32

        convertible_nodes = graph.filter_forward_nodes(_is_convertible_node)
        unfusable_add_nodes = []  # noqa: F841
        log.info(f'rewriting add/mul/cat for {[node.unique_name for node in convertible_nodes]}')
        for idx, node in enumerate(convertible_nodes):
            old_full_name = node.module.full_name  # noqa: F841
            old_is_class = node.module.is_class  # noqa: F841

            op_kind = node.module.kind
            float_functional = nnq.FloatFunctional()

            module_name = f'float_functional_simple_{idx}'

            graph.module_unique_name_dict[id(float_functional)] = module_name
            graph.module_original_name_dict[id(float_functional)] = module_name

            float_functional_cls = type(float_functional)
            module_constructor_lines[id(float_functional)] = f'{qualified_name(float_functional_cls, short=True)}()'

            new_node = TraceNode(float_functional, cur_graph=graph)
            graph.nodes_map[new_node.unique_name] = new_node
            graph.other_init_nodes.append(new_node)

            node.module.is_class = False
            prev_tensor_size = len(node.prev_tensors)
            if op_kind in ('add', 'mul'):
                if prev_tensor_size == 2:
                    op_type = op_kind
                elif prev_tensor_size == 1:
                    op_type = f'{op_kind}_scalar'
                else:
                    log.error(f'Unknown add/mul type for {node.unique_name}, prev tensor size: {prev_tensor_size}')
                    assert False
            else:
                # Don't check anything for other OPs.
                # It is simply too complex for us.
                op_type = op_kind
            node.module.full_name = f'self.{module_name}.{op_type}'
            # We need to convert radd to normal add here
            if node.module.func_type in ['__radd__', '__rmul__']:
                if '=' in node.module.args_string_no_self or ', ' in node.module.args_string_no_self:
                    log.error(f'Don\'t know how to translate {node.module.args_string_no_self} for __radd__/__rmul__')
                    assert False
                if prev_tensor_size == 1:
                    # Here we make a simple guess, if dot is in the string, then it's a floating number.
                    # Otherwise, it is an integral number.
                    if '.' in node.module.args_string_no_self or 'e' in node.module.args_string_no_self:
                        other_arg = float(node.module.args_string_no_self)
                    else:
                        other_arg = int(node.module.args_string_no_self)

                    with override_current_trace_graph(graph):
                        node.module.parse_args(node.prev_tensors[0], other_arg)
                else:
                    with override_current_trace_graph(graph):
                        # It is even simple here. We only need to swap the order of the tensors.
                        node.module.parse_args(node.prev_tensors[1], node.prev_tensors[0])

        # Rewrite other fusable functions
        # e.g. add_relu(x, y) =>
        #            r = torch.add(x, y)
        #            r = torch.nn.functional.relu(r)
        def _is_add_relu_fusable_node(node: TraceNode, custom_data) -> bool:
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                # The intermediate result cannot be used if fused.
                # So we don't fuse the nodes under such circumstances.
                if cur_module.kind != 'add' or len(node.next_nodes) != 1:
                    return False
                # The input for the add operations should be two tensors.
                if len(node.prev_tensors) != 2:
                    return False
                # We accept inplace operations for both add and relu.
                # The inplace property could be elinimated because we track the tensors
                # instead of their names.
                next_node = node.next_nodes[0]
                next_module = next_node.module
                next_class = type(next_module)
                if next_class == TraceFunction:
                    return next_module.kind == 'relu'
                else:
                    return next_class.__name__ == 'ReLU'

        add_relu_fusable_nodes = graph.filter_forward_nodes(_is_add_relu_fusable_node)
        for node in add_relu_fusable_nodes:
            full_name = node.module.full_name.replace('add', 'add_relu')
            next_node = node.next_nodes[0]
            kind = 'add_relu'
            func_type = kind
            is_class = False
            graph.fuse_nodes_to_func([node, node.next_nodes[0]], full_name, kind, func_type, is_class)

        # Rewrite relu, relu6 as nn.ReLU() and nn.ReLU6() for Module fusable rules
        def _is_relu_functional_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind in ('relu', 'relu6')

        relu_nodes_to_rewrite = graph.filter_forward_nodes(_is_relu_functional_node)
        log.info(f'rewriting relu for {[node.unique_name for node in relu_nodes_to_rewrite]}')
        for idx, node in enumerate(relu_nodes_to_rewrite):
            kind = node.module.kind
            inplace = node.module.func_type == f'{kind}_' or 'True' in node.module.args_string
            if node.module.kind == 'relu':
                new_relu = nn.ReLU(inplace=inplace)
            elif node.module.kind == 'relu6':
                new_relu = nn.ReLU6(inplace=inplace)

            graph.module_unique_name_dict[id(new_relu)] = f'rewritten_{kind}_{idx}'
            graph.module_original_name_dict[id(new_relu)] = f'rewritten_{kind}_{idx}'

            relu_cls = type(new_relu)
            if inplace:
                arg_str = 'inplace=True'
            else:
                arg_str = ''
            module_constructor_lines[id(new_relu)] = f'{qualified_name(relu_cls)}({arg_str})'
            graph.replace_node_module(node, new_relu)

        # Rewrite dropout as nn.Dropout() for models in training mode
        def _is_dropout_functional_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'dropout'

        def _dropout_args(p=0.5, training=True, inplace=False):
            return p, training, inplace

        dropout_nodes_to_rewrite = graph.filter_forward_nodes(_is_dropout_functional_node)
        log.info(f'rewriting dropout for {[node.unique_name for node in dropout_nodes_to_rewrite]}')
        for idx, node in enumerate(dropout_nodes_to_rewrite):
            p, _, inplace = eval(f'_dropout_args({node.module.args_string_no_self})')
            kind = node.module.kind
            inplace = node.module.func_type == f'{kind}_' or inplace
            dropout_cls = nn.Dropout
            new_dropout = dropout_cls(p, inplace=inplace)

            graph.module_unique_name_dict[id(new_dropout)] = f'rewritten_{kind}_{idx}'
            graph.module_original_name_dict[id(new_dropout)] = f'rewritten_{kind}_{idx}'

            if inplace:
                arg_str = f'{p}, inplace={inplace}'
            else:
                arg_str = f'{p}'

            module_constructor_lines[id(new_dropout)] = f'{qualified_name(dropout_cls)}({arg_str})'
            graph.replace_node_module(node, new_dropout)

        # Add contiguous nodes for partially-supported OPs
        # Some of the operations support quantization, but they only accept contiguous input tensors.
        def _is_partially_quantizable(node, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == ConstantNode:
                return False
            elif cur_class == TraceFunction:
                return cur_module.kind in ('pad', )
            else:
                return cur_class in (nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d, nn.ZeroPad2d)

        partially_supported_nodes = graph.filter_forward_nodes(_is_partially_quantizable)
        for idx, node in enumerate(partially_supported_nodes):
            assert len(node.prev_nodes) == 1
            for n in node.prev_nodes:
                shared_tensors = list(set(node.prev_tensors).intersection(set(n.next_tensors)))
                if len(shared_tensors) > 1:
                    log.error('rewrite for partially-supported ops supports with nodes with exact one input')
                    assert False
                with override_current_trace_graph(graph):
                    trace_func = TraceFunction('torch.Tensor.contiguous', True,
                                               prefix='fuse_').parse_args(shared_tensors[0])
                next_tensors = [x.contiguous() for x in shared_tensors]
                graph.insert_between(n, node, trace_func, next_tensors)

        # Remove non-leaf `.data` nodes
        def _is_non_leaf_data_nodes(node, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'data' and cur_module.is_property and len(node.next_nodes) > 0
            return False

        non_leaf_data_nodes = graph.filter_forward_nodes(_is_non_leaf_data_nodes)
        for idx, node in enumerate(non_leaf_data_nodes):
            graph.remove_node(node)

        # Add quant/dequant nodes for non-quantizable OPs
        def _is_not_quantizable(node, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == ConstantNode:
                return False
            elif cur_class == TraceFunction:
                return cur_module.kind in ('pow',
                                           'truediv',
                                           'sqrt',
                                           'atan2',
                                           'atan',
                                           'sin',
                                           'cos',
                                           'hardsigmoid',
                                           'silu',
                                           'reciprocal',
                                           'exp',
                                           'layer_norm',
                                           'instance_norm')
            else:
                if LooseVersion(torch.__version__) < LooseVersion('1.7.0'):
                    if cur_class == nn.ConvTranspose2d:
                        return True
                else:
                    if cur_class == nn.SiLU:
                        return True
                return cur_class in (nn.LSTM,
                                     nn.RNN,
                                     nn.GRU,
                                     nn.LayerNorm,
                                     nn.InstanceNorm1d,
                                     nn.InstanceNorm2d,
                                     nn.Hardsigmoid)

        unsupported_nodes = graph.filter_forward_nodes(_is_not_quantizable)
        for idx, node in enumerate(reversed(unsupported_nodes)):
            node_map = dict()
            for inner_idx, next_node in enumerate(node.next_nodes):
                prev_indices = []
                for pt in next_node.prev_tensors:
                    for j, nt in enumerate(node.next_tensors):
                        if id(pt) == id(nt):
                            prev_indices.append(str(j))

                prev_idx = '_'.join(prev_indices)

                if prev_idx in node_map:
                    fake_quant = node_map[prev_idx]

                    graph.insert_between(node, next_node, fake_quant, move_idx=True)
                else:
                    fake_quant = torch_q.QuantStub()

                    fake_quant_name = f'fake_quant_inner_{idx}_{inner_idx}'

                    graph.module_unique_name_dict[id(fake_quant)] = fake_quant_name
                    graph.module_original_name_dict[id(fake_quant)] = fake_quant_name

                    fake_quant_cls = type(fake_quant)
                    module_constructor_lines[id(fake_quant)] = f'{qualified_name(fake_quant_cls)}()'

                    graph.insert_between(node, next_node, fake_quant, move_idx=True)
                    node_map[prev_idx] = graph.nodes_map[fake_quant_name]

        # Finally, we insert the DeQuantStub nodes before every input node of the unsupported ops
        for idx, node in enumerate(unsupported_nodes):
            fake_dequant_cls = torch_q.DeQuantStub
            assert node.rev_index is False
            for inner_idx, prev_node in enumerate(node.prev_nodes):
                fake_dequant = fake_dequant_cls()

                graph.module_unique_name_dict[id(fake_dequant)] = f'fake_dequant_inner_{idx}_{inner_idx}'
                graph.module_original_name_dict[id(fake_dequant)] = f'fake_dequant_inner_{idx}_{inner_idx}'

                module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'

                graph.insert_between(prev_node, node, fake_dequant, move_idx=True)

        graph.quantized = True


class BF16Quantizer(QATQuantizer):
    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new BF16Quantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be generated. \
                Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__(model, dummy_input, work_dir, config)

    def parse_config(self, config: dict):
        super().parse_config(config)

        self.rewrite_graph = False

    def quantize(self) -> nn.Module:
        """ Prepare model for BFloat16 training """

        self.model.train()

        qconfig = torch_q.QConfig(activation=FakeQuantizeBFloat16.with_args(), weight=FakeQuantizeBFloat16.with_args())
        self.model.qconfig = qconfig

        torch_q.prepare_qat(self.model, inplace=True)

        return self.model


class PostQuantizer(QATQuantizer):
    rewrite_graph: bool
    force_overwrite: bool
    is_input_quantized: typing.Optional[typing.Tuple[bool]]
    backend: str
    remove_weights_after_load: bool

    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new PostQuantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be generated. \
                Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__(model, dummy_input, work_dir, config)

    def prepare_qconfig(self, graph: TraceGraph, backend: str):
        """ Prepare qconfig for various configurations.

        Args:
            graph (TraceGraph): The computation graph of the model
            backend (str, optional): The backend of quantization
        """

        log.info('setting qat backend and call prepare_qat')
        qconfig = torch_q.get_default_qconfig(backend)
        qconfig_c = None
        if self.backend == 'qnnpack':
            if not self.asymmetric:
                sym_fq = torch_q.HistogramObserver.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)
                qconfig = torch_q.QConfig(qconfig.activation, sym_fq)
                sym_fq = PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0)
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        else:
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

        torch.backends.quantized.engine = backend
        graph.module.qconfig = qconfig
        if qconfig_c is not None:
            q = queue.Queue()
            q.put(graph.module)

            while not q.empty():
                m = q.get()
                if type(m).__name__ in ('Conv2d', 'ConvBnReLU2d', 'ConvBn2d', 'ConvReLU2d'):
                    m.qconfig = qconfig_c
                else:
                    for c in m.children():
                        q.put(c)

    def prepare_qat(self, graph: TraceGraph, is_input_quantized: typing.Optional[typing.Tuple[bool]] = None, backend: str = 'qnnpack') -> torch.nn.Module:
        """ Prepare model for QAT training

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.

        Returns:
            torch.nn.Module: The QAT-ready model
        """

        graph.module.eval()

        self.prepare_qat_prep(graph, is_input_quantized, backend)

        # Unfornately, the suggested way below will try to fuse all the modules
        # even if some of the nodes are not in a quantized computation graph.
        # So we wrote some alternatives for the function.
        #   torch.quantization.prepare(graph.module, inplace=True)

        if hasattr(torch_q, 'get_default_qconfig_propagation_list'):
            whitelist = torch_q.get_default_qconfig_propagation_list()
        elif hasattr(torch_q, 'get_qconfig_propagation_list'):
            whitelist = torch_q.get_qconfig_propagation_list()
        else:
            whitelist = torch_q.DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST

        if LooseVersion(torch.__version__) < LooseVersion("1.7.0"):
            torch_q.prepare(graph.module, inplace=True)
        else:
            torch_q.propagate_qconfig_(graph.module, qconfig_dict=None)
            for n in graph.forward_nodes:
                if not n.quantized:
                    if hasattr(n.module, "qconfig"):
                        delattr(n.module, "qconfig")
            torch_q.add_observer_(graph.module, qconfig_propagation_list=whitelist)
        for n in graph.forward_nodes:
            if not n.quantized:
                if hasattr(n.module, "_forward_hooks"):
                    if len(n.module._forward_hooks) > 0:
                        n.module._forward_hooks.popitem()
                if hasattr(n.module, "qconfig"):
                    delattr(n.module, "qconfig")
                if hasattr(n.module, "activation_post_process"):
                    delattr(n.module, "activation_post_process")

        if not self.per_tensor:
            self.per_channel_qconfig_post_process(graph)

        return graph.module


class DynamicQuantizer(QATQuantizer):
    rewrite_graph: bool
    force_overwrite: bool
    is_input_quantized: typing.Optional[typing.Tuple[bool]]
    backend: str
    remove_weights_after_load: bool

    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new DynamicQuantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be generated. \
                Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__(model, dummy_input, work_dir, config)

    def parse_config(self, config: dict):
        super().parse_config(config)

        self.rewrite_graph = False

        assert not self.asymmetric, "Asymmetric quantization is not supported for DynamicQuantizer"
        assert self.per_tensor, "Per-channel quantization is not supported for DynamicQuantizer"

    def quantize(self) -> nn.Module:
        """ Prepare model for dynamic quantization """

        self.model.eval()

        torch_q.quantize_dynamic(self.model, inplace=True)

        return self.model


def load_creation_func_names():
    if len(creation_func_names) == 0:
        funcs_d = load_creation_funcs()
        for ns, funcs_v in funcs_d.items():
            ns_str = qualified_name(ns)
            creation_func_names.extend([f'{ns_str}.{x}' for x in funcs_v])
    return creation_func_names


def load_processed_qat_rules():
    if len(processed_qat_rules) == 0:
        # Constructor a prefix tree for the QAT rules
        fuse_rules = sorted(FUSE_RULE_LIST, key=lambda x: len(x), reverse=True)
        rule_dict = {}
        for fuse_rule in fuse_rules:
            base_rule_dict = rule_dict
            for module_cls in reversed(fuse_rule):
                # Node properties (has_key, child_nodes)
                base_rule_dict.setdefault(module_cls, [False, {}])
                base_rule_pair = base_rule_dict[module_cls]
                base_rule_dict = base_rule_pair[1]
            base_rule_pair[0] = True
        processed_qat_rules.update(rule_dict)
        return processed_qat_rules
