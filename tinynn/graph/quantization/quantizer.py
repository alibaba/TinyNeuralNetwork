import copy
import functools
import inspect
import os
import queue
import re
import sys
import typing

from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.quantization as torch_q

from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from tinynn.graph.quantization.fake_quantize import (
    FakeQuantizeBFloat16,
    FakeQuantizeTFLite,
)
from tinynn.graph.quantization.modules import QGLU, QPReLU, QSiLU
from tinynn.graph.quantization.observer import (
    HistogramObserverKL,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from tinynn.graph.quantization.qat_modules import (
    Conv1d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTransposeBn2d,
)
if LooseVersion(torch.__version__) >= LooseVersion("1.13.0"):
    from tinynn.graph.quantization.quantizable.gru import GRU as QuantizableGRU
    from torch.ao.nn.quantizable.modules.rnn import LSTM as QuantizableLSTM
else:
    QuantizableGRU = None
    QuantizableLSTM = None

from tinynn.graph.tracer import (
    ConstantNode,
    TraceFunction,
    TraceGraph,
    TraceNode,
    load_creation_funcs,
    module_constructor_lines,
    override_current_trace_graph,
    qualified_name,
    trace,
)
from tinynn.util.train_util import get_logger, get_module_device
from tinynn.util.util import import_from_path

from . import fused_modules as fm

try:
    import ruamel_yaml as yaml
except ModuleNotFoundError:
    import ruamel.yaml as yaml


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
    (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d),
}

FUSE_RULE_LIST_PTQ_ONLY = {
    (nn.Linear, nn.BatchNorm1d): '1.8.0',
    (nn.ConvTranspose1d, nn.BatchNorm1d): '1.11.0',
    (nn.ConvTranspose3d, nn.BatchNorm3d): '1.11.0',
}

FUSE_RULE_LIST_EXTRA = {
    (torch.nn.Conv1d, torch.nn.BatchNorm1d, torch.nn.ReLU6),
    (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.ReLU6),
    (torch.nn.Conv2d, torch.nn.BatchNorm2d, 'clamp_with_fusion'),
    (torch.nn.Conv3d, torch.nn.BatchNorm3d, torch.nn.ReLU6),
    (torch.nn.Conv1d, torch.nn.ReLU6),
    (torch.nn.Conv2d, torch.nn.ReLU6),
    (torch.nn.Conv2d, 'clamp_with_fusion'),
    (torch.nn.Conv3d, torch.nn.ReLU6),
    (torch.nn.Linear, torch.nn.ReLU6),
    (torch.nn.Linear, 'clamp_with_fusion'),
    (torch.nn.Linear, torch.nn.BatchNorm1d, torch.nn.ReLU6),
    (torch.nn.Linear, torch.nn.BatchNorm1d, 'clamp_with_fusion'),
    (torch.nn.BatchNorm2d, torch.nn.ReLU6),
    (torch.nn.BatchNorm2d, 'clamp_with_fusion'),
    (torch.nn.BatchNorm3d, torch.nn.ReLU6),
    ('add', torch.nn.ReLU6),
    ('add', 'relu6'),
    ('add', 'clamp_with_fusion'),
    (torch.nn.ConvTranspose2d, torch.nn.ReLU),
    (torch.nn.ConvTranspose2d, torch.nn.ReLU6),
    (torch.nn.ConvTranspose2d, 'clamp_with_fusion'),
    (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU),
    (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, torch.nn.ReLU6),
    (torch.nn.ConvTranspose2d, torch.nn.BatchNorm2d, 'clamp_with_fusion'),
}

FUSE_FALLBACK_DICT = {'clamp_with_fusion': 'clamp'}

FUSE_QAT_MODULES = {
    nn.Conv1d: Conv1d,
    nn.ConvTranspose1d: ConvTranspose1d,
    nn.ConvTranspose2d: ConvTranspose2d,
    fm.ConvTransposeBn2d: ConvTransposeBn2d,
}

FUSE_QAT_MODULES_CUSTOM = {}

if LooseVersion(torch.__version__) >= '1.13.0':
    from .quantizable.gru import GRU

    FUSE_QAT_MODULES_CUSTOM.update({nn.GRU: GRU})

FUSE_QAT_MODULES_CVT = {Conv1d: nnq.Conv1d}
if hasattr(nnq, 'ConvTranspose1d'):
    FUSE_QAT_MODULES_CVT.update(
        {
            ConvTranspose1d: nnq.ConvTranspose1d,
            ConvTranspose2d: nnq.ConvTranspose2d,
        }
    )

REWRITE_TO_FUSE_RULE_LIST = {
    (torch.nn.Linear, torch.nn.BatchNorm1d),
}

KNOWN_QSTATS = {
    nn.Softmax: (0, 256.0),
    'softmax': (0, 256.0),
    nn.LogSoftmax: (255, 16.0),
    'log_softmax': (255, 16.0),
}

REWRITE_QUANTIZABLE_RULE_LIST = {
    ('abs',),
    ('sum',),
    ('bmm',),
    ('matmul',),
    ('truediv',),
    ('clamp_min',),
    ('clamp_max',),
    ('sqrt', 'reciprocal'),
}

UNSUPPORTED_PYTORCH_QUANTIZATION_OP_LIST = {
    'pad': '1.7.0',
    'pow': None,
    'truediv': None,
    'sqrt': None,
    'atan2': None,
    'atan': None,
    'sin': None,
    'cos': None,
    'hardsigmoid': None,
    'silu': None,
    'reciprocal': None,
    'exp': None,
    'layer_norm': None,
    'instance_norm': None,
    'softmax': None,
    'log_softmax': None,
    'mm': None,
    'matmul': None,
    'bmm': None,
    'abs': None,
    'sum': None,
    'clamp_min': None,
    'clamp_max': None,
    'prelu': None,
    'elu': None,
    'glu': None,
    'group_norm': None,
    'log': None,
    'std': None,
    'var': None,
    nn.LSTM: '1.13.0',
    nn.ConvTranspose2d: '1.7.0',
    nn.ConstantPad1d: '1.7.0',
    nn.ConstantPad2d: '1.7.0',
    nn.ConstantPad3d: '1.7.0',
    nn.ZeroPad2d: '1.7.0',
    nn.RNN: None,
    nn.GRU: '1.13.0',
    nn.LayerNorm: None,
    nn.InstanceNorm1d: None,
    nn.InstanceNorm2d: None,
    nn.Hardsigmoid: None,
    nn.Softmax: None,
    nn.LogSoftmax: None,
    nn.GLU: None,
    nn.PReLU: None,
    nn.GroupNorm: None,
}

Q_MODULES_MAPPING = {
    nn.PReLU: QPReLU,
    nn.GLU: QGLU,
}

FUNCTIONAL_MODULE_MAPPING = {
    'relu': nn.ReLU,
    'relu6': nn.ReLU6,
    'leaky_relu': nn.LeakyReLU,
    'elu': nn.ELU,
    'prelu': nn.PReLU,
    'glu': nn.GLU,
    'sigmoid': nn.Sigmoid,
    'tanh': nn.Tanh,
    'hardswish': nn.Hardswish,
}

TENSORRT_OBSERVED_NODES = [
    (torch.nn, "Conv1d"),
    (torch.nn, "Conv2d"),
    (torch.nn, "Conv3d"),
    (torch.nn, "ConvTranspose1d"),
    (torch.nn, "ConvTranspose2d"),
    (torch.nn, "ConvTranspose3d"),
    (torch.nn, "Linear"),
    (torch.nn, "LSTM"),
    (torch.nn, "LSTMCell"),
    (torch.nn, "AvgPool1d"),
    (torch.nn, "AvgPool2d"),
    (torch.nn, "AvgPool3d"),
    (torch.nn, "AdaptiveAvgPool1d"),
    (torch.nn, "AdaptiveAvgPool2d"),
    (torch.nn, "AdaptiveAvgPool3d"),
    'avg_pool2d',
    'adaptive_avg_pool2d',
]

if hasattr(nn, 'SiLU'):
    UNSUPPORTED_PYTORCH_QUANTIZATION_OP_LIST.update({nn.SiLU: None})
    Q_MODULES_MAPPING.update({nn.SiLU: QSiLU})
    FUNCTIONAL_MODULE_MAPPING.update({'silu': nn.SiLU})

# Processed QAT fuse rules
processed_qat_rules = {}
processed_ptq_rules = {}
processed_extra_q_rules = {}
processed_all_qat_rules = {}
processed_all_ptq_rules = {}

# Constant func names
creation_func_names = []

# Processed rewrite rules for fusing
processed_rewrite_to_fuse_rules = {}

log = get_logger(__name__, 'WARNING')


class QATQuantizer(object):
    rewrite_graph: bool
    force_overwrite: bool
    is_input_quantized: typing.Optional[typing.Tuple[bool]]
    quantized_input_stats: typing.Optional[typing.List[typing.Optional[typing.Tuple[float, float]]]]
    quantized_op_stats: typing.Optional[typing.List[typing.Optional[typing.Tuple[float, float]]]]
    set_quantizable_op_stats: bool
    backend: str
    remove_weights_after_load: bool
    asymmetric: bool
    per_tensor: bool
    disable_requantization_for_cat: bool
    dynamic_lstm_quant: bool
    rounding_mode: str
    leaf_nodes: typing.Optional[typing.List[nn.Module]]
    swap_nodes: typing.Optional[typing.List[typing.Tuple[nn.Module, nn.Module]]]
    legacy_fq: bool
    extra_tracer_opts: typing.Optional[typing.Dict]
    layerwise_config: typing.Dict[str, bool]
    layerwise_default: bool
    override_qconfig_func: typing.Optional[typing.Callable[[str, nn.Module], typing.Optional[torch_q.QConfig]]]
    effective_layers: typing.List[str]
    ignore_layerwise_config: bool
    fused_layerwise_config: bool
    inplace: bool
    train_mode_dict: typing.Dict[nn.Module, bool]

    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new QATQuantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be \
                generated. Defaults to None, in which case "output" will be used.
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

        if sys.platform == 'win32' and self.backend == 'qnnpack':
            log.error('Quantization backend qnnpack is likely unsupported on Windows. Please use fbgemm instead.')

        if self.backend not in ('fbgemm', 'qnnpack', 'onnx', 'tensorrt'):
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

        if self.backend == 'fbgemm':
            assert self.asymmetric, "Symmetric quantizaton for FBGEMM not supported"
            assert (
                not self.per_tensor
            ), "Per-tensor quantizaton for FBGEMM not supported, please use per-channel quantization instead"

        if self.backend == 'tensorrt':
            if self.asymmetric:
                log.warning('Asymmetric quantizaton for TensorRT not supported')

        if self.disable_requantization_for_cat is None:
            if not self.per_tensor:
                self.disable_requantization_for_cat = True
            else:
                self.disable_requantization_for_cat = False

        self.extra_qparams_mappings = []

        assert (
            self.per_tensor or self.disable_requantization_for_cat
        ), "`disable_requantization_for_cat=True` is required for per-channel quantization"

        if self.legacy_fq:
            version = None
            if type(self).__name__ == 'QATQuantizer':
                version = '1.10.0'
            elif type(self).__name__ == 'PostQuantizer':
                version = '1.12.0'

            if version is None or LooseVersion(torch.__version__) < version:
                log.info(f'legacy_fq=True is only available for QATQuantizer and PostQuantizer with PyTorch {version}+')
                self.legacy_fq = False

        self.leaf_nodes = None
        self.swap_nodes = None

        self.train_mode_dict = {}

        self.layerwise_config = yaml.comments.CommentedMap()
        self.effective_layers = []
        self.layerwise_default = True
        if config is not None and 'layerwise_config' in config:
            self.layerwise_config.update(config['layerwise_config'])

    def parse_config(self, config: typing.Optional[dict]):
        default_values = {
            'rewrite_graph': True,
            'force_overwrite': True,
            'is_input_quantized': None,
            'backend': 'qnnpack',
            'remove_weights_after_load': False,
            'asymmetric': True,
            'per_tensor': True,
            'disable_requantization_for_cat': None,
            'quantized_input_stats': None,
            'dynamic_lstm_quant': False,
            'quantized_op_stats': None,
            'set_quantizable_op_stats': False,
            'rounding_mode': 'pytorch',
            'algorithm': 'l2',
            'fuse_only': False,
            'legacy_fq': True,
            'extra_tracer_opts': {},
            'fused_layerwise_config': True,
            'ignore_layerwise_config': False,
            'inplace': False,
            'override_qconfig_func': None,
        }

        if config is None:
            config = dict()

        for k, v in default_values.items():
            actual_v = config.get(k, v)
            setattr(self, k, actual_v)

    def set_model_mode(self, model: nn.Module = None, preserve: bool = False):
        """Sets the mode (train/eval) of the model

        Args:
            model (nn.Module, optional): A PyTorch model. Defaults to None.
            preserve (bool, optional): Whether the old state is preserved. Defaults to False.
        """
        if model is None:
            model = self.model

        if preserve:
            for m in model.modules():
                training = getattr(m, 'training', None)
                if isinstance(training, bool):
                    self.train_mode_dict[m] = training

        model.train()

    def restore_model_mode(self, model: nn.Module = None):
        """Restores the mode (train/eval) of the model

        Args:
            model (nn.Module, optional): A PyTorch model. Defaults to None.
        """
        if model is None:
            model = self.model

        for m, training in self.train_mode_dict.items():
            if training:
                m.train()
            else:
                m.eval()

        self.train_mode_dict.clear()

    def quantize(self) -> nn.Module:
        """Performs QAT rewrite and preparation

        Returns:
            nn.Module: The QAT-ready model
        """

        # We need a model in training mode so that QAT could take place
        self.set_model_mode(preserve=True)

        if self.rewrite_graph:
            # Retrives the name of the model from type info
            model_name = type(self.model).__name__
            model_name_q = f'Q{model_name}'
            model_name_q_lower = f'{model_name}_q'.lower()
            model_ns = f'tinynn_rewritten_models.{model_name_q}'

            model_code_path = os.path.join(self.work_dir, f'{model_name_q_lower}.py')
            model_weights_path = os.path.join(self.work_dir, f'{model_name_q_lower}.pth')

            config_path = os.path.join(self.work_dir, f'{model_name_q_lower}_config.yml')

            yaml_ = yaml.YAML()
            if len(self.layerwise_config) == 0 and not self.ignore_layerwise_config:
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        self.layerwise_config = yaml_.load(f)

            if 'default' in self.layerwise_config:
                self.layerwise_default = self.layerwise_config['default']
                del self.layerwise_config['default']

            # Generate the code for the modified model
            # We will try to do some op fusion and rewriting in the `rewrite_quantize_graph` function.
            # By default, we will try to insert QuantStubs before every input and DeQuantStubs after every output in
            # the generated graph. If this doesn't suit your needs, e.g. you have intergal/quantized inputs or want to
            # skip the quantization rewrite for some ops, then you may modify the code generated freely and remember to
            # skip this step so it won't be overwritten.
            if self.force_overwrite or not os.path.exists(model_code_path) or not os.path.exists(model_weights_path):
                # After tracing the model, we will get a TraceGraph object
                graph = trace(self.model, self.dummy_input, **self.extra_tracer_opts)
                self.rewrite_quantize_graph(graph)
                if self.inplace:
                    graph.inplace_commit()
                else:
                    graph.generate_code(model_code_path, model_weights_path, model_name_q)

            os.makedirs(self.work_dir, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml_.dump(self.layerwise_config, f)

            if self.inplace:
                rewritten_model = self.model
            else:
                # Import the new model
                rewritten_model = import_from_path(model_ns, model_code_path, model_name_q)()
                rewritten_model.load_state_dict(torch.load(model_weights_path))

                device = get_module_device(self.model)
                if device is not None:
                    rewritten_model.to(device=device)

                # Remove the weights file to save space
                if self.remove_weights_after_load:
                    os.unlink(model_weights_path)

                # Restores the mode of the model
                self.restore_model_mode()

                # Set the model to training mode before tracing again, since we need to perform QAT
                self.set_model_mode(rewritten_model)

            rewritten_graph = trace(rewritten_model, self.dummy_input)

            # Update the model so that the original one can be released
            self.model = rewritten_model
        else:
            # After tracing the model, we will get a TraceGraph object
            graph = trace(self.model, self.dummy_input, **self.extra_tracer_opts)
            rewritten_graph = graph

        # Fuse the modules (e.g conv, bn, relu) in the computation graph according to the fuse rules.
        # By default, we assume all input tensors are of floating type.
        # If you want to use quantized/integral inputs, then you may need to pass in `is_input_quantized`.
        qat_model = self.prepare_qat(rewritten_graph, self.is_input_quantized, self.backend, self.fuse_only)

        return qat_model

    def prepare_qat_prep(
        self,
        graph: TraceGraph,
        is_input_quantized: typing.Optional[typing.Tuple[bool]] = None,
        backend: str = 'qnnpack',
    ):
        """Some common logic before calling torch.quantization.prepare[_qat]

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) \
                quantized. Defaults to None.
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

            # TODO: Enable QAT analysis for TensorRT
            if self.backend == 'tensorrt':
                quantized = True
                node.quantized = True
            elif type(node.module) is torch_q.QuantStub:
                quantized = True
                node.quantized = quantized
            elif type(node.module) is torch_q.DeQuantStub:
                node.quantized = True
                quantized = False
            else:
                node.quantized = quantized
                log.debug(f"[QUANTIZED]{node.unique_name}:{quantized}")

            for i, n in enumerate(node.next_nodes):
                if type(n.module) == TraceFunction:
                    if n.kind() in ('shape', 'size', 'dtype', 'device'):
                        continue
                    if n.kind() == 'expand_as' and i > 0:
                        continue
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

        def _is_params_in_module(node, custom_data):
            if len(node.prev_nodes) == 1 and len(node.next_nodes) == 1:
                if len(node.prev_tensors) == 1 and len(node.next_tensors) == 1:
                    if isinstance(node.prev_tensors[0], nn.Module) and not isinstance(
                        node.prev_tensors[0], nnq.FloatFunctional
                    ):
                        return True
            return False

        param_nodes = graph.filter_forward_nodes(_is_params_in_module)

        for n in graph.constant_nodes + extra_constant_nodes:
            qat_analysis_queue.put((n, not torch.is_floating_point(n.next_tensors[0])))

        while not qat_analysis_queue.empty():
            node, quantized = qat_analysis_queue.get()
            if not graph.quantized:
                graph.quantized = graph.quantized or quantized
            _qat_analysis(node, quantized)

        q_dict = {}
        for n in graph.forward_nodes:
            if isinstance(n.module, nn.Module):
                q_dict[n.module] = q_dict.get(n.module, False) or n.quantized

        for n in graph.forward_nodes:
            if n.module in q_dict:
                n.quantized = q_dict[n.module]

        for n in param_nodes:
            prev_node = n.prev_nodes[0]
            next_node = n.next_nodes[0]
            is_known_mod = prev_node.kind().__name__ in (
                'Conv1d',
                'Conv2d',
                'Linear',
                'ConvTranspose1d',
                'ConvTranspose2d',
            )
            if is_known_mod and n.module.full_name == 'weight' and prev_node.quantized:
                if next_node.type() == torch_q.QuantStub:
                    mod = nn.Sequential(torch_q.DeQuantStub(), torch_q.QuantStub())
                    orig_mod = next_node.module
                    next_node.module = mod
                    setattr(graph.module, next_node.original_name, next_node.module)
                    graph.module_original_name_dict[id(mod)] = graph.module_original_name_dict[id(orig_mod)]
                    graph.module_unique_name_dict[id(mod)] = graph.module_unique_name_dict[id(orig_mod)]
                continue
            qat_analysis_queue.put((n, not torch.is_floating_point(n.next_tensors[0])))

        while not qat_analysis_queue.empty():
            node, quantized = qat_analysis_queue.get()
            if not graph.quantized:
                graph.quantized = graph.quantized or quantized
            _qat_analysis(node, quantized)

        log.debug("qat analysis over")

        if not graph.quantized:
            return

        if isinstance(self, PostQuantizer):
            processed_rules = load_processed_ptq_rules()
        else:
            processed_rules = load_processed_qat_rules()

        is_fusable = functools.partial(self.is_fusable, current_rules=processed_rules, graph=graph)

        def _find_quantized_module_nodes(node: TraceNode, custom_node):
            # Find quantized module nodes
            return node.type() in Q_MODULES_MAPPING and node.quantized

        # Replace module nodes with our custom variants
        quantized_mod_nodes = graph.filter_forward_nodes(_find_quantized_module_nodes)

        type_dict = {}
        for node in quantized_mod_nodes:
            node_type = node.type()
            type_dict.setdefault(node_type, [])
            type_dict[node_type].append(node)

        for node_type, nodes in type_dict.items():
            graph.update_submodule_in_nodes_from_predicate(nodes, Q_MODULES_MAPPING[node_type], self.inplace)

        custom_data = ([], set())
        graph.filter_forward_nodes(is_fusable, custom_data, reverse=True)
        quant_list = custom_data[0]
        log.info(f'found nodes to fuse: {quant_list}')

        new_fuser_func = fm.gen_fuse_known_modules_wrapper(
            sys.modules['torch.quantization.fuse_modules'].fuse_known_modules
        )

        if self.backend != 'tensorrt':
            for quant_nodes in quant_list:
                if self.inplace:
                    quant_nodes = [re.sub('get_submodule\\("(.*?)"\\)', '\\1', x) for x in quant_nodes]
                    quant_nodes = [re.sub('\\[("|)(.*?)("|)\\]', '.\\2', x) for x in quant_nodes]

                if type(self) != PostQuantizer and LooseVersion(torch.__version__) >= '1.11.0':
                    # See https://github.com/pytorch/pytorch/pull/88193
                    if LooseVersion(torch.__version__) < '1.14.0':
                        sys.modules['torch.quantization.fuse_modules']._fuse_modules(
                            graph.module, quant_nodes, is_qat=True, inplace=True, fuser_func=new_fuser_func
                        )
                    else:
                        torch.ao.quantization.fuse_modules_qat(
                            graph.module, quant_nodes, fuser_func=new_fuser_func, inplace=True
                        )
                else:
                    torch_q.fuse_modules(graph.module, quant_nodes, fuser_func=new_fuser_func, inplace=True)

        self.prepare_qconfig(graph, backend)
        self.override_qconfig(graph.module)

    def override_qconfig(self, module: nn.Module):
        """Sets qconfig according to the provided `override_qconfig_func`

        Args:
            module (nn.Module): The model to be prepared for quantization
        """
        if self.override_qconfig_func is not None:
            q = queue.Queue()
            q.put(('', module))

            while not q.empty():
                n, m = q.get()

                new_qconfig = self.override_qconfig_func(n, m)
                if new_qconfig is not None:
                    m.qconfig = new_qconfig
                else:
                    for cn, cm in m.named_children():
                        fn = cn
                        if len(n) > 0:
                            fn = f'{n}.{fn}'
                        q.put((fn, cm))

    def prepare_qconfig(self, graph: TraceGraph, backend: str):
        """Prepare qconfig for various configurations.

        Args:
            graph (TraceGraph): The computation graph of the model
            backend (str, optional): The backend of quantization
        """

        log.info('setting qat backend and call prepare_qat')
        actual_backend = backend
        if backend in ('onnx', 'tensorrt'):
            actual_backend = 'qnnpack'
        if not self.legacy_fq:
            qconfig = torch_q.get_default_qat_qconfig(actual_backend)
        else:
            if LooseVersion(torch.__version__) >= '1.13.0':
                # See https://github.com/pytorch/pytorch/pull/88876
                qconfig = torch_q.QConfig(
                    activation=torch_q.FakeQuantize.with_args(
                        observer=torch_q.MovingAverageMinMaxObserver, quant_min=0, quant_max=255, reduce_range=False
                    ),
                    weight=torch_q.default_weight_fake_quant,
                )
            else:
                version = None
                if LooseVersion(torch.__version__) >= '1.12.0':
                    version = 0
                qconfig = torch_q.get_default_qat_qconfig(actual_backend, version)

        qconfig_c = None
        if self.rounding_mode == 'tflite':
            q_a = FakeQuantizeTFLite.with_args(*qconfig.activation.p.args, **qconfig.activation.p.keywords)
            q_w = FakeQuantizeTFLite.with_args(*qconfig.weight.p.args, **qconfig.weight.p.keywords)
            qconfig = torch_q.QConfig(q_a, q_w)
        if backend == 'qnnpack':
            if not self.asymmetric:
                sym_fq = qconfig.activation.with_args(
                    observer=torch_q.MovingAverageMinMaxObserver,
                    quant_min=0,
                    quant_max=255,
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = qconfig.weight.with_args(
                    observer=torch_q.MovingAveragePerChannelMinMaxObserver.with_args(quant_min=-127, quant_max=127),
                    quant_min=-127,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                    ch_axis=0,
                )
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        elif backend == 'fbgemm':
            fq_type = qconfig.weight.p.func
            sym_fq = fq_type.with_args(
                observer=torch_q.MovingAverageMinMaxObserver,
                quant_min=-128,
                quant_max=127,
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=False,
            )
            qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        elif backend in ('onnx', 'tensorrt'):
            if not self.asymmetric:
                sym_fq = qconfig.activation.with_args(
                    observer=torch_q.MovingAverageMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric,
                    reduce_range=False,
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = qconfig.weight.with_args(
                    observer=torch_q.MovingAveragePerChannelMinMaxObserver,
                    quant_min=-128,
                    quant_max=127,
                    dtype=torch.qint8,
                    qscheme=torch.per_channel_symmetric,
                    reduce_range=False,
                    ch_axis=0,
                )
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        else:
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

        torch.backends.quantized.engine = actual_backend
        graph.module.qconfig = qconfig
        if self.backend == 'qnnpack':
            if qconfig_c is not None:
                q = queue.Queue()
                q.put(graph.module)

                while not q.empty():
                    m = q.get()
                    if type(m).__name__ in (
                        'Conv2d',
                        'ConvBnReLU2d',
                        'ConvBn2d',
                        'ConvReLU2d',
                        'Conv1d',
                        'ConvBnReLU1d',
                        'ConvBn1d',
                    ):
                        m.qconfig = qconfig_c
                    else:
                        for c in m.children():
                            q.put(c)
        elif self.backend == 'fbgemm':
            if qconfig_c is not None:
                q = queue.Queue()
                q.put(graph.module)

                while not q.empty():
                    m = q.get()
                    if type(m).__name__ in ('Linear', 'LinearReLU'):
                        m.qconfig = qconfig_c
                    else:
                        for c in m.children():
                            q.put(c)

        def _lstm_node(node, custom_data):
            return isinstance(node.module, nn.LSTM)

        if self.dynamic_lstm_quant:
            lstm_nodes = graph.filter_forward_nodes(_lstm_node)
            for node in lstm_nodes:
                node.quantized = True
                node.module.qconfig = torch_q.default_dynamic_qconfig

    def prepare_qat(
        self,
        graph: TraceGraph,
        is_input_quantized: typing.Optional[typing.Tuple[bool]] = None,
        backend: str = 'qnnpack',
        fuse_only: bool = False,
    ) -> torch.nn.Module:
        """Prepare model for QAT training

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) \
                quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.
            fuse_only (bool, optional): Whether the returned model is only fused in PostQuantizer. Defaults to False.

        Returns:
            torch.nn.Module: The QAT-ready model
        """

        graph.module.train()

        self.prepare_qat_prep(graph, is_input_quantized, backend)

        if not graph.quantized:
            log.warning('Graph is not quantized, skip preparation')
            return graph.module

        # Unfornately, the suggested way below will try to fuse all the modules
        # even if some of the nodes are not in a quantized computation graph.
        # So we wrote some alternatives for the function.
        #   torch.quantization.prepare_qat(graph.module, inplace=True)

        if hasattr(torch_q, 'get_default_qat_module_mappings'):
            mapping = torch_q.get_default_qat_module_mappings()
        elif hasattr(torch_q, 'get_qat_module_mappings'):
            mapping = copy.deepcopy(torch_q.get_qat_module_mappings())
        else:
            mapping = copy.deepcopy(torch_q.DEFAULT_QAT_MODULE_MAPPING)

        if self.dynamic_lstm_quant:
            mapping = dict(mapping)
            mapping.update({nn.LSTM: nnqd.LSTM})

        mapping.update(FUSE_QAT_MODULES)

        q_dict = {}
        for n in graph.forward_nodes:
            if isinstance(n.module, nn.Module):
                q_dict[n.module] = q_dict.get(n.module, False) or n.quantized

        non_quantized_mods = set(m for m, s in q_dict.items() if s is False)

        if LooseVersion(torch.__version__) < LooseVersion("1.7.0"):
            model = torch_q.prepare(graph.module, inplace=True)
            for m in non_quantized_mods:
                if hasattr(m, "_forward_hooks"):
                    if len(m._forward_hooks) > 0:
                        m._forward_hooks.popitem()
                if hasattr(m, "qconfig"):
                    delattr(m, "qconfig")
                if hasattr(m, "activation_post_process"):
                    delattr(m, "activation_post_process")

            if hasattr(graph.module, 'activation_post_process'):
                if hasattr(graph.module, "_forward_hooks"):
                    if len(graph.module._forward_hooks) > 0:
                        graph.module._forward_hooks.popitem()
                delattr(graph.module, "activation_post_process")

            torch_q.convert(model, mapping, inplace=True)
        else:
            torch_q.propagate_qconfig_(graph.module, qconfig_dict=None)
            for m in non_quantized_mods:
                if hasattr(m, "qconfig"):
                    delattr(m, "qconfig")
            model = torch_q.convert(graph.module, mapping=mapping, inplace=True, remove_qconfig=False)

            if self.dynamic_lstm_quant:
                mapping.pop(nn.LSTM)

            if LooseVersion(torch.__version__) >= LooseVersion("1.13.0"):
                torch_q.propagate_qconfig_(model, qconfig_dict=None)

                for m in non_quantized_mods:
                    if hasattr(m, "qconfig"):
                        delattr(m, "qconfig")

                prepare_custom_config_dict = torch.ao.quantization.get_default_custom_config_dict()
                custom_module_class_mapping = prepare_custom_config_dict.get(
                    "float_to_observed_custom_module_class", {}
                )
                custom_module_class_mapping.update(FUSE_QAT_MODULES_CUSTOM)
                qconfig_propagation_list = torch_q.get_default_qconfig_propagation_list()

                from .quantizable import lstm, gru

                orig_from_float = torch.ao.nn.quantizable.LSTM.from_float

                torch.ao.nn.quantizable.LSTM.from_float = lstm.from_float
                GRU.from_float = gru.from_float

                def patch_observer_set(orig_func):
                    def new_no_observer_set():
                        return set(FUSE_QAT_MODULES_CUSTOM.values()) | orig_func()

                    return new_no_observer_set

                orig_no_observer_set = sys.modules['torch.ao.quantization.quantize'].no_observer_set
                sys.modules['torch.ao.quantization.quantize'].no_observer_set = patch_observer_set(orig_no_observer_set)

                if hasattr(torch_q, 'add_observer_'):
                    add_observer_func = torch_q.add_observer_
                else:
                    add_observer_func = sys.modules['torch.ao.quantization.quantize']._add_observer_

                add_observer_func(
                    model,
                    qconfig_propagation_list,
                    set(mapping.values()),
                    custom_module_class_mapping=custom_module_class_mapping,
                )

                torch.ao.nn.quantizable.LSTM.from_float = orig_from_float
                sys.modules['torch.ao.quantization.quantize'].no_observer_set = orig_no_observer_set

            else:
                torch_q.prepare(model, observer_non_leaf_module_list=set(mapping.values()), inplace=True)
            for m in non_quantized_mods:
                if hasattr(m, "_forward_hooks"):
                    if len(m._forward_hooks) > 0:
                        m._forward_hooks.popitem()
                if hasattr(m, "qconfig"):
                    delattr(m, "qconfig")
                if hasattr(m, "activation_post_process"):
                    delattr(m, "activation_post_process")

        if not self.per_tensor:
            if self.backend == 'qnnpack':
                for n, m in graph.module.named_modules():
                    if n.endswith('.weight_fake_quant'):
                        observer = getattr(m, 'activation_post_process', None)
                        if observer is not None:
                            m.quant_min = -127
                            m.quant_max = 127
                            observer.quant_min = -127
                            observer.quant_max = 127

        self.extra_qat_fusion_postprocess(graph)

        if self.disable_requantization_for_cat:
            self.disable_requantization_for_cat_pass(graph)

        if self.quantized_input_stats is not None:
            self.prepare_quantized_inputs_pass(graph)

        if self.set_quantizable_op_stats:
            if self.quantized_op_stats is None:
                self.quantized_op_stats = {}
            self.quantized_op_stats.update(KNOWN_QSTATS)

        if self.quantized_op_stats is not None:
            self.prepare_quantized_ops_pass(graph)

        if self.backend == 'tensorrt':
            q = queue.Queue()
            q.put(('', graph.module))

            while not q.empty():
                s, m = q.get()
                if isinstance(m, torch_q.QuantStub):
                    m.apply(torch.quantization.enable_fake_quant)
                    m.apply(torch.quantization.enable_observer)

                elif hasattr(m, 'activation_post_process'):
                    m.activation_post_process.apply(torch.quantization.disable_fake_quant)
                    m.activation_post_process.apply(torch.quantization.disable_observer)

                    if hasattr(m, 'weight_fake_quant'):
                        m.weight_fake_quant.apply(torch.quantization.enable_fake_quant)
                        m.weight_fake_quant.apply(torch.quantization.enable_observer)
                else:
                    for n, c in m.named_children():
                        q.put((f'{s}{n}', c))

        if self.backend == 'onnx':
            self.leaf_nodes = []
            self.swap_nodes = []

            q = queue.Queue()
            for node in graph.output_nodes:
                for prev_node in node.prev_nodes:
                    q.put((prev_node, False))

            while not q.empty():
                n, state = q.get()
                if isinstance(n.module, nn.Module):
                    orig_name = graph.module_original_name_dict.get(id(n.module))
                    new_mod, _ = graph.get_submodule_with_parent_from_name(orig_name, self.inplace)
                    if isinstance(new_mod, torch_q.DeQuantStub):
                        state = True
                    else:
                        if state:
                            if isinstance(new_mod, torch_q.QuantStub):
                                state = False
                            elif isinstance(new_mod, nn.Module) and hasattr(new_mod, 'activation_post_process'):
                                self.leaf_nodes.append(new_mod)
                                state = False
                            elif (
                                isinstance(new_mod, nn.Sequential)
                                and type(new_mod).__module__.startswith(nni.__name__)
                                and len(new_mod) > 0
                                and hasattr(new_mod[-1], 'activation_post_process')
                            ):
                                self.leaf_nodes.append(new_mod[-1])
                                state = False
                    for pn in n.prev_nodes:
                        q.put((pn, state))

            q = queue.Queue()
            visited = set()
            for node in graph.input_nodes:
                q.put((node, None, False, 0))

            while not q.empty():
                n, prev_q_mod, state, idx = q.get()
                key = f'{n.unique_name}:{idx}'
                if key in visited:
                    continue
                else:
                    visited.add(key)

                q_mod = prev_q_mod
                if n.quantized:
                    if isinstance(n.module, nn.Module):
                        orig_name = graph.module_original_name_dict.get(id(n.module))
                        new_mod, _ = graph.get_submodule_with_parent_from_name(orig_name, self.inplace)
                        if isinstance(new_mod, nn.Module) and hasattr(new_mod, 'activation_post_process'):
                            q_mod = new_mod
                        elif (
                            isinstance(new_mod, nn.Sequential)
                            and type(new_mod).__module__.startswith(nni.__name__)
                            and len(new_mod) > 0
                            and hasattr(new_mod[-1], 'activation_post_process')
                        ):
                            q_mod = new_mod[-1]
                        elif isinstance(new_mod, torch_q.DeQuantStub):
                            q_mod = new_mod
                        elif type(new_mod) != nn.Identity:
                            state = True
                    else:
                        is_prev_float_functional = (
                            len(n.prev_nodes) > 1 and n.prev_nodes[0].type() == torch.nn.quantized.FloatFunctional
                        )
                        if is_prev_float_functional:
                            q_mod = getattr(n.prev_nodes[0].module, n.kind())
                        else:
                            state = True

                    if state and prev_q_mod is not None and q_mod != prev_q_mod:
                        self.swap_nodes.append((prev_q_mod, q_mod, idx))
                        state = False

                for next_n in n.next_nodes:
                    idx = next_n.prev_nodes.index(n)
                    q.put((next_n, q_mod, state, idx))

        return graph.module

    def extra_qat_fusion_postprocess(self, graph):
        # Process additional fusable nodes
        processed_extra_q_rules = load_processed_extra_q_rules()
        is_extra_fusable = functools.partial(
            self.is_fusable,
            current_rules=processed_extra_q_rules,
            graph=graph,
        )

        custom_data = ([], set())
        graph.filter_forward_nodes(is_extra_fusable, custom_data, reverse=True)
        quant_list = custom_data[0]

        log.debug(f'Extra q postprocess for nodes: {quant_list}')

        rev_dict = dict((v, k) for k, v in graph.module_original_name_dict.items())
        for quant_nodes in quant_list:
            for orig_name in quant_nodes:
                if orig_name in rev_dict:
                    new_mod, _ = graph.get_submodule_with_parent_from_name(orig_name, self.inplace)
                    acp = getattr(new_mod, 'activation_post_process', None)
                    if acp is not None:
                        torch.quantization.disable_fake_quant(acp)
                        torch.quantization.disable_observer(acp)

                        activ_name = quant_nodes[-1]
                        if activ_name in rev_dict:
                            unique_name = graph.module_unique_name_dict[rev_dict[activ_name]]
                        else:
                            unique_name = activ_name
                            activ_name = None
                        node = graph.nodes_map[unique_name]
                        activ_type = node.kind()

                        post_dq = node.next_nodes[0].module
                        post_q = node.next_nodes[0].next_nodes[0].module

                        assert isinstance(post_dq, torch_q.DeQuantStub)
                        assert isinstance(post_q, torch_q.QuantStub)

                        dq_name = graph.module_original_name_dict[id(post_dq)]
                        q_name = graph.module_original_name_dict[id(post_q)]

                        post_acp = getattr(post_q, 'activation_post_process', None)
                        assert post_acp is not None

                        self.extra_qparams_mappings.append([acp, post_acp, dq_name, q_name, activ_name, activ_type])

    def disable_requantization_for_cat_pass(self, graph):
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
            visited_other = dict()
            while not q.empty():
                n, mode, fq_count = q.get()
                if (
                    n.kind() in ('shape', 'size')
                    or n.unique_name in visited_center
                    or visited_other.get(n.unique_name, 2) <= fq_count
                ):
                    continue

                if n.type() == 'cat':
                    visited_center.add(n.unique_name)
                else:
                    visited_other[n.unique_name] = fq_count

                new_fq_count = fq_count

                if isinstance(n.module, nn.Module):
                    is_prev_float_functional = False
                    orig_name = graph.module_original_name_dict.get(id(n.module))
                    new_mod, parent = graph.get_submodule_with_parent_from_name(orig_name, self.inplace)
                    prop = orig_name.split('.')[-1]
                    if QuantizableLSTM is not None and isinstance(new_mod, QuantizableLSTM):
                        if new_fq_count == 0:
                            if new_mod.bidirectional is False:
                                parents.append(new_mod.layers[-1].layer_fw.cell.ogate_cy)
                                names.append(f'{orig_name}.layer_fw.cell.ogate_cy.activation_post_process')
                                props.append('activation_post_process')
                            else:
                                parents.append(new_mod.layers[-1].layer_fw.cell.ogate_cy)
                                names.append(f'{orig_name}.layer_fw.cell.ogate_cy.activation_post_process')
                                props.append('activation_post_process')
                                parents.append(new_mod.layers[-1].layer_bw.cell.ogate_cy)
                                names.append(f'{orig_name}.layer_bw.cell.ogate_cy.activation_post_process')
                                props.append('activation_post_process')
                        new_fq_count += 1
                    elif QuantizableGRU is not None and isinstance(new_mod, QuantizableGRU):
                        if new_fq_count == 0:
                            if new_mod.bidirectional is False:
                                parents.append(new_mod.layers[-1].layer_fw.cell.add4)
                                names.append(f'{orig_name}.layer_fw.cell.add4.activation_post_process')
                                props.append('activation_post_process')
                            else:
                                parents.append(new_mod.layers[-1].layer_fw.cell.add4)
                                names.append(f'{orig_name}.layer_bw.cell.add4.activation_post_process')
                                props.append('activation_post_process')
                                parents.append(new_mod.layers[-1].layer_bw.cell.add4)
                                names.append(f'{orig_name}.layer_bw.cell.add4.activation_post_process')
                                props.append('activation_post_process')
                        new_fq_count += 1
                    elif isinstance(new_mod, (torch_q.FakeQuantize, torch_q.ObserverBase)):
                        if new_fq_count == 0:
                            parents.append(parent)
                            names.append(orig_name)
                            props.append(prop)
                        new_fq_count += 1
                    elif hasattr(new_mod, 'activation_post_process'):
                        if new_fq_count == 0:
                            parents.append(new_mod)
                            names.append(f'{orig_name}.activation_post_process')
                            props.append('activation_post_process')
                        new_fq_count += 1
                    elif (
                        isinstance(new_mod, nn.Sequential)
                        and type(new_mod).__module__.startswith(nni.__name__)
                        and len(new_mod) > 0
                        and hasattr(new_mod[-1], 'activation_post_process')
                    ):
                        if new_fq_count == 0:
                            parents.append(new_mod[-1])
                            names.append(f'{orig_name}[-1].activation_post_process')
                            props.append('activation_post_process')
                        new_fq_count += 1
                    if isinstance(new_mod, (torch_q.DeQuantStub, torch_q.QuantStub)):
                        new_fq_count = 2
                else:
                    is_prev_float_functional = (
                        len(n.prev_nodes) > 1 and n.prev_nodes[0].type() == torch.nn.quantized.FloatFunctional
                    )
                    if n.type() == 'cat':
                        mode = 'both'
                        fq_count = 0
                        new_fq_count = 0
                    if is_prev_float_functional:
                        m = n.prev_nodes[0].module
                        orig_name = graph.module_original_name_dict.get(id(m))
                        if new_fq_count == 0:
                            parents.append(m)
                            names.append(f'{orig_name}.activation_post_process')
                            props.append('activation_post_process')
                        new_fq_count += 1

                if mode in ('both', 'down'):
                    fq_up = fq_count
                    fq_down = new_fq_count
                elif mode == 'up':
                    fq_up = new_fq_count
                    fq_down = fq_count

                if mode == 'up' and len(n.next_nodes) > 1:
                    mode = 'both'
                    fq_down += 1

                if mode in ('both', 'up'):
                    for i, node in enumerate(n.prev_nodes):
                        if is_prev_float_functional and i == 0:
                            continue
                        if fq_up < 2:
                            q.put((node, 'up', fq_up))
                if mode in ('both', 'down'):
                    for node in n.next_nodes:
                        if fq_down < 2:
                            q.put((node, 'down', fq_down))

            if len(names) > 1:
                log.debug(f'Unifying the following nodes into one: {", ".join(names)}')
                unified = getattr(parents[0], props[0])
                for parent, prop in zip(parents[1:], props[1:]):
                    setattr(parent, prop, unified)

    def prepare_quantized_inputs_pass(self, graph):
        if self.quantized_input_stats is not None:
            assert len(self.quantized_input_stats) == len(graph.input_nodes), (
                f"quantized_input_stats contains {len(self.quantized_input_stats)} elements, but"
                f" {len(graph.input_nodes)} is expected"
            )

            for qstats, node in zip(self.quantized_input_stats, graph.input_nodes):
                if qstats is not None:
                    assert (
                        isinstance(qstats, (list, tuple))
                        and len(qstats) == 2
                        and all((isinstance(q, (int, float)) for q in qstats))
                    ), "quantized_input_stats format: [(mean_1, std_1), (mean_2, std_2), ..., (mean_n, std_n)]"

                for next_node in node.next_nodes:
                    if isinstance(next_node.module, torch_q.QuantStub):
                        self.set_module_quantization(next_node.module, qstats[0], qstats[1])

    def prepare_quantized_ops_pass(self, graph):
        if self.quantized_op_stats:

            def _find_quantized_ops_with_type(node, custom_data):
                if isinstance(node.module, nn.Module) and hasattr(node.module, 'activation_post_process'):
                    qstats = self.quantized_op_stats.get(node.kind(), None)
                    if qstats is not None:
                        custom_data.append((qstats, node))
                        return True
                    else:
                        if isinstance(node.module, torch_q.QuantStub):
                            qstats = self.quantized_op_stats.get(node.prev_nodes[0].kind(), None)
                            if qstats is not None:
                                custom_data.append((qstats, node))
                                return True
                return False

            node_with_qstats = []
            graph.filter_forward_nodes(_find_quantized_ops_with_type, node_with_qstats)

            for qstats, node in node_with_qstats:
                if qstats is not None:
                    assert (
                        isinstance(qstats, (list, tuple))
                        and len(qstats) == 2
                        and all((isinstance(q, (int, float)) for q in qstats))
                    ), (
                        "quantized_op_stats format: {'op_0': (mean_1, std_1), 'op_1': (mean_2, std_2), ..., 'op_n':"
                        " (mean_n, std_n)}"
                    )

                self.set_module_quantization(node.module, qstats[0], qstats[1])

    def set_module_quantization(self, module: nn.Module, mean: int, std: float):
        acp = getattr(module, 'activation_post_process', None)
        if acp is not None:
            fq_base_cls = getattr(torch_q, 'FakeQuantizeBase', torch_q.FakeQuantize)
            if isinstance(acp, fq_base_cls):
                fake_quant = acp

                module.apply(torch_q.disable_observer)

                scale = torch.tensor(1.0 / std, dtype=fake_quant.scale.dtype)
                offset = torch.tensor(mean, dtype=fake_quant.zero_point.dtype)

                fake_quant.scale.copy_(scale)
                fake_quant.zero_point.copy_(offset)

                quant_min = fake_quant.quant_min
                quant_max = fake_quant.quant_max

                observer = getattr(fake_quant, 'activation_post_process', None)
            elif isinstance(acp, torch_q.ObserverBase):
                observer = acp

                # We cannot use `disable_observer` which is designed for `FakeQuant` modules`.
                # Instead, we need to monkey-patch the forward function of the observer.
                identity = nn.Identity()
                observer.forward = identity.forward

                scale = torch.tensor(1.0 / std, dtype=torch.float32)
                offset = torch.tensor(mean, dtype=torch.int32)

                quant_min = observer.quant_min
                quant_max = observer.quant_max

            else:
                log.warning(f'Given module {type(module)} doesn\'t seem to be a quantized module')
                return

            if observer is not None:
                if quant_min is None and quant_max is None:
                    if observer.reduce_range:
                        quant_min, quant_max = 0, 127
                    else:
                        quant_min, quant_max = 0, 255

                observer.min_val = scale * (quant_min - offset)
                observer.max_val = scale * (quant_max - offset)

    def rescale_activations_with_quant_min_max(self, quant_min: int, quant_max: int) -> None:
        """Rescales activations with provided quant_min and quant_max"""
        for n, m in self.model.named_modules():
            if '.weight_fake_quant' in n:
                continue

            if isinstance(m, torch.quantization.FakeQuantize):
                observer = getattr(m, 'activation_post_process', None)
                if observer is not None:
                    old_quant_range = m.quant_max - m.quant_min
                    m.quant_min = quant_min
                    m.quant_max = quant_max
                    new_quant_range = m.quant_max - m.quant_min
                    m.scale = m.scale * new_quant_range / old_quant_range
                    zero_point = (new_quant_range + 1) // 2
                    m.zero_point = torch.zeros_like(m.zero_point) + zero_point
                    observer.quant_min = quant_min
                    observer.quant_max = quant_max

    def rewrite_quantize_graph_for_tensorrt(self, graph: TraceGraph) -> None:
        """Rewrites the computation graph for TensorRT quantization"""

        processed_types = set()
        for types in TENSORRT_OBSERVED_NODES:
            if isinstance(types, tuple):
                if hasattr(types[0], types[1]):
                    processed_types.add(getattr(types[0], types[1]))
            else:
                processed_types.add(types)

        def _is_observed_nodes_for_tensorrt(node, custom_data):
            return node.kind() in processed_types

        observed_nodes = graph.filter_forward_nodes(_is_observed_nodes_for_tensorrt)
        node_map = dict()
        for idx, node in enumerate(observed_nodes):
            fake_quant_cls = torch_q.QuantStub
            assert node.rev_index is False
            prev_nodes = {n.unique_name: n for n in node.prev_nodes}.values()
            for inner_idx, prev_node in enumerate(prev_nodes):

                if prev_node.kind() in ('shape', 'device', 'size', 'dtype'):
                    continue

                prev_tensor_ptrs = []
                for pt in node.prev_tensors:
                    for nt in prev_node.next_tensors:
                        if isinstance(nt, (list, tuple)):
                            for k, ntt in enumerate(nt):
                                if id(pt) == id(ntt):
                                    if id(pt) not in prev_tensor_ptrs:
                                        prev_tensor_ptrs.append(id(pt))
                                    break
                        elif id(pt) == id(nt):
                            if id(pt) not in prev_tensor_ptrs:
                                prev_tensor_ptrs.append(id(pt))
                            break

                for ptr_idx, ptr in enumerate(prev_tensor_ptrs):
                    if ptr in node_map:
                        fake_quant = node_map[ptr]

                        graph.insert_between(prev_node, node, fake_quant, move_idx=True, tensor_ptrs=set([ptr]))
                    else:

                        fake_quant = fake_quant_cls()

                        fake_quant_name = f'fake_quant_inner_{idx}_{inner_idx}_{ptr_idx}'

                        graph.module_unique_name_dict[id(fake_quant)] = fake_quant_name
                        graph.module_original_name_dict[id(fake_quant)] = fake_quant_name

                        module_constructor_lines[id(fake_quant)] = f'{qualified_name(fake_quant_cls)}()'

                        graph.insert_between(prev_node, node, fake_quant, move_idx=True, tensor_ptrs=set([ptr]))
                        node_map[ptr] = graph.nodes_map[fake_quant_name]

        graph.quantized = True
        graph.recompute_forward_order()

    def rewrite_quantize_graph(self, graph: TraceGraph) -> None:
        """Rewrites the computation graph for quantization"""
        if graph.quantized:
            return

        graph_quantized = False
        for n in graph.forward_nodes:
            if n.type() in (torch_q.QuantStub, torch_q.DeQuantStub):
                graph_quantized = True
                break

        for n in graph.other_init_nodes:
            if n.type() == nnq.FloatFunctional:
                graph_quantized = True
                break

        if graph_quantized:
            log.warning(
                'Graph is quantized. No need to rewrite. Please pass in `config={"rewrite_graph": False}` to suppress'
                ' this warning'
            )

        if self.backend == 'tensorrt':
            return self.rewrite_quantize_graph_for_tensorrt(graph)

        if self.layerwise_default is False:
            for n in graph.forward_nodes:
                self.layerwise_config.setdefault(n.unique_name, False)

        creation_func_names = load_creation_func_names()

        def _is_extra_constant_nodes(node, custom_data):
            return node.full_name() in creation_func_names

        extra_constant_nodes = graph.filter_forward_nodes(_is_extra_constant_nodes)

        def _is_int_to_float_nodes(node, custom_data):
            if node.full_name() in creation_func_names:
                return False

            if len(node.prev_nodes) == 1 and len(node.next_nodes) == 1:
                if len(node.prev_tensors) == 1 and len(node.next_tensors) == 1:
                    if node.prev_nodes[0].kind() == 'shape' and node.prev_nodes[0].module.is_property:
                        return False
                    if (
                        isinstance(node.prev_tensors[0], torch.Tensor)
                        and isinstance(node.next_tensors[0], torch.Tensor)
                        and node.prev_tensors[0].dtype in (torch.int32, torch.int64)
                        and torch.is_floating_point(node.next_tensors[0])
                    ):
                        return True
            else:
                return False

        int_to_float_nodes = graph.filter_forward_nodes(_is_int_to_float_nodes)

        # When converting float-tensor to int16/int32/int64, we need to add 'fake_dequant' node before convert-node.
        def _is_float_to_non_float_nodes(node, custom_data):
            if isinstance(node.module, TraceFunction) and node.module.kind in ('shape', 'size'):
                return False
            return (
                len(node.next_tensors) == 1
                and len(node.prev_tensors) > 0
                and isinstance(node.prev_tensors[0], torch.Tensor)
                and isinstance(node.next_tensors[0], torch.Tensor)
                and torch.is_floating_point(node.prev_tensors[0])
                and not torch.is_floating_point(node.next_tensors[0])
            )

        float_to_non_float_nodes = graph.filter_forward_nodes(_is_float_to_non_float_nodes)

        def _is_params_in_module(node, custom_data):
            if len(node.prev_nodes) == 1 and len(node.next_nodes) == 1:
                if len(node.prev_tensors) == 1 and len(node.next_tensors) == 1:
                    if isinstance(node.prev_tensors[0], nn.Module) and not isinstance(
                        node.prev_tensors[0], nnq.FloatFunctional
                    ):
                        return True
            return False

        param_nodes = graph.filter_forward_nodes(_is_params_in_module)
        for node in param_nodes:
            prev_node = node.prev_nodes[0]
            is_known_mod = prev_node.kind().__name__ in (
                'Conv1d',
                'Conv2d',
                'Linear',
                'ConvTranspose1d',
                'ConvTranspose2d',
            )
            prop = node.module.full_name

            if is_known_mod and prop in ('weight', 'bias') and self.layerwise_config.get(prev_node.unique_name, True):
                node.module.is_class = False
                node.module.is_property = False

                node.module.full_name = 'tinynn.graph.quantization.utils.get_parameter'
                node.module.args_template = f'{{}}, "{prop}"'
                node.module.args_template_no_self = f'"{prop}"'
                node.module.args_offset = 0
                node.module.update_args_string()

        # First, we insert the QuantStub nodes for every input/constant node
        for idx, node in reversed(
            list(
                enumerate(
                    graph.input_nodes + graph.constant_nodes + extra_constant_nodes + int_to_float_nodes + param_nodes
                )
            )
        ):
            if node.next_tensors[0].dtype in (torch.int32, torch.int64):
                continue

            fake_quant = torch_q.QuantStub()

            graph.module_unique_name_dict[id(fake_quant)] = f'fake_quant_{idx}'
            graph.module_original_name_dict[id(fake_quant)] = f'fake_quant_{idx}'

            fake_quant_cls = type(fake_quant)
            module_constructor_lines[id(fake_quant)] = f'{qualified_name(fake_quant_cls)}()'

            graph.insert_after(node, fake_quant)

        # Second, we insert the DeQuantStub nodes for every output node
        for idx, node in enumerate(graph.output_nodes + float_to_non_float_nodes):
            fake_dequant_cls = torch_q.DeQuantStub
            if node.rev_index:
                modules = []
                for rev_idx in range(len(node.prev_nodes)):
                    if node.prev_tensors[rev_idx].dtype in (torch.int32, torch.int64):
                        fake_dequant = nn.Identity()
                    else:
                        fake_dequant = fake_dequant_cls()

                    graph.module_unique_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}_{rev_idx}'
                    graph.module_original_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}_{rev_idx}'

                    module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'
                    modules.append(fake_dequant)

                graph.insert_before(node, modules, move_idx=True)
            else:
                if node.prev_tensors[0].dtype in (torch.int32, torch.int64):
                    continue

                fake_dequant = fake_dequant_cls()

                graph.module_unique_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}'
                graph.module_original_name_dict[id(fake_dequant)] = f'fake_dequant_{idx}'

                module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'

                if len(node.prev_nodes) > 1:
                    # Insert 'fake_dequant' node before type conversion operators
                    graph.insert_between(node.prev_nodes[0], node, fake_dequant, move_idx=True)
                else:
                    graph.insert_before(node, fake_dequant, move_idx=True)

        # Third, we rewrite neg/sub/div using supported functions(e.g add, mul)
        def _is_neg_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return (
                    cur_module.kind == 'neg'
                    and torch.is_floating_point(cur_module.prev_tensors[0])
                    and self.layerwise_config.get(node.unique_name, True)
                )

        neg_nodes = graph.filter_forward_nodes(_is_neg_node)
        log.info(f'rewriting neg for {[node.unique_name for node in neg_nodes]}')
        for idx, node in enumerate(neg_nodes):
            node.module.func_type = '__mul__'
            node.module.kind = 'mul'

            full_name_parts = node.module.full_name.split('.')
            full_name_parts[-1] = node.module.func_type

            node.module.full_name = '.'.join(full_name_parts)

            with override_current_trace_graph(graph):
                node.module.parse_args(node.prev_tensors[0], -1.0)

        def _is_div_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            # Current, only the following condition could be handled.
            #   a / constant => a * (1 / constant)
            if cur_class == TraceFunction:
                return (
                    (cur_module.kind == 'truediv' or cur_module.func_type in ('div', 'div_'))
                    and len(cur_module.prev_tensors) == 1
                    and torch.is_floating_point(cur_module.prev_tensors[0])
                    and cur_module.func_type != '__rtruediv__'
                    and torch.is_floating_point(node.next_tensors[0])
                    and node.prev_nodes[0].kind() not in ('size', 'shape')
                    and self.layerwise_config.get(node.unique_name, True)
                )

        div_nodes = graph.filter_forward_nodes(_is_div_node)
        log.info(f'rewriting div for {[node.unique_name for node in div_nodes]}')
        for idx, node in enumerate(div_nodes):
            op_type = node.module.func_type
            inplace = op_type in ('__itruediv__', 'itruediv', 'div_')

            if inplace:
                node.module.func_type = '__imul__'
            else:
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
                return (
                    cur_module.kind == 'sub'
                    and torch.is_floating_point(cur_module.prev_tensors[0])
                    and torch.is_floating_point(node.next_tensors[0])
                    and self.layerwise_config.get(node.unique_name, True)
                )

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
            return (
                cur_class == TraceFunction
                and cur_module.kind == 'stack'
                and torch.is_floating_point(node.next_tensors[0])
                and self.layerwise_config.get(node.unique_name, True)
            )

        stack_nodes = graph.filter_forward_nodes(_is_stack_node)

        for idx, node in enumerate(stack_nodes):
            args = getattr(node.module, 'args_string_no_self', '')
            if ',' in args:
                log.error('rewrite doesn\'t support multiple args for torch.stack')
                assert False
            if len(args) > 0:
                if '=' in args:
                    k, v = args.split('=')
                    assert k in ('axis', 'dim')
                    dim = int(v)
                    if k == 'axis':
                        node.module.args_template_no_self = node.module.args_template_no_self.replace('axis', 'dim')
                        node.module.args_template = node.module.args_template.replace('axis', 'dim')
                        node.module.update_args_string()
                else:
                    dim = int(args)
            else:
                dim = 0
            unique_prev_nodes = {n.unique_name: n for n in node.prev_nodes}.values()
            for n in unique_prev_nodes:
                shared_tensors = list(set(node.prev_tensors).intersection(set(n.next_tensors)))
                if len(shared_tensors) == 0:
                    log.debug('tensor rewrite already done, skipping')
                    continue
                for t in shared_tensors:
                    with override_current_trace_graph(graph):
                        trace_func = TraceFunction('torch.unsqueeze', prefix='fuse_').parse_args(t, dim)
                    next_tensors = [torch.unsqueeze(t, dim)]
                    graph.insert_between(n, node, trace_func, next_tensors, move_idx=True, tensor_ptrs=set([id(t)]))

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
                return (
                    cur_module.kind in ('add', 'mul', 'cat')
                    and torch.is_floating_point(node.next_tensors[0])
                    and self.layerwise_config.get(node.unique_name, True)
                )

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
                    if node.prev_nodes[1].prev_nodes[0].kind() not in ('shape', 'size'):
                        op_type = op_kind
                    else:
                        op_type = f'{op_kind}_scalar'
                elif prev_tensor_size == 1:
                    op_type = f'{op_kind}_scalar'
                else:
                    log.error(f'Unknown add/mul type for {node.unique_name}, prev tensor size: {prev_tensor_size}')
                    assert False
            else:
                # Don't check anything for other OPs.
                # It is simply too complex for us.
                op_type = op_kind
            node.module.func_type = op_type
            node.module.full_name = f'self.{module_name}.{op_type}'
            # Inplace operations
            if node.module.func_type in ['__iadd__', '__imul__', 'add_', 'mul_']:
                node.module.add_alias(node.module.tensor_names[0])
                q = queue.Queue()
                q.put(node.prev_nodes[0])
                while not q.empty():
                    n = q.get()
                    if type(n.module) == TraceFunction:
                        prev_aliases = n.module.get_aliases()
                        if prev_aliases is not None:
                            for pa in reversed(prev_aliases):
                                node.module.add_alias(pa, head=True)
                    else:
                        if getattr(n.module, 'inplace', False):
                            q.put(n.prev_nodes[0])
                            node.module.add_alias(n.prev_node_unique_name(0), head=True)

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

        # Rewrite torch.clamp_{min,max} to torch.clamp
        def _is_clamp_min_max_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return (
                    cur_module.kind in ('clamp_min', 'clamp_max')
                    and torch.is_floating_point(node.next_tensors[0])
                    and self.layerwise_config.get(node.unique_name, True)
                )

        clamp_min_max_nodes = graph.filter_forward_nodes(_is_clamp_min_max_node)
        log.info(f'rewriting clamp_{{min,max}} for {[node.unique_name for node in convertible_nodes]}')
        for node in clamp_min_max_nodes:
            kw = node.module.kind[-3:]
            _parse_args = eval(f'lambda {kw}, *args, **kwargs: {kw}')

            val = eval(f'_parse_args({node.module.args_string_no_self})')

            if kw != 'min' or val != 0.0:
                continue

            if kw == 'min':
                arg_str = f'{val}'
            else:
                arg_str = f'None, {val}'

            sub_str = f'_{kw}'
            node.module.kind = node.module.kind.replace(sub_str, '')
            node.module.func_type = node.module.func_type.replace(sub_str, '')
            node.module.full_name = '.'.join(node.module.full_name.split('.')[:-1] + [node.module.func_type])

            node.module.args_template_no_self = arg_str
            node.module.args_template = f'{{}}, {arg_str}'
            node.module.update_args_string()

        # Rewrite torch.clamp to relu, relu6 and clamp_{min, max}
        def _is_clamp_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return (
                    cur_module.kind == 'clamp'
                    and torch.is_floating_point(node.next_tensors[0])
                    and self.layerwise_config.get(node.unique_name, True)
                )

        clamp_nodes = graph.filter_forward_nodes(_is_clamp_node)
        log.info(f'rewriting clamp for {[node.unique_name for node in convertible_nodes]}')
        for node in clamp_nodes:

            def _parse_args(min=None, max=None, *args, **kwargs):
                return min, max

            min, max = eval(f'_parse_args({node.module.args_string_no_self})')

            kind = None
            if min == 0.0:
                if max is None:
                    kind = 'relu'
                elif max == 6.0:
                    kind = 'relu6'

            if kind is None:
                if max is None:
                    kind = 'clamp_min'
                elif min is None:
                    kind = 'clamp_max'
                else:
                    kind = 'clamp_with_fusion'

            if kind in ('clamp_min', 'clamp_max'):
                node.module.kind = kind
                node.module.func_type = node.module.func_type.replace('clamp', kind)
                node.module.full_name = '.'.join(node.module.full_name.split('.')[:-1] + [node.module.func_type])

                if max is None:
                    arg_str = f'{min}'
                else:
                    arg_str = f'{max}'
            elif kind == 'clamp_with_fusion':
                node.module.kind = kind
                node.module.func_type = node.module.func_type.replace('clamp', kind)
                node.module.full_name = f'tinynn.graph.quantization.utils.{node.module.func_type}'

                arg_str = f'{min}, {max}'
            else:
                inplace = node.module.func_type == f'{node.module.kind}_'
                node.module.kind = kind
                node.module.func_type = kind
                node.module.full_name = f'torch.nn.functional.{kind}'

                if inplace:
                    arg_str = 'inplace=True'
                else:
                    arg_str = ''

            node.module.args_template_no_self = arg_str
            node.module.args_template = f'{{}}, {arg_str}'
            node.module.update_args_string()

        # Rewrite other fusable functions
        # e.g. add_relu(x, y) =>
        #            r = torch.add(x, y)
        #            r = torch.nn.functional.relu(r)
        def _is_add_relu_fusable_node(node: TraceNode, custom_data) -> bool:
            cur_module = node.module
            cur_class = type(cur_module)
            visited_nodes = [node]

            if self.layerwise_config.get(node.unique_name, True) is False:
                return False

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
                    fuse = next_module.kind == 'relu'
                else:
                    while next_class == nn.Identity:
                        cur_node = next_node
                        visited_nodes.append(cur_node)
                        if len(cur_node.next_nodes) != 1:
                            return False
                        next_node = cur_node.next_nodes[0]
                        next_module = next_node.module
                        next_class = type(next_module)

                    fuse = next_class.__name__ == 'ReLU'

                if not fuse:
                    return False

                if type(next_node.module) == TraceFunction:
                    inplace = next_node.module.func_type == 'relu_' or 'True' in next_node.module.args_string
                else:
                    inplace = getattr(next_node.module, 'inplace', False)

                # Inplace check
                # If add is inplace and relu is not inplace, we need to ensure that all the aliases of
                # the first operand of add are not used when relu is called.
                if not inplace:
                    aliases = cur_module.get_aliases()
                    if aliases:
                        q = queue.Queue()
                        q.put(node.prev_nodes[0])
                        while not q.empty():
                            n = q.get()
                            last_order = max((x.forward_order for x in n.next_nodes))
                            if last_order > node.forward_order:
                                fuse = False
                                break
                            if type(n.module) == TraceFunction and n.module.get_aliases():
                                q.put(n.prev_nodes[0])
                            elif getattr(n.module, 'inplace', False):
                                q.put(n.prev_nodes[0])

                return fuse

        add_relu_fusable_nodes = graph.filter_forward_nodes(_is_add_relu_fusable_node)
        for node in add_relu_fusable_nodes:
            full_name = node.module.full_name.replace('add', 'add_relu')
            next_node = node.next_nodes[0]
            kind = 'add_relu'
            func_type = kind
            is_class = False
            nodes_to_fuse = [node, next_node]
            while next_node.type() == nn.Identity:
                next_node = next_node.next_nodes[0]
                nodes_to_fuse.append(next_node)
            if type(next_node.module) == TraceFunction:
                inplace = next_node.module.func_type == 'relu_' or 'True' in next_node.module.args_string
            else:
                inplace = next_node.module.inplace
            graph.fuse_nodes_to_func(nodes_to_fuse, full_name, kind, func_type, is_class)
            # Propagate aliases for inplace nodes
            if inplace:
                aliases = node.module.get_aliases()
                if aliases:
                    node.module.add_alias(node.module.tensor_names[0])
            else:
                node.module.aliases = None

        # Rewrite relu, relu6 as nn.ReLU() and nn.ReLU6() for Module fusable rules
        def _is_functional_rewrite_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind in FUNCTIONAL_MODULE_MAPPING and self.layerwise_config.get(
                    node.unique_name, True
                )

        func_nodes_to_rewrite = graph.filter_forward_nodes(_is_functional_rewrite_node)
        log.info(f'rewriting functional to module for {[node.unique_name for node in func_nodes_to_rewrite]}')
        for idx, node in enumerate(func_nodes_to_rewrite):
            kind = node.module.kind
            inplace = node.module.func_type == f'{kind}_' or 'True' in node.module.args_string
            klass = FUNCTIONAL_MODULE_MAPPING[kind]
            arguments = getattr(klass, '__constants__', None)
            if arguments is None:
                new_func = klass()
            elif node.module.kind in ('relu', 'relu6', 'silu', 'hardswish'):
                new_func = klass(inplace=inplace)
            elif node.module.kind in ('elu', 'leaky_relu'):
                if hasattr(node.module, 'args_string_no_self'):

                    def _parse_args(alpha=1.0, *args, **kwargs):  # noqa: F811
                        return alpha

                    alpha = eval(f'_parse_args({node.module.args_string_no_self})')
                    if node.module.kind == 'leaky_relu':
                        new_func = klass(alpha, inplace=inplace)
                    else:
                        new_func = klass(alpha, inplace=inplace)
                else:
                    alpha = None
                    if node.module.kind == 'leaky_relu':
                        new_func = klass(inplace=inplace)
                    else:
                        new_func = klass()
            elif node.module.kind == 'prelu':
                weight_t = node.prev_tensors[1]
                weight_node = node.prev_nodes[1]
                if weight_node.type() == torch_q.QuantStub:
                    weight_node = weight_node.prev_nodes[0]

                if weight_node.type() != ConstantNode or not weight_node.module.is_parameter:
                    log.warning('Rewrite for F.prelu(x, buffer) to nn.PReLU is skipped as it changes the semantics')
                    continue

                num_parameters = weight_t.nelement()
                new_func = klass(num_parameters)
                new_func.weight.data.copy_(node.prev_tensors[1])

                # Drop last input
                node.prev_tensors.pop()
                last_node = node.prev_nodes.pop()
                last_node.next_nodes.remove(node)

                # Remove unused forward functions iteratively
                new_last_node = last_node
                while new_last_node is not None:
                    last_node = new_last_node
                    if (
                        len(last_node.next_nodes) == 0
                        and len(last_node.prev_nodes) < 2
                        and last_node in graph.forward_nodes
                    ):
                        if len(last_node.prev_nodes) == 1:
                            new_last_node = last_node.prev_nodes[0]
                        else:
                            new_last_node = None
                        graph.remove_node(last_node)
                    else:
                        new_last_node = None
            elif node.module.kind == 'glu':
                if hasattr(node.module, 'args_string_no_self'):

                    def _parse_args_dim(dim=-1, *args, **kwargs):  # noqa: F811
                        return dim

                    dim = eval(f'_parse_args_dim({node.module.args_string_no_self})')
                    new_func = klass(dim)
                else:
                    dim = None
                    new_func = klass()
            else:
                raise NotImplementedError(f"Don't know how to parse {klass.__name__} with argument {arguments}")

            graph.module_unique_name_dict[id(new_func)] = f'rewritten_{kind}_{idx}'
            graph.module_original_name_dict[id(new_func)] = f'rewritten_{kind}_{idx}'

            relu_cls = type(new_func)
            if inplace:
                arg_str = 'inplace=True'
            else:
                arg_str = ''

            if node.module.kind in ('elu', 'leaky_relu') and alpha is not None:
                if arg_str:
                    arg_str = f'{alpha}, {arg_str}'
                else:
                    arg_str = f'{alpha}'

            if node.module.kind == 'prelu':
                if num_parameters != 1:
                    arg_str = f'{num_parameters}'
                else:
                    arg_str = ''

            if node.module.kind == 'glu':
                if dim is not None and dim != -1:
                    arg_str = f'{dim}'
                else:
                    arg_str = ''

            module_constructor_lines[id(new_func)] = f'{qualified_name(relu_cls)}({arg_str})'
            graph.replace_node_module(node, new_func)

        # Rewrite dropout as nn.Dropout() for models in training mode
        def _is_dropout_functional_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'dropout' and self.layerwise_config.get(node.unique_name, True)

        def _dropout_args(p=0.5, training=True, inplace=False):
            return p, training, inplace

        dropout_nodes_to_rewrite = graph.filter_forward_nodes(_is_dropout_functional_node)
        log.info(f'rewriting dropout for {[node.unique_name for node in dropout_nodes_to_rewrite]}')
        for idx, node in enumerate(dropout_nodes_to_rewrite):
            args = getattr(node.module, 'args_string_no_self', '')
            p, _, inplace = eval(f'_dropout_args({args})')
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
            if self.layerwise_config.get(node.unique_name, True) is False:
                return False
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == ConstantNode:
                return False
            elif cur_class == TraceFunction:
                return cur_module.kind in ('pad',)
            else:
                return cur_class in (nn.ConstantPad1d, nn.ConstantPad2d, nn.ConstantPad3d, nn.ZeroPad2d)

        if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
            partially_supported_nodes = graph.filter_forward_nodes(_is_partially_quantizable)
            for idx, node in enumerate(partially_supported_nodes):
                for n in node.prev_nodes[:1]:
                    shared_tensors = list(set(node.prev_tensors).intersection(set(n.next_tensors)))
                    if len(shared_tensors) > 1:
                        log.error('rewrite for partially-supported ops supports with nodes with exact one input')
                        assert False
                    with override_current_trace_graph(graph):
                        trace_func = TraceFunction('torch.Tensor.contiguous', True, prefix='fuse_').parse_args(
                            shared_tensors[0]
                        )
                    next_tensors = [x.contiguous() for x in shared_tensors]
                    graph.insert_between(n, node, trace_func, next_tensors)

        # Remove non-leaf `.data` nodes
        def _is_non_leaf_data_nodes(node, custom_data):
            if self.layerwise_config.get(node.unique_name, True) is False:
                return False
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return cur_module.kind == 'data' and cur_module.is_property and len(node.next_nodes) > 0
            return False

        non_leaf_data_nodes = graph.filter_forward_nodes(_is_non_leaf_data_nodes)
        for idx, node in enumerate(non_leaf_data_nodes):
            graph.remove_node(node)

        # Handle PoolNd with kernel_size=1
        def _is_pool_nd_with_one_kernel_size(node, custom_data):
            if self.layerwise_config.get(node.unique_name, True) is False:
                return False
            cur_module = node.module
            cur_class = type(cur_module)
            kernel_size, stride = None, None
            if cur_class == TraceFunction:
                if cur_module.kind in ('avg_pool1d', 'avg_pool2d', 'max_pool1d', 'max_pool2d'):

                    def _avgpool_kernel_size_and_stride(kernel_size, stride=None, *args, **kwargs):
                        return kernel_size, stride

                    kernel_size, stride = eval(f'_avgpool_kernel_size_and_stride({cur_module.args_string_no_self})')
            else:
                if cur_class in (nn.AvgPool1d, nn.AvgPool2d, nn.MaxPool1d, nn.MaxPool2d):
                    kernel_size = cur_module.kernel_size
                    stride = cur_module.stride

            if kernel_size is not None:
                if isinstance(kernel_size, (tuple, list)):
                    is_match = all((ks == 1 for ks in kernel_size))
                else:
                    is_match = kernel_size == 1
            else:
                is_match = False

            if is_match:
                custom_data.append((node, kernel_size, stride))
                return True
            else:
                return False

        pool_one_kernel_size_nodes = []
        graph.filter_forward_nodes(_is_pool_nd_with_one_kernel_size, pool_one_kernel_size_nodes)
        for idx, (node, kernel_size, stride) in enumerate(pool_one_kernel_size_nodes):
            slices = [slice(None)] * 2
            t = node.prev_tensors[0]
            dim = len(t.shape)
            if not isinstance(stride, (list, tuple)):
                stride = [stride] * (dim - 2)
            for s in stride:
                if s == 1 or s is None:
                    slices.append(slice(None))
                else:
                    slices.append(slice(None, None, s))

            with override_current_trace_graph(graph):
                new_func = TraceFunction('torch.Tensor.__getitem__', True).parse_args(t, slices)

                graph.module_unique_name_dict[id(new_func)] = f'rewritten_pool_{idx}'
                graph.module_original_name_dict[id(new_func)] = f'rewritten_pool_{idx}'

                graph.replace_node_module(node, new_func)

        # Rewrite Linear-BatchNorm1d structure to Conv2d-BatchNorm2d
        is_rewrite_to_fuse = functools.partial(
            self.is_fusable,
            current_rules=load_processed_rewrite_to_fuse_rules(),
            check_node_quantized=False,
            graph=graph,
            layerwise_config_default=True,
        )
        custom_data = ([], set())
        graph.filter_forward_nodes(is_rewrite_to_fuse, custom_data, reverse=True)
        rewrite_fuse_names_list = custom_data[0]
        log.debug(f'found names_list that need to rewrite for fusing: {rewrite_fuse_names_list}')

        for idx, names in enumerate(reversed(rewrite_fuse_names_list)):
            # case fc-bn1d
            assert len(names) == 2, 'the rewrite nodes list length != 2'

            node_fc = graph.nodes_map[names[0]]
            node_bn1d = graph.nodes_map[names[1]]
            mod_fc = node_fc.module
            mod_bn = node_bn1d.module

            assert type(mod_fc) == nn.Linear and type(mod_bn) == nn.BatchNorm1d, "the rewrite struct is\'t [fc-bn1d]"

            if len(node_fc.prev_tensors[0].shape) != 2:
                log.debug('the [fc-bn]\'s input dimension != 2')
                continue
            # for fc-bn1d, rewrite [fc-bn1d] to [conv2d-bn2d]
            new_conv2d = torch.nn.Conv2d(
                in_channels=mod_fc.in_features,
                out_channels=mod_fc.out_features,
                kernel_size=[1, 1],
                bias=mod_fc.bias is not None,
            )
            fc_weight = mod_fc.weight
            new_conv2d.weight = nn.Parameter(torch.reshape(fc_weight, [fc_weight.shape[0], fc_weight.shape[1], 1, 1]))
            if mod_fc.bias is not None:
                new_conv2d.bias = mod_fc.bias
            graph.module_unique_name_dict[id(new_conv2d)] = f'rewritten_conv2d_bn2d_conv2d_{idx}'
            graph.module_original_name_dict[id(new_conv2d)] = f'rewritten_conv2d_bn2d_conv2d_{idx}'

            new_bn2d = torch.nn.BatchNorm2d(
                mod_bn.num_features,
                mod_bn.eps,
                mod_bn.momentum,
                affine=mod_bn.affine,
                track_running_stats=mod_bn.track_running_stats,
            )
            new_bn2d.load_state_dict(mod_bn.state_dict())
            graph.module_unique_name_dict[id(new_bn2d)] = f'rewritten_conv2d_bn2d_bn2d_{idx}'
            graph.module_original_name_dict[id(new_bn2d)] = f'rewritten_conv2d_bn2d_bn2d_{idx}'

            # replace new node, then insert reshape before new_conv2d and after new_bn2d
            with override_current_trace_graph(graph):
                graph.replace_node_module(node_fc, new_conv2d)
                graph.replace_node_module(node_bn1d, new_bn2d)

                prev_tensor_shape = node_fc.prev_tensors[0].shape
                prev_func = TraceFunction('torch.reshape').parse_args(
                    node_fc.prev_tensors[0], [prev_tensor_shape[0], prev_tensor_shape[1], 1, 1]
                )
                next_tensor_shape = node_bn1d.next_tensors[0].shape
                next_func = TraceFunction('torch.reshape').parse_args(
                    node_bn1d.next_tensors[0], [next_tensor_shape[0], next_tensor_shape[1]]
                )
            # expand the tensor shape between fc new_conv2d and new_bn2d
            node_fc.next_tensors[0].unsqueeze_(2).unsqueeze_(2)
            node_bn1d.prev_tensors[0].unsqueeze_(2).unsqueeze_(2)
            node_bn1d.next_tensors[0].unsqueeze_(2).unsqueeze_(2)

            prev_out = torch.reshape(
                node_fc.prev_tensors[0],
                [node_fc.prev_tensors[0].shape[0], node_fc.prev_tensors[0].shape[1], 1, 1],
            )
            graph.insert_between(node_fc.prev_nodes[0], node_fc, prev_func, [prev_out])
            next_out = torch.reshape(
                node_bn1d.next_tensors[0],
                [node_bn1d.next_tensors[0].shape[0], node_bn1d.prev_tensors[0].shape[1]],
            )
            graph.insert_after(node_bn1d, next_func, [next_out])

        # Rewrite BatchNorm1d to BatchNorm2d
        def _is_batch_norm_1d(node, custom_data):
            if self.layerwise_config.get(node.unique_name, True) is False:
                return False
            cur_module = node.module
            cur_class = type(cur_module)
            if len(node.prev_nodes) != 1:
                return False

            if node.prev_nodes[0].kind() in ('conv1d', nn.Conv1d):
                return False

            if cur_class == TraceFunction:
                return cur_module.kind == 'batch_norm' and node.prev_tensors[0].ndim == 3
            else:
                return cur_class == nn.BatchNorm1d

        batch_norm_1d_nodes = graph.filter_forward_nodes(_is_batch_norm_1d)
        for idx, node in enumerate(batch_norm_1d_nodes):
            mod = node.module
            if type(mod) == nn.BatchNorm1d:
                new_bn = torch.nn.BatchNorm2d(
                    mod.num_features,
                    mod.eps,
                    mod.momentum,
                    affine=mod.affine,
                    track_running_stats=mod.track_running_stats,
                )
                new_bn.load_state_dict(mod.state_dict())

                graph.module_unique_name_dict[id(new_bn)] = f'rewritten_bn2d_{idx}'
                graph.module_original_name_dict[id(new_bn)] = f'rewritten_bn2d_{idx}'

                with override_current_trace_graph(graph):
                    graph.replace_node_module(node, new_bn)

                    prev_func = TraceFunction('torch.unsqueeze').parse_args(node.prev_tensors[0], 2)
                    next_func = TraceFunction('torch.squeeze').parse_args(node.next_tensors[0], 2)

                node.next_tensors[0].unsqueeze_(2)

                prev_out = torch.unsqueeze(node.prev_tensors[0], 2)
                graph.insert_between(node.prev_nodes[0], node, prev_func, [prev_out])
                next_out = torch.squeeze(node.next_tensors[0], 2)
                graph.insert_after(node, next_func, [next_out])

        type_dict = {}
        for n in graph.forward_nodes:
            if n.type() not in (torch_q.QuantStub, torch_q.DeQuantStub):
                self.layerwise_config.setdefault(n.unique_name, True)
                kind = n.kind()
                type_str = kind if isinstance(kind, str) else kind.__name__
                type_dict[n.unique_name] = type_str

        fuse_mapping = {}
        if self.fused_layerwise_config:
            # Find all fusable nodes
            if type(self).__name__ == 'QATQuantizer':
                processed_all_rules = load_processed_all_qat_rules()
            else:
                processed_all_rules = load_processed_all_ptq_rules()

            is_fusable = functools.partial(
                self.is_fusable,
                current_rules=processed_all_rules,
                check_node_quantized=False,
                use_original_name=False,
            )

            custom_data = ([], set())
            graph.filter_forward_nodes(is_fusable, custom_data, reverse=True)
            activ_names = custom_data[0]

            for idx, names in enumerate(reversed(activ_names)):
                types = [graph.nodes_map[n].kind() for n in names]
                types = [x.__name__ if not isinstance(x, str) else x for x in types]
                types_str = ', '.join(types)
                types_str = f'({types_str})'
                type_dict[names[0]] = types_str
                for name in names[1:]:
                    fuse_mapping[name] = names[0]
                    self.layerwise_config[name] = self.layerwise_config.get(names[0], True)

        for n, t in type_dict.items():
            self.layerwise_config.yaml_add_eol_comment(f'type: {t}', n)

        skip_types = set(k[0] for k in REWRITE_QUANTIZABLE_RULE_LIST if len(k) == 1)
        if self.set_quantizable_op_stats:
            skip_types |= set(KNOWN_QSTATS.keys())
        skip_types_prev = skip_types | set(k[-1] for k in REWRITE_QUANTIZABLE_RULE_LIST if len(k) > 1)
        skip_types_next = skip_types | set(k[0] for k in REWRITE_QUANTIZABLE_RULE_LIST if len(k) > 1)

        # Add quant/dequant nodes for non-quantizable OPs
        def _is_not_quantizable(node, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == ConstantNode:
                return False
            elif cur_class == TraceFunction:
                if node.type() in ('__truediv__', '__itruediv__', 'div', 'div_'):
                    if node.prev_nodes[0].kind() in ('shape', 'size'):
                        return False
                if node.type() in ('shape', 'device', 'size', 'dtype'):
                    if node.unique_name in self.layerwise_config:
                        del self.layerwise_config[node.unique_name]
                    return False
                if self.layerwise_config.get(node.unique_name, True) is False:
                    return True
                supported_version = UNSUPPORTED_PYTORCH_QUANTIZATION_OP_LIST.get(cur_module.kind, torch.__version__)
                return supported_version is None or LooseVersion(torch.__version__) < supported_version
            else:
                if isinstance(cur_module, (torch_q.QuantStub, torch_q.DeQuantStub)):
                    return False
                if self.layerwise_config.get(node.unique_name, True) is False:
                    return True
                unsupported_types = tuple(
                    k
                    for k, v in UNSUPPORTED_PYTORCH_QUANTIZATION_OP_LIST.items()
                    if type(k) != str
                    and k not in Q_MODULES_MAPPING
                    and (v is None or LooseVersion(torch.__version__) < v)
                )
                return isinstance(cur_module, unsupported_types)

        unsupported_nodes = graph.filter_forward_nodes(_is_not_quantizable)
        for idx, node in enumerate(reversed(unsupported_nodes)):
            if node.unique_name in self.layerwise_config:
                if node.kind() not in skip_types and self.layerwise_config[node.unique_name]:
                    del self.layerwise_config[node.unique_name]

            node_map = dict()
            next_nodes = {n.unique_name: n for n in node.next_nodes}.values()
            for inner_idx, next_node in enumerate(next_nodes):
                prev_tensor_ptrs = []
                if type(next_node.module) == TraceFunction and next_node.module.is_property:
                    continue

                for pt in next_node.prev_tensors:
                    for nt in node.next_tensors:
                        if isinstance(nt, (list, tuple)):
                            for k, ntt in enumerate(nt):
                                if id(pt) == id(ntt):
                                    if id(pt) not in prev_tensor_ptrs:
                                        prev_tensor_ptrs.append(id(pt))
                                    break
                        elif id(pt) == id(nt):
                            if id(pt) not in prev_tensor_ptrs:
                                prev_tensor_ptrs.append(id(pt))
                            break

                for ptr_idx, ptr in enumerate(prev_tensor_ptrs):
                    if ptr in node_map:
                        fake_quant = node_map[ptr]

                        graph.insert_between(node, next_node, fake_quant, move_idx=True, tensor_ptrs=[ptr])
                    else:
                        fake_quant = torch_q.QuantStub()

                        fake_quant_name = f'fake_quant_inner_{idx}_{inner_idx}_{ptr_idx}'

                        graph.module_unique_name_dict[id(fake_quant)] = fake_quant_name
                        graph.module_original_name_dict[id(fake_quant)] = fake_quant_name

                        fake_quant_cls = type(fake_quant)
                        module_constructor_lines[id(fake_quant)] = f'{qualified_name(fake_quant_cls)}()'

                        graph.insert_between(node, next_node, fake_quant, move_idx=True, tensor_ptrs=[ptr])
                        node_map[ptr] = graph.nodes_map[fake_quant_name]

        # Insert the DeQuantStub nodes before every input node of the unsupported ops
        for idx, node in enumerate(unsupported_nodes):
            fake_dequant_cls = torch_q.DeQuantStub
            assert node.rev_index is False
            node_map = dict()
            prev_nodes = {n.unique_name: n for n in node.prev_nodes}.values()
            for inner_idx, prev_node in enumerate(prev_nodes):

                if prev_node.kind() in ('shape', 'device', 'size', 'dtype'):
                    continue

                prev_tensor_ptrs = []
                for pt in node.prev_tensors:
                    for nt in prev_node.next_tensors:
                        if isinstance(nt, (list, tuple)):
                            for k, ntt in enumerate(nt):
                                if id(pt) == id(ntt):
                                    if id(pt) not in prev_tensor_ptrs:
                                        prev_tensor_ptrs.append(id(pt))
                                    break
                        elif id(pt) == id(nt):
                            if id(pt) not in prev_tensor_ptrs:
                                prev_tensor_ptrs.append(id(pt))
                            break

                for ptr_idx, ptr in enumerate(prev_tensor_ptrs):
                    if ptr in node_map:
                        fake_dequant = node_map[ptr]

                        graph.insert_between(prev_node, node, fake_dequant, move_idx=True, tensor_ptrs=set([ptr]))
                    else:

                        fake_dequant = fake_dequant_cls()

                        fake_dequant_name = f'fake_dequant_inner_{idx}_{inner_idx}_{ptr_idx}'

                        graph.module_unique_name_dict[id(fake_dequant)] = fake_dequant_name
                        graph.module_original_name_dict[id(fake_dequant)] = fake_dequant_name

                        module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'

                        graph.insert_between(prev_node, node, fake_dequant, move_idx=True, tensor_ptrs=set([ptr]))
                        node_map[ptr] = graph.nodes_map[fake_dequant_name]

        # Remove consecutive dequant quant nodes
        def _is_consecutive_dequant_quant_nodes(node, custom_data):
            cur_type = node.type()
            if cur_type in (torch_q.QuantStub, torch_q.DeQuantStub):
                for next_node in node.next_nodes:
                    next_type = next_node.type()
                    if next_type in (torch_q.QuantStub, torch_q.DeQuantStub):
                        if cur_type != next_type:
                            if cur_type == torch_q.QuantStub:
                                target_node = None
                                if len(node.prev_nodes) == 1 and node.prev_nodes[0].kind() in skip_types_prev:
                                    target_node = node.prev_nodes[0]
                                elif (
                                    len(next_node.next_nodes) == 1 and next_node.next_nodes[0].kind() in skip_types_next
                                ):
                                    target_node = next_node.next_nodes[0]
                                if target_node is not None:
                                    self.layerwise_config.setdefault(target_node.unique_name, True)
                                    if self.layerwise_config[target_node.unique_name]:
                                        return False
                            custom_data.append((node, next_node))
                            return True
            return False

        consecutive_dequant_quant_nodes = []
        consecutive_branch_dequant_quant_nodes = []
        graph.filter_forward_nodes(_is_consecutive_dequant_quant_nodes, consecutive_dequant_quant_nodes)
        for node, next_node in consecutive_dequant_quant_nodes:
            if len(node.next_nodes) == 1 and len(next_node.prev_nodes) == 1:
                graph.remove_node(next_node)
                graph.remove_node(node)
            else:
                consecutive_branch_dequant_quant_nodes.append((node, next_node))

        # TODO: Support complex cases
        for node, next_node in consecutive_branch_dequant_quant_nodes:
            is_removable = all(
                (
                    n.type() in (torch_q.QuantStub, torch_q.DeQuantStub, 'shape', 'device', 'size', 'dtype')
                    and n.type() != node.type()
                    for n in node.next_nodes
                )
            )
            if is_removable:
                graph.remove_node(node)
                for n in node.next_nodes:
                    if n.type() in (torch_q.QuantStub, torch_q.DeQuantStub):
                        graph.remove_node(n)

        # Process additional fusable nodes
        processed_extra_q_rules = load_processed_extra_q_rules()
        is_extra_fusable = functools.partial(
            self.is_fusable,
            current_rules=processed_extra_q_rules,
            check_node_quantized=False,
            use_original_name=False,
        )

        custom_data = ([], set())
        graph.filter_forward_nodes(is_extra_fusable, custom_data, reverse=True)
        activ_names = custom_data[0]
        log.debug(f'found nodes that cannot fuse: {activ_names}')

        for idx, names in enumerate(reversed(activ_names)):
            name = names[-1]
            node = graph.nodes_map[name]
            fake_quant = torch_q.QuantStub()

            graph.module_unique_name_dict[id(fake_quant)] = f'fake_activ_quant_{idx}'
            graph.module_original_name_dict[id(fake_quant)] = f'fake_activ_quant_{idx}'

            fake_quant_cls = type(fake_quant)
            module_constructor_lines[id(fake_quant)] = f'{qualified_name(fake_quant_cls)}()'

            graph.insert_after(node, fake_quant)

            fake_dequant = torch_q.DeQuantStub()

            graph.module_unique_name_dict[id(fake_dequant)] = f'fake_activ_dequant_{idx}'
            graph.module_original_name_dict[id(fake_dequant)] = f'fake_activ_dequant_{idx}'

            fake_dequant_cls = type(fake_dequant)
            module_constructor_lines[id(fake_dequant)] = f'{qualified_name(fake_dequant_cls)}()'

            graph.insert_after(node, fake_dequant)

        fused_clamps = []
        for names in activ_names:
            for name in names:
                node = graph.nodes_map[name]
                if node.kind() == 'clamp_with_fusion':
                    fused_clamps.append(name)

        # Rewrite unfused clamp_with_fusion to torch.clamp
        def _is_unfused_clamp_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return (
                    cur_module.kind == 'clamp_with_fusion'
                    and torch.is_floating_point(node.next_tensors[0])
                    and node.unique_name not in fused_clamps
                )
            return False

        unfused_clamp_nodes = graph.filter_forward_nodes(_is_unfused_clamp_node)
        log.info(f'rewriting unfused_clamp for {[node.unique_name for node in unfused_clamp_nodes]}')
        for node in unfused_clamp_nodes:
            node.module.kind = kind
            node.module.func_type = node.module.func_type.replace('clamp_with_fusion', 'clamp')
            node.module.full_name = f'torch.{node.module.func_type}'

        # Optional tensor shape broadcasting for quantized binary ops
        def _is_broadcastable_binary_quantized_op_node(node: TraceNode, custom_data) -> bool:
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class != TraceFunction:
                return False
            return (
                cur_module.full_name.startswith('self.float_functional_simple_')
                and cur_module.func_type in ('add', 'mul', 'add_relu')
                and node.prev_tensors[0].shape != node.prev_tensors[1].shape
            )

        broadcastable_binary_quantized_op_nodes = graph.filter_forward_nodes(_is_broadcastable_binary_quantized_op_node)
        for node in broadcastable_binary_quantized_op_nodes:
            assert len(node.prev_nodes) == 2
            assert len(node.prev_tensors) == 2

            l_shape = node.prev_tensors[0].shape
            r_shape = node.prev_tensors[1].shape

            if len(l_shape) > len(r_shape):
                ref_index = 0
            elif len(l_shape) < len(r_shape):
                ref_index = 1
            else:
                for l_dim, r_dim in zip(l_shape, r_shape):
                    if l_dim > r_dim:
                        ref_index = 0
                        break
                    elif l_dim < r_dim:
                        ref_index = 1
                        break

            src_index = 1 - ref_index

            with override_current_trace_graph(graph):
                trace_func = TraceFunction('torch.Tensor.expand_as', True, prefix='fuse_').parse_args(
                    node.prev_tensors[src_index], node.prev_tensors[ref_index]
                )
            next_tensors = [node.prev_tensors[src_index].expand_as(node.prev_tensors[ref_index])]
            graph.insert_between(node.prev_nodes[src_index], node, trace_func, next_tensors)

            new_node = graph.nodes_map[trace_func.unique_name]
            new_node.prev_nodes.append(node.prev_nodes[ref_index])
            new_node.prev_tensors.append(node.prev_tensors[ref_index])
            new_node.prev_indices.append(node.prev_indices[ref_index])
            node.prev_nodes[ref_index].next_nodes.append(new_node)

        for name in fuse_mapping:
            if name in self.layerwise_config:
                del self.layerwise_config[name]

        for name in self.layerwise_config:
            node = graph.nodes_map[name]
            if isinstance(node.module, nn.Module) and not isinstance(
                node.module,
                (nn.Dropout, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d, nn.ReLU, nn.Upsample, nn.ConstantPad2d),
            ):
                self.effective_layers.append(name)
            elif node.kind() in ('add', 'mul', 'cat') or node.kind() in FUNCTIONAL_MODULE_MAPPING:
                self.effective_layers.append(name)

        graph.quantized = True
        graph.recompute_forward_order()

    def is_fusable(
        self,
        node,
        custom_data,
        current_rules=None,
        check_node_quantized=True,
        use_original_name=True,
        graph=None,
        layerwise_config_default=False,
    ):
        # Tell whether a TraceNode is fusable with some nearby nodes

        if current_rules is None:
            return False

        if check_node_quantized:
            if not node.quantized:
                return False
        else:
            if self.layerwise_config.get(node.unique_name, layerwise_config_default) is False:
                return False

        cur_node = node
        names = []
        final_names = []
        current_state = False
        while True:
            cur_module = cur_node.module
            cur_class = type(cur_module)
            if isinstance(cur_module, TraceFunction):
                cur_class = cur_module.kind
            prev_nodes = cur_node.prev_nodes
            log.debug(f'cur: {cur_class}')
            if cur_class in current_rules or cur_class == nn.Identity:
                if use_original_name and isinstance(cur_module, nn.Module):
                    cur_name = graph.module_original_name_dict[id(cur_module)]
                else:
                    cur_name = cur_node.unique_name
                if cur_name in custom_data[1]:
                    log.debug('found existing nodes, skipping')
                    break
                if len(prev_nodes) == 0:
                    break
                if cur_class in current_rules:
                    current_state, current_rules = current_rules[cur_class]
                    log.debug('dict: ', current_rules, current_state)
                    names.append(cur_name)
                    if current_state is True:
                        log.debug(f'update best: {names}')
                        final_names.clear()
                        final_names.extend(names)
                if len(cur_node.prev_nodes) != 1:
                    break
                cur_node = cur_node.prev_nodes[0]
            else:
                if cur_class == nn.ConvTranspose2d:
                    if len(final_names) > 0:
                        next_types = []
                        for _ in range(len(final_names)):
                            next_node = cur_node.next_nodes[0]
                            next_types.append(next_node.kind())
                            cur_node = next_node
                        if tuple(next_types) == (nn.BatchNorm2d, nn.ReLU):
                            final_names.clear()
                break

        if len(final_names) > 0:
            final_names.reverse()
            log.debug('final:', final_names)
            custom_data[0].append(final_names)
            for name in final_names:
                custom_data[1].add(name)
            return True

        return False

    def error_analysis(self, qat_model: nn.Module, dummy_input, threshold: float = 20.0):
        """Generates the QAT error report using the SQNR metric

        Args:
            qat_model: The QAT model
            dummy_input: A viable input to the model
            threshold (float): The threshold of SQNR. Defaults to 20.0
        """

        if isinstance(qat_model, DataParallel) or isinstance(qat_model, DistributedDataParallel):
            model = qat_model.module
        else:
            model = qat_model

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
            log.warning('No FakeQuantize modules found. Are you sure you passed in a QAT model?')
            return

        model.apply(torch.quantization.disable_fake_quant)
        model.apply(torch.quantization.disable_observer)

        device = get_module_device(model)

        if type(dummy_input) == torch.Tensor:
            actual_input = [dummy_input]
        elif isinstance(dummy_input, (tuple, list)):
            actual_input = list(dummy_input)
        else:
            log.error(f'Unsupported type {type(dummy_input)} for dummy input')
            assert False

        for i in range(len(actual_input)):
            dummy_input = actual_input[i]
            if type(dummy_input) == torch.Tensor:
                if dummy_input.device != device:
                    actual_input[i] = dummy_input.to(device)

        model.eval()

        with torch.no_grad():
            model(*actual_input)

        for h in hooks:
            h.remove()
        hooks.clear()

        for m, v in fake_quant_enabled_dict.items():
            m.fake_quant_enabled = v

        def sqnr(x, y):
            Ps = torch.norm(x)
            Pn = torch.norm(x - y)
            return (20 * torch.log10(Ps / Pn)).item()

        q_errors_weight = []
        q_errors_activ = []
        while len(float_results) > 0:
            n, f = float_results.popitem()
            mod = modules_list[n]
            with torch.no_grad():
                q = mod(*f)
                loss = sqnr(f[0], q)
            actual_n = '.'.join(n.split('.')[:-1])
            if loss <= threshold:
                if n.endswith('.weight_fake_quant'):
                    q_errors_weight.append((actual_n, mod, loss))
                else:
                    q_errors_activ.append((actual_n, mod, loss))

        q_errors_weight = sorted(q_errors_weight, key=lambda x: x[2])
        q_errors_activ = sorted(q_errors_activ, key=lambda x: x[2])

        logs = []
        if len(q_errors_weight) > 0:
            logs.append('')
            logs.append(f'Weights (SQNR <= {threshold}):')
            for n, m, e in q_errors_weight:
                logs.append(f'{n} SQNR: {e:.4f}, scale: {m.scale.item():.4f}, zero_point: {m.zero_point.item()}')

        if len(q_errors_activ) > 0:
            logs.append('')
            logs.append(f'Activations (SQNR <= {threshold}):')
            for n, m, e in q_errors_activ:
                logs.append(f'{n} SQNR: {e:.4f}, scale: {m.scale.item():.4f}, zero_point: {m.zero_point.item()}')

        if len(q_errors_weight) == 0 and len(q_errors_activ) == 0:
            logs.append('')
            logs.append('All good!')

        if len(logs) > 0:
            logs.insert(0, 'Quantization error report:')
            logs.append('')

            full_log = '\n'.join(logs)
            log.warning(full_log)

        for m, v in observer_enabled_dict.items():
            m.observer_enabled = v

        model.train()

    def restore_to_original(self, q_model: nn.Module):
        """Restores a QAT/PTQ-converted model to original state

        Args:
            qat_model: The QAT/PTQ-converted model

        """

        sub_list = []
        for n, m in q_model.named_children():
            if isinstance(m, (torch.nn.quantized.Quantize, torch.nn.quantized.DeQuantize)):
                sub_list.append((n, nn.Identity()))
            elif isinstance(m, torch.nn.quantized.Linear):
                fc = nn.Linear(m.in_features, m.out_features, m.bias is not None)
                fc.weight = torch.nn.Parameter(m.weight().dequantize())
                fc.bias = torch.nn.Parameter(m.bias())
                sub_list.append((n, fc))
            elif isinstance(m, torch.nn.quantized.Conv2d):
                conv = nn.Conv2d(
                    m.in_channels,
                    m.out_channels,
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    m.dilation,
                    m.groups,
                    m.bias is not None,
                    m.padding_mode,
                )
                conv.weight = torch.nn.Parameter(m.weight().dequantize())
                conv.bias = torch.nn.Parameter(m.bias())
                sub_list.append((n, conv))
            elif isinstance(m, torch.nn.quantized.QFunctional):
                sub_list.append((n, nn.quantized.FloatFunctional()))
            elif hasattr(torch.nn.quantized, 'ReLU') and isinstance(m, torch.nn.quantized.ReLU):
                sub_list.append((n, nn.ReLU()))
            elif isinstance(m, torch.nn.quantized.ReLU6):
                sub_list.append((n, nn.ReLU6()))
            elif type(m).__module__.startswith('torch.nn.modules'):
                continue
            else:
                assert False, f"unsupported type: {type(m).__name__}"

        for n, m in sub_list:
            setattr(q_model, n, m)

    def convert(self, q_model: nn.Module, backend: str = 'tflite') -> nn.Module:
        """Converts a QAT/PTQ-prepared model to an actual quantized model

        Args:
            q_model (nn.Module): The QAT/PTQ-prepared model
            backend (str): The backend to translate for, including `pytorch` and `tflite`. Defaults to `tflite`

        Returns:
            nn.Module: The QAT/PTQ-converted model. When the backend is set to `pytorch`, it is used for validation \
                in PyTorch only.
        """

        for acp, post_acp, dq_name, q_name, activ_name, activ_type in self.extra_qparams_mappings:
            if backend != 'pytorch' and activ_type in ('relu', 'relu6', torch.nn.ReLU, torch.nn.ReLU6):
                continue

            acp.scale = post_acp.scale
            acp.zero_point = post_acp.zero_point
            acp.activation_post_process.min_val = post_acp.activation_post_process.min_val
            acp.activation_post_process.max_val = post_acp.activation_post_process.max_val

            setattr(q_model, dq_name, nn.Identity())
            setattr(q_model, q_name, nn.Identity())
            if activ_name is not None:
                setattr(q_model, activ_name, nn.Identity())

        if type(self).__name__ == 'QATQuantizer':

            q = queue.Queue()
            q.put(q_model)

            while not q.empty():
                m = q.get()
                for n, c in m.named_children():
                    if isinstance(c, ConvTransposeBn2d):
                        setattr(m, n, c.transform(c))
                    else:
                        q.put(c)

            if hasattr(torch_q, 'get_default_static_quant_module_mappings'):
                mapping = torch_q.get_default_static_quant_module_mappings()
            elif hasattr(torch_q, 'get_static_quant_module_mappings'):
                mapping = copy.deepcopy(torch_q.get_static_quant_module_mappings())
            else:
                mapping = copy.deepcopy(torch_q.DEFAULT_MODULE_MAPPING)

            mapping.update(FUSE_QAT_MODULES_CVT)

            float_mods = {}

            for qat_t, q_t in FUSE_QAT_MODULES_CVT.items():
                float_mod = getattr(q_t, '_FLOAT_MODULE', None)
                if float_mod is not None:
                    float_mods[q_t] = float_mod
                    setattr(q_t, '_FLOAT_MODULE', qat_t)

            q_model = torch.quantization.convert(q_model, mapping)

            for q_t, orig_t in float_mods.items():
                setattr(q_t, '_FLOAT_MODULE', orig_t)

            float_mods.clear()
        else:
            q_model = torch.quantization.convert(q_model)

        return q_model

    def optimize_conv_bn_fusion(self, q_model, eps=1e-5):
        """Optimizes the Conv-BatchNorm fusion pattern.
           Sometimes, the running_var of the BatchNorm could be near to zero. If the weights of those channels happen
           to be large, then it may lead to large quantization losses. We choose to ignore those channels when
           `bn.running_var.abs() < eps`.

        Args:
            q_model (nn.Module): The QAT/PTQ-prepared model

        """

        def _pre_hook_func(indices):
            def _pre_hook(mod, input):
                max_val = input[0][~indices].max()
                min_val = input[0][~indices].min()
                input[0].clamp_(min_val, max_val)
                return input

            return _pre_hook

        for m in q_model.modules():
            if type(m).__name__ in ('ConvBnReLU2d', 'ConvBn2d'):
                if m.in_channels == m.out_channels and m.out_channels == m.groups and m.groups > 1:
                    indices = m.bn.running_var < eps
                    if torch.any(indices):
                        m.weight_fake_quant.register_forward_pre_hook(_pre_hook_func(indices))

    def prepare_onnx_export(self, q_model):
        """Prepares for ONNX model export

        Args:
            q_model (nn.Module): The QAT/PTQ-prepared model

        """

        if self.backend == 'onnx':
            for mod in self.leaf_nodes:
                torch.quantization.disable_fake_quant(mod.activation_post_process)

        fq_base_cls = getattr(torch_q, 'FakeQuantizeBase', torch_q.FakeQuantize)
        for n, m in q_model.named_modules():
            if isinstance(m, fq_base_cls):
                if m.qscheme in (torch.per_channel_symmetric, torch.per_tensor_symmetric):
                    m.zero_point.fill_(0)
                else:
                    assert m.qscheme == torch.per_tensor_affine
                    assert isinstance(m, torch_q.FixedQParamsFakeQuantize)

                    m.qscheme = torch.per_tensor_symmetric
                    if torch.all(m.zero_point == 128):
                        m.zero_point.fill_(0)
                    else:
                        assert not torch.is_nonzero(m.zero_point)
                        m.scale *= 2

                m.quant_min = -128
                m.quant_max = 127
                m.dtype = torch.int8

                if hasattr(m, 'activation_post_process'):
                    m.activation_post_process.quant_min = -128
                    m.activation_post_process.quant_max = 127
                    m.activation_post_process.dtype = torch.int8

        if LooseVersion(torch.__version__) >= LooseVersion('1.12.0'):

            def get_wrapper(mod):
                def wrapper(*args, **kwargs):
                    return torch_q.FakeQuantize.forward(mod, *args, **kwargs)

                return wrapper

            fused_fq_cls = getattr(torch_q, 'FusedMovingAvgObsFakeQuantize', None)
            if fused_fq_cls is not None:
                for m in q_model.modules():
                    if isinstance(m, fused_fq_cls):
                        m.forward = get_wrapper(m)
        elif LooseVersion(torch.__version__) >= LooseVersion('1.10.0'):
            mod_dict = {}
            for n, m in q_model.named_modules():
                mod_dict[n] = m

            class _FakeQuantize(nn.Module):
                def __init__(self, fq: torch_q.FakeQuantize) -> None:
                    super().__init__()
                    self.fake_quant_enabled = fq.fake_quant_enabled
                    self.scale = fq.scale
                    self.zero_point = fq.zero_point
                    self.quant_min = fq.quant_min
                    self.quant_max = fq.quant_max
                    self.is_per_channel = fq.is_per_channel
                    self.ch_axis = fq.ch_axis

                def forward(self, X):
                    if self.fake_quant_enabled[0] == 1:
                        if self.is_per_channel:
                            X = torch.fake_quantize_per_channel_affine(
                                X, self.scale, self.zero_point, self.ch_axis, self.quant_min, self.quant_max
                            )
                        else:
                            X = torch.fake_quantize_per_tensor_affine(
                                X, self.scale, self.zero_point, self.quant_min, self.quant_max
                            )
                    return X

            action_list = []
            for n, m in q_model.named_modules():
                if isinstance(m, torch_q.FakeQuantize):
                    if '.' in n:
                        n_split = n.split('.')
                        prop = n_split[-1]
                        pm_name = '.'.join(n_split[:-1])
                        pm = mod_dict[pm_name]
                    else:
                        prop = n
                        pm = q_model

                    new_m = _FakeQuantize(m)

                    action_list.append((pm, prop, new_m))

            for parent, prop, new_mod in action_list:
                setattr(parent, prop, new_mod)

        for n, m in q_model.named_modules():

            def conv_fused_wrapper(mod, scale_factor):
                type_name = type(mod).__name__

                def new_wrapper(input):
                    weight_shape = [1] * len(mod.weight.shape)
                    weight_shape[0] = -1
                    y = type(mod)._conv_forward(
                        mod, input, mod.weight_fake_quant(mod.weight * scale_factor.reshape(weight_shape)), mod.bias
                    )
                    if 'Bn' in type_name:
                        y = mod.bn(y)
                    if 'ReLU' in type_name:
                        y = torch.nn.functional.relu(y)
                    return y

                return new_wrapper

            type_name = type(m).__name__
            if type_name in ('ConvBn1d', 'ConvBnReLU1d', 'ConvBn2d', 'ConvBnReLU2d', 'ConvBn3d', 'ConvBnReLU3d'):
                running_std = torch.sqrt(m.bn.running_var + m.bn.eps)
                scale_factor = m.bn.weight / running_std
                if m.bias is not None:
                    m.bn.running_mean -= m.bias
                    m.bias = None
                m.bn.running_mean *= scale_factor
                m.bn.weight /= scale_factor
                m.forward = conv_fused_wrapper(m, scale_factor)

        def get_pre_hook(acps, indices):
            def pre_hook(module, input):
                new_input = list(input)
                for acp, idx in zip(acps, indices):
                    new_input[idx] = acp(new_input[idx])
                return tuple(new_input)

            return pre_hook

        def get_hook_func(acps, indices, orig_func):
            def pre_hook(*input, **kwargs):
                new_input = list(input)
                for acp, idx in zip(acps, indices):
                    if orig_func.__name__ == 'cat':
                        new_input[0][idx] = acp(new_input[0][idx])
                    else:
                        new_input[idx] = acp(new_input[idx])
                input = tuple(new_input)
                return orig_func(*input, **kwargs)

            return pre_hook

        if self.backend == 'onnx':
            end_mod_dict = {}
            for start_mod, end_mod, idx in self.swap_nodes:
                end_mod_dict.setdefault(end_mod, ([], []))
                end_mod_dict[end_mod][0].append(start_mod)
                end_mod_dict[end_mod][1].append(idx)

            for end_mod, (start_mods, indices) in end_mod_dict.items():
                acps = []
                for start_mod in start_mods:
                    if inspect.isroutine(start_mod) and isinstance(start_mod.__self__, nnq.FloatFunctional):
                        acp = start_mod.__self__.activation_post_process
                    else:
                        acp = start_mod.activation_post_process
                    acps.append(acp)

                if inspect.isroutine(end_mod) and isinstance(end_mod.__self__, nnq.FloatFunctional):
                    ff = end_mod.__self__

                    for i in range(len(indices)):
                        indices[i] -= 1

                    ff.cat = get_hook_func(acps, indices, ff.cat)
                    ff.add = get_hook_func(acps, indices, ff.add)
                    ff.mul = get_hook_func(acps, indices, ff.mul)
                    ff.add_scalar = get_hook_func(acps, indices, ff.add_scalar)
                    ff.mul_scalar = get_hook_func(acps, indices, ff.mul_scalar)
                    ff.add_relu = get_hook_func(acps, indices, ff.add_relu)
                else:
                    assert isinstance(
                        end_mod, nn.Module
                    ), "Only end nodes with `nn.Module` are supported duing module swapping"

                    end_mod.register_forward_pre_hook(get_pre_hook(acps, indices))

        q_model.apply(torch_q.disable_observer)

    def freeze_weights(self, q_model, dummy_input):
        """Freezes the weights before model export without `quantizer.convert`

        Args:
            q_model (nn.Module): The QAT/PTQ-prepared model

        """
        hooks = []

        def collect_fake_quantize_hook(mod, inp, outp):
            mod.register_parameter('_tinynn_frozen_q_weight', nn.Parameter(outp))

        for n, m in q_model.named_modules():
            if n.endswith('.weight_fake_quant'):
                hooks.append(m.register_forward_hook(collect_fake_quantize_hook))

        if isinstance(dummy_input, (list, tuple)):
            q_model(*dummy_input)
        else:
            q_model(dummy_input)

        for hook in hooks:
            hook.remove()

        hooks.clear()

        def freeze_fake_quantize_hook(mod, inp):
            return mod._tinynn_frozen_q_weight

        for n, m in q_model.named_modules():
            if n.endswith('.weight_fake_quant'):
                hooks.append(m.register_forward_pre_hook(freeze_fake_quantize_hook))


class BF16Quantizer(QATQuantizer):
    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new BF16Quantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be \
                generated. Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__(model, dummy_input, work_dir, config)

    def parse_config(self, config: dict):
        super().parse_config(config)

        self.rewrite_graph = False

    def quantize(self) -> nn.Module:
        """Prepare model for BFloat16 training"""

        self.model.train()

        qconfig = torch_q.QConfig(activation=FakeQuantizeBFloat16.with_args(), weight=FakeQuantizeBFloat16.with_args())
        self.model.qconfig = qconfig

        torch_q.prepare_qat(self.model, inplace=True)

        return self.model


class PostQuantizer(QATQuantizer):
    rewrite_graph: bool
    force_overwrite: bool
    is_input_quantized: typing.Optional[typing.Tuple[bool]]
    quantized_input_stats: typing.Optional[typing.List[typing.Optional[typing.Tuple[float, float]]]]
    backend: str
    remove_weights_after_load: bool
    asymmetric: bool
    per_tensor: bool
    disable_requantization_for_cat: bool
    dynamic_lstm_quant: bool
    algorithm: str

    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new PostQuantizer object

        Args:
            model: The model to be quantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be \
                generated. Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__(model, dummy_input, work_dir, config)

    def parse_config(self, config: typing.Optional[dict]):
        if config and 'algorithm' not in config.keys():
            config['algorithm'] = 'l2'

        super().parse_config(config)

    def set_model_mode(self, model: nn.Module = None, preserve: bool = False):
        """Sets the mode (train/eval) of the model

        Args:
            model (nn.Module, optional): A PyTorch model. Defaults to None.
            preserve (bool, optional): Whether the old state is preserved. Defaults to False.
        """
        if model is None:
            model = self.model

        if preserve:
            for m in model.modules():
                training = getattr(m, 'training', None)
                if isinstance(training, bool):
                    self.train_mode_dict[m] = training

        model.eval()

    def prepare_qconfig(self, graph: TraceGraph, backend: str):
        """Prepare qconfig for various configurations.

        Args:
            graph (TraceGraph): The computation graph of the model
            backend (str, optional): The backend of quantization
        """

        log.info('setting qat backend and call prepare_qat')
        if not self.legacy_fq or LooseVersion(torch.__version__) < '1.12.0':
            qconfig = torch_q.get_default_qconfig(backend)
        else:
            qconfig = torch_q.get_default_qconfig(backend, 0)
        qconfig_c = None
        if self.backend == 'qnnpack':
            if not self.asymmetric:
                sym_fq = torch_q.HistogramObserver.with_args(
                    dtype=torch.quint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
                )
                qconfig = torch_q.QConfig(qconfig.activation, sym_fq)
                sym_fq = PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0
                )
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        elif self.backend == 'fbgemm':
            sym_fq = torch_q.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
            qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        elif self.backend in ('onnx', 'tensorrt'):
            if not self.asymmetric:
                sym_fq = torch_q.HistogramObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
                )
                qconfig = torch_q.QConfig(sym_fq, qconfig.weight)
            if not self.per_tensor:
                sym_fq = torch_q.MinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False
                )
                qconfig = torch_q.QConfig(qconfig.activation, sym_fq)
                sym_fq = torch_q.PerChannelMinMaxObserver.with_args(
                    dtype=torch.qint8, qscheme=torch.per_channel_symmetric, reduce_range=False, ch_axis=0
                )
                qconfig_c = torch_q.QConfig(qconfig.activation, sym_fq)
        else:
            log.warning(f'Quantization backend {self.backend} is not tested. Please use at your risk.')

        if self.algorithm != 'l2':
            if self.algorithm == 'kl':
                if self.backend == 'qnnpack':
                    alg_sym_fq = HistogramObserverKL.with_args(qscheme=torch.per_tensor_symmetric, reduce_range=False)
                    alg_asym_fq = HistogramObserverKL.with_args(reduce_range=False)
                elif self.backend == 'fbgemm':
                    alg_sym_fq = HistogramObserverKL.with_args(qscheme=torch.per_tensor_symmetric, reduce_range=True)
                    alg_asym_fq = HistogramObserverKL.with_args(reduce_range=True)
                else:
                    alg_sym_fq = qconfig.activation
                    alg_asym_fq = qconfig.activation
                if not self.asymmetric:
                    qconfig = torch_q.QConfig(alg_sym_fq, qconfig.weight)
                else:
                    qconfig = torch_q.QConfig(alg_asym_fq, qconfig.weight)

        torch.backends.quantized.engine = backend
        graph.module.qconfig = qconfig
        if self.backend == 'qnnpack':
            if qconfig_c is not None:
                q = queue.Queue()
                q.put(graph.module)

                while not q.empty():
                    m = q.get()
                    if type(m).__name__ in (
                        'Conv2d',
                        'ConvBnReLU2d',
                        'ConvBn2d',
                        'ConvReLU2d',
                        'Conv1d',
                        'ConvBnReLU1d',
                        'ConvBn1d',
                    ):
                        m.qconfig = qconfig_c
                    else:
                        for c in m.children():
                            q.put(c)
        elif self.backend == 'fbgemm':
            if qconfig_c is not None:
                q = queue.Queue()
                q.put(graph.module)

                while not q.empty():
                    m = q.get()
                    if type(m).__name__ in ('Linear', 'LinearReLU'):
                        m.qconfig = qconfig_c
                    else:
                        for c in m.children():
                            q.put(c)

        def _lstm_node(node, custom_data):
            return isinstance(node.module, nn.LSTM)

        if self.dynamic_lstm_quant:
            lstm_nodes = graph.filter_forward_nodes(_lstm_node)
            for node in lstm_nodes:
                node.quantized = True
                node.module.qconfig = torch_q.default_dynamic_qconfig

    def prepare_qat(
        self,
        graph: TraceGraph,
        is_input_quantized: typing.Optional[typing.Tuple[bool]] = None,
        backend: str = 'qnnpack',
        fuse_only: bool = False,
    ) -> torch.nn.Module:
        """Prepare model for QAT training

        Args:
            graph (TraceGraph): The computation graph of the model
            is_input_quantized (typing.Union[typing.Tuple[bool]], optional): Whether the input tensor(s) is (are) \
                quantized. Defaults to None.
            backend (str, optional): The backend of quantization. Defaults to 'qnnpack'.
            fuse_only (bool, optional): Whether the returned model is only fused in PostQuantizer. Defaults to False.

        Returns:
            torch.nn.Module: The QAT-ready model
        """

        graph.module.eval()

        self.prepare_qat_prep(graph, is_input_quantized, backend)
        if fuse_only:
            return graph.module

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

            if hasattr(graph.module, 'activation_post_process'):
                if hasattr(graph.module, "_forward_hooks"):
                    if len(graph.module._forward_hooks) > 0:
                        graph.module._forward_hooks.popitem()
                delattr(graph.module, "activation_post_process")
        else:
            torch_q.propagate_qconfig_(graph.module, qconfig_dict=None)
            for n in graph.forward_nodes:
                if not n.quantized:
                    if hasattr(n.module, "qconfig"):
                        delattr(n.module, "qconfig")

            if LooseVersion(torch.__version__) >= LooseVersion("1.8.0"):
                if LooseVersion(torch.__version__) >= LooseVersion("1.13.0"):
                    prepare_custom_config_dict = torch.ao.quantization.get_default_custom_config_dict()
                else:
                    prepare_custom_config_dict = {}

                custom_module_class_mapping = prepare_custom_config_dict.get(
                    "float_to_observed_custom_module_class", {}
                )

                torch_q.add_observer_(
                    graph.module,
                    qconfig_propagation_list=whitelist,
                    custom_module_class_mapping=custom_module_class_mapping,
                )
            else:
                torch_q.add_observer_(
                    graph.module,
                    qconfig_propagation_list=whitelist,
                )

        if self.dynamic_lstm_quant:
            mapping = {nn.LSTM: nnqd.LSTM}
            torch_q.convert(graph.module, mapping=mapping, inplace=True, remove_qconfig=False)

        for n in graph.forward_nodes:
            if not n.quantized:
                if hasattr(n.module, "_forward_hooks"):
                    if len(n.module._forward_hooks) > 0:
                        n.module._forward_hooks.popitem()
                if hasattr(n.module, "qconfig"):
                    delattr(n.module, "qconfig")
                if hasattr(n.module, "activation_post_process"):
                    delattr(n.module, "activation_post_process")

        if self.disable_requantization_for_cat:
            self.disable_requantization_for_cat_pass(graph)

        if self.quantized_input_stats is not None:
            self.prepare_quantized_inputs_pass(graph)

        if self.set_quantizable_op_stats:
            if self.quantized_op_stats is None:
                self.quantized_op_stats = {}
            self.quantized_op_stats.update(KNOWN_QSTATS)

        if self.quantized_op_stats is not None:
            self.prepare_quantized_ops_pass(graph)

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
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be \
                generated. Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the quantizer
        """

        super().__init__(model, dummy_input, work_dir, config)

    def parse_config(self, config: dict):
        super().parse_config(config)

        self.rewrite_graph = False

        assert not self.asymmetric, "Asymmetric quantization is not supported for DynamicQuantizer"
        assert self.per_tensor, "Per-channel quantization is not supported for DynamicQuantizer"

    def quantize(self) -> nn.Module:
        """Prepare model for dynamic quantization"""

        self.model.eval()

        torch_q.quantize_dynamic(self.model, inplace=True)

        return self.model


class DeQuantizer(object):
    rewrite_graph: bool
    force_overwrite: bool
    remove_weights_after_load: bool
    extra_tracer_opts: typing.Optional[typing.Dict]
    target: str

    def __init__(self, model, dummy_input, work_dir: typing.Optional[str] = None, config: typing.Optional[dict] = None):
        """ Constructs a new DeQuantizer object

        Args:
            model: The model to be dequantized
            dummy_input: A viable input to the model
            work_dir (typing.Optional[str], optional): The working directory in which the intermediate files will be \
                generated. Defaults to None, in which case "output" will be used.
            config (typing.Optional[dict]): Options for the dequantizer
        """

        if isinstance(model, DataParallel) or isinstance(model, DistributedDataParallel):
            self.model = model.module
        else:
            self.model = model

        self.dummy_input = dummy_input
        self.work_dir = 'out' if work_dir is None else work_dir

        self.parse_config(config)

    def dequantize(self) -> nn.Module:
        # We need a model in training mode so that QAT could take place
        if self.target == 'qat':
            self.model.train()
        else:
            self.model.eval()

        # After tracing the model, we will get a TraceGraph object
        graph = trace(self.model, self.dummy_input, **self.extra_tracer_opts)

        if self.rewrite_graph:
            # Retrives the name of the model from type info
            model_name = type(self.model).__name__
            if model_name.endswith('_qat'):
                model_name = model_name[:-4]
            model_name_float = f'{model_name}_float'
            model_name_float_lower = model_name_float.lower()
            model_ns = f'tinynn_rewritten_models.{model_name_float}'

            model_code_path = os.path.join(self.work_dir, f'{model_name_float_lower}.py')
            model_weights_path = os.path.join(self.work_dir, f'{model_name_float_lower}.pth')

            # Generate the code for the modified model
            # We will try to do some op rewriting in the `rewrite_quantize_graph` function.
            if self.force_overwrite or not os.path.exists(model_code_path) or not os.path.exists(model_weights_path):
                self.rewrite_quantize_graph(graph)
                graph.generate_code(model_code_path, model_weights_path, model_name_float)

            # Import the new model
            rewritten_model = import_from_path(model_ns, model_code_path, model_name_float)()
            rewritten_model.load_state_dict(torch.load(model_weights_path))

            device = get_module_device(self.model)
            if device is not None:
                rewritten_model.to(device=device)

            # Remove the weights file to save space
            if self.remove_weights_after_load:
                os.unlink(model_weights_path)

            # Set the model to training mode before tracing again, since we need to perform QAT
            if self.target == 'qat':
                rewritten_model.train()
            else:
                rewritten_model.eval()

            # Update the model so that the original one can be released
            self.model = rewritten_model
        else:
            rewritten_model = self.model

        return rewritten_model

    def parse_config(self, config: typing.Optional[dict]):
        default_values = {
            'rewrite_graph': True,
            'force_overwrite': True,
            'extra_tracer_opts': {},
            'remove_weights_after_load': False,
            'target': 'qat',
        }

        if config is None:
            config = dict()

        for k, v in default_values.items():
            actual_v = config.get(k, v)
            setattr(self, k, actual_v)

    def rewrite_quantize_graph(self, graph: TraceGraph) -> None:
        """Rewrites the computation graph for quantization"""
        if graph.quantized:
            return

        graph_quantized = False
        for n in graph.forward_nodes:
            if n.type() in (torch_q.QuantStub, torch_q.DeQuantStub):
                graph_quantized = True
                break

        for n in graph.other_init_nodes:
            if n.type() == nnq.FloatFunctional:
                graph_quantized = True
                break

        if not graph_quantized:
            log.warning(
                'Graph is not quantized. No need to rewrite. Please pass in `config={"rewrite_graph": False}` to'
                ' suppress this warning'
            )
            return

        # Remove QuantStubs and DeQuantStubs
        names = [n.unique_name for n in graph.forward_nodes]
        for name in names:
            n = graph.nodes_map[name]
            if n.type() in (torch_q.QuantStub, torch_q.DeQuantStub):
                graph.remove_node(n)

        def _is_add_relu_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return (
                    cur_module.kind == 'add_relu'
                    and len(node.prev_nodes) > 1
                    and node.prev_nodes[0].type() == nnq.FloatFunctional
                )

        # Split FloatFunctional.add_relu to FloatFunctional.add and torch.relu
        add_relu_nodes = graph.filter_forward_nodes(_is_add_relu_node)
        for n in add_relu_nodes:
            n.module.kind = 'add'
            n.module.func_type = 'add'

            parts = n.module.full_name.split('.')[:-1] + [n.module.func_type]
            n.module.full_name = '.'.join(parts)

            with override_current_trace_graph(graph):
                next_func = TraceFunction('torch.relu').parse_args(n.next_tensors[0])

            next_out = torch.relu(n.next_tensors[0])
            graph.insert_after(n, next_func, [next_out])

        # Replace FloatFunctional.{add, mul, cat} with torch.{add, mul, cat}
        def _is_add_mul_cat_node(node: TraceNode, custom_data):
            cur_module = node.module
            cur_class = type(cur_module)
            if cur_class == TraceFunction:
                return (
                    cur_module.kind in ('add', 'mul', 'cat')
                    and len(node.prev_nodes) > 1
                    and node.prev_nodes[0].type() == nnq.FloatFunctional
                )

        add_mul_cat_nodes = graph.filter_forward_nodes(_is_add_mul_cat_node)
        for n in add_mul_cat_nodes:
            parts = ['torch'] + [n.module.func_type]
            n.module.full_name = '.'.join(parts)
            n.module.is_class = False

            n.prev_nodes[0].next_nodes.remove(n)
            n.prev_nodes.pop(0)
            n.prev_indices.pop(0)

            n.module.tensor_names.pop(0)
            n.module.args_parsed.pop(0)
            n.module.args_to_string(n.module.args_parsed)

        # Remove FloatFunctional nodes
        names = [n.unique_name for n in graph.other_init_nodes]
        for name in names:
            n = graph.nodes_map[name]
            if n.type() == nnq.FloatFunctional:
                graph.other_init_nodes.remove(n)
                del graph.nodes_map[n.unique_name]

        graph.quantized = False
        graph.recompute_forward_order()


def load_creation_func_names():
    if len(creation_func_names) == 0:
        funcs_d = load_creation_funcs()
        for ns, funcs_v in funcs_d.items():
            ns_str = qualified_name(ns)
            creation_func_names.extend([f'{ns_str}.{x}' for x in funcs_v])
    return creation_func_names


def get_dict_from_rules(rules):
    rule_dict = {}
    for rule in rules:
        base_rule_dict = rule_dict
        for module_cls in reversed(rule):
            # Node properties (has_key, child_nodes)
            base_rule_dict.setdefault(module_cls, [False, {}])
            base_rule_pair = base_rule_dict[module_cls]
            base_rule_dict = base_rule_pair[1]
        base_rule_pair[0] = True
    return rule_dict


def load_processed_qat_rules():
    if len(processed_qat_rules) == 0:
        # Constructor a prefix tree for the QAT rules
        fuse_rules = sorted(FUSE_RULE_LIST, key=lambda x: len(x), reverse=True)
        rule_dict = get_dict_from_rules(fuse_rules)
        processed_qat_rules.update(rule_dict)
    return processed_qat_rules


def load_processed_ptq_rules():
    if len(processed_ptq_rules) == 0:
        # Constructor a prefix tree for the QAT rules
        filtered_ptq_rules = {
            k for k, v in FUSE_RULE_LIST_PTQ_ONLY.items() if LooseVersion(torch.__version__) >= LooseVersion(v)
        }
        ptq_rules = set(FUSE_RULE_LIST).union(set(filtered_ptq_rules))
        fuse_rules = sorted(ptq_rules, key=lambda x: len(x), reverse=True)
        rule_dict = get_dict_from_rules(fuse_rules)
        processed_ptq_rules.update(rule_dict)
    return processed_ptq_rules


def load_processed_extra_q_rules():
    if len(processed_extra_q_rules) == 0:
        # Constructor a prefix tree for the QAT rules
        fuse_rules = sorted(FUSE_RULE_LIST_EXTRA, key=lambda x: len(x), reverse=True)
        rule_dict = get_dict_from_rules(fuse_rules)
        processed_extra_q_rules.update(rule_dict)
    return processed_extra_q_rules


def load_processed_rewrite_to_fuse_rules():
    if len(processed_rewrite_to_fuse_rules) == 0:
        # Constructor a prefix tree for the rewrite rules
        fuse_rules = sorted(REWRITE_TO_FUSE_RULE_LIST, key=lambda x: len(x), reverse=True)
        rule_dict = get_dict_from_rules(fuse_rules)
        processed_rewrite_to_fuse_rules.update(rule_dict)
    return processed_rewrite_to_fuse_rules


def load_processed_all_qat_rules():
    if len(processed_all_qat_rules) == 0:
        # Constructor a prefix tree for the QAT rules
        qat_rules = set(FUSE_RULE_LIST).union(set(FUSE_RULE_LIST_EXTRA))
        fuse_rules = sorted(qat_rules, key=lambda x: len(x), reverse=True)
        rule_dict = get_dict_from_rules(fuse_rules)
        processed_all_qat_rules.update(rule_dict)
    return processed_all_qat_rules


def load_processed_all_ptq_rules():
    if len(processed_all_ptq_rules) == 0:
        # Constructor a prefix tree for the QAT rules
        filtered_ptq_rules = {
            k for k, v in FUSE_RULE_LIST_PTQ_ONLY.items() if LooseVersion(torch.__version__) >= LooseVersion(v)
        }
        ptq_rules = set(FUSE_RULE_LIST).union(set(filtered_ptq_rules)).union(set(FUSE_RULE_LIST_EXTRA))
        fuse_rules = sorted(ptq_rules, key=lambda x: len(x), reverse=True)
        rule_dict = get_dict_from_rules(fuse_rules)
        processed_all_ptq_rules.update(rule_dict)
    return processed_all_ptq_rules
