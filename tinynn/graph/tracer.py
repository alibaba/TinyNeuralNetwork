import contextlib
import copy
import importlib
import inspect
import io
import logging
import os
import traceback
import typing

import torch
import torch.nn as nn
import yaml
import numpy as np
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel

from tinynn.util.util import get_logger, import_from_path, tensors2ndarray
from ._utils import patch_getitem, revert_getitem

# Basic types


class GlobalData(object):
    """ The data structure to store data that can be used in this script,
        which is a wrapper of a object of a built-in type. """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def get_value(self):
        """ Returns the inner value of the wrapper """
        return self.value

    def set_value(self, value):
        """ Sets the inner value of the wrapper """
        self.value = value

    def __str__(self):
        """ Returns the string representation of the inner object """
        return self.value.__str__()

    def __repr__(self):
        """ Returns the string representation of the inner object """
        return self.value.__repr__()

    def __call__(self, *args):
        """ Simplifies the usage of the wrapper
            e.g. a = GlobalData(3)
                 a()  -> a.get_value()
                 a(1) -> a.set_value(1) """
        if len(args) == 0:
            return self.get_value()
        elif len(args) == 1:
            return self.set_value(*args)

    def __bool__(self):
        """ Returns the actual boolean value of the inner object """
        return self.value.__bool__()


# Constants


# Template for a traced module
MODULE_TEMPLATE = """
%(import_block)s


class %(name_block)s(torch.nn.Module):
    def __init__(self):
        super().__init__()

%(init_block)s

%(forward_block)s


if __name__ == "__main__":
    model = %(name_block)s()
%(load_weight_block)s

    model.eval()
    model.cpu()

%(input_block)s

    output = model(%(input_names)s)
    print(output)

"""

# Special math operators
SPECIAL_OPERATORS = ['add',
                     'and',
                     'div',
                     'floordiv',
                     'lshift',
                     'mul',
                     'or',
                     'pow',
                     'rshift',
                     'sub',
                     'xor']


# Global objects

# Logger
log = get_logger(__name__, 'WARNING')

# Loaded overriable items from the config file
overridable_funcs = {}
overridable_modules = []
overridable_creation_funcs = {}

# Load state for the override items
overridable_funcs_loaded = GlobalData(False)
overridable_modules_loaded = GlobalData(False)
overridable_creation_funcs_loaded = GlobalData(False)

funcs_overrided = GlobalData(False)
modules_overrided = GlobalData(False)
creation_funcs_overrided = GlobalData(False)

# Lock for tracing
lock = GlobalData(False)

# Whether the constructors get traced
module_constructor_traced = set()

# Current traced graph
current_graph = GlobalData(None)

# Generated module constructor lines
module_constructor_lines = {}

# Directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Original module constructor signatures
module_constructor_signatures = {}

# Original values of tracked objects
original_values_for_tracked_objects = {}


class TraceNode(object):
    """ A basic data structure to represent a node in the computation graph """
    module: typing.Union[torch.nn.Module, 'TraceFunction', 'ConstantNode']
    prev_nodes: typing.List['TraceNode']
    next_nodes: typing.List['TraceNode']
    prev_tensors: typing.List[torch.Tensor]
    next_tensors: typing.List[torch.Tensor]
    prev_indices: typing.List[typing.Optional[int]]
    rev_index: bool
    unique_name: str
    active: bool

    def __init__(self, module: typing.Union[torch.nn.Module, 'ConstantNode', 'TraceFunction'], cur_graph: typing.Optional['TraceGraph'] = None):
        # Inner module, could either be a `nn.Module`, `ConstantNode` or `TraceFunction`
        self.module = module

        # Previous and next nodes in the computation graph
        self.prev_nodes = []
        self.next_nodes = []

        # The input and output tensors for the node
        self.prev_tensors = []
        self.next_tensors = []

        # The indices used to retrieve the corresponding tensor
        # e.g. torch.chunk() returns [A, B], in which A and B are PyTorch tensors.
        #      so if we use A in this node, then the corresponding prev_index is 0.
        # If the tensor is not a sub item, then `None` should be used.
        self.prev_indices = []

        # In some nodes, the indices are reversed. For example, for an output node,
        # the indices are not used to fetch the items, but to construct a list that
        # contains them.
        self.rev_index = False

        # The current TraceGraph to be processed
        # In the trace phase, it can be obtained through `current_graph()`
        # Otherwise, you need to pass it explicitly
        if cur_graph is None:
            cur_graph = current_graph()

        # Unique name of the node (the key of the node in the node map in TraceGraph)
        if type(module) in (ConstantNode, TraceFunction):
            self.unique_name = module.unique_name
        else:
            self.unique_name = cur_graph.module_unique_name_dict[id(module)]

        # Whether the node is active in the computation graph
        self.active = True

        # The index of the node in the graph
        self.forward_order = 0

        # Whether the node is in a quantized graph
        self.quantized = False

        # Numbering of the name of the node
        if cur_graph.global_nodes.get(self.unique_name) is not None:
            cur_graph.global_nodes[self.unique_name] += 1
            self.unique_name = "_".join([self.unique_name, str(cur_graph.global_nodes[self.unique_name])])
        else:
            cur_graph.global_nodes[self.unique_name] = 0

    def type(self):
        """ Returns the original name of the function or the type of the module """
        if type(self.module) == TraceFunction:
            return self.module.func_type

        return type(self.module)

    def kind(self):
        """ Returns the kind of the function or the type of the module """
        if type(self.module) == TraceFunction:
            return self.module.kind

        return type(self.module)

    def is_class(self) -> bool:
        """ Judges whether it is a class function or not """
        if type(self.module) == TraceFunction:
            return self.module.is_class
        else:
            return False

    def full_name(self) -> str:
        """ Returns the original full name of the function (including namespace) """
        if type(self.module) in (TraceFunction, ConstantNode):
            return self.module.full_name
        else:
            return f'{type(self.module).__module__}.{type(self.module).__name__}'

    def __hash__(self) -> str:
        """ Uses the unique name as the hash for the node """
        return self.unique_name

    def prev_node_unique_name(self, idx) -> str:
        """ A utility function to generate the name of the previous node with index """
        if idx < len(self.prev_nodes) and idx < len(self.prev_indices):
            node_name = self.prev_nodes[idx].unique_name
            node_idx = self.prev_indices[idx]
            ns = ''
            if type(self.prev_nodes[idx].module) == ConstantNode:
                ns = 'self.'
            if node_idx is None:
                return f'{ns}{node_name}'
            else:
                if type(node_idx) in (list, tuple):
                    indices_str = ''.join([f'[{i}]' for i in node_idx])
                    return f'{ns}{node_name}{indices_str}'
                else:
                    return f'{ns}{node_name}[{node_idx}]'
        else:
            return ''


class ConstantNode(object):
    """ A data structure for runtime-defined constants """

    def __init__(self, data: typing.List, dtype: torch.dtype, shape: torch.Size):
        # Raw data (list)
        self.data = data

        # Data shape
        self.shape = tuple(shape)

        # Data type
        self.dtype = str(dtype)

        # Please refer to the the description of those properties in `TraceFunction`
        self.kind = 'tensor'
        self.func_type = 'tensor'
        self.full_name = 'torch.tensor'
        self.is_class = False

        # Numbering of the name of the node
        if not current_graph().global_functions.get(self.kind, None):
            current_graph().global_functions[self.kind] = 1
        else:
            current_graph().global_functions[self.kind] += 1

        self.unique_name = "_".join([self.kind, str(current_graph().global_functions[self.kind])])

    def parse(self, convert_to_parameter: bool = False):
        def _stringify_list(content) -> str:
            """ Convert a list of objects to a string """
            if type(content) in (list, tuple):
                sub_contents = []
                for item in content:
                    sub_contents.append(_stringify_list(item))
                inner_content = ', '.join(sub_contents)
                return f'[{inner_content}]'
            elif type(content) in (int, float, bool):
                return str(content)
            elif type(content) == str:
                return f'"{content}"'

        # If `convert_to_parameter` is `True`, we set `module_constructor_line` for this node.
        # So that the content of the data will not be written inline.
        if not convert_to_parameter:
            module_constructor_lines[id(self)] = f'torch.tensor({_stringify_list(self.data)}, dtype={self.dtype})'

        return self


class TraceFunction(object):
    """ A data structure for traced functions """

    def __init__(self, full_name: str, is_class: bool = False, is_property: bool = False):
        super().__init__()

        # The base name of the function
        self.func_type = full_name.split('.')[-1]

        # The class name of the function
        # It can be acquired by removing underlines in the base name of the function for special math
        # operators and inline functions
        self.kind = None
        if self.func_type.endswith('__') and self.func_type.startswith('__'):
            inner_name = self.func_type[2:-2]
            if len(inner_name) > 1 and inner_name[0] in ('i', 'r'):
                inner_op = inner_name[1:]
                if inner_op in SPECIAL_OPERATORS:
                    self.kind = inner_op
            if self.kind is None:
                self.kind = inner_name

        if self.kind is None:
            if self.func_type.endswith('_'):
                self.kind = self.func_type[:-1]
            else:
                self.kind = self.func_type

        # Numbering of the nodes
        if not current_graph().global_functions.get(self.kind, None):
            current_graph().global_functions[self.kind] = 1
        else:
            current_graph().global_functions[self.kind] += 1

        # Unique name
        self.unique_name = "_".join([self.kind, str(current_graph().global_functions[self.kind])])

        # The input tensors of the function
        self.prev_tensors = []

        # The name of the function (including namespace)
        self.full_name = full_name

        # Whether it is a class function/property
        self.is_class = is_class

        # Whether it is a property
        self.is_property = is_property

        # Arguments
        self.args = None
        self.kargs = None
        self.args_string = None
        self.args_parsed = None

    def parse_args(self, *args, **kwargs):
        """ Sets the string representation of the arguments """

        def _tensor_name(a, convert_to_parameter=False):
            """ Get the tensor name from the computation graph """
            ns = ''
            if id(a) not in current_graph().tensor_pre_node_dict:
                if not a.is_leaf:
                    log.error(
                        f'Connection is lost when generating code for {self.unique_name} of type {self.full_name}')
                else:
                    # Constant generation
                    log.warning('Constant generation is experimental and may yield error')
                    if a.numel() > 50 and a.is_floating_point():
                        convert_to_parameter = True
                    raw_data = a.tolist()
                    constant_node = ConstantNode(raw_data, a.dtype, a.shape).parse(convert_to_parameter)
                    trace_node = TraceNode(constant_node)
                    add_constant_node(trace_node, a)
                    ns = 'self.'
                    pre_node_name = current_graph().tensor_pre_node_dict[id(a)]
            else:
                pre_node_name = current_graph().tensor_pre_node_dict[id(a)]
                node = current_graph().nodes_map[pre_node_name]
                if type(node.module) == ConstantNode:
                    ns = 'self.'
            if id(a) in current_graph().tensor_pre_index_dict:
                pre_node_index = current_graph().tensor_pre_index_dict[id(a)]
                log.debug(f'pre_index gen func {self.kind}: {pre_node_index}')
                return f"{ns}{pre_node_name}[{pre_node_index}]"
            else:
                return f"{ns}{pre_node_name}"

        def _parse_args(arg):
            """ Converts the argument to a list of strings """
            new_arg = []

            for a in arg:
                if type(a) in [list, tuple, torch.Size]:
                    new_arg.append(_parse_args(a))
                elif type(a) in (torch.Tensor, torch.nn.Parameter) or \
                        (type(a) in (torch.dtype, torch.device, torch.Size) and
                         id(a) in current_graph().tensor_pre_node_dict):
                    aa = a
                    convert_to_parameter = False
                    if type(a) == torch.nn.Parameter:
                        aa = a.data
                        convert_to_parameter = True
                    self.prev_tensors.append(aa)
                    new_arg.append(_tensor_name(aa, convert_to_parameter))
                elif type(a) in (str, torch.device):
                    new_arg.append(f"\'{a}\'")
                elif type(a) in (int, float, bool, torch.dtype):
                    new_arg.append(str(a))
                elif a is None:
                    new_arg.append('None')
                elif a is Ellipsis:
                    new_arg.append('...')
                elif type(a) == slice:
                    t = (a.start, a.stop, a.step)
                    parts = []
                    for x in t:
                        if x is None:
                            parts.append('')
                        else:
                            parts.extend(_parse_args([x]))
                    r = ':'.join(parts)
                    if r.endswith(':'):
                        r = r[:-1]
                    new_arg.append(r)
                else:
                    log.error(f"unsupported type {type(a)} while generating arg for func {self.full_name}")
                    assert False

            return new_arg

        def _flatten_list(content):
            """ Flatten a list of nested list or string into a string """
            if type(content) == list:
                sub_contents = []
                for item in content:
                    sub_contents.append(_flatten_list(item))
                inner_content = ', '.join(sub_contents)
                return f'[{inner_content}]'
            else:
                return content

        self.prev_tensors.clear()
        arg_str = _parse_args(args)

        kw_items = kwargs.items()
        if kw_items:
            kw_keys, kw_vals = zip(*kw_items)
            kw_val_strs = _parse_args(kw_vals)

            for (k, v) in zip(kw_keys, kw_val_strs):
                if type(v) is list:
                    v_str = _flatten_list(v)
                    arg_str.append(f"{k}={v_str}")
                else:
                    arg_str.append(f"{k}={v}")

        self.args_parsed = arg_str

        for i in range(len(arg_str)):
            if type(arg_str[i]) is list:
                arg_str[i] = _flatten_list(arg_str[i])

        try:
            self.args_string = ", ".join(arg_str)
            self.args_string_no_self = ", ".join(arg_str[1:])
        except Exception:
            log.error(f"Error generating argument string for function {self.full_name}")
            assert False

        return self


@contextlib.contextmanager
def no_catch():
    """ Context manager for tracing nodes. Use it to avoid tracing the nodes recursively. """
    if lock():
        yield False
    else:
        lock(True)
        yield True
        lock(False)


def new_setattr_gen(orig_setattr, key: str):
    """ Wrapper function for the __setattr__ functions of the modules in PyTorch """
    log.debug(f'registered module setattr wrapper: {key}')

    def new_setattr(obj, name, value):
        log.debug(f'{key} in setattr function wrapper')
        related = True
        log.debug(f'{key} before with block, lock: {lock}')
        with no_catch() as res:
            if res:
                if id(obj) not in module_constructor_traced:
                    related = False
                class_type = type(obj)
                if related and not hasattr(class_type, '__constants__'):
                    related = False
                if related and name not in class_type.__constants__:
                    related = False
                if related:
                    log.warning(
                        f'The constant property `{name}` of {qualified_name(class_type)} is changed. We need to drop the original constructor line.')
                    module_constructor_traced.remove(id(obj))
                    del module_constructor_lines[id(obj)]
            return orig_setattr(obj, name, value)
        log.debug(f'{key} after with block, lock: {lock}')

    return new_setattr


def new_getattr_gen(orig_getattr, key: str, is_class: bool):
    """ Wrapper function for the __getattribute__ functions of the modules in PyTorch """
    log.debug(f'registered module getattr wrapper: {key}')

    def new_getattr(obj, name):
        log.debug(f'{key} in getattr function wrapper')
        related = False
        log.debug(f'{key} before with block, lock: {lock}')
        with no_catch() as res:
            result = orig_getattr(obj, name)
            if current_graph() is None:
                related = False
            if name in ('device', 'shape', 'data', 'dtype'):
                related = True
            if res:
                if related:
                    # Also the property should be constant if the result object is unchanged.
                    # Only create a new node when there isn't one.
                    if id(result) in current_graph().tensor_pre_node_dict and \
                            id(result) in original_values_for_tracked_objects and \
                            original_values_for_tracked_objects[id(result)] == result:
                        node_name = current_graph().tensor_pre_node_dict[id(result)]
                        trace_node = current_graph().nodes_map[node_name]
                        if trace_node.module.is_property and trace_node.module.func_type == 'shape':
                            result = trace_node.next_tensors
                    else:
                        # Handling dynamic shape

                        # If the torch.Size object is generated by a tensor,
                        # then we connect it to the graph.
                        # Otherwise, don't track it.
                        old_result = None
                        if type(result) == torch.Size and type(obj) == torch.Tensor:
                            # Create a list of new tensors for the sake of tracking
                            # The reason to use that instead of a tensor is stated below.
                            # e.g. Users may use the following clause to deal with sizes
                            #      x, y = tensor.size()
                            # Currently, there is no way to trace it.
                            # However, by doing this, if user calls `numel` on the `torch.Size`
                            # object, it will now throw an exception.
                            # TODO: Fix the case if user calls `numel` on `torch.Size`
                            original_values_for_tracked_objects[id(result)] = copy.deepcopy(result)
                            new_result = []
                            for elem in result:
                                new_result.append(torch.tensor(elem))
                                current_graph().tensor_pre_node_dict[id(new_result[-1])
                                                                     ] = current_graph().tensor_pre_node_dict[id(obj)]
                            old_result = result
                            result = tuple(new_result)

                        log.debug(f'{key} is called with {name}')
                        new_key = key.replace('__getattribute__', name)
                        trace_func = TraceFunction(new_key, True, True).parse_args(obj)
                        trace_node = TraceNode(trace_func)

                        if old_result is not None:
                            current_graph().tensor_pre_node_dict[id(old_result)] = trace_node.unique_name

                        add_forward_node(trace_node, trace_func.prev_tensors, result)

            log.debug(f'{key} after with block, lock: {lock}')
            return result

    return new_getattr


def new_init_gen(orig_init, key: str):
    """ Wrapper function for the init functions of the modules in PyTorch """
    log.debug(f'registered module init wrapper: {key}')

    def new_init(obj, *args, **kwargs):
        log.debug(f'{key} in init function wrapper')
        module_constructor_traced.add(id(obj))
        init_fullname = key
        class_fullname = '.'.join(init_fullname.split('.')[:-1])
        log.debug(f'{key} before with block, lock: {lock}')
        with no_catch() as res:
            if id(obj) not in module_constructor_lines:
                if not res:
                    log.warning(
                        f'Failed to acquire the tracing lock while tracing {init_fullname}, which is unexpected.')
                log.debug(f'{key} in with block, lock: {lock}')

                rckwa = check_types(kwargs.values())
                rca = check_types(args)
                err_type = rca or rckwa
                if err_type:
                    log.warning(
                        f'Constructor of {class_fullname} has arguments of type {err_type} which is unsupported')
                    log.warning(f'  Args: {args}')
                    log.warning(f'  Keyword args: {kwargs}')
                else:
                    log.info(f'Constructor of {class_fullname} registered')
                    cleaned_args = [f'"{arg}"' if type(arg) == str else str(arg) for arg in args]
                    args_content = ', '.join(cleaned_args)
                    kwargs_content = ', '.join((f'{k}="{v}"' if type(
                        v) is str else f'{k}={v}' for k, v in kwargs.items()))
                    args_connector = '' if args_content == '' or kwargs_content == '' else ', '
                    full_args_content = f'{args_content}{args_connector}{kwargs_content}'
                    orig_constructor_line = f'{class_fullname}({full_args_content})'
                    module_constructor_lines[id(obj)] = orig_constructor_line
            orig_init(obj, *args, **kwargs)
        log.debug(f'{key} after with block, lock: {lock}')

    return new_init


def new_func_gen(orig_func, key: str, is_class: bool):
    """ Wrapper function for functions in PyTorch """
    log.debug(f'registered function wrapper: {key}')

    def new_func(*args, **kwargs):
        log.debug(f'{key} in function wrapper')
        related = False
        log.debug(f'{key} before with block, lock: {lock}')
        with no_catch() as res:
            result = orig_func(*args, **kwargs)
            if res and current_graph() is not None:
                log.debug(f'{key} in with block, lock: {lock}')
                if key == 'torch.Tensor.size' and len(args) > 1:
                    # Tracking torch.Tensor.size with optional int argument
                    result = torch.tensor(result)
                if type(result) == torch.Size:
                    # Handling dynamic shape

                    # If the torch.Size object is generated by a tensor,
                    # then we connect it to the graph.
                    # Otherwise, don't track it.
                    if len(args) > 0 and type(args[0]) == torch.Tensor:
                        # Create a list of new tensors for the sake of tracking
                        # The reason to use that instead of a tensor is stated below.
                        # e.g. Users may use the following clause to deal with sizes
                        #      x, y = tensor.size()
                        # Currently, there is no way to trace it.
                        # However, by doing this, if user calls `numel` on the `torch.Size`
                        # object, it will now throw an exception.
                        # TODO: Fix the case if user calls `numel` on `torch.Size`
                        new_result = []
                        for elem in result:
                            new_result.append(torch.tensor(elem))
                            current_graph().tensor_pre_node_dict[id(new_result[-1])
                                                                 ] = current_graph().tensor_pre_node_dict[id(args[0])]
                        result = tuple(new_result)
                        related = True
                elif type(result) in (torch.dtype, torch.device):
                    related = True
                else:
                    related = check_tensor_type(result)
            log.debug(f'{key} after with block, lock: {lock}')
        if related:
            log.debug(f'tracing {key} in function wrapper')
            with no_catch() as res:
                if res:
                    trace_func = TraceFunction(key, is_class).parse_args(*args, **kwargs)
                    trace_node = TraceNode(trace_func)

                    add_forward_node(trace_node, trace_func.prev_tensors, result)
            log.debug(f'tracing {key} function wrapper complete')
        return result

    return new_func


def new_creation_func_gen(orig_func, key: str, is_class: bool):
    """ Wrapper function for functions in PyTorch """
    log.debug(f'registered creation function wrapper: {key}')

    def new_func(*args, **kwargs):
        log.debug(f'{key} in creation function wrapper')
        with no_catch() as res:
            result = orig_func(*args, **kwargs)
        log.debug(f'tracing {key} in creation function wrapper')
        with no_catch() as res:
            if res:
                trace_func = TraceFunction(key, is_class).parse_args(*args, **kwargs)
                trace_node = TraceNode(trace_func)

                add_forward_node(trace_node, trace_func.prev_tensors, result)
        return result

    return new_func


def fetch_modules(config: typing.Optional[str] = None):
    """ Fetches the functions from the config. """
    if config is None:
        config = os.path.join(current_dir, 'configs/torch_module_override.yml')
    modules = []
    with open(config, 'r') as f:
        module_dict = yaml.load(f, yaml.SafeLoader)
        for ns, module_names in module_dict.items():
            scope = importlib.import_module(ns)
            for module_name in module_names:
                if hasattr(scope, module_name):
                    module = getattr(scope, module_name)
                    modules.append(module)
                    if hasattr(module, '__init__'):
                        constructor = module.__init__
                        module_constructor_signatures[module] = inspect.signature(constructor).parameters.values()
    return modules


def fetch_funcs(config: typing.Optional[str] = None):
    """ Fetches the functions from the config. """
    if config is None:
        version_parts = torch.__version__.split('.')
        if int(version_parts[1]) < 6:
            version_parts[1] = '6'
        version_str = '_'.join(version_parts[:2])
        config = os.path.join(current_dir, f'configs/torch_func_override_{version_str}.yml')
    modules = []
    with open(config, 'r') as f:
        module_dict = yaml.load(f, yaml.SafeLoader)
        new_dict = {}
        for ns, module_names in module_dict.items():
            log.debug(f'Attempting to load {ns}')
            spec = importlib.util.find_spec(ns)
            if spec is None:
                modules = ns.split('.')
                ns = '.'.join(modules[:-1])
                typename = modules[-1]
                spec = importlib.util.find_spec(ns)
                if spec is None:
                    log.warning(f"Error importing {ns}, which may not be a module")
                    continue
                scope = importlib.import_module(ns)
                if hasattr(scope, typename):
                    scope = getattr(scope, typename)
                else:
                    log.warning(f"Error importing {ns}.{typename}")
                    continue
            else:
                scope = importlib.import_module(ns)
            modules = []
            for module_name in module_names:
                if hasattr(scope, module_name):
                    modules.append(module_name)
            new_dict[scope] = modules
    return new_dict


def qualified_name(module, item: typing.Optional[str] = None):
    if hasattr(module, '__module__'):
        obj_key = f'{module.__module__}.{module.__name__}'
    else:
        obj_key = module.__name__
    if item is not None:
        return f'{obj_key}.{item}'
    else:
        return obj_key


@contextlib.contextmanager
def patch(object, name, gen):
    """ Temporarily monkeypatches an object. """

    pre_patched_value = getattr(object, name)
    setattr(object, name, gen(pre_patched_value))
    yield object
    setattr(object, name, pre_patched_value)


@contextlib.contextmanager
def patch_modules(objects, names, gens):
    """ Temporarily monkeypatches the modules in PyTorch. """
    if type(names) not in (tuple, list) and type(gens) not in (tuple, list):
        names = (names,)
        gens = (gens,)
    pre_patched_values = {}
    for obj in objects:
        for name, gen in zip(names, gens):
            key = qualified_name(obj, name)
            pre_patched_values[key] = getattr(obj, name)
            setattr(obj, name, gen(pre_patched_values[key], key))
    yield objects
    for obj in objects:
        for name in names:
            key = qualified_name(obj, name)
            pre_patched_value = pre_patched_values[key]
            setattr(obj, name, pre_patched_value)


@contextlib.contextmanager
def patch_funcs(object_dicts, gens):
    """ Temporarily monkeypatches the functions in PyTorch. """
    if type(object_dicts) not in (tuple, list) and type(gens) not in (tuple, list):
        object_dicts = (object_dicts,)
        gens = (gens,)
    pre_patched_value_dict = {}
    for object_dict, gen in zip(object_dicts, gens):
        for obj, names in object_dict.items():
            for name in names:
                key = qualified_name(obj, name)
                if key in pre_patched_value_dict:
                    log.warning(f'{key} declared more than once in torch_func_override.yml, skipping')
                else:
                    if key == 'torch.Tensor.__getitem__':
                        objs = list(obj.__bases__) + [obj]
                        pre_patched_value_dict[key] = getattr(torch._C._TensorBase, '__getitem__')
                        new_func = gen(pre_patched_value_dict[key], key, hasattr(obj, '__module__'))
                        for o in objs:
                            key = qualified_name(o, name)
                            pre_patched_value_dict[key] = patch_getitem(o, new_func)
                    else:
                        pre_patched_value_dict[key] = getattr(obj, name)
                        setattr(obj, name, gen(pre_patched_value_dict[key], key, hasattr(obj, '__module__')))
    yield object_dict
    for object_dict, gen in zip(object_dicts, gens):
        for obj, names in object_dict.items():
            for name in names:
                key = qualified_name(obj, name)
                if key == 'torch.Tensor.__getitem__':
                    objs = list(obj.__bases__) + [obj]
                    for o in objs:
                        key = qualified_name(o, name)
                        revert_getitem(o, pre_patched_value_dict[key])
                else:
                    setattr(obj, name, pre_patched_value_dict[key])


def get_constructor_args(actual_class_type):
    """ Gets the args of the original constructor for a known module class """
    if actual_class_type in module_constructor_signatures:
        return module_constructor_signatures[actual_class_type]
    else:
        return inspect.signature(actual_class_type.__init__).parameters.values()


def gen_module_constrctor_line(module):
    """ Generates the constructor line for a loaded module """
    class_type = type(module)
    class_name = f'{class_type.__module__}.{class_type.__name__}'

    # Sometimes the class is a child class that doesn't have an actual __init__ functions
    # So we need to search the base classes as well
    actual_class_type = class_type
    while True:
        normal_arg_found = False
        positional_arg_found = False
        arg_info = get_constructor_args(actual_class_type)
        for i, p in enumerate(arg_info):
            # Skip first element
            if i == 0:
                continue

            # Skip *args and **kwargs
            if p.kind == inspect.Parameter.VAR_POSITIONAL:
                positional_arg_found = True
                continue

            if p.kind == inspect.Parameter.VAR_KEYWORD and positional_arg_found:
                continue

            # If a normal argument is found, then it should be okay
            normal_arg_found = True
            break

        # Search in the base class if we didn't find one in the current class type
        if not normal_arg_found:
            if len(actual_class_type.__bases__) > 0:
                actual_class_type = actual_class_type.__bases__[0]
            else:
                # We couldn't find one so we restore to the default type here
                actual_class_type = class_type
                break
        else:
            break

    arg_str = ''
    custom_prop_func = {'torch.nn.modules.conv.Conv2d_bias': lambda x: x is not None,
                        'torch.nn.modules.linear.Linear_bias': lambda x: x is not None,
                        'torch.nn.modules.conv.ConvTranspose2d_bias': lambda x: x is not None,
                        'torch.nn.modules.conv.Conv1d_bias': lambda x: x is not None, }
    skip_props = {'torch.nn.modules.rnn.RNN_mode',
                  'torch.nn.modules.rnn.LSTM_mode',
                  'torch.nn.modules.rnn.GRU_mode', }
    unknown_pairs = set()
    if hasattr(class_type, '__constants__') and hasattr(actual_class_type, '__init__'):
        known_constants = set(class_type.__constants__)
        arg_info = get_constructor_args(actual_class_type)
        args = []
        for p in arg_info:
            prop_name = p.name
            custom_key = f'{class_name}_{prop_name}'
            if prop_name == 'self':
                continue
            if prop_name not in known_constants:
                if custom_key not in custom_prop_func:
                    if custom_key not in unknown_pairs:
                        unknown_pairs.add(custom_key)
                        log.warning(f'Argument "{prop_name}" of the constructor of {class_name} is not a known constant, skipping')
                    continue
            if custom_key in skip_props:
                continue
            prop_value = getattr(module, prop_name)
            if custom_key in custom_prop_func:
                prop_value = custom_prop_func[custom_key](prop_value)
            # Appending positional args
            if p.default is p.empty:
                args.append(f'{prop_value}')
            else:
                # Appending keyword args
                default_value = p.default
                # Skip the arg if it has the same value with the default one
                if default_value == prop_value:
                    continue
                if type(prop_value) == str:
                    prop_value_str = f'"{prop_value}"'
                else:
                    prop_value_str = prop_value
                args.append(f'{prop_name}={prop_value_str}')
        # Argument string concatenation
        arg_str = ', '.join(args)
    return f'{class_name}({arg_str})'


def add_input_node(node: TraceNode, output_tensors):
    """ Adds an input node to the current computation graph """
    assert node is not None
    if type(output_tensors) not in [list, tuple]:
        output_tensors = [output_tensors]

    node.next_tensors.extend(output_tensors)

    for t in output_tensors:
        current_graph().tensor_pre_node_dict[id(t)] = node.unique_name

    current_graph().input_nodes.append(node)
    current_graph().nodes_map[node.unique_name] = node


def add_constant_node(node: TraceNode, output_tensor):
    """ Adds a constant node to the current computation graph """
    assert node is not None
    node.next_tensors = [output_tensor]

    current_graph().tensor_pre_node_dict[id(output_tensor)] = node.unique_name

    current_graph().constant_nodes.append(node)
    current_graph().nodes_map[node.unique_name] = node


def add_output_node(node: TraceNode, input_tensors):
    """ Adds an output node to the current computation graph """
    assert node is not None
    need_idx = True
    if type(input_tensors) not in [list, tuple]:
        input_tensors = [input_tensors]
        need_idx = False

    node.prev_tensors.extend(input_tensors)
    node.rev_index = need_idx

    for i, t in enumerate(input_tensors):
        node.prev_nodes.append(current_graph().nodes_map[current_graph().tensor_pre_node_dict[id(t)]])
        if id(t) in current_graph().tensor_pre_index_dict:
            node.prev_indices.append(current_graph().tensor_pre_index_dict[id(t)])
        else:
            node.prev_indices.append(None)

    current_graph().output_nodes.append(node)
    current_graph().nodes_map[node.unique_name] = node


def add_forward_node(node: TraceNode, input_tensors, output_tensors):
    """ Adds a forward node to the current computation graph """
    assert node is not None
    if type(input_tensors) not in [list, tuple]:
        input_tensors = [input_tensors]

    need_idx = True
    if type(output_tensors) not in [list, tuple]:
        output_tensors = [output_tensors]
        need_idx = False

    node.prev_tensors.extend(input_tensors)
    node.next_tensors.extend(output_tensors)

    for i, t in enumerate(input_tensors):
        assert type(t) in (torch.dtype, torch.device, torch.Size, torch.Tensor), \
            f'Input #{i} of {node.unique_name}({node.type()}) should be one of the following type \
            [torch.dtype, torch.device, torch.Size, torch.Tensor], but got {type(t)}'
        if id(t) not in current_graph().tensor_pre_node_dict:
            if not t.is_leaf:
                log.error(f'Connection is lost when generating code for {node.unique_name} of type {node.full_name()}')
            else:
                # constant tensor generation
                log.warning('Constant generation is experimental and may yield error')
                convert_to_parameter = False
                if t.numel() > 50 and t.is_floating_point():
                    convert_to_parameter = True
                raw_data = t.tolist()
                constant_node = ConstantNode(raw_data, t.dtype, t.shape).parse(convert_to_parameter)
                trace_node = TraceNode(constant_node)
                add_constant_node(trace_node, t)
        pre_node_name = current_graph().tensor_pre_node_dict[id(t)]
        node.prev_nodes.append(current_graph().nodes_map[pre_node_name])
        if id(t) in current_graph().tensor_pre_index_dict:
            pre_node_index = current_graph().tensor_pre_index_dict[id(t)]
            log.debug(f'propagate pre_index tensor {pre_node_name} {pre_node_index}')
            node.prev_indices.append(pre_node_index)
        else:
            node.prev_indices.append(None)

    for i, t in enumerate(output_tensors):
        if type(t) in (list, tuple):
            for j, rt in enumerate(t):
                assert type(rt) in (torch.dtype, torch.device, torch.Size, torch.Tensor), \
                    f'Output [{i}][{j}] of {node.unique_name}({node.type()}) should be one of the following type \
                    [torch.dtype, torch.device, torch.Size, torch.Tensor], but got {type(rt)}'
                current_graph().tensor_pre_node_dict[id(rt)] = node.unique_name
                if need_idx:
                    log.debug(f'set pre_index tensor {i}, {j}')
                    current_graph().tensor_pre_index_dict[id(rt)] = [i, j]
        else:
            assert type(t) in (torch.dtype, torch.device, torch.Size, torch.Tensor), \
                f'Output #{i} of {node.unique_name}({node.type()}) should be one of the following type \
                [torch.dtype, torch.device, torch.Size, torch.Tensor], but got {type(t)}'
            current_graph().tensor_pre_node_dict[id(t)] = node.unique_name
            if need_idx:
                log.debug(f'set pre_index tensor {i}')
                current_graph().tensor_pre_index_dict[id(t)] = i

    current_graph().forward_nodes.append(node)
    current_graph().nodes_map[node.unique_name] = node


@contextlib.contextmanager
def hook_modules(module):
    """ Temporarily adds the hooks to a `nn.Module` for tracing """
    hooks = []

    def register_submodule_tracer(module):
        def _submodule_pre_tracer(module, input):
            log.debug(f'pre tracer in _submodule_pre_tracer in {type(module).__name__}')
            lock(True)

        def _submodule_tracer(module, inputs, outputs):
            log.debug(f'tracer in _submodule_tracer in {type(module).__name__}')
            lock(False)
            node = TraceNode(module)
            add_forward_node(node, inputs, outputs)

        module_unique_name = current_graph().module_unique_name_dict[id(module)]
        if module_unique_name in current_graph().traced_modules:
            log.debug(f"module {module_unique_name} is traced")
            return None

        related = False
        if id(module) in module_constructor_traced:
            if id(module) in module_constructor_lines:
                related = True
        else:
            if type(module) in overridable_modules:
                related = True
            else:
                for m in overridable_modules:
                    if isinstance(module, m):
                        related = True
                        break

        if related:
            hooks.append(module.register_forward_pre_hook(_submodule_pre_tracer))
            hooks.append(module.register_forward_hook(_submodule_tracer))

        current_graph().traced_modules.append(module_unique_name)
        return None

    def _model_pre_tracer(module, inputs):
        log.debug('pre tracer in _model_pre_tracer')
        for i in inputs:
            node = TraceNode(TraceFunction("input"))
            add_input_node(node, i)

    def _model_tracer(module, inputs, outputs):
        log.debug('tracer in _model_tracer')
        if type(outputs) == torch.Tensor:
            node = TraceNode(TraceFunction("output"))
            add_output_node(node, outputs)
        elif type(outputs) in (list, tuple):
            for i in outputs:
                if type(i) == torch.Tensor or (type(i) in (list, tuple) and all((type(x) == torch.Tensor for x in i))):
                    node = TraceNode(TraceFunction("output"))
                    add_output_node(node, i)
                else:
                    log.warning(
                        "Only tensors or list, tuple of tensors are supported when nested in a class, dict, list or tuple")
        elif type(outputs) == dict:
            for k, v in outputs.items():
                if type(v) == torch.Tensor or (type(v) in (list, tuple) and all((type(x) == torch.Tensor for x in v))):
                    node = TraceNode(TraceFunction("output"))
                    add_output_node(node, v)
                else:
                    log.warning(
                        "Only tensors or list, tuple of tensors are supported when nested in a class, dict, list or tuple")
        else:
            log.warning(f'Output type is not supported: {type(outputs).__name__}, try to extract tensors from it')
            for k in outputs.__dir__():
                v = getattr(outputs, k)
                if type(v) == torch.Tensor or (type(v) in (list, tuple) and all((type(x) == torch.Tensor for x in v))):
                    node = TraceNode(TraceFunction("output"))
                    add_output_node(node, v)

    log.debug('trace: apply register_submodule_tracer')
    module.apply(register_submodule_tracer)

    log.debug('trace: add hooks')
    hooks.append(module.register_forward_pre_hook(_model_pre_tracer))
    hooks.append(module.register_forward_hook(_model_tracer))

    yield module

    for hook in hooks:
        hook.remove()


@contextlib.contextmanager
def tracer_context():
    """ Basic context manager for tracing """
    yield True
    lock(False)
    module_constructor_traced.clear()
    module_constructor_lines.clear()
    original_values_for_tracked_objects.clear()


@contextlib.contextmanager
def model_constructor_tracer():
    """ Basic context manager for capturing constructors for `nn.Module` """
    with patch_helper(wrap_funcs=False, wrap_creation_funcs=False):
        yield True


@contextlib.contextmanager
def model_tracer():
    """ Simple context manager for tracing. Also captures module constructors """
    with tracer_context():
        with model_constructor_tracer():
            yield True


@contextlib.contextmanager
def construct_trace_graph(module, dummy_input: torch.Tensor, eliminate_dead_graph: bool) -> 'TraceGraph':
    """ Simple context manager for creating a new TraceGraph """
    current_graph(TraceGraph(module, dummy_input, eliminate_dead_graph))
    yield current_graph.get_value()
    current_graph(None)


@contextlib.contextmanager
def override_current_trace_graph(new_graph: 'TraceGraph') -> 'TraceGraph':
    """ Simple context manager for creating a new TraceGraph """
    old_graph = current_graph.get_value()
    current_graph(new_graph)
    yield current_graph.get_value()
    current_graph(old_graph)


class TraceGraph(object):
    """ A data structure for storing a computation graph """
    global_functions: typing.Dict[str, int]
    global_nodes: typing.Dict[str, int]
    module_unique_name_dict: typing.Dict[int, torch.nn.Module]
    module_original_name_dict: typing.Dict[int, str]
    traced_modules: typing.List[str]
    input_nodes: typing.List[TraceNode]
    forward_nodes: typing.List[TraceNode]
    output_nodes: typing.List[TraceNode]
    constant_nodes: typing.List[TraceNode]
    other_init_nodes: typing.List[TraceNode]
    nodes_map: typing.Dict[str, TraceNode]
    tensor_pre_node_dict: typing.Dict[int, str]
    tensor_pre_index_dict: typing.Dict[int, int]
    module: torch.nn.Module
    dummy_input: torch.Tensor
    eliminate_dead_graph: bool
    inited: bool
    quantized: bool
    code: str

    def __init__(self, module: torch.nn.Module, dummy_input: torch.Tensor, eliminate_dead_graph: bool = False):
        # Used for function / node numbering
        self.global_functions = {}
        self.global_nodes = {}

        # Unique name for modules and submodules
        self.module_unique_name_dict = {}
        self.module_original_name_dict = {}

        # Recording traced modules
        self.traced_modules = []

        # Recording nodes
        self.input_nodes = []
        self.forward_nodes = []
        self.output_nodes = []
        self.constant_nodes = []
        self.other_init_nodes = []

        # Node <-> name mapping
        self.nodes_map = {}

        # Recording the previous node of the tensors
        self.tensor_pre_node_dict = {}

        # Recording the previous index of the tensors
        self.tensor_pre_index_dict = {}

        # Input module
        if isinstance(module, DataParallel) or isinstance(module, DistributedDataParallel):
            log.error(
                'You are tracing a parallel module, which is unsupported. Please pass in a raw model using `.module`.')
            assert False
        else:
            self.module = module

        # Input data
        self.dummy_input = dummy_input

        # Whether to keep inactive nodes after tracing
        self.eliminate_dead_graph = eliminate_dead_graph

        # Let's give the module and its children a name.
        self.__tag_nodes()

        # Whether the tracing is completed or not
        self.inited = False

        # Whether the graph is rewrited to be a quantized one
        self.quantized = False

        # Generated code
        self.code = "None"

    def all_nodes(self) -> typing.List[TraceNode]:
        """ Returns all the nodes in a computation graph during forward process """
        return self.input_nodes + self.forward_nodes + self.output_nodes + self.constant_nodes

    def __tag_nodes(self) -> None:
        """ Gives the modules and the submodules a unique name """
        # Tag submodules
        for n, m in self.module.named_modules():
            self.module_original_name_dict[id(m)] = n
            n = n.replace(".", "_")
            n = n.replace("-", "_")
            n = 'module_' + n if n.isnumeric() else n
            self.module_unique_name_dict[id(m)] = n
        # Tag the module itself
        self.module_unique_name_dict[id(self.module)] = type(self.module).__name__

    def __active_detection(self, node: TraceNode):
        """ Detects whether the node is active or not """
        if not node.active:
            node.active = True
            for i in node.prev_nodes:
                self.__active_detection(i)

    def init(self) -> None:
        """ Builds a computation graph """
        if self.inited:
            return
        with self.__numbering_context():
            if type(self.dummy_input) == torch.Tensor:
                actual_input = [self.dummy_input]
            elif type(self.dummy_input) in (tuple, list):
                actual_input = list(self.dummy_input)
            else:
                log.error(f'Unsupported type {type(self.dummy_input)} for dummy input')
                assert False

            for i in range(len(actual_input)):
                dummy_input = actual_input[i]
                if type(dummy_input) == torch.Tensor:
                    new_input = dummy_input.detach().clone()
                    if new_input.is_floating_point():
                        new_input.requires_grad = True
                    actual_input[i] = new_input

            original_state_dict = copy.deepcopy(self.module.state_dict())

            with patch_helper():
                with hook_modules(self.module):
                    self.module(*actual_input)

            self.module.load_state_dict(original_state_dict)

            if self.eliminate_dead_graph:
                for n in self.input_nodes + self.forward_nodes + self.output_nodes:
                    n.active = False

                for i in self.output_nodes:
                    self.__active_detection(i)

            active_input_nodes = [i for i in self.input_nodes if i.active]
            active_forward_nodes = [i for i in self.forward_nodes if i.active]

            self.input_nodes = active_input_nodes
            self.forward_nodes = active_forward_nodes

            forward_order = 0
            for n in self.input_nodes + self.forward_nodes + self.output_nodes:
                n.forward_order = forward_order
                forward_order += 1
                for prev in n.prev_nodes:
                    if n not in prev.next_nodes:
                        prev.next_nodes.append(n)
        self.inited = True

    @contextlib.contextmanager
    def __numbering_context(self):
        """ A simple context manager for numbering nodes """
        yield True

        self.global_functions.clear()
        self.global_nodes.clear()

    def __gen_init_code(self) -> str:
        """ Generates the code for the init function for a `nn.Module` """
        generated_node = []
        lines = []
        for node in self.constant_nodes + self.forward_nodes + self.other_init_nodes:
            if node.unique_name in generated_node:
                log.info(f"skip dumplicate node code gen {node.unique_name}")
                continue

            generated_node.append(node.unique_name)
            if id(node.module) in module_constructor_lines:
                orig_constructor_line = module_constructor_lines[id(node.module)]
                line = f'        self.{node.unique_name} = {orig_constructor_line}'
                lines.append(line)
            elif type(node.module) == ConstantNode:
                # Parameter generation
                line = f'        self.register_parameter("{node.unique_name}", torch.nn.Parameter(torch.empty({node.module.shape}, dtype={node.module.dtype})))'
                lines.append(line)
            elif type(node.module) != TraceFunction:
                # Generate the module even if the constructor is not caught
                log.info(
                    f'the constructor of the module {node.unique_name} of type {type(node.module).__name__} is not traced, trying the experimental way')
                line = f'        self.{node.unique_name} = {gen_module_constrctor_line(node.module)}'
                lines.append(line)

        block = "\n".join(lines)
        return block

    def __gen_forward_code(self) -> str:
        """ Generates the code for the forward function for a `nn.Module` """
        lines = [
            f"    def forward(self, {','.join([i.unique_name for i in self.input_nodes])}):"]

        for node in self.forward_nodes:
            output = ", ".join([node.unique_name])
            param = ", ".join([node.prev_node_unique_name(i)
                               for i in range(len(node.prev_nodes))])

            if type(node.module) == TraceFunction:
                node_type = node.type()
                if node.is_class():
                    if node.module.is_property:
                        line = f"        {output} = {node.prev_node_unique_name(0)}.{node_type}"
                    elif node_type == '__getitem__':
                        args = node.module.args_string_no_self
                        if not args.startswith('[') and not args.endswith(']'):
                            args = f'[{args}]'
                        line = f"        {output} = {node.prev_node_unique_name(0)}{args}"
                    else:
                        # Since the node is a class and the member function is called, the first argument should be removed.
                        args = node.module.args_string_no_self
                        line = f"        {output} = {node.prev_node_unique_name(0)}.{node_type}({args})"
                else:
                    args = node.module.args_string
                    line = f"        {output} = {node.full_name()}({args})"
            else:
                line = f"        {output} = self.{node.unique_name}({param})"
            lines.append(line)

        def _gen_output_node(node):
            if node.rev_index:
                return f'[{", ".join([node.prev_node_unique_name(i) for i in range(len(node.prev_nodes))])}]'
            else:
                return node.prev_node_unique_name(0)

        lines.append(f"        return {', '.join([_gen_output_node(i) for i in self.output_nodes])}")

        block = "\n".join(lines)

        return block

    def __gen_import_code(self) -> str:
        """ Generates the code for the import section for a `nn.Module` """
        # TODO: Selective module importing
        import_block = """import torch\nimport torch.nn\nimport torch.functional\nimport torch.nn.functional"""
        if self.quantized is True:
            import_block += '\nimport torch.quantization\nimport torch.nn.quantized'
        return import_block

    def __gen_input_code(self) -> str:
        """ Generates the code for the input section for the code to invoke `nn.Module` """
        input_block = ""
        for i, node in enumerate(self.input_nodes):
            shape = ", ".join((str(i) for i in node.next_tensors[0].shape))
            dtype = node.next_tensors[0].dtype

            if i != 0:
                input_block += "\n"

            input_block += f"    dummy_input_{i} = torch.ones(({shape}), dtype={dtype})"

        return input_block

    def generate_code(self, output_script_path: typing.Optional[str], output_weight_path: typing.Optional[str],
                      model_name: str = 'DefaultModel', check: bool = False) -> bool:
        """ The main function for code generation """

        output_paths = (output_script_path, output_weight_path)
        for output_path in output_paths:
            if not output_path:
                continue
            output_dir = os.path.dirname(output_path)
            if output_dir != '' and not os.path.exists(output_dir):
                os.makedirs(output_dir)

        DummyModel = type('DummyModel', (torch.nn.Module,), {})
        dummy_model = DummyModel()
        for node in self.forward_nodes:
            setattr(dummy_model, node.unique_name, node.module)

        for node in self.constant_nodes:
            if id(node.module) not in module_constructor_lines:
                dtype = getattr(torch, node.module.dtype.split('.')[-1])
                weight = torch.nn.Parameter(torch.tensor(node.module.data, dtype=dtype))
                dummy_model.register_parameter(node.unique_name, weight)

        if output_weight_path:
            torch.save(dummy_model.state_dict(), output_weight_path)

        init_block = self.__gen_init_code()
        forward_block = self.__gen_forward_code()
        import_block = self.__gen_import_code()
        input_block = self.__gen_input_code()

        context = {
            "import_block": import_block,
            "init_block": init_block,
            "forward_block": forward_block,
            "name_block": model_name,
            "load_weight_block": "" if output_weight_path is None else f"    model.load_state_dict(torch.load('{output_weight_path}s'))",
            "input_block": input_block,
            "input_names": ", ".join([f"dummy_input_{i}" for i in range(len(self.input_nodes))])
        }

        code = MODULE_TEMPLATE % context

        if output_script_path:
            if os.path.exists(output_script_path):
                os.remove(output_script_path)
            with io.open(output_script_path, 'w') as f:
                f.write(code)

            if check:
                training_state = self.module.training
                valid = True

                with torch.no_grad():
                    self.module.eval()
                    original_outputs = tensors2ndarray(self.module(*self.dummy_input))

                    new_module = import_from_path(f'tracer_check.{model_name}', output_script_path, model_name)()
                    new_module.eval()
                    if output_weight_path:
                        new_module.load_state_dict(torch.load(output_weight_path))

                    new_outputs = tensors2ndarray(new_module(*self.dummy_input))

                    for i in range(len(original_outputs)):
                        output = original_outputs[i]
                        new_output = new_outputs[i]

                        if not np.allclose(output, new_output):
                            log.warning(f"[WARNING] Output {i} is not equal.")
                            valid = False

                if training_state:
                    self.module.train()

                return valid
        else:
            return True

    def update_submodule_in_nodes_from_predicate(self, nodes: typing.List[TraceNode],
                                                 module_gen_predicate: typing.Callable[[nn.Module], nn.Module]):
        """ update a submodule from the nodes using the predicate given """
        for node in nodes:
            module = node.module
            new_module = module_gen_predicate(module)
            self.update_submodule_in_node(node, new_module)

    def update_submodule_in_node(self, node: TraceNode, module: nn.Module):
        """ update a submodule from the nodes using the module given """
        module_name = self.module_original_name_dict[id(node.module)]
        module_name_parts = module_name.split('.')
        cur_obj = self.module

        for ns in module_name_parts[1:-1]:
            if type(cur_obj) == nn.ModuleList:
                cur_obj = cur_obj[int(ns)]
            elif type(cur_obj) == nn.ModuleDict:
                cur_obj = cur_obj[ns]
            else:
                cur_obj = getattr(cur_obj, ns)

        ns = module_name_parts[-1]
        new_obj = module
        if type(cur_obj) == nn.ModuleList:
            cur_obj[int(ns)] = new_obj
        elif type(cur_obj) == nn.ModuleDict:
            cur_obj[ns] = new_obj
        else:
            setattr(cur_obj, ns, new_obj)

    def filter_forward_nodes(self, predicate, custom_data=None, reverse=False) -> typing.List[TraceNode]:
        """ A utility function for filtering forward nodes """
        nodes = []
        iter_nodes = self.forward_nodes
        if reverse:
            iter_nodes = reversed(iter_nodes)
        for node in iter_nodes:
            if predicate(node, custom_data):
                nodes.append(node)
        return nodes

    def insert_after(self, node: TraceNode, module, next_tensors: typing.Optional[typing.List[torch.Tensor]] = None):
        """ Insert a module or an existing node after a node in the computation graph """
        # Create a new node and connects it to the next node/tensors
        if type(module) != TraceNode:
            new_node = TraceNode(module, cur_graph=self)
            if node in self.input_nodes or node in self.constant_nodes:
                self.forward_nodes.insert(0, new_node)
            elif node in self.output_nodes:
                log.error('You cannot insert a node after output nodes')
                assert False
            else:
                idx = self.forward_nodes.index(node)
                self.forward_nodes.insert(idx + 1, new_node)
            self.nodes_map[new_node.unique_name] = new_node
        else:
            new_node = module

        new_node.prev_nodes.append(node)
        new_node.next_nodes.extend(node.next_nodes)
        if next_tensors is None:
            next_tensors = [None] * len(node.next_tensors)
        for t, new_t in zip(node.next_tensors, next_tensors):
            if new_t is None:
                new_t = t.clone()
            self.tensor_pre_node_dict[id(new_t)] = new_node.unique_name
            new_node.prev_tensors.append(t)
            new_node.next_tensors.append(new_t)
            new_node.prev_indices.append(None)

        # Make input/constant nodes connects to the new node
        node.next_nodes.clear()
        node.next_nodes.append(new_node)

        # Connect the next nodes to the new node
        tensor_replace_dict = dict(zip(new_node.prev_tensors, new_node.next_tensors))
        for next_node in new_node.next_nodes:
            for i, n in enumerate(next_node.prev_nodes):
                if n == node:
                    next_node.prev_nodes[i] = new_node
                    break

            # Make sure the data is writable
            if type(next_node.prev_tensors) == tuple:
                next_node.prev_tensors = list(next_node.prev_tensors)

            for i, t in enumerate(next_node.prev_tensors):
                if t in tensor_replace_dict:
                    next_node.prev_tensors[i] = tensor_replace_dict[t]

            # Since the function calls are rendered beforehand,
            # we need to change them as well.
            if type(next_node.module) == TraceFunction:
                if next_node.module.args_string is not None:
                    ns = ''
                    if type(node.module) == ConstantNode:
                        ns = 'self.'
                    node_unique_name = f'{ns}{node.unique_name}'
                    next_node.module.args_string = next_node.module.args_string.replace(
                        node_unique_name, new_node.unique_name)
                    next_node.module.args_string_no_self = next_node.module.args_string_no_self.replace(
                        node_unique_name, new_node.unique_name)

    def insert_between(self, prev_node: TraceNode, next_node: TraceNode, module,
                       next_tensors: typing.Optional[typing.List[torch.Tensor]] = None, move_idx: bool = False):
        """ Insert a module or an existing node between two nodes in the computation graph """
        # Create a new node and connects it to the previous node/tensors
        old_unique_name = prev_node.unique_name
        is_constant_node = type(prev_node.module) == ConstantNode

        if type(module) != TraceNode:
            new_node = TraceNode(module, cur_graph=self)

            if prev_node not in next_node.prev_nodes or next_node not in prev_node.next_nodes:
                log.error('You cannot insert a node between two nodes that is not connected')
                assert False

            idx = self.forward_nodes.index(next_node)
            self.forward_nodes.insert(idx, new_node)

            self.nodes_map[new_node.unique_name] = new_node
            new_node.prev_nodes.append(prev_node)
            new_node.next_nodes.append(next_node)
        else:
            new_node = module

        # Gather tensors from previous nodes
        prev_tensors = []
        prev_indices = []
        for pt, pidx in zip(next_node.prev_tensors, next_node.prev_indices):
            for nt in prev_node.next_tensors:
                if id(pt) == id(nt):
                    prev_tensors.append(pt)
                    prev_indices.append(pidx)
                    break

        if next_tensors is None:
            next_tensors = [None] * len(prev_tensors)

        for idx, (t, new_t, pidx) in enumerate(zip(prev_tensors, next_tensors, prev_indices)):
            if new_t is None:
                new_t = t.clone()
                next_tensors[idx] = new_t

            self.tensor_pre_node_dict[id(new_t)] = new_node.unique_name
            new_node.prev_tensors.append(t)
            new_node.next_tensors.append(new_t)
            new_node.prev_indices.append(pidx if move_idx else None)

        # Make output nodes connects to the new node
        for idx in range(len(next_node.prev_nodes)):
            if next_node.prev_nodes[idx] == prev_node:
                next_node.prev_nodes[idx] = new_node
                break

        # Update tensors in output nodes
        for idx, t in enumerate(next_node.prev_tensors):
            for pt, nt in zip(prev_tensors, next_tensors):
                if id(t) == id(pt):
                    next_node.prev_tensors[idx] = nt
                    if move_idx:
                        next_node.prev_indices[idx] = None
                    break

        # Connect the previous nodes to the new node
        for prev_node in new_node.prev_nodes:
            for i, n in enumerate(prev_node.next_nodes):
                if n == next_node:
                    prev_node.next_nodes[i] = new_node
                    break

        # Update previous node name for next nodes (TraceFunction)
        if type(next_node.module) == TraceFunction:
            ns = ''
            if is_constant_node:
                ns = 'self.'
            prev_unique_name = f'{ns}{old_unique_name}'
            log.debug('node rename: ', old_unique_name, '->', new_node.unique_name)
            if n.module.args_string_no_self is not None:
                n.module.args_string_no_self = n.module.args_string_no_self.replace(
                    prev_unique_name, new_node.unique_name)
                n.module.args_string = n.module.args_string.replace(prev_unique_name, new_node.unique_name)

    def insert_before(self, node: TraceNode, module, next_tensors: typing.Optional[typing.List[torch.Tensor]] = None):
        """ Insert a module or an existing node before a node in the computation graph """
        # Create a new node and connects it to the previous node/tensors
        if type(module) != TraceNode:
            if type(module) not in (tuple, list):
                modules = [module]
            else:
                if not node.rev_index:
                    log.error('You can only insert nodes with a list modules when node.rev_index=True')
                    assert False

                if len(module) != len(node.prev_nodes):
                    log.error(f'The number of the modules provided is wrong, expected: {len(node.prev_nodes)}')
                    assert False

                modules = module

            new_nodes: typing.List[TraceNode] = []
            for module in modules:
                new_node = TraceNode(module, cur_graph=self)
                new_nodes.append(new_node)
        else:
            new_nodes = [module]

        if node in self.input_nodes or node in self.constant_nodes:
            log.error('You cannot insert a node before input/constant nodes')
            assert False
        elif node in self.output_nodes:
            for new_node in new_nodes:
                self.forward_nodes.append(new_node)
        else:
            for new_node in new_nodes:
                idx = self.forward_nodes.index(node)
                self.forward_nodes.insert(idx, new_node)

        for idx, new_node in enumerate(new_nodes):
            self.nodes_map[new_node.unique_name] = new_node
            if len(new_nodes) == 1:
                new_node.prev_nodes.extend(node.prev_nodes)
            else:
                new_node.prev_nodes.append(node.prev_nodes[idx])
            new_node.next_nodes.append(node)

        for idx, new_node in enumerate(new_nodes):
            if len(new_nodes) == 1:
                prev_tensors = node.prev_tensors
            else:
                prev_tensors = []
                for t in node.prev_tensors:
                    for nt in new_node.prev_nodes[0].next_tensors:
                        if id(t) == id(nt):
                            prev_tensors.append(t)

            if next_tensors is None:
                next_tensors = [None] * len(prev_tensors)
            for t, new_t in zip(node.prev_tensors, next_tensors):
                if new_t is None:
                    new_t = t.clone()
                self.tensor_pre_node_dict[id(new_t)] = new_node.unique_name
                new_node.prev_tensors.append(t)
                new_node.next_tensors.append(new_t)
                new_node.prev_indices.append(None)

        # Make output nodes connects to the new node
        node.prev_nodes.clear()
        node.prev_nodes.extend(new_nodes)
        node.prev_tensors.clear()
        for new_node in new_nodes:
            node.prev_tensors.extend(new_node.next_tensors)

        # Connect the previous nodes to the new node
        for new_node in new_nodes:
            for prev_node in new_node.prev_nodes:
                for i, n in enumerate(prev_node.next_nodes):
                    if n == node:
                        prev_node.next_nodes[i] = new_node
                        break

        # Update previous node name for next nodes (TraceFunction)
        if type(node.module) == TraceFunction and node not in self.output_nodes:
            new_node = new_nodes[0]
            old_unique_name = new_node.prev_nodes[0].unique_name
            is_constant_node = type(new_node.prev_nodes[0].module) == ConstantNode
            ns = ''
            if is_constant_node:
                ns = 'self.'
            prev_unique_name = f'{ns}{old_unique_name}'
            log.debug('node rename: ', old_unique_name, '->', new_node.unique_name)
            if node.module.args_string_no_self is not None:
                node.module.args_string_no_self = node.module.args_string_no_self.replace(
                    prev_unique_name, new_node.unique_name)
                node.module.args_string = node.module.args_string.replace(prev_unique_name, new_node.unique_name)

    def replace_node_module(self, node: TraceNode, module: torch.nn.Module) -> None:
        """ Replaces a module in a node with another """
        # Update unique name for node
        old_unique_name = node.unique_name
        is_constant_node = type(node.module) == ConstantNode
        node.unique_name = self.module_unique_name_dict[id(module)]

        # Update module for node
        node.module = module

        # Update node map
        del self.nodes_map[old_unique_name]
        self.nodes_map[node.unique_name] = node

        # Update previous node name for next tensors
        for t in node.next_tensors:
            self.tensor_pre_node_dict[id(t)].replace(old_unique_name, node.unique_name)

        # Update previous node name for next nodes (TraceFunction)
        for n in node.next_nodes:
            if type(n.module) == TraceFunction:
                ns = ''
                if is_constant_node:
                    ns = 'self.'
                prev_unique_name = f'{ns}{old_unique_name}'
                log.debug('node rename: ', old_unique_name, '->', node.unique_name)
                if n.module.args_string_no_self is not None:
                    n.module.args_string_no_self = n.module.args_string_no_self.replace(
                        prev_unique_name, node.unique_name)
                    n.module.args_string = n.module.args_string.replace(prev_unique_name, node.unique_name)

    def fuse_nodes_to_func(self, nodes: typing.List[TraceNode], full_name: str, kind: str, func_type: str,
                           is_class: bool) -> None:
        """ Fuses several nodes into one function """
        if len(nodes) > 1:
            # Set the full name if the first node is already a TraceFunction
            # Otherwise, we need to construct one.
            next_nodes = []
            next_tensors = []
            if type(nodes[0].module) == TraceFunction:
                next_nodes.extend(nodes[-1].next_nodes)
                next_tensors.extend(nodes[-1].next_tensors)

                last_node_unique_name = nodes[-1].unique_name
                first_node_unique_name = nodes[0].unique_name

                for node in nodes[1:]:
                    name = node.unique_name
                    del self.nodes_map[name]
                    self.forward_nodes.remove(node)

                node = nodes[0]

                node.next_nodes.clear()
                node.next_tensors.clear()

                node.next_nodes.extend(next_nodes)
                node.next_tensors.extend(next_tensors)

                node.module.func_type = func_type
                node.module.is_class = is_class
                node.module.kind = kind
                node.module.full_name = full_name

                # Update next tensors
                for t in node.next_tensors:
                    self.tensor_pre_node_dict[id(t)] = self.tensor_pre_node_dict[id(
                        t)].replace(last_node_unique_name, first_node_unique_name)

                for n in node.next_nodes:
                    # Update next nodes
                    for i, pn in enumerate(n.prev_nodes):
                        if pn.unique_name == last_node_unique_name:
                            n.prev_nodes[i] = node
                    # Rewrite func calls in next nodes
                    if type(n.module) == TraceFunction:
                        if n.module.args_string is not None:
                            n.module.args_string = n.module.args_string.replace(
                                last_node_unique_name, first_node_unique_name)
                            n.module.args_string_no_self = n.module.args_string_no_self.replace(
                                last_node_unique_name, first_node_unique_name)

            else:
                # TODO: Implement this codepath
                log.error('Module fusion requires the first node to be a TraceFunction.')
                raise NotImplementedError
        else:
            log.warning('Calling fuse with less than 2 nodes is no-op.')

    def remove_node(self, node: TraceNode) -> None:
        """ Remove a node from the computation graph """
        if node not in self.forward_nodes:
            log.error('Only forward nodes can be removed')
            assert False

        if len(node.prev_nodes) != 1:
            log.error('You cannot remove a node with multiple input nodes')
            assert False

        if len(node.prev_tensors) != len(node.next_tensors):
            log.error('You cannot remove a node in which the size of input tensors and the output tensors is different')
            assert False

        for idx, (prev_tensor, next_tensor) in enumerate(zip(node.prev_tensors, node.next_tensors)):
            if prev_tensor.shape != next_tensor.shape:
                log.error(f'The shape of the input/output at index {idx} mismatches')
                log.error(f'The shape of the input tensor is {prev_tensor.shape}')
                log.error(f'The shape of the output tensor is {next_tensor.shape}')
                assert False

        prev_node = node.prev_nodes[0]
        next_nodes = node.next_nodes
        tensor_dict = dict(zip(node.next_tensors, node.prev_tensors))
        is_constant_node = type(node.module) == ConstantNode

        # Deal with previous nodes
        if node in prev_node.next_nodes:
            prev_node.next_nodes.remove(node)
        else:
            log.error('Current node is not in the next nodes of the previous node')
            assert False
        prev_node.next_nodes.extend(next_nodes)

        # Deal with next nodes
        for n in next_nodes:
            # Handle previous nodes
            for i, pn in enumerate(n.prev_nodes):
                if pn == node:
                    n.prev_nodes[i] = prev_node
                    break
            # Handle previous tensors
            for i, pt in enumerate(n.prev_tensors):
                if pt in tensor_dict:
                    n.prev_tensors[i] = tensor_dict[pt]
            # Rewrite func calls in next nodes
            if type(n.module) == TraceFunction:
                if n.module.args_string is not None:
                    ns = ''
                    if is_constant_node:
                        ns = 'self.'
                    prev_unique_name = f'{ns}{node}'
                    n.module.args_string = n.module.args_string.replace(prev_unique_name, prev_node.unique_name)
                    n.module.args_string_no_self = n.module.args_string_no_self.replace(
                        prev_unique_name, prev_node.unique_name)

        # Remove this node
        self.forward_nodes.remove(node)
        del self.node_map[node.unique_name]


@contextlib.contextmanager
def patch_helper(wrap_modules: bool = True, wrap_funcs: bool = True, wrap_creation_funcs: bool = True):
    """ Temporarily monkeypatches the functions and the modules in PyTorch. """
    if wrap_modules:
        if not modules_overrided():
            if overridable_modules_loaded():
                modules = overridable_modules
            else:
                modules = fetch_modules()
                overridable_modules.extend(modules)
                overridable_modules_loaded(True)
            modules_overrided(True)
        else:
            wrap_modules = False
    if wrap_funcs:
        if not funcs_overrided():
            if overridable_funcs_loaded():
                funcs = overridable_funcs
            else:
                funcs = fetch_funcs()
                overridable_funcs.update(funcs)
                overridable_funcs_loaded(True)
            funcs_overrided(True)
        else:
            wrap_funcs = False
    if wrap_creation_funcs:
        if not creation_funcs_overrided():
            if overridable_creation_funcs_loaded():
                creation_funcs = overridable_creation_funcs
            else:
                creation_funcs = fetch_funcs(os.path.join(current_dir, 'configs/torch_creation_funcs_override.yml'))
                overridable_creation_funcs.update(creation_funcs)
                overridable_creation_funcs_loaded(True)
            creation_funcs_overrided(True)
        else:
            wrap_creation_funcs = False
    if wrap_modules:
        module_manager = patch_modules(modules, ('__init__', '__setattr__'), (new_init_gen, new_setattr_gen))
        module_manager.__enter__()
    if wrap_funcs:
        func_manager = patch_funcs((funcs, {torch.Tensor: ['__getattribute__']}), (new_func_gen, new_getattr_gen))
        func_manager.__enter__()
    if wrap_creation_funcs:
        creation_func_manager = patch_funcs(creation_funcs, new_creation_func_gen)
        creation_func_manager.__enter__()
    yield True
    if wrap_modules:
        module_manager.__exit__(None, None, None)
        modules_overrided(False)
    if wrap_funcs:
        func_manager.__exit__(None, None, None)
        funcs_overrided(False)
    if wrap_creation_funcs:
        creation_func_manager.__exit__(None, None, None)
        creation_funcs_overrided(False)


def check_types(values: typing.Iterable) -> bool:
    """ Checks whether unsupported types are in the args. """
    for value in values:
        if type(value) in (tuple, list):
            res = check_types(value)
            if res is not None:
                return res
        elif type(value) not in (int, float, bool, str, type(None)):
            return type(value).__name__
    return None


def check_tensor_type(value) -> bool:
    """ Check whether types are related to torch.Tensor. """
    if type(value) in (tuple, list):
        for item in value:
            res = check_tensor_type(item)
            if res:
                return res
    elif type(value) == torch.Tensor:
        return True
    return False


def check_creation_args(args: typing.Iterable) -> typing.Tuple:
    """ Cast arguments of type of Tensor to normal values """
    new_args = []
    for arg in args:
        if type(arg) in (tuple, list):
            new_args.append(check_creation_args(arg))
        elif type(arg) == torch.Tensor:
            if arg.dim() == 0:
                new_args.append(arg.item())
            else:
                new_args.append(arg.tolist())
        else:
            new_args.append(arg)
    return tuple(new_args)


def trace(module: torch.nn.Module, dummy_input: torch.Tensor, eliminate_dead_graph: bool = False) -> TraceGraph:
    """ main function for tracing """
    try:
        with construct_trace_graph(module, dummy_input, eliminate_dead_graph) as new_graph:
            new_graph.init()
            return new_graph
    except Exception:
        traceback.print_exc()
        if current_graph() is not None:
            log.error(f'inputs: {[n.unique_name for n in current_graph().input_nodes]}')
            log.error(f'forwards: {[n.unique_name for n in current_graph().forward_nodes]}')
            log.error(f'outputs: {[n.unique_name for n in current_graph().output_nodes]}')
            log.error(f'constants: {[n.unique_name for n in current_graph().constant_nodes]}')
        quit()
