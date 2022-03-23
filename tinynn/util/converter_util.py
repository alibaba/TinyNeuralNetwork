import json
import os
import typing
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from tinynn.util.util import get_logger

log = get_logger(__name__)


def tensor_config(tensors: typing.List[torch.Tensor], transpose: typing.List[bool], with_shape: bool) -> OrderedDict:
    """Generate the tensor info needed for the config file for the TinyNeuralNetwork converter

    Args:
        tensors (typing.List[torch.Tensor]): The tensors to gather info
        transpose (typing.List[bool]): Whether to perform nchw-nhwc transpose for the tensors
        with_shape (bool): Whether to dump the shape of the tensor

    Returns:
        OrderedDict: The info of the tensors
    """

    tensor_list = []
    for t, trans in zip(tensors, transpose):
        tensor_dict = OrderedDict()
        if with_shape:
            tensor_dict['shape'] = list(t.shape)
        try:
            tensor_dict['type'] = str(t.detach().numpy().dtype)
        except Exception:
            type_str = str(t.dtype)
            if 'torch.q' in type_str:
                type_str = type_str.replace('torch.q', '')
            else:
                type_str = type_str.replace('torch.', '')
            tensor_dict['type'] = type_str
        if trans is None:
            trans = len(t.shape) == 4
        tensor_dict['transpose'] = trans
        tensor_list.append(tensor_dict)
    return tensor_list


def generate_converter_config(
    inputs: typing.List[torch.Tensor],
    outputs: typing.List[torch.Tensor],
    input_transpose: typing.Union[typing.Iterable[bool], bool],
    output_transpose: typing.Union[typing.Iterable[bool], bool],
    export_file: str,
    tflite_path: typing.Optional[str] = None,
    config_path: typing.Optional[str] = None,
):
    """ Generate a config file that will work for the TinyNeuralNetwork converter

    Args:
        inputs (typing.List[torch.Tensor]): The input tensors
        outputs (typing.List[torch.Tensor]): The output tensors
        input_transpose (typing.Union[typing.Iterable[bool], bool]): The flag whether to insert transpose after the \
            input nodes
        output_transpose (typing.Union[typing.Iterable[bool], bool]): The flag whether to insert transpose after the \
            output nodes
        export_file (str): The path of the generate torchscript model
        tflite_path (str): The path of the generate tflite model. Defaults to None.
        config_path (str): The path of the generate config. Defaults to None.

    Raises:
        AssertionError: input transpose should either be boolean or list of booleans, under the latter condition, \
             the size of the list should be the same of that of the inputs
    """
    if type(input_transpose) in (tuple, list):
        if len(input_transpose) != len(inputs) or not all((type(x) == bool for x in input_transpose)):
            raise AssertionError('input transpose should either be boolean or list of booleans')
    elif type(input_transpose) == bool or input_transpose is None:
        input_transpose = [input_transpose] * len(inputs)
    else:
        raise AssertionError('input transpose should either be boolean or list of booleans')

    if type(output_transpose) in (tuple, list):
        if len(output_transpose) != len(outputs) or not all((type(x) == bool for x in output_transpose)):
            raise AssertionError('output transpose should either be boolean or list of booleans')
    elif type(output_transpose) == bool or output_transpose is None:
        output_transpose = [output_transpose] * len(inputs)
    else:
        raise AssertionError('output transpose should either be boolean or list of booleans')

    if tflite_path is None:
        tflite_path = export_file.replace('.pt', '.tflite')

    json_obj = OrderedDict(
        {
            'src_model': export_file,
            'dst_model': tflite_path,
            'inputs': tensor_config(inputs, input_transpose, True),
            'outputs': tensor_config(outputs, output_transpose, False),
        }
    )

    if config_path is None:
        config_path = export_file.replace('.pt', '.json')

    with open(config_path, 'w') as f:
        json.dump(json_obj, f, indent=4)


def export_converter_files(
    model: nn.Module,
    dummy_input: typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]],
    export_dir: typing.Optional[str] = None,
    model_name: typing.Optional[str] = None,
    input_transpose: typing.Optional[typing.Union[bool, typing.Iterable[bool]]] = None,
    output_transpose: typing.Optional[typing.Union[bool, typing.Iterable[bool]]] = None,
    dump_graph: bool = False,
):
    """ Automatically generate required files for the model converter

    Args:
        model (nn.Module): The input model
        dummy_input (typing.Union[torch.Tensor, typing.Iterable[torch.Tensor]]): A viable input to the model
        export_dir (typing.Optional[str], optional): Directory to use for exporting. Defaults to None(os.getcwd()).
        model_name (typing.Optional[str], optional): File name for exporting. Defaults to None("jit_model").
        input_transpose (typing.Optional[typing.Union[bool, typing.Iterable[bool]]], optional): Whether to transpose \
            the input(s). Defaults to None(True for 4d-input, False otherwise).
        output_transpose (typing.Optional[typing.Union[bool, typing.Iterable[bool]]], optional): Whether to transpose \
            the input(s). Defaults to None(True for 4d-output, False otherwise).
        dump_graph (bool, optional): Whether to print the traced graph. Defaults to False.
    """

    if export_dir is None:
        export_dir = os.getcwd()

    if model_name is None:
        model_name = 'jit_model'

    export_file = os.path.abspath(os.path.join(export_dir, f'{model_name}.pt'))

    script = torch.jit.trace(model, dummy_input)

    if dump_graph:
        log.info(script.inlined_graph)

    os.makedirs(export_dir, exist_ok=True)
    torch.jit.save(script, export_file)

    if type(dummy_input) not in (tuple, list):
        inputs = [dummy_input]
    else:
        inputs = dummy_input

    output = model(*inputs)

    if type(output) not in (tuple, list):
        outputs = [output]
    else:
        new_output = []
        for item in output:
            if type(item) in (tuple, list):
                new_output.extend(item)
            else:
                new_output.append(item)
        outputs = new_output

    generate_converter_config(inputs, outputs, input_transpose, output_transpose, export_file)


def get_tensor_details(config):
    input_shapes = []
    input_transpose = []
    input_dtypes = []
    output_transpose = []

    input_shapes.extend((inp['shape'] for inp in config['inputs']))
    input_transpose.extend((inp['transpose'] for inp in config['inputs']))
    input_dtypes.extend((inp['type'] for inp in config['inputs']))
    output_transpose.extend((outp['transpose'] for outp in config['outputs']))

    return input_shapes, input_transpose, input_dtypes, output_transpose


def prepare_input_arrays(input_shapes, input_transpose, input_dtypes):
    inputs = []
    for i in range(len(input_shapes)):
        input_shape = input_shapes[i]
        tranpose = input_transpose[i]
        input_data = np.zeros(input_shape, dtype=input_dtypes[i])
        if tranpose:
            input_data = input_data.transpose((0, 2, 3, 1))
        inputs.append(input_data)

    return inputs


def data_to_pytorch(inputs, input_transpose):
    torch_inputs = list(map(torch.from_numpy, inputs))
    for i in range(len(torch_inputs)):
        if input_transpose[i]:
            torch_inputs[i] = torch_inputs[i].permute(0, 3, 1, 2)
    return torch_inputs


def parse_config(
    json_file: str, prepare_inputs: bool = True
) -> typing.Tuple[str, str, typing.List[bool], typing.List[torch.Tensor], typing.List[bool]]:
    """Parses the configuration file for converter

    Args:
        json_file (str): The path of the configuration file
        prepare_inputs (bool, optional): Whether to prepare inputs. Defaults to True.

    Returns:
        tflite_model_path (str): The path used to place the generated tflite model
        torch_model_path (str): The path of the torchscript model
        input_transpose (typing.List[bool]): Flag variables whether the inputs will be transposed (nchw -> nhwc)
        torch_inputs (typing.List[torch.Tensor]): The prepared inputs if prepare_inputs is True, otherwise None
        output_transpose (typing.List[bool]): Flag variables whether the outputs will be transposed (nchw -> nhwc)
    """

    with open(json_file, 'r') as f:
        config = json.load(f)

    tflite_model_path = config['dst_model']
    torch_model_path = config['src_model']

    if prepare_inputs:
        input_shapes, input_transpose, input_dtypes, output_transpose = get_tensor_details(config)
        inputs = prepare_input_arrays(input_shapes, input_transpose, input_dtypes)
        torch_inputs = data_to_pytorch(inputs, input_transpose)
    else:
        torch_inputs = None

    return torch_model_path, tflite_model_path, input_transpose, torch_inputs, output_transpose
