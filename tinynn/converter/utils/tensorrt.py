import os
import re

from typing import Optional

import onnx
import onnxruntime as ort
import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt

from contextlib import contextmanager


# A logger with the specific log level
# Available levels: WARNING, ERROR, VERBOSE, INFO
TRT_BUILD_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_EVAL_LOGGER = trt.Logger(trt.Logger.ERROR)

# Constant batch, currently it cannot be disabled
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


class MyLogger(trt.ILogger):
    def __init__(self, inner):
        trt.ILogger.__init__(self)
        self.info_log = None
        self.inner = inner

    def log(self, severity, msg):
        if msg.startswith('Engine Layer Information:'):
            self.info_log = msg
        self.inner.log(severity, msg)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, dynamic_shapes_for_build, dynamc_shapes_for_eval):
    """Allocates all buffers required for an engine, i.e. host/device inputs/outputs."""
    inputs = []
    outputs = []
    bindings = []
    names_map_i = {}
    names_map_o = {}
    stream = cuda.Stream()
    for idx, binding in enumerate(engine):
        if binding in dynamc_shapes_for_eval:
            size = trt.volume(dynamc_shapes_for_eval[binding])
        elif binding in dynamic_shapes_for_build:
            size = trt.volume(dynamic_shapes_for_build[binding][1])
        else:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Input/Output node name
        if engine.binding_is_input(binding):
            names_map_i[binding] = len(inputs)
            inputs.append(HostDeviceMem(host_mem, device_mem))
            print('input:', engine.get_binding_shape(binding), dtype, binding)
        else:
            names_map_o[binding] = len(outputs)
            outputs.append(HostDeviceMem(host_mem, device_mem))
            print('output:', engine.get_binding_shape(binding), dtype, binding)
    return inputs, outputs, bindings, stream, names_map_i, names_map_o


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """This function is generalized for multiple inputs/outputs.
    inputs and outputs are expected to be lists of HostDeviceMem objects."""
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def GiB(val):
    """Numerical value for GiB"""
    return val * 1 << 30


def add_profile(builder, config, dynamic_shapes_for_build):
    if len(dynamic_shapes_for_build) > 0:
        profile = builder.create_optimization_profile()
        for inp, (min_shape, opt_shape, max_shape) in dynamic_shapes_for_build.items():
            profile.set_shape(inp, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)


@contextmanager
def build_engine(model_path, logger, build_with_fp16, build_with_int8, build_with_workspace, dynamic_shapes_for_build):
    """Build the TensorRT engine from a ONNX model"""
    with trt.Builder(logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(
        network, logger
    ) as parser:
        with open(model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        with builder.create_builder_config() as config:
            if build_with_fp16:
                config.flags = 1 << (int)(trt.BuilderFlag.FP16)
            if build_with_int8:
                config.flags = 1 << (int)(trt.BuilderFlag.INT8)
            config.max_workspace_size = GiB(build_with_workspace)

            add_profile(builder, config, dynamic_shapes_for_build)

            yield builder.build_engine(network, config)


def get_needed_outputs(lines):
    needed_outputs = set()
    dtype_start = None
    shape_start = None
    start_end_dict = {'[': ']', '(': ')'}
    dtype_end = None
    shape_end = None
    sep = None
    for line in lines:
        if line.startswith('Layer('):
            line = line.rstrip('\n')

            op = re.findall(r'Layer\((.*?)\)', line)[0]

            if '->' not in line:
                continue

            pos = line.find('-> ')
            outputs = line[pos + 3 :]

            if dtype_start is None:
                match = next(re.finditer(r'( *(\[|\())(Int32|Float|Half|Int8)(\(|\[)', outputs))
                dtype_start = match.group(1)
                shape_start = match.group(4)
                dtype_end = start_end_dict[dtype_start.lstrip()]
                shape_end = start_end_dict[shape_start]

            if sep is None:
                end_str = f'{shape_end}{dtype_end}'
                pos = outputs.find(end_str)
                assert pos > 0
                start = pos + len(shape_end) + len(dtype_end)
                if start != len(outputs):
                    end = outputs.find(' ', start)
                    sep = f'{end_str}{outputs[start:end]} '

            if sep is not None and sep in outputs:
                outputs = outputs.split(sep)
            else:
                outputs = [outputs]

            for output in outputs:
                if dtype_start not in output:
                    continue

                pos = output.rfind(dtype_start)
                output_name = output[:pos]

                if '[' in output_name or '(' in output_name or '+' in output_name:
                    continue

                if re.match('Reformatted .* to .*', output):
                    continue

                print('Observing', op, repr(output_name))
                needed_outputs.add(output_name)
    return needed_outputs


def add_outputs_for_onnx_model(needed_outputs, onnx_path, new_onnx_path):
    model = onnx.load(onnx_path)

    orig_outputs = set()
    for node in model.graph.output:
        orig_outputs.add(node.name)
    print()

    for node in model.graph.node:
        for output in node.output:
            if output in needed_outputs and output not in orig_outputs:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])

                print('Added output:', output)

    inferred_model = onnx.shape_inference.infer_shapes(model)

    onnx.save_model(inferred_model, new_onnx_path)
    print('Modified model saved at', new_onnx_path)


def compare_onnx_tensorrt(
    onnx_path: str,
    build_with_fp16: bool,
    build_with_int8: bool,
    build_with_workspace: int = 4,
    dynamic_shapes_for_build: Optional[dict] = None,
    dynamc_shapes_for_eval: Optional[dict] = None,
    input_path_mapping: Optional[dict] = None,
):
    if dynamic_shapes_for_build is None:
        dynamic_shapes_for_build = {}

    if dynamc_shapes_for_eval is None:
        dynamc_shapes_for_eval = {}

    if input_path_mapping is None:
        input_path_mapping = {}

    onnx_fn, onnx_ext = os.path.splitext(onnx_path)
    new_onnx_path = f'{onnx_fn}_with_outputs{onnx_ext}'
    new_trt_path = f'{onnx_fn}_with_outputs{onnx_ext}'

    print('Building TensorRT engine with', onnx_path)
    logger = MyLogger(TRT_BUILD_LOGGER)
    with build_engine(
        onnx_path, logger, build_with_fp16, build_with_int8, build_with_workspace, dynamic_shapes_for_build
    ) as engine:
        pass

    assert logger.info_log is not None, "Engine layer information is missing"
    lines = logger.info_log.splitlines()
    needed_outputs = get_needed_outputs(lines)
    add_outputs_for_onnx_model(needed_outputs, onnx_path, new_onnx_path)

    with build_engine(
        new_onnx_path, TRT_BUILD_LOGGER, build_with_fp16, build_with_int8, build_with_workspace
    ) as engine:
        with open(new_trt_path, 'wb') as f:
            f.write(bytearray(engine.serialize()))

    runtime = trt.Runtime(TRT_EVAL_LOGGER)
    with open(new_trt_path, 'rb') as f:
        engine_bytes = f.read()
        engine = runtime.deserialize_cuda_engine(engine_bytes)

    print('=' * 60)
    print('input output tensors:')
    inputs, outputs, bindings, stream, names_map_i, names_map_o = allocate_buffers(
        engine, dynamic_shapes_for_build, dynamc_shapes_for_eval
    )
    print('=' * 60)

    input_data = {}

    with engine.create_execution_context() as context:
        if True:
            for binding in names_map_i:
                dtype = trt.nptype(engine.get_binding_dtype(binding))

                if binding in input_path_mapping:
                    assert (
                        binding in dynamc_shapes_for_eval
                    ), "input_path_mapping and dynamc_shapes_for_eval should be specified together"

                    shape = dynamc_shapes_for_eval[binding]
                    data = np.fromfile(input_path_mapping[binding], dtype='uint8')
                    data = np.reshape(data.view(dtype), shape)
                    input_data[binding] = data

                    context.set_binding_shape(engine.get_binding_index(binding), shape)
                else:
                    if binding in dynamc_shapes_for_eval:
                        shape = dynamc_shapes_for_eval[binding]
                        context.set_binding_shape(engine.get_binding_index(binding), shape)
                    else:
                        shape = engine.get_binding_shape(binding)
                        if -1 in shape:
                            shape = dynamic_shapes_for_build[binding][1]
                            context.set_binding_shape(engine.get_binding_index(binding), shape)

                    data = np.random.random(shape).astype(dtype)
                    input_data[binding] = data

                np.copyto(inputs[names_map_i[binding]].host, data.ravel())

            output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

            output_keys = list(names_map_o)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(new_onnx_path, providers=['CUDAExecutionProvider'], sess_options=opts)

    res = sess.run(output_keys, input_data)

    for i, binding in enumerate(output_keys):
        onnx_val = res[i]

        if onnx_val is None:
            continue

        dtype = trt.nptype(engine.get_binding_dtype(binding))

        if dtype != onnx_val.dtype:
            trt_val = np.reshape(output[names_map_o[binding]].view(dtype).astype(onnx_val.dtype), onnx_val.shape)
        else:
            trt_val = np.reshape(output[names_map_o[binding]].view(onnx_val.dtype), onnx_val.shape)

        is_aligned = np.allclose(onnx_val, trt_val)

        if is_aligned:
            print(binding, 'matches:', is_aligned)
            continue

        onnx_val_ravel = onnx_val.ravel()
        trt_val_ravel = trt_val.ravel()

        cross_sim = np.dot(onnx_val_ravel, trt_val_ravel) / (
            np.linalg.norm(onnx_val_ravel) * np.linalg.norm(trt_val_ravel)
        )
        is_aligned = is_aligned or cross_sim > 0.999
        print(binding, 'matches:', is_aligned, 'cross_sim =', cross_sim)

        if not is_aligned:
            print('Top 10 values with maximum differences')
            max_diff_indices = np.argsort(np.abs(onnx_val - trt_val).ravel())[::-1][:10]
            print('TensorRT:')
            print(trt_val_ravel[max_diff_indices])
            print('ONNX:')
            print(onnx_val_ravel[max_diff_indices])

        print('-' * 60)
