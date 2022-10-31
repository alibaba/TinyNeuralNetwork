from ..schemas.tflite import schema_generated as tflite


def parse_model(path):
    with open(path, 'rb') as f:
        buf = f.read()

    model = tflite.Model.GetRootAsModel(buf, 0)
    return model


def parse_lstm_states(model):
    if isinstance(model, str):
        model = parse_model(model)
    elif isinstance(model, bytes):
        model = tflite.Model.GetRootAsModel(model, 0)
    elif not isinstance(model, tflite.Model):
        assert False, f"expected type str, bytes and tflite.Model but got {type(model).__name__}"

    assert model.SubgraphsLength() == 1, "Only one subgraph is supported"

    subgraph = model.Subgraphs(0)
    state_idx = []

    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)

        opcode = model.OperatorCodes(op.OpcodeIndex())
        if opcode.DeprecatedBuiltinCode() in (
            tflite.BuiltinOperator.BIDIRECTIONAL_SEQUENCE_LSTM,
            tflite.BuiltinOperator.UNIDIRECTIONAL_SEQUENCE_LSTM,
        ):
            for j in range(op.InputsLength()):
                tensor_idx = op.Inputs(j)
                if tensor_idx < 0:
                    continue
                op_input = subgraph.Tensors(tensor_idx)
                if op_input.IsVariable():
                    state_idx.append(tensor_idx)

    return state_idx
