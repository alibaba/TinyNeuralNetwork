from .generated_ops import CustomOperator


class Atan2Operator(CustomOperator):
    def __init__(self, inputs, outputs) -> None:
        super().__init__(inputs, outputs)
        self.op.custom_code = "Atan2"
