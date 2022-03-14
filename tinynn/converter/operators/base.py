import inspect
import sys
from enum import IntEnum

from ..schemas.tflite.schema_generated import ActivationFunctionType, BuiltinOperator

# In Python 3.6, we cannot make ExtendedOperator derive from IntEnum
if sys.version_info >= (3, 7):
    bases = (IntEnum,)
else:
    bases = ()


class _ExtendedOperatorBase(BuiltinOperator, *bases):
    INPUT_NODE = -1
    OUTPUT_NODE = -2
    CONSTANT_NODE = -3
    UNUSED_NODE = -4

    BATCH_NORM = -10
    GENERIC_CONV = -11
    GENERIC_DECONV = -12

    def type_name(self):
        return self.name.replace('_NODE', '')


# In Python 3.6, the elements in the parent class are not collected in IntEnum,
# so we have to do that dynamically.
if sys.version_info >= (3, 7):
    ExtendedOperator = _ExtendedOperatorBase
else:
    ExtendedOperator = IntEnum(
        'ExtendedOperator', dict(filter(lambda x: not x[0].startswith('__'), inspect.getmembers(_ExtendedOperatorBase)))
    )

FUSE_ACTIVATION_MAP = {
    BuiltinOperator.RELU: ActivationFunctionType.RELU,
    BuiltinOperator.RELU6: ActivationFunctionType.RELU6,
    BuiltinOperator.TANH: ActivationFunctionType.TANH,
}
