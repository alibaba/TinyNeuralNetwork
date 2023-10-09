## TFLite2TF

A utility for model conversion between TFLite and Tensorflow SavedModel.

### Supported operators
- PAD
- ADD
- RESIZE_BILINEAR
- RESHAPE
- CONV_2D
- AVERAGE_POOL_2D
- DEPTHWISE_CONV_2D
- FULLY_CONNECTED
- DEPTH_TO_SPACE
- TRANSPOSE_CONV
- SLICE
- RELU

### Usage
```py
import sys
import subprocess
from tflite2tf import parse_tflite

parse_tflite('test.tflite')

# It will dump a script `generate_tf_savedmodel.py` in the current directory.
subprocess.call([sys.executable, 'generate_tf_savedmodel.py'])

# And then, the saved model will be saved in the "saved_model" directory.
```

### Limitations
1. It may not work if the weights are shared between certain ops
2. The generated model may not be optimal
