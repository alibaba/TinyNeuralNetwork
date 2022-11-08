import tensorflow_hub as hub
import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib

mobilenet = tf.keras.Sequential([
  keras.layers.InputLayer(input_shape=(224, 224, 3)),
  hub.KerasLayer("/home/ubuntu/lyn/TinyNeuralNetwork/examples/converter/in8")
])

converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet)

# Convert to TF Lite without quantization
#resnet_tflite_file = tflite_models_dir/"resnet_v2_101.tflite"
#resnet_tflite_file.write_bytes(converter.convert())

# Convert to TF Lite with quantization
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#resnet_quantized_tflite_file = "/home/ubuntu/lyn/TinyNeuralNetwork/examples/converter/in8/mobilenet_quantized.tflite"
#resnet_quantized_tflite_file.write_bytes(converter.convert())

def representative_data_gen():
  for _ in range(10):
    input_value = np.random.rand(1, 224, 224, 3).astype(np.float32)
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_int8 = converter.convert()
with open('mobilenet_int8.tflite', 'wb') as f:
  f.write(tflite_model_int8)
