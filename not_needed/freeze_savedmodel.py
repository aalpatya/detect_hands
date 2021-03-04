import tensorflow as tf 
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

loaded_model = tf.saved_model.load(r'C:\Users\ap5\Documents\hands\detect_hands\model_data\ssd_mobilenet_v2_fpn_320\saved_model')
# Convert Keras model to ConcreteFunction

full_model = tf.function(lambda x: loaded_model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(
        loaded_model.signatures['serving_default'].inputs[0].shape,
        loaded_model.signatures['serving_default'].inputs[0].dtype
    )
)
 
# Get frozen ConcreteFunction
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()
 
# Print out model inputs and outputs
print("Frozen model inputs: ", frozen_func.inputs)
print("Frozen model outputs: ", frozen_func.outputs)
 
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir="./frozen_models",
                  name="frozen_graph1.pb",
                  as_text=True)