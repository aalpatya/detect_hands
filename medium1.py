# Needs tensorflow 1.5
import tensorflow as tf
optimized_graph_path = "medium\out_frozen.pb"
output_pbtxt = "medium\optmized_graph.pbtxt"
# Read the graph.
with tf.compat.v1.gfile.GFile(optimized_graph_path, "rb") as f:
    # graph_def = tf.GraphDef()
    # graph_def.ParseFromString(f.read())
    graph_def = f.graph.as_graph_def()
# Remove Const nodes.
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]
    for attr in ['T', 'data_format', 'Tshape', 'N', 'Tidx', 'Tdim',
                 'use_cudnn_on_gpu', 'Index', 'Tperm', 'is_training',
                 'Tpaddings']:
        if attr in graph_def.node[i].attr:
            del graph_def.node[i].attr[attr]
# Save as text.
tf.io.write_graph(graph_def, "", output_pbtxt, as_text=True)