import os

import cv2
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def convert_h5_to_frozen_pb():
    model = tf.keras.models.load_model('model.h5', compile=False)
    tf.keras.backend.set_learning_phase(0)
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=".",
        name="frozen_graph.pb",
        as_text=False
    )

    net = cv2.dnn.readNet(r'frozen_graph.pb')
    for layer_name in net.getLayerNames():
        print(layer_name)


def convert_pb_to_caffemodel():
    os.system('mmconvert -sf tf -iw frozen_graph.pb -df caffe -om ./model --inputShape=416,416,3 --inNodeName=x --dstNodeName=Identity')


if __name__ == '__main__':
    convert_h5_to_frozen_pb()
    convert_pb_to_caffemodel()

