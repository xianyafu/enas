import tensorflow as tf
from src.cifar10.data_utils import read_data
import numpy as np
import sys
import time
import psutil
from tensorflow.core.framework import attr_value_pb2
sys.path.append(r'/home/BH/sy1706331/NAS/qint_enas/enas')
from src.cifar10.quant_branch import create_eval_graph
def freeze_graph_test(pb_path, x_test, y_test):
    '''
    :param pb_path:pb文件的路径
    :param image_path:测试图片的路径
    :return:
    '''
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        #qint_ops = ['child/layer_9/conv_1x1/Conv2D', 'child/layer_9/conv_3x3/Conv2D']
        #create_eval_graph(tf.get_default_graph(), qint_ops)
        with tf.Session() as sess:
            input_graph_def = sess.graph_def
            for node in input_graph_def.node:
                #if node.name in ['child_1/layer_8/conv_1x1/Conv2D', 'child_1/layer_8/conv_3x3/Conv2D','child_1/layer_9/conv_1x1/w_qint8','map/TensorArrayStack/TensorArrayGatherV3']:
                #   print(node)
                #if node.name in ['child_1/layer_8/conv_3x3/bn/FusedBatchNorm', 'child_1/layer_8/conv_3x3/Relu']:
                #   print(node)
                #if node.op =="Dequantize":
                #   print(node)
                if node.name in  ["child_1/dropout/Shape"]:
                   node.attr["dtype"].CopyFrom(node.attr["T"])
                   print(node)
                if node.name.find('MovingAvgQuantize')!=-1 and node.attr['_class']:
                   #print(node)
                   del node.attr['_class']
                #if node.name == 'child_1/layer_8/MovingAvgQuantize/AssignMinEma/child/layer_8/MovingAvgQuantize/min/sub':
                #   print(node)
                if node.op == 'RandomUniform':
                   #print(node)
                   node.attr['seed2'].CopyFrom(attr_value_pb2.AttrValue(i=1))
                   node.attr['seed'].CopyFrom(attr_value_pb2.AttrValue(i=1))
                   #print(node)
                #if node.op == 'Const':
                #   del node.attr()
            '''
            for node in input_graph_def.node:
                if node.name.find('AssignMinEma/decay')!=-1 or node.name.find('AssignMaxEma/decay')!=-1:
                   print("----------")
                   print(node)
            '''
            sess.run(tf.global_variables_initializer())
 
            input_image_tensor = sess.graph.get_tensor_by_name("map/TensorArrayStack/TensorArrayGatherV3:0")
            tmp = sess.graph.get_operation_by_name("map/TensorArrayStack/TensorArrayGatherV3")

            output_tensor_name = sess.graph.get_tensor_by_name("child_1/fc/MatMul:0")
            out=sess.run(output_tensor_name, feed_dict={'map/TensorArrayStack/TensorArrayGatherV3:0' : x_test})
            score = tf.nn.softmax(out, name='pre')
            class_id = tf.argmax(score, 1)
            y_pred = sess.run(class_id)
            print(y_pred)
            print(y_test)
            d = np.argwhere(y_pred==y_test)
            print(len(d))
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                 sess=sess,
                 input_graph_def=input_graph_def,
                 output_node_names=['child_1/fc/MatMul'])
            with tf.gfile.GFile('tmp_v1_8_1.pb', "wb") as f: 
                f.write(output_graph_def.SerializeToString())


images, labels = read_data('data/cifar10')
x_test, y_test = tf.train.batch(
      [images["test"], labels["test"]],
      #input_queue,
      batch_size=10,
      capacity=10000,
      enqueue_many=True,
      num_threads=1,
      allow_smaller_final_batch=True,
)
sess = tf.Session()
x_100 = tf.split(images['test'], 10000, 0)
y_100 = tf.split(labels['test'], 10000, 0)
#x = sess.run(tf.transpose(x_100[0], [0,3,1,2]))
x = sess.run(x_100[0])
y = sess.run(y_100[0])
freeze_graph_test('./tmp_v1_8.pb', x, y)

