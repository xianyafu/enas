import tensorflow as tf
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
import sys
from src.cifar10.data_utils import read_data
import numpy as np
sys.path.append(r'/home/BH/sy1706331/NAS/qint_enas/enas')
from src.cifar10.quant_branch import create_eval_graph
def freeze_graph(input_checkpoint,output_graph, x_test, y_test):
    '''
    :param input_checkpoint:
    :param output_graph
    :return:
    '''
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        '''
        qint_ops = ['child/layer_9/conv_1x1/Conv2D', 'child/layer_9/conv_3x3/Conv2D']
        create_eval_graph(tf.get_default_graph(), qint_ops)
        sess.run(tf.global_variables_initializer())
        output_tensor_name = sess.graph.get_tensor_by_name("child/fc/MatMul:0")
        out=sess.run(output_tensor_name, feed_dict={'map/TensorArrayStack/TensorArrayGatherV3:0' : x_test})
        score = tf.nn.softmax(out, name='pre')
        class_id = tf.argmax(score, 1)
        y_pred = sess.run(class_id)
        d = np.argwhere(y_pred==y_test)
        print(len(d))
        '''
        output_graph_def = tf.graph_util.convert_variables_to_constants(
             sess=sess,
             input_graph_def=sess.graph_def,
             output_node_names=['child_1/fc/MatMul'])
        n = tf.NodeDef()
        n.op = 'Placeholder'
        n.name = 'map/TensorArrayStack/TensorArrayGatherV3'
        shape = []#[1, 32, 32, 3]
        dims = tensor_shape_pb2.TensorShapeProto(
           dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=x) for x in shape])
        n.attr['shape'].CopyFrom(attr_value_pb2.AttrValue(shape=dims))
        for node in output_graph_def.node:
                if node.op == 'RefSwitch':
                   node.op = 'Switch'
                   for index in range(len(node.input)):
                       if 'moving_' in node.input[index]:
                          node.input[index] = mode.input[index] +'/read'
                elif node.op == 'AssignSub':
                   node.op = 'Sub'
                   if 'use_locking' in node.attr: del node.attr['use_locking']
                elif node.op == 'AssignAdd':
                   node.op = 'Add'
                   if 'use_locking' in node.attr: del node.attr['use_locking']


        for node in output_graph_def.node:
            #if node.name == 'child/stem_conv/bn/AssignMovingAvg/child/stem_conv/bn/moving_mean/child/stem_conv/bn/child/stem_conv/bn/moving_mean':
            #   print(node)
            #print(node.name)
            if node.name == 'map/TensorArrayStack/TensorArrayGatherV3':
               print(node)
               node.op = n.op
               del node.input[0]
               del node.input[0]
               del node.input[0]
               del node.attr['_class']
               del node.attr['element_shape']
               node.attr['shape'].CopyFrom(n.attr['shape'])
            if node.op == 'Assign':
               node.op = 'Const'
               node.attr['dtype'].CopyFrom(node.attr['T'])
               for no in output_graph_def.node:
                   if no.name == node.input[0]:
                      node.attr['value'].CopyFrom(no.attr['value'])
               if 'use_locking' in node.attr: del node.attr['use_locking']
               del node.input[0]
               del node.input[0]
               del node.attr['validate_shape']
               del node.attr['_class']
               del node.attr['T']
            #   print(node)
            if node.name == 'child_1/fc/MatMul':
               print(node)
        with tf.gfile.GFile(output_graph, "wb") as f: 
            f.write(output_graph_def.SerializeToString()) 

images, labels = read_data('data/cifar10')
sess = tf.Session()
x_100 = tf.split(images['test'], 10000, 0)
y_100 = tf.split(labels['test'], 10000, 0)
x = sess.run(x_100[0])
y = sess.run(y_100[0])
#freeze_graph('cifar10_macro_final_12l_qint8_ms/model.ckpt-155000', '12l_qint8_1.pb',x,y)
#freeze_graph('./cifar10_macro_final_12l_float_nhwc/model.ckpt-155000','./12l_float_1.pb', x, y)
#freeze_graph('cifar10_macro_final_12l_qint8/model.ckpt-155000','./12l.pb')
freeze_graph('outputs/model.ckpt-155000','./12l_float_4.pb', x, y)
