import tensorflow as tf
from src.cifar10.data_utils import read_data
import numpy as np
import sys
import time
import psutil
sys.path.append(r'/home/BH/sy1706331/NAS/qint_enas/enas')
from src.cifar10.quant_branch import create_eval_graph

def qint8_test(pb_path, images, labels):
   sess1 = tf.Session()
   x_100 = tf.split(images['test'], 100, 0)
   y_100 = tf.split(labels['test'], 100, 0)
   tmp_times = np.zeros(100)
   tmp_mem = np.zeros(100)
   tmp_acc = np.zeros(100)
   #for i in range(0,100):
   #  x_test = sess1.run(tf.transpose(x_100[i], [0,3,1,2]))
   #  y_test = sess1.run(y_100[i])
   with tf.Graph().as_default():
      output_graph_def = tf.GraphDef()
      with open(pb_path, "rb") as f:
          output_graph_def.ParseFromString(f.read())
          tf.import_graph_def(output_graph_def, name="")
          for node in output_graph_def.node:
              if node.name in ['child/stem_conv/Conv2D/eightbit','child_1/layer_8/conv_1x1/Conv2D/eightbit','child_1/layer_8/conv_1x1/Conv2D/eightbit/requantize']:
                 print(node)
      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          output_tensor_name = sess.graph.get_tensor_by_name("child_1/fc/MatMul:0")
          for i in range(0, 100):
              #x_test = sess1.run(tf.transpose(x_100[i], [0,3,1,2]))
              x_test = sess1.run(x_100[i])
              y_test = sess1.run(y_100[i])
              start = time.time()
              mem_s = psutil.virtual_memory()
              out=sess.run(output_tensor_name, feed_dict={'map/TensorArrayStack/TensorArrayGatherV3:0' : x_test})
              mem_e = psutil.virtual_memory()
              cur = time.time()
              tmp_times[i]=cur-start
              tmp_mem[i] = mem_e.used-mem_s.used
              #score = tf.nn.softmax(out, name='pre')
              class_id = tf.argmax(out, 1)
              y_pred = sess.run(class_id)
              d = np.argwhere(y_pred==y_test)
              acc = len(d)/100
              tmp_acc[i] = acc
              print("epoch: ",i, "  acc: ",tmp_acc[i], "  time: ",tmp_times[i])#, "  mem: ",tmp_mem[i])
   print(tmp_acc)
   t = (tmp_times.sum()-tmp_times[0])/100
   a = np.sum(tmp_acc)/100
   a_1 = 0
   for i in range(0,100):
     a_1 += tmp_acc[i]
   a = a_1/100
   m = tmp_mem.mean()
   print("total acc: ", a)
   print("inference time: ", t)
   #print("memory: ", m)
images, labels = read_data('data/cifar10')
#sess = tf.Session()
#x_100 = tf.split(images['test'], 100, 0)
#y_100 = tf.split(labels['test'], 100, 0)
#x = sess.run(tf.transpose(x_100[0], [0,3,1,2]))
#y = sess.run(y_100[0])
qint8_test('./12l_float_4.pb', images, labels)
#qint8_test('./12l_qint8_ms_it_v1.pb', images, labels)
