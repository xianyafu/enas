from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import cPickle as pickle
import pickle
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.cifar10.data_utils import read_data
from src.cifar10.general_child import GeneralChild
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.framework import dtypes
import numpy as np
flags = tf.app.flags
FLAGS = flags.FLAGS


DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("data_format", "NHWC", "'NHWC' or 'NCWH'")
DEFINE_string("search_for", None, "Must be [macro|micro]")

DEFINE_integer("batch_size", 32, "")

DEFINE_integer("num_epochs", 300, "")
DEFINE_integer("child_lr_dec_every", 100, "")
DEFINE_integer("child_num_layers", 5, "")
DEFINE_integer("child_num_cells", 5, "")
DEFINE_integer("child_filter_size", 5, "")
DEFINE_integer("child_out_filters", 48, "")
DEFINE_integer("child_out_filters_scale", 1, "")
DEFINE_integer("child_num_branches", 4, "")
DEFINE_integer("child_num_aggregate", None, "")
DEFINE_integer("child_num_replicas", 1, "")
DEFINE_integer("child_block_size", 3, "")
DEFINE_integer("child_lr_T_0", None, "for lr schedule")
DEFINE_integer("child_lr_T_mul", None, "for lr schedule")
DEFINE_integer("child_cutout_size", None, "CutOut size")
DEFINE_float("child_grad_bound", 5.0, "Gradient clipping")
DEFINE_float("child_lr", 0.1, "")
DEFINE_float("child_lr_dec_rate", 0.1, "")
DEFINE_float("child_keep_prob", 0.5, "")
DEFINE_float("child_drop_path_keep_prob", 1.0, "minimum drop_path_keep_prob")
DEFINE_float("child_l2_reg", 1e-4, "")
DEFINE_float("child_lr_max", None, "for lr schedule")
DEFINE_float("child_lr_min", None, "for lr schedule")
DEFINE_string("child_skip_pattern", None, "Must be ['dense', None]")
DEFINE_string("child_fixed_arc", None, "")
DEFINE_boolean("child_use_aux_heads", False, "Should we use an aux head")
DEFINE_boolean("child_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("child_lr_cosine", False, "Use cosine lr schedule")

DEFINE_boolean("controller_search_whole_channels", False, "")
DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")

def get_ops(images, labels):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  assert FLAGS.search_for is not None, "Please specify --search_for"


  ChildClass = GeneralChild

  child_model = ChildClass(
    images,
    labels,
    use_aux_heads=FLAGS.child_use_aux_heads,
    cutout_size=FLAGS.child_cutout_size,
    whole_channels=FLAGS.controller_search_whole_channels,
    num_layers=FLAGS.child_num_layers,
    num_cells=FLAGS.child_num_cells,
    num_branches=FLAGS.child_num_branches,
    fixed_arc=FLAGS.child_fixed_arc,
    out_filters_scale=FLAGS.child_out_filters_scale,
    out_filters=FLAGS.child_out_filters,
    keep_prob=FLAGS.child_keep_prob,
    drop_path_keep_prob=FLAGS.child_drop_path_keep_prob,
    num_epochs=FLAGS.num_epochs,
    l2_reg=FLAGS.child_l2_reg,
    data_format=FLAGS.data_format,
    batch_size=FLAGS.batch_size,
    clip_mode="norm",
    grad_bound=FLAGS.child_grad_bound,
    lr_init=FLAGS.child_lr,
    lr_dec_every=FLAGS.child_lr_dec_every,
    lr_dec_rate=FLAGS.child_lr_dec_rate,
    lr_cosine=FLAGS.child_lr_cosine,
    lr_max=FLAGS.child_lr_max,
    lr_min=FLAGS.child_lr_min,
    lr_T_0=FLAGS.child_lr_T_0,
    lr_T_mul=FLAGS.child_lr_T_mul,
    optim_algo="momentum",
    sync_replicas=FLAGS.child_sync_replicas,
    num_aggregate=FLAGS.child_num_aggregate,
    num_replicas=FLAGS.child_num_replicas,
  )

  child_model.connect_controller(None)
  controller_ops = None

  child_ops = {
    "global_step": child_model.global_step,
    "loss": child_model.loss,
    "train_op": child_model.train_op,
    "lr": child_model.lr,
    "grad_norm": child_model.grad_norm,
    "train_acc": child_model.train_acc,
    "optimizer": child_model.optimizer,
    "num_train_batches": child_model.num_train_batches,
    "model_size": child_model.model_size,
    "infer_time": child_model.infer_time,
    "test_acc": child_model.test_acc,
    "x_test": child_model.x_test,
    "y_test": child_model.y_test,
  }

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }

  return ops

def test():
  images, labels = read_data(FLAGS.data_path, num_valids=0)
  include = []
  ckpt_path = '/home/BH/sy1706331/NAS/qint_enas/enas/cifar10_macro_final_12l_float_nhwc/model.ckpt-155000'
  reader=pywrap_tensorflow.NewCheckpointReader(ckpt_path) 
  var_to_shape_map=reader.get_variable_to_shape_map() 
  for key in var_to_shape_map: 
   include.append(key.split(':')[0])
   if key == "child/layer_5/conv_3x3/w":
      print(reader.get_tensor("child/layer_5/conv_3x3/w"))
  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels)
    child_ops = ops["child"]
    #variables_to_restore = slim.get_variables_to_restore(include=include)
    variables = tf.contrib.framework.get_variables_to_restore()
    variables_to_resotre = [v for v in variables if v.name.find('Variable')==-1]
    saver = tf.train.Saver(variables_to_resotre)
    #saver =  tf.train.import_meta_graph('/home/BH/sy1706331/NAS/qint_enas/enas/cifar10_macro_final_12l_qint8_ms/model.ckpt-155000.meta')

    print("-" * 80)
    print("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, '/home/BH/sy1706331/NAS/qint_enas/enas/cifar10_macro_final_12l_float_nhwc/model.ckpt-155000')
      x_100 = tf.split(images['test'], 10000, 0)
      y_100 = tf.split(labels['test'], 10000, 0)
      for i in range(0, 1):
        x = sess.run(x_100[0])
        y = sess.run(y_100[0])

        print('start')
        #acc = sess.run(child_ops["test_acc"])
      
        ops["eval_func"](sess,"test", 
                        {child_ops["x_test"]:x, child_ops["y_test"]:y})
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
              elif node.op == "FusedBatchNorm":
                 node.attr['is_training'].CopyFrom(attr_value_pb2.AttrValue(b=False))
                 node.input[3]='child'+node.input[3].split('child_1')[1].split('Const')[0]+'moving_mean/read'
                 node.input[4]='child'+node.input[4].split('child_1')[1].split('Const')[0]+'moving_variance/read'
                 print(node)
              elif node.op == "BatchNorm":
                 print(node)
              elif node.op == 'AssignSub':
                 node.op = 'Sub'
                 if 'use_locking' in node.attr: del node.attr['use_locking']
              elif node.op == 'AssignAdd':
                 node.op = 'Add'
                 if 'use_locking' in node.attr: del node.attr['use_locking']
      
      for node in output_graph_def.node:
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
          if node.name == 'child_1/fc/MatMul':
             print(node)
      with tf.gfile.GFile('./tmp_32.pb', "wb") as f: 
          f.write(output_graph_def.SerializeToString()) 

     
def main(_):
  print("-" * 80)
  test()

if __name__ == "__main__":
  tf.app.run()
