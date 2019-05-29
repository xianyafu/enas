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
sys.path.append('/home/BH/sy1706331/github/enas')
from src import utils
from src.utils import Logger
from src.utils import DEFINE_boolean
from src.utils import DEFINE_float
from src.utils import DEFINE_integer
from src.utils import DEFINE_string
from src.utils import print_user_flags

from src.cifar100.data_utils import read_data_by_order
from src.cifar100.general_controller import GeneralController
from src.cifar100.general_child import GeneralChild
from src.cifar100.general_child_v1 import GeneralChildV1

from src.cifar100.micro_controller import MicroController
from src.cifar100.micro_child import MicroChild

from src.cifar100.monitored_session import SingularMonitoredSession 
flags = tf.app.flags
FLAGS = flags.FLAGS

DEFINE_boolean("reset_output_dir", False, "Delete output_dir if exists.")
DEFINE_string("data_path", "", "")
DEFINE_string("output_dir", "", "")
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

DEFINE_float("controller_lr", 1e-3, "")
DEFINE_float("controller_lr_dec_rate", 1.0, "")
DEFINE_float("controller_keep_prob", 0.5, "")
DEFINE_float("controller_l2_reg", 0.0, "")
DEFINE_float("controller_bl_dec", 0.99, "")
DEFINE_float("controller_tanh_constant", None, "")
DEFINE_float("controller_op_tanh_reduce", 1.0, "")
DEFINE_float("controller_temperature", None, "")
DEFINE_float("controller_entropy_weight", None, "")
DEFINE_float("controller_skip_target", 0.8, "")
DEFINE_float("controller_skip_weight", 0.0, "")
DEFINE_integer("controller_num_aggregate", 1, "")
DEFINE_integer("controller_num_replicas", 1, "")
DEFINE_integer("controller_train_steps", 50, "")
DEFINE_integer("controller_forwards_limit", 2, "")
DEFINE_integer("controller_train_every", 2,
               "train the controller after this number of epochs")
DEFINE_boolean("controller_search_whole_channels", False, "")
DEFINE_boolean("controller_sync_replicas", False, "To sync or not to sync.")
DEFINE_boolean("controller_training", True, "")
DEFINE_boolean("controller_use_critic", False, "")

DEFINE_integer("log_every", 50, "How many steps to log")
DEFINE_integer("eval_every_epochs", 1, "How many epochs to eval")
DEFINE_integer("cl_group", 10, "how many class per incremental train")
DEFINE_integer("total_classes", 100, "total classes of the dataset")

def get_controller():
  ControllerClass = GeneralController

  if FLAGS.child_fixed_arc is None:
    controller_model = ControllerClass(
      search_for=FLAGS.search_for,
      search_whole_channels=FLAGS.controller_search_whole_channels,
      skip_target=FLAGS.controller_skip_target,
      skip_weight=FLAGS.controller_skip_weight,
      num_cells=FLAGS.child_num_cells,
      num_layers=FLAGS.child_num_layers,
      num_branches=FLAGS.child_num_branches,
      out_filters=FLAGS.child_out_filters,
      lstm_size=64,
      lstm_num_layers=1,
      lstm_keep_prob=1.0,
      tanh_constant=FLAGS.controller_tanh_constant,
      op_tanh_reduce=FLAGS.controller_op_tanh_reduce,
      temperature=FLAGS.controller_temperature,
      lr_init=FLAGS.controller_lr,
      lr_dec_start=0,
      lr_dec_every=1000000,  # never decrease learning rate
      l2_reg=FLAGS.controller_l2_reg,
      entropy_weight=FLAGS.controller_entropy_weight,
      bl_dec=FLAGS.controller_bl_dec,
      use_critic=FLAGS.controller_use_critic,
      optim_algo="momentum",
      sync_replicas=FLAGS.controller_sync_replicas,
      num_aggregate=FLAGS.controller_num_aggregate,
      num_replicas=FLAGS.controller_num_replicas)
  return controller_model

def get_ops_v1(images, labels, controller_model,index_num, images_i, labels_i):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  assert FLAGS.search_for is not None, "Please specify --search_for"

  if FLAGS.search_for == "micro":
    ControllerClass = MicroController
    ChildClass = MicroChild
  else:
    ControllerClass = GeneralController
    ChildClass = GeneralChild

  child_model = GeneralChildV1(
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
    class_num=index_num*FLAGS.cl_group,
    total_classes=FLAGS.total_classes,
    cl_group=FLAGS.cl_group,
    image_i=images_i,
    label_i=labels_i
  )
  child_model.connect_controller(controller_model)
  controller_model.build_trainer(child_model)

  controller_ops = {
    "train_step": controller_model.train_step,
    "loss": controller_model.loss,
    "train_op": controller_model.train_op,
    "lr": controller_model.lr,
    "grad_norm": controller_model.grad_norm,
    "valid_acc": controller_model.valid_acc,
    "optimizer": controller_model.optimizer,
    "baseline": controller_model.baseline,
    "entropy": controller_model.sample_entropy,
    "sample_arc": controller_model.sample_arc,
    "skip_rate": controller_model.skip_rate,
    "arc_mem": controller_model.arc_mem,
    "in_time": controller_model.in_time,
  }

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
    "pred": child_model.pred,
    "x_train": child_model.x_train,
    "y_train": child_model.y_train,
    "variables_graph":child_model.variables_graph,
    "variables_graph2":child_model.variables_graph2,
    "log_probs": child_model.log_probs,
    "log_probs_v1": child_model.log_probs_v1,
    "pred_old_cl": child_model.pred_old_cl,
    "label_old_classes": child_model.label_old_classes,
    "pred_new_cl": child_model.pred_new_cl,
    "label_new_classes": child_model.label_new_classes,
  }

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }

  return ops


def get_ops_v2(images, labels, controller_model,index_num):
  """
  Args:
    images: dict with keys {"train", "valid", "test"}.
    labels: dict with keys {"train", "valid", "test"}.
  """

  assert FLAGS.search_for is not None, "Please specify --search_for"

  if FLAGS.search_for == "micro":
    ControllerClass = MicroController
    ChildClass = MicroChild
  else:
    ControllerClass = GeneralController
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
    class_num=index_num*FLAGS.cl_group,
    total_classes=FLAGS.total_classes,
    cl_group=FLAGS.cl_group,
  )
  child_model.connect_controller(controller_model)
  controller_model.build_trainer(child_model)

  controller_ops = {
    "train_step": controller_model.train_step,
    "loss": controller_model.loss,
    "train_op": controller_model.train_op,
    "lr": controller_model.lr,
    "grad_norm": controller_model.grad_norm,
    "valid_acc": controller_model.valid_acc,
    "optimizer": controller_model.optimizer,
    "baseline": controller_model.baseline,
    "entropy": controller_model.sample_entropy,
    "sample_arc": controller_model.sample_arc,
    "skip_rate": controller_model.skip_rate,
    "arc_mem": controller_model.arc_mem,
    "in_time": controller_model.in_time,
  }

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
    "pred": child_model.pred,
    "x_train": child_model.x_train,
    "y_train": child_model.y_train,
  }

  ops = {
    "child": child_ops,
    "controller": controller_ops,
    "eval_every": child_model.num_train_batches * FLAGS.eval_every_epochs,
    "eval_func": child_model.eval_once,
    "num_train_batches": child_model.num_train_batches,
  }

  return ops

def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
 
    for i in not_initialized_vars:
            print(i.name)
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def train(images, labels):
  g = tf.Graph()
  g.as_default()
  controller_model = get_controller()
  for class_index in range(0, 1):
      print("!!!!!!!!!!!!!!!!!!")
      print("for ",(class_index+1)*FLAGS.cl_group, " class of cifar100")
      ops = get_ops_v2(images[class_index], labels[class_index], controller_model, class_index+1)
      child_ops = ops["child"]
      controller_ops = ops["controller"]
       
      saver = tf.train.Saver(max_to_keep=2)
      checkpoint_saver_hook = tf.train.CheckpointSaverHook(
        FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)
 
      hooks = [checkpoint_saver_hook]
      
      if FLAGS.child_sync_replicas:
        sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
        hooks.append(sync_replicas_hook)
      if FLAGS.controller_training and FLAGS.controller_sync_replicas:
        sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
        hooks.append(sync_replicas_hook)
      
      print("-" * 80)
      print("Starting session")
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.999)
      config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
      with SingularMonitoredSession(
        config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
          start_time = time.time()
          while True:
            run_ops = [
              child_ops["loss"],
              child_ops["lr"],
              child_ops["grad_norm"],
              child_ops["train_acc"],
              child_ops["train_op"],
              child_ops["model_size"],
              child_ops["infer_time"],
            ]
            loss, lr, gn, tr_acc, _, ms, intm = sess.run(run_ops)
            global_step = sess.run(child_ops["global_step"])
 
            if FLAGS.child_sync_replicas:
              actual_step = global_step * FLAGS.num_aggregate
            else:
              actual_step = global_step
            epoch = actual_step // ops["num_train_batches"]
            curr_time = time.time()
            if global_step % FLAGS.log_every == 0:
              log_string = ""
              log_string += "epoch={:<6d}".format(epoch)
              log_string += "ch_step={:<6d}".format(global_step)
              log_string += " loss={:<8.6f}".format(loss)
              log_string += " lr={:<8.4f}".format(lr)
              log_string += " |g|={:<8.4f}".format(gn)
              log_string += " tr_acc={:<3d}/{:>3d}".format(
                  tr_acc, FLAGS.batch_size)
              log_string += " mins={:<10.2f}".format(
                  float(curr_time - start_time) / 60)
              log_string += " mem= "+str(ms)
              log_string += " infer_time="+str(intm)
              print(log_string)
              
            if actual_step % ops["eval_every"] == 0:
              if (FLAGS.controller_training and
                  epoch % FLAGS.controller_train_every == 0):
                print("Epoch {}: Training controller".format(epoch))
                for ct_step in range(FLAGS.controller_train_steps *
                                      FLAGS.controller_num_aggregate):
                  run_ops = [
                    controller_ops["loss"],
                    controller_ops["entropy"],
                    controller_ops["lr"],
                    controller_ops["grad_norm"],
                    controller_ops["valid_acc"],
                    controller_ops["baseline"],
                    controller_ops["skip_rate"],
                    controller_ops["train_op"],
                    controller_ops["arc_mem"],
                    controller_ops["in_time"],
                  ]
                  loss, entropy, lr, gn, val_acc, bl, skip, _,mem, intm = sess.run(run_ops)
                  controller_step = sess.run(controller_ops["train_step"])
 
                  if ct_step % FLAGS.log_every == 0:
                    curr_time = time.time()
                    log_string = ""
                    log_string += "ctrl_step={:<6d}".format(controller_step)
                    log_string += " loss={:<7.3f}".format(loss)
                    log_string += " ent={:<5.2f}".format(entropy)
                    log_string += " lr={:<6.4f}".format(lr)
                    log_string += " |g|={:<8.4f}".format(gn)
                    log_string += " acc={:<6.4f}".format(val_acc)
                    log_string += " bl={:<5.2f}".format(bl)
                    log_string += " mins={:<.2f}".format(
                        float(curr_time - start_time) / 60)
                    log_string += " mem_c= "+str(mem)
                    log_string += " infer_time= "+str(intm)
                    print(log_string)
 
                print("Here are 10 architectures")
                for _ in range(10):
                  arc, acc, arc_mem,intm = sess.run([
                    controller_ops["sample_arc"],
                    controller_ops["valid_acc"],
                    controller_ops["arc_mem"],
                    controller_ops["in_time"],
                  ])
                  if FLAGS.search_for == "micro":
                    normal_arc, reduce_arc = arc
                    print(np.reshape(normal_arc, [-1]))
                    print(np.reshape(reduce_arc, [-1]))
                  else:
                    start = 0
                    for layer_id in range(FLAGS.child_num_layers):
                      if FLAGS.controller_search_whole_channels:
                        end = start + 1 + layer_id
                      else:
                        end = start + 2 * FLAGS.child_num_branches + layer_id
                      print(np.reshape(arc[start: end], [-1]))
                      start = end
                  print("val_acc={:<6.4f}".format(acc))
                  print("arc_mem="+str(arc_mem))
                  print("infer_time="+str(intm))
                  print("-" * 80)
 
              print("Epoch {}: Eval".format(epoch))
              if FLAGS.child_fixed_arc is None:
                ops["eval_func"](sess, "valid")
              start_time = time.time()
              ops["eval_func"](sess, "test")
              curr_time = time.time()
              print(float(curr_time-start_time))

            if epoch == FLAGS.num_epochs:
              images_i = np.zeros((4500+500, 32, 32, 3), dtype=np.float32)
              labels_i = np.zeros((4500+500), dtype=np.int32)
              for i in range(0, images[class_index+1]["train"].shape[0]):
                      images_i[i] = images[class_index+1]["train"][i]
                      labels_i[i] = labels[class_index+1]["train"][i]
              num = images[class_index+1]["train"].shape[0]
              for index in range(0,25):
                  x_train, y_train, pred = sess.run([child_ops["x_train"],child_ops["y_train"],child_ops["pred"]])
                  pred.sort(axis=0)
                  tmp = np.zeros(FLAGS.batch_size)
                  for i in range(0, FLAGS.batch_size):
                      tmp[i]=pred[i][0]
                  tmp.sort(axis=0)
                  small_20 = tmp[20]
                  m = 0
                  for j in range(0, FLAGS.batch_size):
                      if pred[j][0] <= small_20 and m<20:
                         images_i[num] = np.transpose(x_train[j],[1,2,0])
                         labels_i[num] = y_train[j]
                         num += 1
                         m += 1
              images[class_index+1]["train"] = images_i
              labels[class_index+1]["train"] = labels_i
              print(num)
            if epoch >= FLAGS.num_epochs:
              break
      tf.reset_default_graph()
  return images_i, labels_i

def train_incre(index, images, labels, images_i, labels_i):
  g = tf.Graph()
  g.as_default()
  controller_model = get_controller()

  for class_index in range(index, index+1):
      print("!!!!!!!!!!!!!!!!!!")
      print("for ",(class_index+1)*FLAGS.cl_group, " class of cifar100")
      ops = get_ops_v1(images[class_index], labels[class_index], controller_model, class_index+1, images_i, labels_i)
      child_ops = ops["child"]
      controller_ops = ops["controller"]
      
      saver = tf.train.Saver(max_to_keep=2)
      checkpoint_saver_hook = tf.train.CheckpointSaverHook(
        FLAGS.output_dir, save_steps=child_ops["num_train_batches"], saver=saver)
 
      hooks = [checkpoint_saver_hook]
      
      if FLAGS.child_sync_replicas:
        sync_replicas_hook = child_ops["optimizer"].make_session_run_hook(True)
        hooks.append(sync_replicas_hook)
      if FLAGS.controller_training and FLAGS.controller_sync_replicas:
        sync_replicas_hook = controller_ops["optimizer"].make_session_run_hook(True)
        hooks.append(sync_replicas_hook)
      
      print("-" * 80)
      print("Starting session")
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.999)
      config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
      with SingularMonitoredSession(
        config=config, hooks=hooks, checkpoint_dir=FLAGS.output_dir) as sess:
          variables = tf.contrib.framework.get_variables_to_restore()
          #variables_to_restore = [v for v in variables if v.name.split('/')[0] == 'controller']
          variables_to_restore = variables
          saver1 = tf.train.Saver(variables_to_restore)
          saver.restore(sess, tf.train.latest_checkpoint('/home/fuxianya/github/enas/outputs/') )
          op_assign = [(child_ops["variables_graph2"][i]).assign(child_ops["variables_graph"][i]) for i in range(len(child_ops["variables_graph"]))]
          start_time = time.time()
          while True:
            sess.run(op_assign)
            run_ops = [
              child_ops["loss"],
              child_ops["lr"],
              child_ops["grad_norm"],
              child_ops["train_acc"],
              child_ops["train_op"],
              child_ops["model_size"],
              child_ops["infer_time"],
            ]
            loss, lr, gn, tr_acc, _, ms, intm = sess.run(run_ops)
            global_step = sess.run(child_ops["global_step"])
 
            if FLAGS.child_sync_replicas:
              actual_step = global_step * FLAGS.num_aggregate
            else:
              actual_step = global_step
            epoch = actual_step // ops["num_train_batches"]
            curr_time = time.time()
            if global_step % FLAGS.log_every == 0:
              log_string = ""
              log_string += "epoch={:<6d}".format(epoch)
              log_string += "ch_step={:<6d}".format(global_step)
              log_string += " loss={:<8.6f}".format(loss)
              log_string += " lr={:<8.4f}".format(lr)
              log_string += " |g|={:<8.4f}".format(gn)
              log_string += " tr_acc={:<3d}/{:>3d}".format(
                  tr_acc, FLAGS.batch_size)
              log_string += " mins={:<10.2f}".format(
                  float(curr_time - start_time) / 60)
              log_string += " mem= "+str(ms)
              log_string += " infer_time="+str(intm)
              print(log_string)
            if actual_step % ops["eval_every"] == 0:
              if (FLAGS.controller_training and
                  epoch % FLAGS.controller_train_every == 0):
                print("Epoch {}: Training controller".format(epoch))
                for ct_step in range(FLAGS.controller_train_steps *
                                      FLAGS.controller_num_aggregate):
                  run_ops = [
                    controller_ops["loss"],
                    controller_ops["entropy"],
                    controller_ops["lr"],
                    controller_ops["grad_norm"],
                    controller_ops["valid_acc"],
                    controller_ops["baseline"],
                    controller_ops["skip_rate"],
                    controller_ops["train_op"],
                    controller_ops["arc_mem"],
                    controller_ops["in_time"],
                  ]
                  loss, entropy, lr, gn, val_acc, bl, skip, _,mem, intm = sess.run(run_ops)
                  controller_step = sess.run(controller_ops["train_step"])
 
                  if ct_step % FLAGS.log_every == 0:
                    curr_time = time.time()
                    log_string = ""
                    log_string += "ctrl_step={:<6d}".format(controller_step)
                    log_string += " loss={:<7.3f}".format(loss)
                    log_string += " ent={:<5.2f}".format(entropy)
                    log_string += " lr={:<6.4f}".format(lr)
                    log_string += " |g|={:<8.4f}".format(gn)
                    log_string += " acc={:<6.4f}".format(val_acc)
                    log_string += " bl={:<5.2f}".format(bl)
                    log_string += " mins={:<.2f}".format(
                        float(curr_time - start_time) / 60)
                    log_string += " mem_c= "+str(mem)
                    log_string += " infer_time= "+str(intm)
                    print(log_string)
 
                print("Here are 10 architectures")
                for _ in range(10):
                  arc, acc, arc_mem,intm = sess.run([
                    controller_ops["sample_arc"],
                    controller_ops["valid_acc"],
                    controller_ops["arc_mem"],
                    controller_ops["in_time"],
                  ])
                  if FLAGS.search_for == "micro":
                    normal_arc, reduce_arc = arc
                    print(np.reshape(normal_arc, [-1]))
                    print(np.reshape(reduce_arc, [-1]))
                  else:
                    start = 0
                    for layer_id in range(FLAGS.child_num_layers):
                      if FLAGS.controller_search_whole_channels:
                        end = start + 1 + layer_id
                      else:
                        end = start + 2 * FLAGS.child_num_branches + layer_id
                      print(np.reshape(arc[start: end], [-1]))
                      start = end
                  print("val_acc={:<6.4f}".format(acc))
                  print("arc_mem="+str(arc_mem))
                  print("infer_time="+str(intm))
                  print("-" * 80)
 
              print("Epoch {}: Eval".format(epoch))
              if FLAGS.child_fixed_arc is None:
                ops["eval_func"](sess, "valid")
              start_time = time.time()
              ops["eval_func"](sess, "test")
              curr_time = time.time()
              print(float(curr_time-start_time))

            if epoch == FLAGS.num_epochs*(1+class_index):
              images_i_new = np.zeros((4500+(1+index)*500, 32, 32, 3), dtype=np.float32)
              labels_i_new = np.zeros((4500+(1+index)*500), dtype=np.int32)
              for i in range(0, images[class_index+1]["train"].shape[0]):
                      images_i_new[i] = images[class_index+1]["train"][i]
                      labels_i_new[i] = labels[class_index+1]["train"][i]
              num = images[class_index+1]["train"].shape[0]
              for index in range(0,25):
                  x_train, y_train, pred = sess.run([child_ops["x_train"],child_ops["y_train"],child_ops["pred"]])
                  pred.sort(axis=0)
                  tmp = np.zeros(FLAGS.batch_size)
                  for i in range(0, FLAGS.batch_size):
                      tmp[i]=pred[i][0]
                  tmp.sort(axis=0)
                  small_20 = tmp[20]
                  m = 0
                  for j in range(0, FLAGS.batch_size):
                      if pred[j][0] <= small_20 and m<20:
                         images_i_new[num] = np.transpose(x_train[j],[1,2,0])
                         labels_i_new[num] = y_train[j]
                         num += 1
                         m += 1
              for i in range(4500, images[class_index]["train"].shape[0]):
                    images_i_new[num] = images[class_index]["train"][i]
                    labels_i_new[num] = labels[class_index]["train"][i]
                    num += 1
              images[class_index+1]["train"] = images_i_new
              labels[class_index+1]["train"] = labels_i_new
              print('num: ',num)
            if epoch >= FLAGS.num_epochs*(1+class_index):
              break
      tf.reset_default_graph()
  return images_i_new, labels_i_new

def main(_):
  print("-" * 80)
  if not os.path.isdir(FLAGS.output_dir):
    print("Path {} does not exist. Creating.".format(FLAGS.output_dir))
    os.makedirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print("Path {} exists. Remove and remake.".format(FLAGS.output_dir))
    shutil.rmtree(FLAGS.output_dir)
    os.makedirs(FLAGS.output_dir)

  print("-" * 80)
  log_file = os.path.join(FLAGS.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)

  if FLAGS.child_fixed_arc is None:
    images, labels = read_data_by_order(FLAGS.data_path)
  else:
    images, labels = read_data_by_order(FLAGS.data_path, num_valids=0)


  utils.print_user_flags()
  image_i, label_i = train(images, labels)
  for i in range(1, int(FLAGS.total_classes/FLAGS.cl_group)):
      image_i, label_i = train_incre(i, images, labels, image_i, label_i)


if __name__ == "__main__":
  tf.app.run()
