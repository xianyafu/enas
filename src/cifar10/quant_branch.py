#from tensorflow.contrib.quantize.python import copy_graph
#from tensorflow.contrib.quantize.python import fold_batch_norms
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
import sys
sys.path.append(r'/home/fuxianya/github/enas/src/cifar')
import quantize
import fold_batch_norms
def _create_graph(input_graph=None,
                  qint_ops=[],
                  is_training=True,
                  weight_bits=8,
                  activation_bits=8,
                  quant_delay=None,
                  freeze_bn_delay=None,
                  scope=None):
  """Rewrites an input_graph in place for simulated quantization.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.
    is_training: Whether quantizing training or eval graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.
    freeze_bn_delay: Number of steps after which moving mean and variance are
      frozen and used instead of batch statistics during training.
      freeze_bn_delay should be greater than quant_delay and should correspond
      to the number of steps when training has almost converged
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """
  #print("333333333333: ", weight_bits)
  if input_graph is None:
    input_graph = ops.get_default_graph()

  with input_graph.as_default():
    fold_batch_norms.FoldBatchNorms(
        input_graph)
    quantize.Quantize(
        input_graph,
        qint_ops,
        weight_bits=8,
        weight_narrow_range=False,
        activation_bits=8,
        is_training=is_training)
        #quant_delay=quant_delay,
        #weight_bits=weight_bits,
        #activation_bits=activation_bits,
        #scope=scope)


def create_training_graph(input_graph=None, qint_ops=[], quant_delay=None):
  """Rewrites a training input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  This function must be invoked prior to insertion of gradient ops in a graph
  as quantization should be modeled in both forward and backward passes.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  The default value of quant_delay is suitable for finetuning an already trained
  floating point model (recommended).
  If one wants to train a quantized model from scratch, quant_delay should be
  set to the number of steps it take the floating point model to converge.
  Quantization will be activated at this point and effectively finetune the
  model. If quant_delay is not provided when training from scratch, training can
  often fail.

  Args:
    input_graph: The tf.Graph to be transformed.
    quant_delay: Number of steps after which weights and activations are
      quantized during training.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """
  # TODO(raghuramank) Need to have freeze_bn_delay be a function of batch size
  # Currently the values below are hardcoded for mobilenetV1 on imagenet
  # Please use the experimental API if you need to tune these values.
  freeze_bn_delay = None
  _create_graph(
      input_graph=input_graph,
      qint_ops=qint_ops, 
      is_training=True,
      quant_delay=quant_delay,
      freeze_bn_delay=freeze_bn_delay)


def create_eval_graph(input_graph=None, qint_ops=[]):
  """Rewrites an eval input_graph in place for simulated quantization.

  Variables added by the rewrite get added to the global variables collection.

  The graph has fake quantization ops inserted to simulate the error
  introduced by quantization. Since the graph is transformed in place,
  the expected behavior of previously held references to nodes and tensors may
  change.

  Args:
    input_graph: The tf.Graph to be transformed, if None then defaults to the
      default graph.

  Raises:
    ValueError: If elements contains an element that isn't a tf.Tensor or
      tf.Operation.
  """
  _create_graph(input_graph=input_graph, qint_ops=qint_ops, is_training=False)


