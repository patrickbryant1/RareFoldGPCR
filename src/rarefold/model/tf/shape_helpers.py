"""Utilities for dealing with shapes of TensorFlow tensors."""
import tensorflow.compat.v1 as tf


def shape_list(x):
  """Return list of dimensions of a tensor, statically where possible.

  Like `x.shape.as_list()` but with tensors instead of `None`s.

  Args:
    x: A tensor.
  Returns:
    A list with length equal to the rank of the tensor. The n-th element of the
    list is an integer when that dimension is statically known otherwise it is
    the n-th element of `tf.shape(x)`.
  """
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)

  static = x.get_shape().as_list()
  shape = tf.shape(x)

  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret
