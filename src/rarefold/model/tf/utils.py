"""Shared utilities for various components."""
import tensorflow.compat.v1 as tf


def tf_combine_mask(*masks):
  """Take the intersection of float-valued masks."""
  ret = 1
  for m in masks:
    ret *= m
  return ret


class SeedMaker(object):
  """Return unique seeds."""

  def __init__(self, initial_seed=0):
    self.next_seed = initial_seed

  def __call__(self):
    i = self.next_seed
    self.next_seed += 1
    return i

seed_maker = SeedMaker()


def make_random_seed():
  return tf.random.uniform([2],
                           tf.int32.min,
                           tf.int32.max,
                           tf.int32,
                           seed=seed_maker())
