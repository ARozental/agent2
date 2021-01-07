import tensorflow as tf
import numpy as np
from random import randint
import functools

def normal_init(shape):
  return np.random.normal(loc=0.0, scale=0.125/(shape[-1]**0.5), size=shape)


def add_eos_and_padding(pad, eos, config, level):
  def p(x):
    padding_matrix = tf.concat([tf.expand_dims(eos, axis=0),
                                tf.reshape(tf.tile(pad, [config.sequence_lengths[level]]),
                                           (config.sequence_lengths[level], config.vector_sizes[level]))], axis=0)
    return tf.concat([x, padding_matrix], axis=0)[:config.sequence_lengths[level]]
  return p

def add_padding(pad, config, level):
  def p(x):
    padding_matrix = tf.reshape(tf.tile(pad, [config.sequence_lengths[level]]),
                                           (config.sequence_lengths[level], config.vector_sizes[level]))
    return tf.concat([x, padding_matrix], axis=0)[:config.sequence_lengths[level]]
  return p

def add_eos(eos, config, level):
  def p(x):
    eos_matrix = tf.expand_dims(eos, axis=0)
    return tf.concat([x, eos_matrix], axis=0)[:config.sequence_lengths[level]]
  return p

def create_look_ahead_mask(size):
  mask = 1.0 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

#todo: change it! as we use vectors here
# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
#
#   # add extra dimensions so that we can add the padding
#   # to the attention logits.
#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_input_mask(config,level, sequence_length):

  input_mask = tf.concat([tf.ones(shape=[sequence_length], dtype=config.dtype),
                    tf.zeros(shape=[config.sequence_lengths[level]], dtype=config.dtype)],axis=0)[0:config.sequence_lengths[level]]
  return tf.expand_dims(input_mask,axis=-1)


def gelu(input_tensor):
  """This Activation was used for origianl BERT MLM
  paper: https://arxiv.org/abs/1606.08415
  """
  cdf = 0.5 * (1.0 + tf.math.erf(input_tensor / 1.4142135623730951))
  return input_tensor * cdf

def merge_matrixes(m1,m2,x):
  if str(type(m1)).lower().find("ragged") != -1:  # todo: remove later when all is ragged
    m1 = tf.reshape(m1.flat_values, [-1,m2.shape[-1]])
  return tf.concat([m1[0:x],m2,m1[x:tf.shape(m1)[0]]],axis=0)

def one_hot_merge_matrix(m1,m2,x):
  return tf.concat([tf.zeros(tf.shape(m1[0:x])),tf.ones(tf.shape(m2)),tf.zeros(tf.shape(m1[x:tf.shape(m1)[0]]))],axis=0)

def dist(v1,v2):
  tf.losses.mean_squared_error(v1,v2)

def flatten(S):
  if S == []:
    return S
  if isinstance(S[0], list):
    return flatten(S[0]) + flatten(S[1:])
  return S[:1] + flatten(S[1:])





def add_gradient_hashes(h1,h2):
  both = [k for k in h1.keys() if k in h2.keys()]
  h1k = [k for k in h1.keys() if not(k in h2.keys())]
  h2k = [k for k in h2.keys() if not(k in h1.keys())]
  return {**{k:h1[k]+h2[k] for k in both},**{k:h1[k] for k in h1k},**{k:h2[k] for k in h2k}}

def add_gradients(grad_list):
  g_hashes = [{t: t for t in g if t is not None} for g in grad_list] #.name/id ;(
  h = functools.reduce(add_gradient_hashes, g_hashes,{})
  res = list(h.values())
  return res

    #todo: add layer_norm