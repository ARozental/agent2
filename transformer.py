
"""
Transformer taken from TF-2.0.0-alpha doc.
At least it is better than nothing.
"""


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
from tensorflow.python.keras.api._v2 import keras
from agent_utils import *
from config import *


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / float(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  sines = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  cosines = np.cos(angle_rads[:, 1::2])

  pos_encoding = np.concatenate([sines, cosines], axis=-1)

  pos_encoding = pos_encoding[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=Config().dtype) #todo: pass it in parapms


def create_padding_mask(seq):
  """
  create_padding_mask(tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])) =>
  array([[[[0., 0., 1., 1., 0.]]],[[[0., 0., 0., 1., 1.]]],[[[1., 1., 1., 0., 0.]]]], dtype=float32)>
  """
  seq = tf.cast(tf.math.equal(seq, 0), Config().dtype) #todo: pass it in parapms

  # add extra dimensions so that we can add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  """
  for left-to-right decoding
  x = tf.random.uniform((1, 3))
  temp = create_look_ahead_mask(x.shape[1]) =>
  array([[0., 1., 1.],[0., 0., 1.],[0., 0., 0.]], dtype=float32)>
  """
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], Config().dtype) #todo: pass it in parapms
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += tf.cast(mask * -1e9, Config().dtype) #todo: pass it in parapms

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth_v)

  return output, attention_weights

class MultiHeadAttention(keras.layers.Layer):
  def __init__(self, d_model, num_heads,level,layer_index):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    self.level = level

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model,name="agent_level" + str(level) + "_wq"+ str(layer_index))
    self.wk = tf.keras.layers.Dense(d_model,name="agent_level" + str(level) + "_wk"+ str(layer_index))
    self.wv = tf.keras.layers.Dense(d_model,name="agent_level" + str(level) + "_wv"+ str(layer_index))

    self.dense = tf.keras.layers.Dense(d_model,name="agent_level" + str(level) + "_mha"+ str(layer_index))

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = len([x for x in q]) #todo: make less dumb, works for both ragged and regular tensors
    #batch_size = tf.shape(q)[0]


    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
      q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff,level,layer_index):
  return keras.Sequential([
      keras.layers.Dense(dff, activation='relu',name="agent_level" + str(level) + "_ff1_l"+ str(layer_index)),  # (batch_size, seq_len, dff)
      keras.layers.Dense(d_model,name="agent_level" + str(level) + "_ff2_l"+ str(layer_index))  # (batch_size, seq_len, d_model)
  ])


#  def __init__(self, config,level, d_model, num_heads, dff, rate=0.1):
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, config,level,layer_index):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(config.vector_sizes[level], config.num_heads[level],level,layer_index)
    self.ffn = point_wise_feed_forward_network(config.vector_sizes[level], config.fnn_sizes[level],level,layer_index) #reasonable default params {heads:8,d_size:512,dff:2042}

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name="agent_level" + str(level) + "_LayerNormalization1e_l")
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name="agent_level" + str(level) + "_LayerNormalization1e_l")

    self.dropout1 = tf.keras.layers.Dropout(config.drop_rate)
    self.dropout2 = tf.keras.layers.Dropout(config.drop_rate)

  def call(self, x, training, mask):
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


class Encoder(tf.keras.layers.Layer):
  def __init__(self, config,level):
    super(Encoder, self).__init__()

    self.d_model = config.vector_sizes[level]
    self.num_layers = config.num_transformer_layers[level]
    self.config = config
    self.level = level

    #self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(config.sequence_lengths[level], self.d_model)

    self.enc_layers = [EncoderLayer(config,level,i)
                       for i in range(self.num_layers)]

    self.dropout = tf.keras.layers.Dropout(config.drop_rate)

  def call(self, x, training, mask):
    hidden_size = self.config.vector_sizes[self.level]
    if str(type(x)).lower().find("ragged")==-1: #todo: remove later when all is ragged
      seq_len = self.config.sequence_lengths[self.level]
    else:
      seq_len = int(len(x.flat_values)/hidden_size)
      x = x.flat_values
    x = tf.reshape(x,[1,seq_len,hidden_size])

    # adding embedding and position encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, self.config.dtype))
    x += self.pos_encoding[:, :seq_len, :]


    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,config,level,layer_index):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(config.vector_sizes[level], config.num_heads[level],level,layer_index)
    self.mha2 = MultiHeadAttention(config.vector_sizes[level], config.num_heads[level],level,layer_index)

    self.ffn = point_wise_feed_forward_network(config.vector_sizes[level], config.fnn_sizes[level],level,layer_index)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name="agent_level" + str(level) + "_LayerNormalization1d_l"+ str(layer_index))
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name="agent_level" + str(level) + "_LayerNormalization2d_l"+ str(layer_index))
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6,name="agent_level" + str(level) + "_LayerNormalization3d_l"+ str(layer_index))

    self.dropout1 = tf.keras.layers.Dropout(config.drop_rate)
    self.dropout2 = tf.keras.layers.Dropout(config.drop_rate)
    self.dropout3 = tf.keras.layers.Dropout(config.drop_rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
      enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
  def __init__(self, config,level):
    super(Decoder, self).__init__()
    self.config = config
    self.level = level


    self.d_model = config.vector_sizes[level]
    self.num_layers = config.num_transformer_layers[level]

    #self.embedding = tf.keras.layers.Embedding(config.vocab_sizes[level], self.d_model)
    self.pos_encoding = positional_encoding(config.sequence_lengths[level], self.d_model)

    self.dec_layers = [DecoderLayer(config,level,i)
                       for i in range(self.num_layers)]
    self.dropout = tf.keras.layers.Dropout(config.drop_rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    hidden_size = self.config.vector_sizes[self.level]
    if str(type(x)).lower().find("ragged")==-1: #todo: remove later when all is ragged
      seq_len = self.config.sequence_lengths[self.level]
    else:
      seq_len = int(len(x.flat_values)/hidden_size)
      x = x.flat_values
    x = tf.reshape(x,[1,seq_len,hidden_size])

    attention_weights = {}

    #x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, self.config.dtype))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
      attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights
