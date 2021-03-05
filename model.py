
import tensorflow as tf
from functools import partial
from tensorflow.python.keras.api._v2 import keras
from tensorflow import autograph
from collections import defaultdict
import numpy as np
from loss_functions import *
from agent_utils import *
import transformer
from prepare_dataset import create_reverse_tokenizer, join_text
import random
# import tensor2tensor.models.evolved_transformer as evolved_transformer
# from tensor2tensor.layers import transformer_layers, common_attention
# from tensor2tensor.models.research.universal_transformer_util import *
# from tensor2tensor.models.research.universal_transformer import update_hparams_for_universal_transformer
# from tensor2tensor.models.transformer import transformer_tiny
# from tensor2tensor.models.evolved_transformer import evolved_transformer_base
# from tensor2tensor.models import transformer, evolved_transformer
# from tensor2tensor.models.research.universal_transformer_util import *
# from tensor2tensor.models.research.universal_transformer import update_hparams_for_universal_transformer
# from tensor2tensor.models.transformer import transformer_tiny
# from tensor2tensor.models.evolved_transformer import evolved_transformer_base
# from tensor2tensor.models import transformer, evolved_transformer
# from tensor2tensor.layers import transformer_layers, common_attention


# encoder_padding = common_attention.embedding_to_padding(inputs)
# ignore_padding = common_attention.attention_bias_ignore_padding(encoder_padding)
# bias = ignore_padding
# x = self.encoding_layer(x)
# x = self.encoding_layer(x, bias, self.hparams)
# if training:
# x = self.dropout(x, training=training)

def do_to_tree(func,lvl):
  def recur(t):
    depth = tf.shape(tf.shape(t))[0]
    if depth == lvl:
      return func(t)
    else:
      return tf.stack([recur(x) for x in t])
  return autograph.to_graph(recur, recursive=True)


def create_dvts(agent,token_tree_batch):
  """
  input in token_tree and agent model
  output is: DVT,high_level embedding matrixes,A matrix for PNDB
  """
  embedding_matrixes = {}
  for i in range(agent.depth)[1:]:
    embedding_matrixes[i] = []
  loss_sum_hashes = {} #WTF when it was vectors we lost the gradients???
  text_num_hash = {}
  for i in range(agent.depth):
    loss_sum_hashes[i] = {}
    text_num_hash[i] = 0.0
    for j in ["mlm","coherence","reconstruction","compressor"]: #no "compressor" here?
      loss_sum_hashes[i][j] = 0.0
  token_trees = [token_tree for token_tree in token_tree_batch]


  def construct_dvt(tree,current_lvl):
    add_eos_layer = add_eos(agent.encoders[current_lvl].eos, agent.config, current_lvl)
    if current_lvl==0:
      matrix = tf.RaggedTensor.from_tensor(add_eos_layer(tf.ragged.map_flat_values(tf.nn.embedding_lookup, agent.embedding_matrixes[0], tree)))
      #print(matrix.shape[0])
    else:
      ddd = [construct_dvt(sub_tree, current_lvl-1) for sub_tree in tree]
      if len(ddd)==1:
        matrix = tf.RaggedTensor.from_tensor(ddd)
      else:
        matrix = tf.ragged.stack(ddd)
      embedding_matrixes.get(current_lvl, list()).extend([matrix])
      matrix = add_eos_layer(matrix)


    input_mask = tf.squeeze(create_input_mask(agent.config, current_lvl, min(matrix.shape[0] + 1,agent.config.sequence_lengths[current_lvl])))  # +1 for eos token
    padding_mask = 1.0 - input_mask
    encoded = agent.encoders[current_lvl](matrix,mask=padding_mask)
    #print(["tt  " + str(type(encoded)),str(type(matrix))])
    look_ahead_mask = agent.look_ahead_masks[current_lvl]


    matrix = tf.reshape(matrix.flat_values, [-1, agent.config.vector_sizes[current_lvl]])


    mlm_loss = calc_mlm_loss(agent, current_lvl, encoded, matrix)
    coherence_loss = calc_coherence_loss(agent, current_lvl, matrix)
    vector = agent.compressors[current_lvl](encoded)

    decompresssed_matrix = agent.decompressors[current_lvl](vector)
    compressor_loss = calc_autoencoder_regularization_loss(matrix,decompresssed_matrix[0:len(matrix)],agent) #sad, either find a good way to measure distance between matrixes or don't use it


    training = True #Hardcoded in train mode for now

    reconstructed_matrix = agent.decoders[current_lvl](decompresssed_matrix,decompresssed_matrix, training,look_ahead_mask, padding_mask)
    reconstruction_loss = calc_reconstruction_loss(agent, current_lvl, matrix, reconstructed_matrix)

    loss_sum_hashes[current_lvl]["mlm"] = loss_sum_hashes[current_lvl]["mlm"] + mlm_loss
    loss_sum_hashes[current_lvl]["coherence"] = loss_sum_hashes[current_lvl]["coherence"] + coherence_loss
    loss_sum_hashes[current_lvl]["compressor"] = loss_sum_hashes[current_lvl]["compressor"] + compressor_loss
    loss_sum_hashes[current_lvl]["reconstruction"] = loss_sum_hashes[current_lvl]["reconstruction"] + reconstruction_loss
    text_num_hash[current_lvl] = text_num_hash[current_lvl] + 1.0

    return vector
  # print(len(token_trees))
  # print(token_trees[0])
  # print("JJJJJ")
  # print(construct_dvt(token_trees[0],agent.depth-1)) #4
  # print("JJJJJ")


  dvts = tf.stack([construct_dvt(token_tree,agent.depth-1) for token_tree in token_trees])

  embedding_matrixes = {k: tf.concat([z.with_row_splits_dtype('int32') for z in v], 0) for k, v in embedding_matrixes.items()} #ValueError: Input RaggedTensors have mismatched row_splits dtypes; use RaggedTensor.with_row_splits_dtype() to convert them to compatible dtypes.

  #losses = [loss_sum_arrays[i]/text_num_arrays[i] for i in range(agent.depth)]
  total_loss = 0.0
  for i in range(agent.depth):
    loss_sum_hashes[i]['total'] = 0.0
    for j in ["mlm","coherence","reconstruction","compressor"]: #no "compressor" here?
      loss_sum_hashes[i][j] /= text_num_hash[i]
      total_loss += loss_sum_hashes[i][j]
      loss_sum_hashes[i]['total'] += loss_sum_hashes[i][j]


  return dvts,embedding_matrixes,loss_sum_hashes,total_loss



def keras_wrapper(l):
  def w(x, training=None,mask=None):
    x = tf.expand_dims(x, axis=0)
    x = l(x,training=training,mask=mask)
    x = tf.squeeze(x)
    return x
  return w


class AgentEncoder(keras.layers.Layer):
  def __init__(self, config,level):
    super(AgentEncoder, self).__init__()
    self.config = config
    self.level = level
    self.pad = tf.Variable(normal_init([config.vector_sizes[level]]), name="agent_level" + str(level) + "_pad", dtype=config.dtype)
    self.eos = tf.Variable(normal_init([config.vector_sizes[level]]), name="agent_level" + str(level) + "_eos", dtype=config.dtype)
    self.mlm_mask = tf.Variable(normal_init([config.vector_sizes[level]]), name="agent_level" + str(level) + "_mask", dtype=config.dtype)

    self.padding_layer = keras.layers.Lambda(add_padding(self.pad,config,level))

    #self.keras_layer = transformer.EncoderLayer(config,level)
    self.keras_layer = transformer.Encoder(config,level)
    self.layer = keras_wrapper(self.keras_layer)
    #self.layer = transformer.EncoderLayer(config,level)

  def call(self, inputs, training=None,mask=None):
    #mask = tf.squeeze(1.0 - create_input_mask(self.config, self.level, inputs.shape[0]+1)) #1 for all pad (but not EoS tokens
    #x = self.layer(self.padding_layer(inputs))
    x = self.padding_layer(inputs)
    x = self.layer(x,training, mask) #todo: get mask from padding
    return x

class AgentCompressor(keras.layers.Layer):
  def __init__(self, config,level):
    super(AgentCompressor, self).__init__()
    self.keras_layer = keras.layers.RNN(tf.keras.layers.LSTMCell(config.vector_sizes[level+1], activation="linear"), dtype=config.dtype,name="agent_level" + str(level)+"_AgentCompressor_lstm")
    #todo: add a dense layer so we can control the size and return memory and bidirectional, or have no activation
    self.d1 = 7 #todo: have a dropout layer here
    self.layer = keras_wrapper(self.keras_layer)

  def call(self, inputs, training=None, mask=None):
    #_,last_h,last_c = self.layer(inputs)
    last_h = self.layer(inputs)
    return last_h


class AgentDecompressor(keras.layers.Layer):
  def __init__(self, config,level):
    super(AgentDecompressor, self).__init__()
    self.mem = tf.Variable([normal_init([config.vector_sizes[level+1]])], name="agent_level" + str(level) + "_mem", dtype=config.dtype) #initial state of memory cell
    self.keras_layer = keras.layers.RNN(tf.keras.layers.LSTMCell(config.vector_sizes[level+1]),return_state=True, dtype=config.dtype,name="agent_level" + str(level)+"_AgentDecompressor_lstm") #LSTM(latent_dim, return_sequences=True, return_state=True) =>, activation="linear"?????
    self.out_projection =  keras.layers.Dense(config.vector_sizes[level], name="agent_level" + str(level) + "_dp", activation="linear", dtype=config.dtype)
    self.d1 = 7 #todo: have a dropout layer here

    def decode_keras_lstm_wrapper(lstm):
      def w(text_vector):
        res = []
        last_h, last_c = tf.expand_dims(text_vector,axis=0),self.mem
        for i in range(config.sequence_lengths[level]):
          decoder_outputs, last_h, last_c = lstm(
            tf.expand_dims(last_h,axis=0), initial_state=[last_h, last_c])
          res.append(tf.squeeze(decoder_outputs))
        res = self.out_projection(tf.stack(res))
        return res
      return w
    self.layer = decode_keras_lstm_wrapper(self.keras_layer)

  def call(self, inputs, training=None, mask=None):
    x = self.layer(inputs)
    return x


class AgentDecoder(keras.layers.Layer):
  def __init__(self, config,level):
    super(AgentDecoder, self).__init__()
    self.config = config
    self.level = level

    def keras_wrapper(l):
      def w(x, enc_output, training,look_ahead_mask, padding_mask):
        x = tf.expand_dims(x, axis=0)
        x,_ = l(x, enc_output, training,look_ahead_mask, padding_mask)
        x = tf.squeeze(x)
        return x

      return w

    self.keras_layer = transformer.Decoder(config,level)
    self.layer = keras_wrapper(self.keras_layer)

  def call(self, inputs, enc_output, training,look_ahead_mask, padding_mask):
    #mask = tf.squeeze(1.0 - create_input_mask(self.config, self.level, inputs.shape[0]+1)) #1 for all pad (but not EoS tokens
    #x = self.layer(self.padding_layer(inputs))
    x = inputs
    x = self.layer(x,enc_output, training,look_ahead_mask, padding_mask) #todo: get mask from padding
    return x

class AgentCoherenceChecker(keras.layers.Layer):
  def __init__(self, config,level):
    super(AgentCoherenceChecker, self).__init__()
    self.d1 = keras.layers.Dense(config.vector_sizes[level+1], activation=tf.nn.tanh, dtype=config.dtype, name="agent_level" + str(level) + "_cc_d1")
    self.d2 = keras.layers.Dense(config.vector_sizes[level+1], activation=tf.nn.tanh, dtype=config.dtype, name="agent_level" + str(level) + "_cc_d2")
    self.out = keras.layers.Dense(1, activation=tf.nn.sigmoid, dtype=config.dtype, name="agent_level" + str(level) + "_cc_out")

  def call(self, inputs, training=None, mask=None):
    x = self.d1(inputs)
    x = self.d2(x)
    x = self.out(x)
    return x

class AgentEmbedding(keras.layers.Layer):
  def __init__(self, config):
    super(AgentEmbedding, self).__init__()
    self.keras_layer = keras.layers.Embedding(config.vocab_sizes[0], config.vector_sizes[0], input_length=config.sequence_lengths[0],
                             input_shape=(config.vector_sizes[0],), dtype=config.dtype)

    self.layer = keras_wrapper(self.keras_layer)

  def call(self, inputs, training=None, mask=None):
    x = tf.RaggedTensor.from_tensor(self.layer(inputs)) #todo: check if we actually need to enforce it here
    return x

def init_embedding_matrix(config,level,const=True):
  n = normal_init([config.vocab_sizes[level],config.vector_sizes[level]])
  if const:
    return tf.constant(n, name="agent_level" + str(level) + "_embedding", dtype=config.dtype)
  return tf.Variable(n, name="agent_level" + str(level) + "_embedding", dtype=config.dtype)


# #todo: finish them
# class AgentGenerator(keras.layers.Layer):
#   def __init__(self, config,level):
#     super(AgentGenerator, self).__init__()
#     self.d1 = keras.layers.Dense(config.vector_sizes[level+1], activation=tf.nn.tanh, dtype=config.dtype)
#     self.d2 = keras.layers.Dense(config.vector_sizes[level+1], activation=tf.nn.tanh, dtype=config.dtype)
#     self.out = keras.layers.Dense(config.vector_sizes[level+1], activation="linear", dtype=config.dtype)
#
#   def call(self, inputs, training=None, mask=None):
#     #todo: lambda:  prompt==input
#     x = self.d1(inputs)
#     x = self.d2(x)
#     x = self.out(x)
#     return x
#
# class AgentDiscriminator(keras.layers.Layer):
#   def __init__(self, config,level):
#     super(AgentDiscriminator, self).__init__()
#     self.d1 = keras.layers.Dense(config.vector_sizes[level+1], activation=tf.nn.tanh, dtype=config.dtype)
#     self.d2 = keras.layers.Dense(config.vector_sizes[level+1], activation=tf.nn.tanh, dtype=config.dtype)
#     self.out = keras.layers.Dense(config.vector_sizes[level+1], activation="linear", dtype=config.dtype)
#
#   def call(self, generated_input, dvt_input, decompressor,decoder,training=True):
#     return generated_input

class AgentModel(keras.Model):

  def __init__(self,config):
    super(AgentModel, self).__init__()
    self.config = config
    self.depth = len(config.num_heads)-1

    self.rt = create_reverse_tokenizer("chars.txt")

    self.embedding_layer = AgentEmbedding(config)


    # self.embedding_matrixes = { **{0: init_embedding_matrix(config,0,False)}, **{k: init_embedding_matrix(config,k) for k in range(1,self.depth)}}
    # self.encoders = {k: AgentEncoder(config,k) for k in range(self.depth)} #[AgentEncoder(config,k) for k in range(self.depth)]
    # self.encoders_transforms = {k: keras.layers.Dense(config.vector_sizes[k],activation=gelu,dtype=config.dtype
    #                                                   ,name="agent_level" + str(k) + "_transforms") for k in range(self.depth)}
    # self.compressors = {k: AgentCompressor(config, k) for k in range(self.depth)}
    # self.coherence_checkers = {k: AgentCoherenceChecker(config, k) for k in range(self.depth)}
    # self.decompressors = {k: AgentDecompressor(config, k) for k in range(self.depth)}
    # self.decoders = {k: AgentDecoder(config, k) for k in range(self.depth)}


    self.embedding_matrixes = [init_embedding_matrix(config,0,False)]+[init_embedding_matrix(config,k) for k in range(1,self.depth)]
    self.encoders = [AgentEncoder(config,k) for k in range(self.depth)]
    self.encoders_transforms = [keras.layers.Dense(config.vector_sizes[k],activation=gelu,dtype=config.dtype,name="agent_level" + str(k) + "_transforms") for k in range(self.depth)]
    self.compressors = [AgentCompressor(config, k) for k in range(self.depth)]
    self.coherence_checkers = [AgentCoherenceChecker(config, k) for k in range(self.depth)]
    self.decompressors = [AgentDecompressor(config, k) for k in range(self.depth)]
    self.decoders = [AgentDecoder(config, k) for k in range(self.depth)]
    self.look_ahead_masks = [create_look_ahead_mask(config.sequence_lengths[k]) for k in range(self.depth)]


    self.global_step = 0


    self.constants = {}#tf.ones and tf.zeros that we use across the graph and don't want to create anew each time, should it be here??
    #self.look_ahead_masks = {k: create_look_ahead_mask(config.sequence_lengths[k]) for k in range(self.depth)}


  def decode(self,vec,level):
    def cut_at_eos(matrix,level):
      """when decoding we need to know when to stop. That is take just 5 sentences for a paragraph instead of max_sentences."""
      eos = self.encoders[level].eos
      good_vectors = []
      for v in matrix:
        d_stop = tf.losses.mean_squared_error(v,eos)
        embedding = self.embedding_matrixes[level]
        closest_vec_dist = min([tf.losses.mean_squared_error(v,embedding[i]) for i in range(self.config.vocab_sizes[level])])
        if len(good_vectors)==0 or closest_vec_dist<d_stop : #and also d_stop should probably be smaller than some absolute value.
          good_vectors.append(v)
        else:
          break
      return good_vectors

    matrix = self.decompressors[level](vec)
    cut_matrix = cut_at_eos(matrix,level)
    if level == 0:

      # print("1111")
      res = [tf.argmin([tf.losses.mean_squared_error(self.embedding_matrixes[level][i], t) for i in range(self.config.vocab_sizes[level])]) for t in cut_matrix]
      # print(res)
      # print("2222")
      #return vec
      #x=[tf.argmin([tf.losses.mean_squared_error(self.embedding_matrixes[level][i], t) for i in range(self.config.vocab_sizes[level])]) for t in matrix]
      #print(x)
      return list(map(lambda x: x.numpy(),res))
      #return 8
    else:
 #all vectors until EoS (not included)

      input_mask = tf.squeeze(create_input_mask(self.config, level, len(cut_matrix)))
      padding_mask = 1.0 - input_mask
      matrix = self.decoders[level](matrix,matrix,False,self.look_ahead_masks[level], padding_mask)[0:len(cut_matrix)]
      # def tt(v):
      #   try:
      #     print("good")
      #     print(v,level)
      #     print("fine")
      #     return self.decode(v,level-1)
      #   except:
      #     print("shit")
      #     print(v,level)
      #     print("crap")
      #
      #     return self.decode(v, level - 1)
      return [self.decode(v,level-1) for v in matrix]

  def call(self, inputs, training=None, mask=None):
    dvts, batch_embedding_matrixes,loss_object,total_loss = create_dvts(self,inputs)
    #shifted = {k: tf.concat([batch_embedding_matrixes[k],self.embedding_matrixes[k]],axis=0)[:self.config.vocab_sizes[k]] for k in range(1,self.depth)} #todo: add eos vec to matrixes => here???
    shifted = [tf.concat([batch_embedding_matrixes[k],self.embedding_matrixes[k]],axis=0)[:self.config.vocab_sizes[k]] for k in range(1,self.depth)] #todo: add eos vec to matrixes => here???

    #total_loss = combine_level_loss(self, loss_object)
    y5=self.decode(dvts[0],4)
    print("decoding")
    t1=batch_embedding_matrixes[1][0]
    t2=batch_embedding_matrixes[2][0]
    y1 = self.decode(t1,0)
    y2 = self.decode(t2,1)
    #print([join_text(y5, self.rt),y5,inputs[0]])
    print(["word: ",join_text(y1, self.rt),y1,t1])
    print(["sentence: ",join_text(y2, self.rt),y2,t2])
    print("done")

    #x = self.decode(dvts[0],2) #sometimes probably if matrix is not of full length when sent to decode: tensorflow.python.framework.errors_impl.InvalidArgumentError: In[0] mismatch In[1] shape: 5 vs. 1: [1,4,5,5] [1,4,1,2] 0 0 [Op:BatchMatMulV2] name: agent_model/agent_decoder_2/decoder_2/decoder_layer_4/multi_head_attention_14/MatMul
    #print("kkk")
    #print(x) 

    for k in range(1,self.depth):
      self.embedding_matrixes[k] = shifted[k-1]

    #self.embedding_matrixes.update(shifted)

    #print(loss_object)
#    print(dvts[1])
    #loss = tf.reduce_mean([tf.reduce_mean(loss_arrays[i]) for i in range(self.depth)])

    self.global_step += 1
    return dvts,loss_object,total_loss

