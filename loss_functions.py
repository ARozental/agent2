import tensorflow as tf
from tensorflow.python.keras.api._v2 import keras
from tensorflow import autograph
import numpy as np
from agent_utils import *



def create_mlm_mask(config,level):
  mlm_keep_mask = tf.nn.dropout(tf.ones(shape=[config.sequence_lengths[level]], dtype=config.dtype),
                                rate=config.mlm_rate) * (1.0-config.mlm_rate) #undo scaling
  return tf.expand_dims(mlm_keep_mask,axis=-1)

def choose_token_mlm(original_token, mask_token, embedding):
  """
  hard coded 10%,10%,80% (like BERT) move to config??
  :param original_token:
  :param mask_token:
  :param embedding:
  :return:
  """
  r = np.random.uniform()
  if r>0.2:
    return mask_token
  elif r>0.1:
    return original_token
  else:
    return embedding[int(r*10.0*np.shape(embedding)[0])]

def calc_mlm_loss(agent, level, encoded_dvt_input, input_tensor):
  """

  :param agent:
  :param level:
  :param encoded_dvt_input:
  :param input_tensor: a Ragged Matrix
  :return:
  """
  input_mask  = create_input_mask(agent.config, level, min(input_tensor.shape[0], agent.config.sequence_lengths[level]))
  mlm_keep_mask = create_mlm_mask(agent.config, level)

  masked_token_number = tf.reduce_sum((1.0-mlm_keep_mask)*input_mask)
  if masked_token_number<1.0:
    return tf.constant(0.0, dtype=agent.config.dtype)

  fn = lambda x: choose_token_mlm(x, agent.encoders[level].mlm_mask, agent.embedding_matrixes[level])
  pad = add_padding(agent.encoders[level].pad,
                    #agent.encoders[level].eos,
                                    agent.config, level)

  padded_input_tensor = pad(input_tensor)
  padded_replacment_tensor = pad(list(map(fn,input_tensor)))


  #apply mask
  mlm_mask = (1.0-mlm_keep_mask)
  m1 = mlm_keep_mask*input_mask*padded_input_tensor #vectors to keep
  m2 = mlm_mask*input_mask*padded_replacment_tensor #vectors replacements,
  m3 = (1.0-input_mask)*padded_input_tensor #paddings to keep (all of them)
  masked_input = m1+m2+m3
  input_tensor = agent.encoders[level](masked_input, mask=tf.squeeze(1.0 - input_mask))
  input_tensor = agent.encoders_transforms[level](input_tensor)

  x = randint(0, agent.config.vocab_sizes[level])
  embeddings = merge_matrixes(agent.embedding_matrixes[level], encoded_dvt_input, x) #

  #one_hot_labels is a matrix where one_hot_labels[i][j]==1 if the i-th row of the embedding matrix==the j-th row of encoded_dvt_input
  one_hot_labels = merge_matrixes(tf.zeros([agent.config.vocab_sizes[level], agent.config.sequence_lengths[level]], dtype=agent.config.dtype),
                                  tf.eye(agent.config.sequence_lengths[level], dtype=agent.config.dtype), x)

  logits = tf.matmul(input_tensor, embeddings, transpose_b=True) #v_size*num_v, v_size*num_e
  log_probs = tf.nn.log_softmax(logits, axis=-1)
  per_word_loss = -tf.reduce_sum(log_probs * mlm_mask*tf.transpose(one_hot_labels),axis=-1) #loss for unmasked words = 0.0
  res=tf.reduce_mean(per_word_loss)
  return res


def create_segment_b_mask(sequence_length,dtype):
  #p=np.round(np.random.uniform())*np.random.uniform()
  p=tf.round(tf.random.uniform([],dtype=dtype))*tf.random.uniform([],dtype=dtype)
  if p==0:
    return tf.expand_dims(np.zeros(shape=[sequence_length],dtype=dtype),axis=-1)
  r = np.floor(np.random.uniform(size=[sequence_length])/(1.0-p))
  #r = tf.floor(tf.random.uniform([sequence_length],dtype=dtype)/(1.0-p))

  segment_b = tf.cast(np.divide(r,r,out=r, where=r>0),dtype=dtype)
  return  tf.expand_dims(segment_b * np.round(np.random.uniform(size=[sequence_length])),axis=-1)
  #return  tf.expand_dims(segment_b * np.round(tf.random.uniform([sequence_length],dtype=dtype)),axis=-1)

def choose_token_coherence(embedding):
  """
  :param embedding:
  :return:
  """
  return embedding[randint(0,np.shape(embedding)[0]-1)]


def calc_coherence_loss(agent, level, input_tensor):
  input_mask  = create_input_mask(agent.config, level, input_tensor.shape[0] - 1)#eos token is not replaced.
  pad = add_padding(agent.encoders[level].pad,
                    #agent.encoders[level].eos,
                                    agent.config, level)

  padded_input_tensor = pad(input_tensor)

  segment_b_mask = create_segment_b_mask(padded_input_tensor.shape[0],agent.config.dtype)
  replaced_tokens_ratio = tf.reduce_mean(segment_b_mask)  #the label


  fn = lambda x: choose_token_coherence(agent.embedding_matrixes[level])
  padded_replacment_tensor = pad(list(map(fn,input_tensor)))
  #input_tensor = input_tensor*input_mask*(1.0-segment_b_mask)
  # print(type(input_tensor))
  # print(input_tensor)
  # print(segment_b_mask)
  # print(input_mask)
  # print(padded_input_tensor)
  m1 = (1.0-segment_b_mask)*input_mask*padded_input_tensor #vectors to keep :todo fails because padded_input_tensor is ragged
  m2 = segment_b_mask*input_mask*padded_replacment_tensor #vectors replacment
  m3 = (1.0-input_mask)*padded_input_tensor #paddings to keep (all of them) #todo: can we pad only once here
  masked_input = m1 + m2 + m3

  padding_mask  = tf.squeeze(1.0 - create_input_mask(agent.config, level, min(input_tensor.shape[0], agent.config.sequence_lengths[level])))

  encoded = agent.encoders[level](masked_input, mask=padding_mask)
  vector = agent.compressors[level](encoded) #should be equal to dvt vector if p==0

  cc_res = agent.coherence_checkers[level](tf.expand_dims(vector, axis=0))
  res = tf.squeeze(((replaced_tokens_ratio-cc_res)**2)**0.5)
  return res


def calc_autoencoder_regularization_loss(m1,m2,agent):
  """with l1 norm for now"""
  return tf.norm(m1/tf.norm(m1) - m2/tf.norm(m2)) * agent.config.auto_encoder_regularization

def calc_reconstruction_loss(agent, level, dvt_matrix, reconstructed_matrix):
  real_length = len(dvt_matrix) #including eos token
  reconstructed_matrix = reconstructed_matrix[0:real_length] #loss doesn't care about tokens after eos
  x = randint(0, agent.config.vocab_sizes[level])
  embeddings = merge_matrixes(agent.embedding_matrixes[level], dvt_matrix, x) #
  one_hot_labels = merge_matrixes(tf.zeros([agent.config.vocab_sizes[level], real_length], dtype=agent.config.dtype),
                                  tf.eye(real_length, dtype=agent.config.dtype), x)[:,:real_length]

  #we don't have an additional projection layer here.


  logits = tf.matmul(reconstructed_matrix, embeddings, transpose_b=True) #v_size*num_v, v_size*num_e
  log_probs = tf.nn.log_softmax(logits, axis=-1)
  per_word_loss = -tf.reduce_sum(log_probs * tf.transpose(one_hot_labels), axis=-1)
  res = tf.reduce_mean(per_word_loss)
  return res


#with kinda shitty results...
def delay_loss(loss,global_step,delay,delay_step_size):
  return loss*tf.sigmoid(-delay+delay_step_size*global_step)
def combine_level_loss(agent,loss_object):
  total_loss = 0.0
  for i in range(agent.depth):
    total_loss+=(i+1)*delay_loss(loss_object[i]['total'],agent.global_step,i*5,0.01)
  return total_loss
