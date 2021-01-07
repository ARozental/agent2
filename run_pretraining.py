# coding=utf-8
"""Text file goes in tf record for AGENT comes out."""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from absl import flags
from prepare_dataset import Dataset,BookDataset
from model import AgentModel
import numpy as np
import math
import os
import glob
import random
from tensorflow.python.ops import control_flow_util
control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import pickle

from tensorflow.python.keras.api._v2 import keras
from config import *
from agent_utils import *


FLAGS = flags.FLAGS

flags.DEFINE_string("stuff", None,"")



def flatten(S):
  if S == []:
    return S
  if isinstance(S[0], list):
    return flatten(S[0]) + flatten(S[1:])
  return S[:1] + flatten(S[1:])




def main(_):
  [tf.config.experimental.set_memory_growth(x, enable=True) for x in tf.config.list_physical_devices('GPU')]
  print("start run_pretraining main")
  print(tf.version)
  #ds = Dataset()
  c = Config()

  bb = BookDataset(c)
  model_input_hash = bb.iter.get_next()
  model_input = model_input_hash["tokens"]
  keras.backend.set_floatx(c.dtype)
  #mirrored_strategy = tf.distribute.MirroredStrategy()
  #with mirrored_strategy.scope():
  model = AgentModel(c)
  model.compile()
  model(model_input) #WTF without this line the trainable_parameters don't include the Keras Vars
  print("WWWWW")
  #x = list(map(lambda v: v.name,model.trainable_weights))
  #y = [v for v in x if v.find("agent_level1")!=-1]
  #print(y)


  # with kinda shitty results...
  def delay_loss(loss, global_step, delay, delay_step_size):
    return loss * tf.sigmoid(-delay + delay_step_size * global_step)

  def combine_level_loss(agent, loss_object):
    total_loss = 0.0
    for i in range(agent.depth):
      total_loss += (i + 1) * delay_loss(loss_object[i]['total'], agent.global_step, i * 5, 0.01)
    return total_loss

  #todo: efficiency, create a 2d list of relevant vars loss_object[2]["total"] is dependent on encoder 0 but not on decoder 0
  def get_layer_variables(i,j):
    return [v for v in model.trainable_weights if v.name.find("agent_level"+str(i))!=-1]

  #layer_variables = [get_layer_variables(i,0) for i in range(model.depth)]
  layer_variables = [[[]] * model.depth] * model.depth
  for i in range(model.depth):
    for j in range(i,model.depth):
      layer_variables[i][j]=get_layer_variables(i,j)


  def apply_loss(model,loss_object,tape,optimizer):
    all_grads = []
    all_variables = []
    for i in range(model.depth):
      for j in range(i,model.depth):
        grads = tape.gradient(loss_object[j]["total"]*pow(0.25,j-i), layer_variables[i][j])
        #optimizer.apply_gradients(zip(grads, layer_variables[i][j])) #;todo: understand which gradients are useless here i=2, j=1 decoder1 coherence1, decoder0 coherence0
        all_grads.extend(grads)
        all_variables.extend(layer_variables[i][j])
    optimizer.apply_gradients(zip(all_grads, all_variables))
    return


  optimizer = keras.optimizers.Adam(learning_rate=0.001)
  #tf.compat.v1.disable_eager_execution()

  def train_one_step(model, optimizer,global_step, x):
    with tf.GradientTape(persistent=True) as tape:
      dvts, loss_object, total_loss = model(x)
      #print(dvts)
      #print(loss)
      #loss = tf.reduce_mean(dvts)
      #print("loss")
      #print(loss)

      #grads = tape.gradient(total_loss, model.trainable_variables)
      #todo? maybe tape here the grads from lower level losses so each loss effect its own layer the most the earlier layers less
      #optimizer.apply_gradients(zip(grads, model.trainable_variables))
      apply_loss(model,loss_object, tape,optimizer)

      print("total_loss: " + str(total_loss.numpy()))
      # for k,v in loss_object.items():
      #   loss_object[k]["total"] = 0.0
      #   for k1,v1 in v.items():
      #     if k1!="total":
      #       loss_object[k]["total"]+=v1.numpy()
      #       loss_object[k][k1] = v1.numpy()


      print("all: " + str(loss_object))
    return total_loss,loss_object


  #print(type(model.trainable_weights))
  #print(list(map(lambda x: [x.name,x.shape],model.trainable_weights)))
  #print(list(map(lambda x: [x.name,x.shape],model.weights)))
  print("trainable_weights_number: " + str(len(np.concatenate(list(map(lambda t: t.numpy().flatten(), model.trainable_weights)))))) #no   print(model.summary()) as there is no model.build() as there is no input shape. Keras sucks

  def record_with_tensorboard(total_loss,loss_object,train_summary_writer,total_loss_list,all_losses_lists):
    total_loss_list.append(total_loss)
    for lvl in range(3):
      all_losses_lists[lvl]['mlm'].append(loss_object[lvl]['mlm'])
      all_losses_lists[lvl]['coherence'].append(loss_object[lvl]['coherence'])
      all_losses_lists[lvl]['compressor'].append(loss_object[lvl]['compressor'])
      all_losses_lists[lvl]['reconstruction'].append(loss_object[lvl]['reconstruction'])
      all_losses_lists[lvl]['total'].append(loss_object[lvl]['total'])
    if global_step%5==0:
      print("round " + str(global_step))
      with train_summary_writer.as_default():
        tf.summary.scalar('total_loss', sum(total_loss_list)/len(total_loss_list), step=global_step)
        total_loss_list = []
        for lvl in range(3):
          tf.summary.scalar(str(lvl)+'_mlm_loss', sum(all_losses_lists[lvl]['mlm']) / len(all_losses_lists[lvl]['mlm']), step=global_step)
          tf.summary.scalar(str(lvl)+'_coherence_loss', sum(all_losses_lists[lvl]['coherence']) / len(all_losses_lists[lvl]['coherence']), step=global_step)
          tf.summary.scalar(str(lvl)+'_compressor_loss', sum(all_losses_lists[lvl]['compressor']) / len(all_losses_lists[lvl]['compressor']), step=global_step)
          tf.summary.scalar(str(lvl)+'_reconstruction_loss', sum(all_losses_lists[lvl]['reconstruction']) / len(all_losses_lists[lvl]['reconstruction']), step=global_step)
          tf.summary.scalar(str(lvl)+'_total_loss', sum(all_losses_lists[lvl]['total']) / len(all_losses_lists[lvl]['total']), step=global_step)
          all_losses_lists[lvl]['mlm'] = []
          all_losses_lists[lvl]['coherence'] = []
          all_losses_lists[lvl]['compressor'] = []
          all_losses_lists[lvl]['reconstruction'] = []
          all_losses_lists[lvl]['total'] = []
    return


  ### train loop and log tensor loss

  train_summary_writer = tf.summary.create_file_writer("log/layered_loss2_larger_batch_and_0.25")
  total_loss_list,all_losses_lists = [],{}
  for lvl in range(len(c.vector_sizes)-1):
    all_losses_lists[lvl] = {'mlm': [], 'coherence': [], 'compressor': [], 'reconstruction': [], 'total': []}
  for global_step in range(100000):
    model_input_hash = bb.iter.get_next()
    model_input = model_input_hash["tokens"]
    total_loss,loss_object = train_one_step(model, optimizer,global_step,model_input)
    record_with_tensorboard(total_loss, loss_object, train_summary_writer, total_loss_list, all_losses_lists)
    if global_step%3==2:
      7
      #model.save_weights("weights.crap", save_format='tf') #works on cpu,   The wrapped dictionary contains a non-string key which maps to a trackable object or mutable data structure
      #model.save_weights("weights.crap", save_format='h5') #RuntimeError: Unable to create link (name already exists) File "h5py/h5o.pyx", line 202, in h5py.h5o.link
      #model.load_weights("weights.crap")
  #print(model.trainable_weights[-1])

if __name__ == "__main__":
  tf.compat.v1.app.run()