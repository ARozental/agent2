#some doc for what we need
import torch.nn as nn
import torch
from src.pre_processing import TreeTokenizer

class TreeDataset:
    def __init__(self, max_level=2):
        self.tree_tokenizer = TreeTokenizer()
        self.texts = ["I like big butts. I can not lie.","some other song"] #hard coded for tests
    def iterator(self):
        yield [self.tree_tokenizer.batch_texts_to_trees(item) for item in self.texts]

class Config():
  def __init__(self):
    self.top_level = 2
    self.sequence_lengths = [5,6,7,3,4] #[10,12,6,20,20]
    self.vector_sizes = [8,10,12,14,16,18] #[4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    self.num_heads = [2,2,2,2,2,2]#[2,3,4,5] #for transformers
    self.fnn_sizes = [8,10,12,14,16,18] #[2,3,4,5] #for fnn in transformers
    self.vocab_sizes = [80,20,10,8,8,8]#[1000,21,6,5]
    self.num_transformer_layers = [2,2,2,2,2,2]#[2,2,2,2]
    self.dtype = 'float32'
    self.mlm_rate = 0.1
    self.batch_size = 2 #todo: use it to create the actual dataset, it is also hardcoded there
    self.drop_rate = 0.1
    self.auto_encoder_regularization = 0.0

  class AgentLevel(nn.Module):
    def __init__(self,level,config):
      super().__init__()
      self.level = level
      self.encoder = Encoder(level,config)
      self.encoder_transform = nn.Linear(level,config)  # For the MLM Loss only
      self.decoder = Decoder(level,config)
      self.compressor = Compressor(level,config)
      self.decompressor = Decompressor(level,config)
      self.coherence_checker = CoherenceChecker(parent_embed)
      self.eos_vector = None

      #these functions change the modes
      def realize_vectors(self,node_batch):
        return
      def calc_mlm_loss(self,embedding_matrix,node_batch):
        return
      def calc_coherence_loss(self,embedding_matrix,node_batch):
        return
      def calc_reconstruction_loss(self,embedding_matrix,node_batch):
        return




class AgentModel():
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.levels = [AgentLevel()]

    def realize_batch_tree_vectors(self,batch_tree):
      return
    def realize_batch_tree_losses(self,batch_tree):
      return
    def create_loss_object(self,batch_tree):
      return {}

    def decode(self,level,vec):
      return "bla"
