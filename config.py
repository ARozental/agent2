


#todo: different versions of config local/plumbus/big
# class Config():
#   def __init__(self):
#     self.sequence_lengths = [32,8,16]
#     self.vector_sizes = [32,64,128,256]
#     self.num_heads = [4,4,8,8] #for transformers
#     self.fnn_sizes = [32,64,128,256] #for fnn in transformers
#     self.vocab_sizes = [1000,60,60,60]
#     self.num_transformer_layers = [2,2,2,2]
#     self.dtype = 'float32'
#     self.mlm_rate = 0.15 #0.15 like BERT
#     self.drop_rate = 0.1
#     self.auto_encoder_regularization = 0.01


class Config():
  def __init__(self, sequence_lengths=[12,4,5], vector_sizes=[4,6,8,10],num_heads = [2,3,4,5],fnn_sizes = [2,3,4,5],vocab_sizes = [1000,21,6,5],num_transformer_layers = [2,2,2,2],
               dtype='float32',batch_size=1,mlm_rate = 0.15,drop_rate = 0.1,auto_encoder_regularization = 0.01):
    self.sequence_lengths = [5,6,7,3,4] #[10,12,6,20,20]
    self.vector_sizes = [8,10,12,14,16,18] #[4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    self.num_heads = [2,2,2,2,2,2]#[2,3,4,5] #for transformers
    self.fnn_sizes = [8,10,12,14,16,18] #[2,3,4,5] #for fnn in transformers
    self.vocab_sizes = [80,20,10,8,8,8]#[1000,21,6,5]
    self.num_transformer_layers = [2,2,2,2,2,2]#[2,2,2,2]
    self.dtype = 'float32'
    self.mlm_rate = mlm_rate #0.15 like BERT
    self.batch_size = 2 #todo: use it to create the actual dataset, it is also hardcoded there
    self.drop_rate = drop_rate
    self.auto_encoder_regularization = auto_encoder_regularization

    # self.hparams = evolved_transformer_base()
    # self.hparams.hidden_size = 8
    # self.hparams.num_heads = 1
    # self.hparams.filter_size = 4 * self.hparams.hidden_size
    # self.hparams.num_hidden_layers = 2