# coding=utf-8
"""Text file goes in tf record for AGENT comes out."""

import tensorflow as tf
from absl import flags

import random
from xlnet_prepro_utils import preprocess_text, encode_ids
import sentencepiece as spm
import nltk.data
from config import Config
import re

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,"")
p_split_token = "<newline> <newline>" #"2sep2"


#todo: make a real one
def tokenize(sentence,sp):
  return [random.randint(0,7) for x in sentence.strip().split(" ")]

def tokenize_fn(text,sp):
  text = preprocess_text(text, lower=True)
  return encode_ids(sp, text)

mod = 1000
def doc_line_to_lists(line,sp,ss_tokenizer):
  doc  = []
  for p in line[:-1].split(p_split_token):
    current_paragraph = []
    for s in ss_tokenizer.tokenize(p):
      s=tokenize_fn(s,sp)
      s = [n%mod for n in s] #todo: delete this when training
      if len(s)>0:
        current_paragraph.append(s)
    if len(p)>0:
      doc.append(current_paragraph)
  return doc

class LineGenerator(object):
  def __init__(self):
    self.sp = spm.SentencePieceProcessor()
    self.sp.load("spiece.model")
    self.nltk_ss_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  def get_next_line(self):
    #todo: read several files, make file_object=current_file, rotate
    file_object = open("dummy_data3.txt","r", encoding='utf-8')
    while True:
      data = file_object.readline()
      if not data:
        file_object = open("dummy_data3.txt", "r", encoding='utf-8')
        data = file_object.readline()
      out = doc_line_to_lists(data,self.sp,self.nltk_ss_tokenizer)
      ragged_out =tf.ragged.constant(out)

      yield ({
              "flat_values":ragged_out.flat_values,
              "nested_row_splits": ragged_out.nested_row_splits,
              "teacher1": tf.constant(7.0, dtype=tf.float32)
              })

class Dataset(object):
  def __init__(self):
    self.generator = LineGenerator()
    self.next_element = self.build_iterator(self.generator)

  def build_iterator(self, gen: LineGenerator):

    batch_size = 8
    prefetch_batch_buffer = 24
    dataset = tf.data.Dataset.from_generator(gen.get_next_line,output_types = {
                                                                               "teacher1": tf.float32,
                                                                               "flat_values":tf.int64, "nested_row_splits":(tf.int64,tf.int64)
                                                                               })
    dataset = dataset.map(lambda h: {"teacher1": h["teacher1"],
                                     "tokens": tf.RaggedTensor.from_nested_row_splits(h["flat_values"],(h["nested_row_splits"]))
                                     })

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_batch_buffer)

    #iter = dataset.make_one_shot_iterator()
    iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    element = iter.get_next()

    return element

#--------- char model under development
from collections import defaultdict
import glob

def create_tokenizer(char_file):  # "../chars.txt"
  x = open(char_file, "r", encoding='utf-8').readlines()
  tokenizer = defaultdict(int, dict(zip([l.strip() for l in x], range(1, 7777))))
  def tokenize(word):
    return [tokenizer[l] for l in word]
  return tokenize

def create_reverse_tokenizer(char_file):  # "../chars.txt"
  x = open(char_file, "r", encoding='utf-8').readlines()
  reverse_tokenizer = dict(zip(range(1, 500), [l.strip() for l in x]))
  reverse_tokenizer[0] = ''

  def tokenize(token_list):
    return [reverse_tokenizer[t] for t in token_list]

  return tokenize


delimiters = ['', ' ', '  ', '\n','\n\n']  # letters,words,sentences,paragraphs,chapters (though it is "Chapter " really)
def split_text(text,tokenize_fn, current_level,config):
  ll = config.sequence_lengths[current_level]-1
  if current_level == 0:
    return tokenize_fn(text)[0:ll]
  f = lambda text: text.split(delimiters[current_level])
  return [split_text(t,tokenize_fn, current_level - 1,config) for t in f(text)][0:ll]


def is_empty(a):
  if a == [0]:
    return True
  if type(a) != list:
    return False
  elif len(a) > 1:
    return False
  elif len(a) == 0:
    return True
  elif type(a[0]) != list:
    return False
  else:
    return is_empty(a[0])

def remove_empty(a):
  if type(a) != list:
    return a
  return [remove_empty(x) for x in a if is_empty(x) == False]

def do_book(location,config):
  tokenize_fn = create_tokenizer("chars.txt")
  lines = open(location, "r", encoding='utf-8').readlines()
  #book_title = lines[10].strip()
  text = "".join(lines)
  chapters = re.split('(?i)chapter ',text)
  ll = config.sequence_lengths[4]-1
  chapters = [split_text("Chapter " + x,tokenize_fn, 3,config) for x in chapters][1:ll+1]
  #chapters[0] = split_text(book_title,tokenize_fn, 3,config)
  return remove_empty(chapters)


def max_lengths(book):
  #max length for a book: 2 dataset examples of short books: ([21, 52, 9, 135, 15, 1], [21, 57, 11, 182, 12, 1])
  chapters = book
  l1 = len(chapters)
  l2 = max([len(x) for x in chapters])
  paragraphs = [item for sublist in chapters for item in sublist]
  l3 = max([len(x) for x in paragraphs])
  sentences = [item for sublist in paragraphs for item in sublist]
  l4 = max([len(x) for x in sentences])
  words = [item for sublist in sentences for item in sublist]
  l5 = max([len(x) for x in words])
  return [l5, l4, l3, l2, l1, 1]

def get_depth(nested, res=-1):
  if type(nested) != list:
    return res
  return get_depth(nested[0], res + 1)


def join_text(nested, reverse_tokenizer_fn):
  current_level = get_depth(nested)
  f = lambda text: delimiters[current_level].join(text)
  if current_level == 0:
    res = reverse_tokenizer_fn(nested)
    return f(res)
  return f([join_text(arr, reverse_tokenizer_fn) for arr in nested])


class BookDataset(object):
  def __init__(self,config):
    self.config = config
    self.iter = self.build_iterator()


  def get_next_book(self):
    all_books = glob.glob("datasets/no_titles/*.txt")
    l = len(all_books)
    num = 0
    while True:
      out = do_book(all_books[num % l],self.config)
      ragged_out = tf.ragged.constant(out)
      yield ({
        "flat_values": ragged_out.flat_values,
        "nested_row_splits": ragged_out.nested_row_splits
      })
      num += 1

  def build_iterator(self):
    batch_size = self.config.batch_size
    prefetch_batch_buffer = 2
    dataset = tf.data.Dataset.from_generator(self.get_next_book, output_types={
      "flat_values": tf.int64, "nested_row_splits": (tf.int64, tf.int64, tf.int64, tf.int64)
    })

    dataset = dataset.map(lambda h: {"tokens": tf.RaggedTensor.from_nested_row_splits(h["flat_values"],(h["nested_row_splits"]))})


    #iter = dataset.make_one_shot_iterator()
    iter = tf.compat.v1.data.make_one_shot_iterator(dataset)
    element = iter.get_next()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_batch_buffer)
    it = tf.compat.v1.data.make_one_shot_iterator(dataset)

    return it


#---------

def main(_):
  sp = spm.SentencePieceProcessor()
  sp.load("spiece.model")
  text = "I like big butts"
  ids = encode_ids(sp, text)
  print(ids)
  text2 = sp.DecodeIds([31999])
  print(text2)
  print("========================")

  tokenize = create_tokenizer("chars.txt")
  rt = create_reverse_tokenizer("chars.txt")
  print(tokenize("sh#t"))
  c = Config()
  #print(c.batch_size)
  #print("zz")
  #nested = do_book("datasets/no_titles/Adapt (A Touch of Power Book 2)_trial.txt",c)
  #print(nested)
  #text = join_text(nested, rt)
  #print(text)
  bb = BookDataset(c)
  print("ZZZ")
  print(bb.iter.get_next())
  print(bb.iter.get_next())
  print(bb.iter.get_next())
  print(bb.iter.get_next())
  print(bb.iter.get_next())
  print(bb.iter.get_next())
  print("lll")
  b = [[[[[76, 38, 59, 59, 21], [59, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 21], [12, 37, 37, 37, 37], [53, 53, 53, 53, 78]], [[76, 38, 35, 24, 24], [59, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 21], [12, 37, 37, 37, 37], [53, 53, 53, 53, 78]], [[73, 35, 24, 24, 24], [21, 59, 59, 59, 21], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 6]]], [[[36, 35, 35, 24, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[73, 35, 24, 24, 24], [21, 59, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[24, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 59], [21, 21, 58, 58, 33], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[24, 24, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 7], [21, 21, 58, 79, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[24, 24, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 7], [21, 21, 58, 79, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 35, 24, 24, 24], [21, 21, 59, 21, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 35, 35, 24, 24], [54, 59, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]]]], [[[[76, 38, 59, 59, 24], [59, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 21, 58], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[76, 38, 35, 24, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[73, 35, 24, 24, 24], [21, 59, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[57, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 59, 59], [21, 21, 58, 58, 33], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[57, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 59, 59], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[76, 38, 59, 21, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[76, 38, 59, 21, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]]], [[[36, 38, 35, 24, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[73, 35, 24, 24, 24], [21, 59, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[24, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 59], [21, 21, 58, 58, 33], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[24, 24, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 7], [21, 21, 58, 79, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[26, 24, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 7], [21, 21, 58, 79, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 35, 35, 24, 24], [21, 59, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 35, 35, 24, 24], [54, 21, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]]]], [[[[76, 38, 59, 21, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 35, 35, 24, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[73, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 33], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[57, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 59, 59], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[57, 35, 24, 24, 24], [54, 21, 21, 21, 21], [21, 21, 21, 59, 59], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 38, 59, 24, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 38, 59, 24, 24], [54, 59, 59, 59, 59], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]]], [[[36, 38, 35, 24, 24], [21, 59, 59, 59, 59], [21, 59, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[73, 35, 24, 24, 24], [21, 59, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 14], [12, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[24, 35, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 21], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[47, 24, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 7], [21, 21, 58, 79, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[26, 24, 24, 24, 24], [21, 21, 21, 21, 21], [21, 21, 21, 21, 7], [21, 75, 58, 79, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 35, 35, 24, 24], [21, 21, 59, 21, 21], [21, 21, 59, 59, 59], [21, 21, 21, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]], [[36, 73, 35, 24, 24], [54, 21, 59, 59, 21], [21, 21, 59, 59, 59], [21, 21, 58, 58, 33], [2, 37, 37, 37, 37], [53, 53, 53, 53, 53]]]]]
  dd=get_depth(b)
  print(dd)


  # sp.load("spiece.model")
  # with tf.io.gfile.GFile("dummy_data3.txt", "r") as reader:
  #   all_docs = []
  #   all_lines = []
  #   while True:
  #     line = reader.readline()
  #     if not line:
  #       break
  #     doc_list = doc_line_to_lists(line,sp)
  #     all_docs.append(doc_list)
  #     all_lines += line
    # print("prepare_dataset")
    # ds = Dataset()
    # model_input = ds.next_element

if __name__ == "__main__":
  tf.compat.v1.app.run()

#with tf.compat.v1.Session() as sess:
#  sess.run(tf.constant([1, 2, 3, 4, 5, 6, 7]))