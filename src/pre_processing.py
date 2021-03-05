#this file should have a function that takes text and returns Tree with nodes
import re
from collections import defaultdict
import nltk


class Config(): #move to file, join with the big config class
  def __init__(self):
    self.sequence_lengths = [5,99,9] #[letters_in_word, max_words_in_sentence, max_sentences_in_paragraph]
    self.pad_token_id = -1
    self.eos_token_id = -2
    self.join_token_id = -3
    self.agent_level = 2 #most complex vector agent can create 2=paragraph

config=Config()


class Node():
  def __init__(self,id = None, parent = None, children = None, level = 2,struct = None,
               tokens=None, vector = None,realized = False, type = None, tree_id = None, config = config):
    self.id = id #pre order id
    self.struct = struct
    self.config = config #so expand_struct will know how to pad letters
    self.parent = parent
    self.children = children
    self.level = level
    self.tokens = tokens
    self.vector = vector
    self.realized = realized
    self.type = type #leaf, inner, root, join_token, eos_token
    self.tree_id = tree_id #not sure if we'll need it

  def bebug_get_tree(self,attr="id"):
    if self.children == None:
      return getattr(self, attr)
    else:
      return [x.bebug_get_tree(attr) for x in self.children]

  def join_struct_children(self):
    # for level > 1 => a paragraph(2) can join its sentences
    new_struct = [self.struct[0]]
    max_length = self.config.sequence_lengths[self.level-1]
    for sub in self.struct[1:]:
      if len(new_struct[-1]) + 1 + len(sub) < max_length:
        new_struct[-1].append(config.join_token_id)
        new_struct[-1].extend(sub)
      else:
        new_struct.append(sub)
    self.struct = new_struct
    return new_struct


  def expand_struct(self):  # move set ids to after we are done with joins
    counter = self.id
    def expand_struct1(self):
      # for each level create nodes; give them struct; delete own struct
      # word is the leaf here not letter; it'll make it faster
      nonlocal counter
      counter += 1
      self.id = counter
      if self.struct == self.config.join_token_id:
        self.type = "join_token"
        self.tokens = self.config.join_token_id
      elif self.level==0: #word
        self.tokens = (self.struct + [config.eos_token_id] + [config.pad_token_id] * config.sequence_lengths[0])[
                      0:config.sequence_lengths[0]]
        self.type = "leaf"
      else:
        if self.level >= 2 and self.type!="batch root":
          self.join_struct_children()
        self.children = [expand_struct1(Node(struct=x, parent=self, level=self.level - 1, config=self.config, type="inner")) for x in self.struct]
      # self.struct = None
      return self

    return expand_struct1(self)


class BatchTree():
  def __init__(self,batch_root,config=config):
    self.config = config
    self.level_nodes = {i: [] for i in range(config.agent_level+1)} #{0: [sorted nodes for words], 1: [sorted nodes for sentences]}
    self.batch_root = batch_root

  def __batch_up_nodes1(self,node):
    self.level_nodes[node.level].append(node)
    if node.children != None:
      [self.__batch_up_nodes1(c) for c in node.children]

  def batch_up_nodes(self):
    [self.__batch_up_nodes1(c) for c in self.batch_root.children]

class TreeTokenizer:
  def __init__(self,char_file = "../chars.txt",config=config):
    self.letter_tokenizer = defaultdict(int, dict(zip([l.strip() for l in open(char_file, "r", encoding='utf-8').readlines()], range(1, 7777))))
    self.sentence_spliter = nltk.data.load('tokenizers/punkt/english.pickle')
    self.split_functions = [self.paragraph_to_sentences, self.sentence_to_words]
    self.max_depth = len(self.split_functions)

  def tokenize_word(self,word):
    #"sheeבt" => [68, 57, 54, 54, 0, 69]
    return [self.letter_tokenizer[l] for l in word]

  def sentence_to_words(self,sentence):
    #"I like big butts." => ['I', 'like', 'big', 'butts.']
    return re.split(' ',sentence)

  def paragraph_to_sentences(self,p):
    #"I like big butts. I can not lie." => ['I like big butts.', 'I can not lie.']
    return self.sentence_spliter.tokenize(p)


  def text_to_tree_struct(self,text,level=2):
    #"I like big butts. I can not lie." => [[[32], [61, 58, 60, 54], [51, 58, 56], [51, 70, 69, 69, 68, 10]], [[32], [52, 50, 63], [63, 64, 69], [61, 58, 54, 10]]]
    if level>0:
      return [self.text_to_tree_struct(x,level-1) for x in self.split_functions[self.max_depth-level](text) if len(x)>0]
    else:
      return self.tokenize_word(text)

  def batch_texts_to_trees(self,texts,config=config):
    #input: ["I like big butts. I can not lie.","You other brothers can't deny"]
    structs = [self.text_to_tree_struct(text) for text in texts]
    batch_root = Node(struct=structs,type="batch root", id=0, level=config.agent_level+1)
    batch_root.expand_struct()
    batch_tree = BatchTree(batch_root)
    batch_tree.batch_up_nodes()
    return batch_tree


tt = TreeTokenizer()
# x = tt.tokenize_word("sheeבt")
# x = tt.text_to_tree_struct("I like big   butts. I can not lie.")
#x = tt.batch_texts_to_trees(["I like big butts. I can not lie.","some other song"] )
#x = tt.batch_texts_to_trees(["I am big. you are too.","I am big. you are too."] )
#print([[k,len(v)] for (k,v) in x.level_nodes.items()])
#print(x.struct)
# print(x.children[0].children[0].children[0].tokens)
# print(x.bebug_get_tree("tokens"))
node = Node(struct=tt.text_to_tree_struct("I like big butts. I can not lie."),id=0,level=2,type="debug root") #level 0 is word node
node.expand_struct()
# #print(node.children[-1].children[-1].tokens) #see that tokens are correct :)
# print("struct",node.struct)
# print("word ids",node.bebug_get_tree(attr="id"))
print("tokens",node.bebug_get_tree(attr="tokens"))
# #print(node.bebug_get_tree())
# print({i:3 for i in range(5)})