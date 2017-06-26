#!/usr/bin/env python3

import sys, os, time
import pandas as pd
import numpy as np
from collections import Counter, Iterable
from nltk.tokenize import casual_tokenize
from IPython import embed
import textwrap, pickle
from pprint import pprint

def load_reader(path):
  """Loads a reader from a previously saved run"""
  if path[-4:] != '.pkl':
    path+='.pkl'
  with open(path,"r+b") as f:
    log("Loading reader from {}".format(path))
    r = pickle.load(f)
  return r

def log(*message, f=sys.stdout):
  message = " ".join(map(str,message))
  time_message = "["+time.strftime("%Y/%m/%d %H:%M:%S")+"] "
  l = len(time_message)
  t = textwrap.TextWrapper(width = 100, subsequent_indent=" "*l)
  print(t.fill(time_message+str(message)), file=f)

class Reader():
  cols = ["sentence1", "sentence2", "sentence3", "sentence4"]
  train_cols = cols+["sentence5"]
  test_val_cols = cols+["sentence5_1", "sentence5_2"]
  
  def __init__(self,data_dir,vocab_cutoff):
    train_path = os.path.join(data_dir,"train.csv")
    test_path = os.path.join(data_dir,"test.csv")
    val_path = os.path.join(data_dir,"val.csv")
    
    self.train = pd.read_csv(train_path,index_col=0)
    self.val = pd.read_csv(val_path,index_col=0)
    self.test = pd.read_csv(test_path,index_col=0)
    self.vocab_cutoff = vocab_cutoff
    self.context_size = len(self.cols)

  def __str__(self):
    a = ["=".join([str(k), str(v)]) for k,v in self.__dict__.items()
         if k[0] != '_' and (not isinstance(v,Iterable)
                             or isinstance(v,str)
         )]
    return self.__class__.__name__+"("+', '.join(a)+")"

  def save(self,fn):
    """Saves the reader to a pickle file"""
    fn = fn if fn[-4:] == ".pkl" else fn+".pkl"
    with open(fn,"wb+") as f:
      pickle.dump(self,f)
      log("Saved reader to {}".format(fn))

  def oov(self):
    return self._tok_to_id["<OOV>"]

  def pad(self):
    return self._tok_to_id["<PAD>"]
  
  def bom(self):
    return self._tok_to_id["<BOM>"]

  def eom(self):
    return self._tok_to_id["<EOM>"]
  
  def tok_to_id(self,tok):
    return self._tok_to_id.get(tok,self.oov())

  def id_to_tok(self,i):
    return self._id_to_tok.get(i,"<OOV>")
  
  def tokenize(self,string,lowercase=True):
    if lowercase:
      string = string.lower()
    return casual_tokenize(string)

  def str_to_ids(self,string):
    return [self.tok_to_id(t) for t in self.tokenize(string)]

  def docs_to_ids(self, docs):
    return [self.str_to_ids(d) for d in docs]
  
  def ids_to_toks(self,ids,remove_pads=True):
    """Converts a list of ids to a list of tokens,
    and optionally removes padding and EOM symbols """
    ids = list(ids)
    index = len(ids)
    if remove_pads:
      if self.bom() == ids[0]:
        ids = ids[1:]
      if self.eom() in ids:
        index = ids.index(self.eom())
      if self.pad() in ids and ids.index(self.pad()) < index:
        index = ids.index(self.pad())
        
    return [self._id_to_tok[i] for i in ids[:index]]

  def ids_to_str(self,ids):
    return " ".join(self.ids_to_toks(ids))
  
  def seqs_to_toks(self,id_array,remove_pads=True):
    """Converts a 2D array of ids to a list of list of tokens"""
    return [self.ids_to_toks(seq,remove_pads)
            for seq in id_array]

  def seqs_to_strs(self,id_array,remove_pads=True):
    """Converts a 2D array of ids to a list of strings"""
    l = self.seqs_to_toks(id_array,remove_pads)
    return [' '.join(i) for i in l]

  def make_vocab(self,extra_toks=["<BOM>","<EOM>","<PAD>","<OOV>"]):
    words = [w for c in self.train_cols for s in self.train[c]
             for w in self.tokenize(s)]
    
    c = Counter(words)
    log("Number of distinct words: {}".format(len(c)))
    cutoff = [k for k,v in c.items() if v >= self.vocab_cutoff]
    for t in extra_toks:
      cutoff.insert(0,t)
    
    self._tok_to_id = dict(zip(cutoff,range(len(cutoff))))
    self._id_to_tok = {v:k for k,v in self._tok_to_id.items()}
    self.vocab_size = len(self._tok_to_id)
    log("Vocab size after cutoff of {}: {}".format(self.vocab_cutoff,self.vocab_size))
    
  def tokenize_docs(self):
    """Tokenizes sentences, integerizes them. For the context sentences,
    they will be all put into one column of the dataframe.
    """
    if not hasattr(self,"max_seq_len"):
      self.max_seq_len = 0
    
    context_cols = self.cols

    for df in [self.train, self.val, self.test]:
      df["tok_context"] = df[context_cols].values.tolist()
      df["tok_context"] = df["tok_context"].apply(self.docs_to_ids)
      self.max_seq_len = max(self.max_seq_len,
                              df["tok_context"].apply(
                                lambda x: max(map(len,x))).max())
    
    self.train["tok_ending"] = self.train[self.train_cols[-1]].apply(
      self.str_to_ids)
    self.val["tok_ending_1"] = self.val[self.test_val_cols[-2]].apply(
      self.str_to_ids)
    self.val["tok_ending_2"] = self.val[self.test_val_cols[-1]].apply(
      self.str_to_ids)
    self.test["tok_ending_1"] = self.test[self.test_val_cols[-2]].apply(
      self.str_to_ids)
    self.test["tok_ending_2"] = self.test[self.test_val_cols[-1]].apply(
      self.str_to_ids)
    self.max_seq_len = max(
      self.max_seq_len,
      self.train["tok_ending"].apply(len).max(),
      self.val["tok_ending_1"].apply(len).max(),
      self.val["tok_ending_2"].apply(len).max(),
      self.test["tok_ending_1"].apply(len).max(),
      self.test["tok_ending_2"].apply(len).max(),
    )
    # Max_seq_len is without BOM, EOM
    self.max_seq_len += 2
    log("Maximum sequence length: {}".format(self.max_seq_len))

  def pad_sequence(self, s, max_len):
      return s+[self.pad()]*(max_len-len(s))

  def split_train_val(self,ratio=.1):
    """Splits training data into training and validation.
    This is necessary cause the training data was made through
    a different process than the validation / test data, so 
    testing for convergence isn't trivial.
    """
    lim = int(np.ceil(len(self.train) * ratio))
    order = list(range(len(self.train)))
    np.random.shuffle(order)
    self.train_train = self.train.ix[order[lim:]]
    self.train_val = self.train.ix[order[:lim]]
    log("Split data into training/val: {} -> {} {}".format(
      len(self.train),len(self.train_train),lim))
    
  def LMBatchYielder(self,batch_size,d="train_train"):
    """Constructs batches where context is merged into 1 sentence."""
    data = None
    if d=="train_train":
      ending_cols = ["tok_ending"]
      data = self.train_train[["tok_context"]+ending_cols].copy()
    elif d=="train_val":
      ending_cols = ["tok_ending"]
      data = self.train_val[["tok_context"]+ending_cols].copy()
    elif d=="val":
      ending_cols = ["tok_ending_1","tok_ending_2"]
      data = self.val[["rightending","tok_context"]+ending_cols].copy()
    elif d=="test":
      ending_cols = ["tok_ending_1","tok_ending_2"]
      data = self.test[["rightending","tok_context"]+ending_cols].copy()
    
    n_yields = int(np.ceil(len(data)/batch_size))
    log("Yielding {} '{}' batches".format(n_yields,d))

    cont_list = [[[self.bom()]+s+[self.eom()] for s in cs]
                 for cs in data.tok_context]
    data["tok_context_cont"] = [[t for s in cs for t in s]
                   for cs in cont_list]

    if not hasattr(self,"max_targ_len"):
      self.max_targ_len = 0
      
    for c in ending_cols:
      data["tok_"+c] = [[self.bom()]+s+[self.eom()]
                             for s in data[c]]
      max_targ_len = max(data["tok_"+c].apply(len))
      self.max_targ_len = max(self.max_targ_len,max_targ_len)

    if not hasattr(self,"max_cont_len"):
      self.max_cont_len = 0
    max_cont_len = max(data["tok_context_cont"].apply(len))
    self.max_cont_len = max(max_cont_len,self.max_cont_len)
    
    # print(self.max_cont_len,self.max_targ_len)
    
    for i in range(n_yields):
      chunk = data.iloc[i*batch_size:(i+1)*batch_size]
      b_size = len(chunk)

      cont_len = np.array([len(c) for c in chunk["tok_context_cont"]])
      assert cont_len.shape == (b_size,)
      
      cont_padded = [self.pad_sequence(c,self.max_cont_len)
                     for c in chunk["tok_context_cont"]]
      context = np.array(cont_padded)
      assert context.shape == (b_size,self.max_cont_len)
      
      if len(ending_cols) == 1: # train
        targ_len = np.array([len(s) for s in chunk["tok_"+ending_cols[0]]])
        targ_padded = [self.pad_sequence(s,self.max_targ_len)
                       for s in chunk["tok_"+ending_cols[0]]]
        target = np.array(targ_padded)
        # if target.shape != (b_size,self.max_targ_len): embed()
        assert target.shape == (b_size,self.max_targ_len)
        
        b = Batch(
          context_seq=context,
          context_len=cont_len,
          target_seq=target,
          target_len=targ_len,
          ids=chunk.index,
          size=b_size)
        
      elif len(ending_cols) == 2:
        targ1_len = np.array([len(s) for s in chunk["tok_"+ending_cols[0]]])
        targ1_padded = [self.pad_sequence(s,self.max_targ_len)
                       for s in chunk["tok_"+ending_cols[0]]]
        target1 = np.array(targ1_padded)
        # if target.shape != (b_size,self.max_targ_len): embed()
        assert target1.shape == (b_size,self.max_targ_len)

        targ2_len = np.array([len(s) for s in chunk["tok_"+ending_cols[1]]])
        targ2_padded = [self.pad_sequence(s,self.max_targ_len)
                        for s in chunk["tok_"+ending_cols[1]]]
        target2 = np.array(targ2_padded)
        # if target.shape != (b_size,self.max_targ_len): embed()
        assert target2.shape == (b_size,self.max_targ_len)

        # Right ending
        right_ending = chunk["rightending"]
        right_target = np.array(
          [t1 if i == 1 else t2
           for i,t1,t2 in zip(right_ending,target1,target2)])
        right_target_len = np.array(
          [l1 if i == 1 else l2
           for i,l1,l2 in zip(right_ending,targ1_len,targ2_len)])
        wrong_target = np.array(
          [t1 if i == 2 else t2
           for i,t1,t2 in zip(right_ending,target1,target2)])
        wrong_target_len = np.array(
          [l1 if i == 2 else l2
           for i,l1,l2 in zip(right_ending,targ1_len,targ2_len)])
        
        b = Batch(
          context_seq=context,
          context_len=cont_len,
          right_target_seq=right_target,
          right_target_len=right_target_len,
          wrong_target_seq=wrong_target,
          wrong_target_len=wrong_target_len,
          target1_seq=target1,
          target1_len=targ1_len,
          target2_seq=target2,
          target2_len=targ2_len,
          size=b_size,
          rightending=right_ending.as_matrix(),
          ids=right_ending.index
        )
        
      yield b


class CharROCReader(Reader):
  def tokenize(self, string, lowercase=True):
    if lowercase:
      string = string.lower()
    return list(string)
      
class Batch():
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

