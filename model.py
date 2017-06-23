#!/usr/bin/env python3

import os, time
import tensorflow as tf
from reader import *
from IPython import embed
from nltk import bleu
from sklearn.metrics import \
  accuracy_score, precision_recall_fscore_support
from scipy.stats import entropy,\
  pearsonr,ttest_ind,ttest_rel
import statsmodels.api as sm

###############################################
######## Saving and loading functions #########
###############################################

def save_model(fn, model, ckpt=None):
  """Saves the TensorFlow variables to file"""
  # if fn[-3] != ".tf":
  #   fn += ".tf"
  if model.saver is None:
    with model.graph.as_default():
      model.saver = tf.train.Saver()
  if ckpt is None:
    ckpt = fn+".ckpt"
  ckpt = os.path.basename(ckpt)
  log("Saving model to {}".format(fn))
  model.saver.save(model.session, fn, latest_filename=ckpt)

def load_model(fn, model):
  """Loads the TensorFlow variables into the model (has to be constructed)"""
  # if fn[-3] != ".tf":
  #   fn += ".tf"
  if model.saver is None:
    with model.graph.as_default():
      model.saver = tf.train.Saver()
  log("Loading model from {}".format(fn))
  model.saver.restore(model.session, fn)
  log("Done loading!")
  
###############################################

PRINT_FREQ = 100

def LSTMTupleSlice(tup,i):
  c = np.array([tup.c[i]])
  h = np.array([tup.h[i]])
  return tf.contrib.rnn.LSTMStateTuple(c=c,h=h)

def LSTMTupleSplit(tup,n):
  cs = np.split(tup.c,n)
  hs = np.split(tup.h,n)
  return [tf.contrib.rnn.LSTMStateTuple(c=c,h=h)
          for c,h in zip(cs,hs)]

def LSTMTupleConcat(tups):
  cs = [t.c for t in tups]
  hs = [t.h for t in tups]
  return tf.contrib.rnn.LSTMStateTuple(c=np.concatenate(cs),h=np.concatenate(hs))

class BaseModel():
  def list_variables(self,scope=None,not_optimizer=True):
    opt_str = "AdamOptimizer"
    scope = type(self).__name__ if scope is None else scope
    return [v.name for v in self.graph.get_collection(
      tf.GraphKeys.VARIABLES, scope=scope) if opt_str not in v.name]
  
  def dict_variables(self,scope=None,not_optimizer=True):
    opt_str = "AdamOptimizer"
    scope = type(self).__name__ if scope is None else scope
    return {v.name:v for v in self.graph.get_collection(
      tf.GraphKeys.VARIABLES, scope=scope) if opt_str not in v.name}


  def rnn_with_embedding(self,cell,init_state,input_seq,
                             input_seq_len,is_training,
                             reuse=None,scope="RNN"):
    """Given a sequence, embeds the symbols and runs it through a RNN.
    Returns:
      the unembedded outputs & final state at the right time step.

    Note: unembedded outputs are length-1 compared to input_seq_len!
    """    
    with tf.variable_scope(scope,reuse=reuse) as vs:
      log(vs.name+"/Encoding sequences")
      with tf.device('/cpu:0'):
        emb_range = 0.05
        emb = tf.get_variable("emb",
                              [self.vocab_size,self.hidden_size],
                              #initializer=tf.random_uniform_initializer(
                              #  -emb_range,emb_range),
                              dtype=tf.float32)
        un_emb = tf.get_variable("unemb",
                                 [self.hidden_size,self.vocab_size],
                                 tf.float32)
        # We need a bias
        un_emb_b = tf.get_variable("unemb_b",
                                   [self.vocab_size],
                                   # initializer=tf.constant_initializer(1.0),
                                   dtype=tf.float32)
        
        assert scope+"/emb:0" in emb.name,\
          "Making sure the reusing is working"
        emb_input_seq = tf.nn.embedding_lookup(
          emb,input_seq)
        emb_input_list = tf.unstack(
          tf.transpose(emb_input_seq,[1,0,2]))
        
      # RNN pass
      if init_state is None:
        init_state = cell.zero_state(
          tf.shape(emb_input_list[0])[0],tf.float32)
      
      emb_output_list, final_state = tf.contrib.rnn.static_rnn(
        cell,emb_input_list,initial_state=init_state,
        sequence_length=input_seq_len)

      # We shift the predicted outputs, because at
      # each word we're trying to predict the next.
      emb_output_list = emb_output_list[:-1]
      
      # Unembedding
      output_list = [tf.matmul(t,un_emb) + un_emb_b
                     for t in emb_output_list]
      outputs = tf.transpose(tf.stack(output_list),[1,0,2])

    return outputs, final_state
  
  def birnn_with_embedding(self,fwd_cell,bwd_cell,fwd_init_state,
                             bwd_init_state,input_seq,input_seq_len,
                             reuse=None,scope="RNN"):
    """Given a sequence, embeds the symbols and runs it through a biRNN.
    Returns:
      the unembedded outputs & final state at the right time step.

    Note: unembedded outputs are length-1 compared to input_seq_len!
    """    
    with tf.variable_scope(scope,reuse=reuse) as vs:
      log(vs.name+"/Encoding sequences")
      with tf.device('/cpu:0'):
        emb_range = 0.05
        emb = tf.get_variable("emb",
                              [self.vocab_size,self.hidden_size],
                              #initializer=tf.random_uniform_initializer(
                              #  -emb_range,emb_range),
                              dtype=tf.float32)
        un_emb = tf.get_variable("unemb",
                                 [self.hidden_size*2,self.vocab_size],
                                 tf.float32)
        # We need a bias
        un_emb_b = tf.get_variable("unemb_b",
                                   [self.vocab_size],
                                   # initializer=tf.constant_initializer(1.0),
                                   dtype=tf.float32)
        
        assert scope+"/emb:0" in emb.name,\
          "Making sure the reusing is working"
        emb_input_seq = tf.nn.embedding_lookup(
          emb,input_seq)
        emb_input_list = tf.unstack(
          tf.transpose(emb_input_seq,[1,0,2]))
        
      # RNN pass
      if fwd_init_state is None:
        fwd_init_state = fwd_cell.zero_state(
          tf.shape(emb_input_list[0])[0],tf.float32)
      if bwd_init_state is None:
        bwd_init_state = bwd_cell.zero_state(
          tf.shape(emb_input_list[0])[0],tf.float32)

      emb_output_list,fwd_final_state,bwd_final_state = \
        tf.contrib.rnn.static_bidirectional_rnn(
          fwd_cell,bwd_cell,emb_input_list,fwd_init_state,
          bwd_init_state,sequence_length=input_seq_len)
      
      # We shift the predicted outputs, because at
      # each word we're trying to predict the next.
      emb_output_list = emb_output_list[:-1]
      
      # Unembedding
      output_list = [tf.matmul(t,un_emb) + un_emb_b
                     for t in emb_output_list]
      outputs = tf.transpose(tf.stack(output_list),[1,0,2])

    return outputs, fwd_final_state, bwd_final_state
 
  def sequence_prob(
      self,logits,target,seq_len,
      max_seq_len,reduce_mean=True):
    """Computes the probability of a sequence."""
    
    softmax_list=  [tf.log(tf.nn.softmax(t)) for t in logits]
    i = target[0]
    lsp = tf.expand_dims(tf.cast(tf.linspace(
      0.,tf.cast(tf.shape(i)[0]-1,tf.float32),
      tf.shape(i)[0]),tf.int64),1)

    probs = []
    for sm, i in zip(softmax_list,target):
      g = tf.concat(axis=1,values=[lsp,tf.expand_dims(i,1)])
      p = tf.gather_nd(sm,g)
      probs.append(p)
    
    ones = tf.ones_like(seq_len)
    ones_float = tf.ones_like(seq_len,dtype=tf.float32)
    zeros = ones_float*0
    probs_list = [
      tf.where(
        tf.less_equal(
          ones*i,seq_len-1),
        p,zeros)
      for i,p in enumerate(probs)]
    
    return tf.reduce_sum(tf.stack(probs_list),0)
    
  def softmax_xent_loss_sequence(
      self,logits,target,seq_len,
      max_seq_len,reduce_mean=True):
    """Given a target sentence (and length) and
    un-normalized probabilities (logits) for predicted
    sentence, computes the cross entropy, excluding the
    padded symbols.
    Note: All arguments must be lists.
    """
    # assert logits[0].get_shape()[1] == self.vocab_size, logits[0].get_shape()
    
    # Loss weights; don't wanna penalize PADs!
    ones = tf.ones_like(seq_len)
    ones_float = tf.ones_like(seq_len,dtype=tf.float32)
    zeros = ones_float*0
    weight_list = [
      tf.where(
        tf.less_equal(
          ones*i,seq_len-1),
        ones_float,zeros)
      for i in range(max_seq_len-1)]
    self._weights = tf.transpose(tf.stack(weight_list))

    # Loss function
    xent = tf.contrib.seq2seq.sequence_loss(
      logits,target,self._weights,average_across_batch=reduce_mean)

    return xent

  def sequence_pred_report(self,target_seqs,pred_seqs,bleu_weights=None,return_list=False):
    """Computes evaluation metrics for the predicted sequence."""
    if bleu_weights is None:
      BLEU_WEIGHTS = [1]*3
    else:
      BLEU_WEIGHTS = bleu_weights
      
    # turning into words cause it's easier to interpret,
    # but also removes padding
    target_list = self.reader.seqs_to_toks(target_seqs)
    pred_list = self.reader.seqs_to_toks(pred_seqs)
    
    bleu_scores = []
    for y,y_ in zip(target_list,pred_list):
      try:
        score = bleu([y],y_,BLEU_WEIGHTS)
      except ZeroDivisionError:
        # One (or both) of the sentences isn't long enough...
        # we could back off to smaller BLEU, but w/e
        if len(y) == len(y_) and y == y_:
          score = 1
        else:
          score = 0
      bleu_scores.append(score)

    # TODO: get METEOR scores too?
    if return_list:
      return bleu_scores
    else:
      return np.mean(bleu_scores)

  def train_epoch(self, batch_iterator, cost_only=False, verbose=True, global_step=0):
    """Returns average cost for a unit (not a batch)"""
    total_xent_right = 0.
    count = 0
    pred_seqs, target_seqs = [],[]
    target_lens = []
    total_enc_state = []
    total_ids = []
    
    ## Changing learning rate!
    # if global_step > 0 and global_step % 15 == 0:
    #   self.halve_lr()
    
    for step, b in enumerate(batch_iterator):
      count += b.size
      total_ids = total_ids.append(b.ids) if len(total_ids) else b.ids
      
      xent_right,pred_seq,enc_state = self.train_batch(b, cost_only)
      
      total_enc_state = np.concatenate([total_enc_state,enc_state]) \
                        if len(total_enc_state) else enc_state

      
      target_lens.append(b.target_len-1) # cause BOM isnt in pred_seq
      
      if not pred_seq is None:
        target_seqs.append(b.target_seq[:,1:]) # cause BOM isnt in pred_seq
        pred_seqs.append(pred_seq)
        
      total_xent_right += xent_right * b.size
      
      if (1+step) % PRINT_FREQ == 0 and verbose:
        m  = "   Step {:3d}, cost: {:.4f}, avg cost: {:.4f}".format(
          step+1, xent_right, total_xent_right/count)
        log(m)
        
    if (1+step) > PRINT_FREQ and (1+step) % PRINT_FREQ != 0 and verbose:
      m  = "   Step {:3d}, cost: {:.4f}, avg cost: {:.4f}".format(
        step+1, xent_right, total_xent_right/count)
      log(m)

    target_lens = np.concatenate(target_lens)
      
    if len(target_seqs) > 0:
      target_seqs = np.concatenate(target_seqs)
      pred_seqs = np.concatenate(pred_seqs)
      avg_BLEU = self.sequence_pred_report(target_seqs,pred_seqs)

      print("Avg BLEU:",avg_BLEU)

    if cost_only:
      pass
      # For exporting?
      # df = pd.DataFrame(total_enc_state,index=total_ids)
      # embed()
      
    if cost_only and hasattr(self,"summaryWriter"):
      feed_dict = {
        self._xent_right_ending: total_xent_right/count,
      }
      if len(target_seqs) > 0:
        feed_dict[self._bleu] = avg_BLEU
        
      summ = self.session.run(
        self.merged_summaries,
        feed_dict)
      self.summaryWriter.add_summary(summ,global_step)
    
    return (total_xent_right)/(count)

  def test_epoch(self,batch_iterator,reverse_prob=False,global_step=0):
    count = 0
    total_xent = []
    total_pe = []
    total_pc = []
    total_pe_given_c = []
    targ_endings = []
    pred_endings = []
    total_grad = []
    total_grad_r = []
    total_grad_w = []
    total_enc_state = []
    total_ids = []
    
    for step, b in enumerate(batch_iterator):
      count += b.size
      total_ids = total_ids.append(b.ids) if len(total_ids) else b.ids
      
      # xent_r,xent_w,p_r,p_w,pred_ending,grad_r,grad_w = \ 
      # xent_r,xent_w,p_r,p_w,pred_ending,grad = \
      xent,pe,pc,pe_given_c,pred_ending,grad,enc_state = \
        self.test_batch(b,reverse_prob=reverse_prob)
      
      total_enc_state = LSTMTupleConcat([total_enc_state,enc_state]) \
                        if len(total_enc_state) else enc_state
      
      total_xent = np.concatenate([total_xent,xent]) if len(total_xent) else xent
      # total_xent_wrong = np.concatenate([total_xent_wrong,xent_w])
      
      if not pe is None:
        total_pe = np.concatenate([total_pe,pe]) if len(total_pe) else pe
      if not pc is None:
        total_pc = np.concatenate([total_pc,pc]) if len(total_pc) else pc
      
      total_pe_given_c = np.concatenate([total_pe_given_c,pe_given_c]) \
                         if len(total_pe_given_c) else pe_given_c
      
      if not grad is None:
        total_grad = np.concatenate([total_grad, grad]) if len(total_grad) else grad
      # total_grad_r = np.concatenate([total_grad_r,grad_r]) if len(total_grad_r)>0 else grad_r
      # total_grad_w = np.concatenate([total_grad_w,grad_w]) if len(total_grad_w)>0 else grad_w

      targ_endings.extend(b.rightending.tolist())
      pred_endings.extend(pred_ending.tolist())

    targ_endings = np.array(targ_endings)
    pred_endings = np.array(pred_endings)

    ### Computing pc_given_e
    if len(total_pe):
      total_pc_given_e = total_pe_given_c - total_pe
    
    ### Selecting the right xents and probs
    xent_pred_r = np.array([ # ŷ = 1
      x[e-1] for x,e in zip(total_xent,pred_endings)])
    xent_pred_w = np.array([ # ŷ = 0
      x[1-(e-1)] for x,e in zip(total_xent,pred_endings)])
    xent_true_r = np.array([ # y = 1
      x[e-1] for x,e in zip(total_xent,targ_endings)])
    xent_true_w = np.array([ # y = 0
      x[1-(e-1)] for x,e in zip(total_xent,targ_endings)])
    
    probs = {
      "pe_given_c_pred_r": np.array([ # ŷ = 1
        x[e-1] for x,e in zip(total_pe_given_c,pred_endings)]),
      "pe_given_c_pred_w": np.array([ # ŷ = 0
        x[1-(e-1)] for x,e in zip(total_pe_given_c,pred_endings)]),
      "pe_given_c_true_r": np.array([ # y = 1
        x[e-1] for x,e in zip(total_pe_given_c,targ_endings)]),
      "pe_given_c_true_w": np.array([ # y = 0
        x[1-(e-1)] for x,e in zip(total_pe_given_c,targ_endings)])      
    }

    score_pred_r = probs["pe_given_c_pred_r"]
    score_pred_w = probs["pe_given_c_pred_w"]
    score_true_r = probs["pe_given_c_true_r"]
    score_true_w = probs["pe_given_c_true_w"]
    
    if len(total_pc):
      probs.update({"pc": total_pc})
      
    if len(total_pe):
      probs.update({
        "pe_pred_r": np.array([ # ŷ":  1
          x[e-1] for x,e in zip(total_pe,pred_endings)]),
        "pe_pred_w": np.array([ # ŷ":  0
          x[1-(e-1)] for x,e in zip(total_pe,pred_endings)]),
        "pe_true_r": np.array([ # y":  1
          x[e-1] for x,e in zip(total_pe,targ_endings)]),
        "pe_true_w": np.array([ # y = 0
          x[1-(e-1)] for x,e in zip(total_pe,targ_endings)])
      })
      ## Scoring function
      probs.update({
        "pc_given_e_pred_r": np.array([ # ŷ":  1
          x[e-1] for x,e in zip(total_pc_given_e,pred_endings)]),
        "pc_given_e_pred_w": np.array([ # ŷ":  0
          x[1-(e-1)] for x,e in zip(total_pc_given_e,pred_endings)]),
        "pc_given_e_true_r": np.array([ # y":  1
          x[e-1] for x,e in zip(total_pc_given_e,targ_endings)]),
        "pc_given_e_true_w": np.array([ # y":  0
          x[1-(e-1)] for x,e in zip(total_pc_given_e,targ_endings)])
      })
      
      score_pred_r = probs["pc_given_e_pred_r"]
      score_pred_w = probs["pc_given_e_pred_w"]
      score_true_r = probs["pc_given_e_true_r"]
      score_true_w = probs["pc_given_e_true_w"]
      
    probs = pd.DataFrame(probs,index=total_ids)
    
    ### Classification reports
    support = targ_endings.mean()
    acc = accuracy_score(targ_endings,pred_endings)
    prec,rec,f1,_ = precision_recall_fscore_support(
      targ_endings,pred_endings,average="binary")
    res = pd.Series(
      [acc,prec,rec,f1,support],
      index=["Accuracy","Precision","Recall","F1","Support"])
    
    xent_pred_r_mean = xent_pred_r.mean()
    xent_pred_w_mean = xent_pred_w.mean()
    xent_true_r_mean = xent_true_r.mean()
    xent_true_w_mean = xent_true_w.mean()

    mean_log_p_r = score_true_r.mean()
    mean_log_p_w = score_true_w.mean()

    # Averaged across hidden dims
    ### Return gradients based on the index (1 or 2) and
    ### split here by true right or pred right
    if len(total_grad):
      grad_pred_r = np.array([ # ŷ = 1
        x[e-1] for x,e in zip(total_grad,pred_endings)])
      grad_pred_w = np.array([ # ŷ = 0
        x[1-(e-1)] for x,e in zip(total_grad,pred_endings)])

      grad_true_r = np.array([ # y = 1
        x[e-1] for x,e in zip(total_grad,targ_endings)])
      grad_true_w = np.array([ # y = 0
        x[1-(e-1)] for x,e in zip(total_grad,targ_endings)])

      grad_true_r_mean = grad_true_r.mean()
      grad_true_w_mean = grad_true_w.mean()
      grad_pred_r_mean = grad_pred_r.mean()
      grad_pred_w_mean = grad_pred_w.mean()
    
      # pred_endings -=1
      # targ_endings -=1
      
      log("ŷ=1 v.s. ŷ=0 (paired t-test)",
          ttest_rel(grad_pred_r.mean(axis=1),grad_pred_w.mean(axis=1)))
      log("ŷ=1 v.s. y=1 (paired t-test)",
          ttest_rel(grad_pred_r.mean(axis=1),grad_true_r.mean(axis=1)))
      log("y=1 v.s. y=0 (paired t-test)",
          ttest_rel(grad_true_r.mean(axis=1),grad_true_w.mean(axis=1)))
      log("ŷ=0 v.s. y=0 (paired t-test)",
          ttest_rel(grad_pred_w.mean(axis=1),grad_true_w.mean(axis=1)))
      
    if len(total_pe):
      self.statTests(probs)

      ## For Exporting:
      df_r = probs[["pe_true_r","pe_given_c_true_r"]].copy()
      df_r.columns = ["pe","pe_given_c"]
      df_r["label"] = 1
      df_w = probs[["pe_true_w","pe_given_c_true_w"]].copy()
      df_w.columns = ["pe","pe_given_c"]
      df_w["label"] = 0
      df = pd.concat([df_r,df_w])
      # embed()
    
    if self.summ:
      feed_dict = {
        self._xent_right   : xent_true_r_mean,
        self._xent_wrong   : xent_true_w_mean,
        self._xent_diff    : (xent_true_w_mean-xent_true_r_mean),
        self._log_p_right  : mean_log_p_r,
        self._log_p_wrong  : mean_log_p_w,
        self._log_p_diff   : (mean_log_p_w-mean_log_p_r), 
        self._acc          : acc,
        self._prec         : prec,
        self._rec          : rec,
        self._f1           : f1,
        self._grad_true_r_mean  : grad_true_r_mean,
        self._grad_true_w_mean  : grad_true_w_mean,
        self._grad_pred_r_mean  : grad_pred_r_mean,
        self._grad_pred_w_mean  : grad_pred_w_mean, 
        self._grad_true_r   : grad_true_r,
        self._grad_true_w   : grad_true_w,
        self._grad_pred_r   : grad_pred_r,
        self._grad_pred_w   : grad_pred_w,
      }
      summ = self.session.run(
        self._merged_summaries,feed_dict)
      self.summaryWriter.add_summary(summ,global_step)
    else:
      feed_dict = {
        "xent_right" : xent_true_r_mean,
        "xent_wrong" : xent_true_w_mean,
        "xent_diff"  : (xent_true_w_mean-xent_true_r_mean),
        "log_p_right": mean_log_p_r,
        "log_p_wrong": mean_log_p_w,
        "log_p_diff" : (mean_log_p_w-mean_log_p_r),
        "acc"        : acc,
        "prec"       : prec,
        "rec"        : rec,
        "f1"         : f1,
        "grad_true_r_mean"   : grad_true_r_mean,
        "grad_true_w_mean"   : grad_true_w_mean,
        "grad_pred_r_mean"   : grad_pred_r_mean,
        "grad_pred_w_mean"   : grad_pred_w_mean,
      }
      pprint(feed_dict)


    return xent_true_r_mean,xent_true_w_mean, res # p_diff
    # return (mean_log_p_w-mean_log_p_r), res # p_diff
    # return (mean_log_p_r), res # p_right
    # return (mean_log_p_r+mean_log_p_w)/2, res # p_mean
    # return (mean_xent_r-mean_xent_w), res # xent_diff
    # return (mean_xent_r+mean_xent_w)/2, res # xent_mean
    # return mean_xent_r, res # xent_right

  def logSymmKLdiffExp(self,a,b):
    """log KL-divergence except not summed"""
    return np.log(np.exp(a)*(a-b) + np.exp(b)*(b-a))
    
  def symmKLdivExp(self,a,b):
    """Takes the symmetrized KL-divergence of two vectors,
    after np.exp ing them"""
    return entropy(np.exp(a),np.exp(b)) + entropy(np.exp(b),np.exp(a))

  def mutualInformation(self,pe,pc,pe_given_c):
    """These probs are all log space"""
    pc_s = np.exp(pc+pe_given_c)
    return np.log((pc_s * (pe_given_c - pe)).sum())

  def pmi(self,pe,pe_given_c):
    return (pe_given_c-pe).mean()
  
  def statTests(self,probs):
    """Does various KL-divergence computations"""

    if "pc" in probs.columns:
      log("ŷ=1 MI(e,c)",
          self.mutualInformation(probs.pe_pred_r,probs.pc,probs.pe_given_c_pred_r))
      log("ŷ=0 MI(e,c)",
          self.mutualInformation(probs.pe_pred_w,probs.pc,probs.pe_given_c_pred_w))
      log("y=1 MI(e,c)",
          self.mutualInformation(probs.pe_true_r,probs.pc,probs.pe_given_c_true_r))
      log("y=0 MI(e,c)",
          self.mutualInformation(probs.pe_true_w,probs.pc,probs.pe_given_c_true_w))

    log("ŷ=1 pmi(e,c)",
        self.pmi(probs.pe_pred_r,probs.pe_given_c_pred_r))
    log("ŷ=0 pmi(e,c)",
        self.pmi(probs.pe_pred_w,probs.pe_given_c_pred_w))
    log("y=1 pmi(e,c)",
        self.pmi(probs.pe_true_r,probs.pe_given_c_true_r))
    log("y=0 pmi(e,c)",
        self.pmi(probs.pe_true_w,probs.pe_given_c_true_w))

    """
    log("ŷ=1 KL-div[p(s),p(s|c)]",
        self.symmKLdivExp(probs.ps_pred_r, probs.pe_given_c_pred_r))
    log("ŷ=0 KL-div[p(s),p(s|c)]",
        self.symmKLdivExp(probs.ps_pred_w, probs.pe_given_c_pred_w))
    log("y=1 KL-div[p(s),p(s|c)]",
        self.symmKLdivExp(probs.ps_true_r, probs.pe_given_c_true_r))
    log("y=0 KL-div[p(s),p(s|c)]",
        self.symmKLdivExp(probs.ps_true_w, probs.pe_given_c_true_w))
    """
    
  def halve_lr(self):
    new_lr = self.session.run(self._lr) / 2
    log("Halving learning rate to {:.4f}".format(new_lr))
    self.session.run(self._lr.assign(new_lr))
    
class EncDec(BaseModel):
  """Encoder decoder framework. Encodes the context
  sentences and then decodes the target training sentence.
  This only learns from the 5th sentence; no error is 
  propagated from the context sentences (1-4).
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )

      encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)
      decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = encoder_cell.zero_state(
        tf.shape(self._context_len)[0],tf.float32)
      
      if is_training:
        inp_p = .6
        out_p = .6
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
          decoder_cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("decoder_cell={}(inp_p={},out_p={})".format(
          type(decoder_cell).__name__,inp_p,out_p))
      
      # encoding the context sentences.
      _, self._enc_state = self.rnn_with_embedding(
        encoder_cell,self._init_state,self._context_seq,
        self._context_len,is_training,scope="encoder")

      # Running RNN through target sentence
      logit, _ = self.rnn_with_embedding(
        decoder_cell,self._enc_state,self._target_seq,self._target_len,
        is_training=True,reuse=None,scope="decoder")
      
      logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      target_list = target_list[1:]

      if is_training:
        self._lr = tf.Variable(self.learning_rate,
                               name="lr",trainable=False)
        # Loss computation
        xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,max_targ_len)
        
        self._cost = xent
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))
        
        self._xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,
          max_targ_len,reduce_mean=False)

        # 1st derivative saliency?
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._xent,[self._enc_state.c, self._enc_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]
        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())


  def add_summaries(self,args):
    self._xent_right = tf.placeholder(
      tf.float32,[],name="xent_right")
    self._xent_wrong = tf.placeholder(
      tf.float32,[],name="xent_wrong")
    
    self._grad_true_r_mean = tf.placeholder(
      tf.float32,[],name="grad_true_r_mean")
    self._grad_true_w_mean = tf.placeholder(
      tf.float32,[],name="grad_true_w_mean")
    self._grad_pred_r_mean = tf.placeholder(
      tf.float32,[],name="grad_pred_r_mean")
    self._grad_pred_w_mean = tf.placeholder(
      tf.float32,[],name="grad_pred_w_mean")
    
    self._xent_diff = tf.placeholder(
      tf.float32,[],name="xent_diff")
    self._log_p_right = tf.placeholder(
      tf.float32,[],name="log_p_right")
    self._log_p_wrong = tf.placeholder(
      tf.float32,[],name="log_p_wrong")
    self._log_p_diff = tf.placeholder(
      tf.float32,[],name="log_p_diff")
    self._acc = tf.placeholder(
      tf.float32,[],name="acc")
    self._prec = tf.placeholder(
      tf.float32,[],name="prec")
    self._rec = tf.placeholder(
      tf.float32,[],name="rec")
    self._f1 = tf.placeholder(
      tf.float32,[],name="f1")

    self._grad_pred_r = tf.placeholder(
      tf.float32,[None,self.hidden_size*2])
    self._grad_pred_w = tf.placeholder(
      tf.float32,[None,self.hidden_size*2])
    self._grad_true_r = tf.placeholder(
      tf.float32,[None,self.hidden_size*2])
    self._grad_true_w= tf.placeholder(
      tf.float32,[None,self.hidden_size*2])
    tf.summary.histogram("grad_pred_r",self._grad_pred_r)
    tf.summary.histogram("grad_pred_w",self._grad_pred_w)
    tf.summary.histogram("grad_true_r",self._grad_true_r)
    tf.summary.histogram("grad_true_w",self._grad_true_w)
    
    tf.summary.scalar("grad_pred_r_mean",self._grad_pred_r_mean)
    tf.summary.scalar("grad_pred_w_mean",self._grad_pred_w_mean)
    tf.summary.scalar("grad_true_r_mean",self._grad_true_r_mean)
    tf.summary.scalar("grad_true_w_mean",self._grad_true_w_mean)

    tf.summary.scalar("xent_right",self._xent_right)
    tf.summary.scalar("xent_wrong",self._xent_wrong)
    tf.summary.scalar("xent_diff",self._xent_diff)
    tf.summary.scalar("log_p_right",self._log_p_right)
    tf.summary.scalar("log_p_wrong",self._log_p_wrong)
    tf.summary.scalar("log_p_diff",self._log_p_diff)
    tf.summary.scalar("class_accuracy",self._acc)
    tf.summary.scalar("class_precision",self._prec)
    tf.summary.scalar("class_recall",self._rec)
    tf.summary.scalar("class_f1",self._f1)
    self._merged_summaries = tf.summary.merge_all()
    if args.train or args.test:
      path = args.train if args.train else args.test
      self.summaryWriter = tf.summary.FileWriter(
        "summaries/"+os.path.basename(path)+"/"+time.strftime(
          "%Y-%m-%d_%H-%M-%S"),
        self.graph)

  def train_batch(self,b,cost_only=False):
    """ """
    #embed()
    fetch_vars = [self._cost, self._predicted_seq, self._enc_state]
    feed_dict = {
      self._context_seq: b.context_seq,
      self._context_len: b.context_len,
    }
    
    if not cost_only:
      fetch_vars.insert(0,self._train_op)
      
    feed_dict.update({
      self._target_seq: b.target_seq,
      self._target_len: b.target_len
    })

    out = self.session.run(fetch_vars,feed_dict)

    if not cost_only: out.pop(0) # None from train_op
    cost = out[0]
    pred_seq = out[1]
    enc_state = np.concatenate(out[2],axis=1)
    return cost, pred_seq, enc_state

  def score_sentence(self,probs,targ, targ_len, norm_len=False):
    ps = [[b_time[t_time] for t, (b_time, t_time) in enumerate(zip(b_row,t_row))
           if t < l]
       for b_row,t_row,l in zip(probs,targ,targ_len)]
    # list(zip(targ[0,1:],ps[0])) gives (word, p(word|history)) tuple
    # ps includes the EOM symbol

    # p(w1,w2,...,EOM|BOM) = p(w1|BOM) * p(w2|BOM,w1) * ...
    # log space -> sum instead of multiply
    ps = [sum(l) for l in ps]
    if norm_len:
      # Normalizing for length
      ps = [p/l for p,l in zip(ps,targ_len)]
      
    return ps
  
  def test_batch(self,b,reverse_prob=True):
    """Runs a batch through, scores the 2 endings and chooses 1"""

    if reverse_prob:
      # Get initial state
      init_state = self.session.run(self._init_state,{
        self._context_len:np.concatenate([b.context_len,b.context_len])})
      
      #### p(s)
      cond_log_prob,xent = self.session.run(
        [self._softmaxes, self._xent],{
          self._enc_state: init_state,
          self._target_seq: np.concatenate([b.target1_seq,b.target2_seq]),
          self._target_len: np.concatenate([b.target1_len,b.target2_len]),
        }
      )
      
      cond_log_prob1, cond_log_prob2 = np.split(
        cond_log_prob,2)
      
      xent1, xent2 = np.split(xent,2)
      ps1 = self.score_sentence(
        cond_log_prob1,b.target1_seq[:,1:],b.target1_len-1)
      ps2 = self.score_sentence(
        cond_log_prob2,b.target2_seq[:,1:],b.target2_len-1)
      ps = np.array([ps1,ps2]).T
      
    #### p(sentence|context) ####
    # Step 1: run RNN over context
    feed_dict = {
      self._context_seq: np.concatenate([b.context_seq]*2),
      self._context_len: np.concatenate([b.context_len]*2),
    }
    enc_state = self.session.run(
      self._enc_state,feed_dict)

    # Step 1.5 get p(c)
    if hasattr(self,"_context_softmaxes"):
      cond_log_prob_context = self.session.run(
        self._context_softmaxes,{
          self._context_seq:b.context_seq,
          self._context_len: b.context_len})
      pc = self.score_sentence(
        cond_log_prob_context,b.context_seq[:,1:],b.context_len-1)
    else:
      pc = None
      
    # Step 2: run RNN over both target sentences
    cond_log_prob,xent,grad = self.session.run(
      [self._softmaxes, self._xent, self._grad],{
        self._enc_state: enc_state,
        self._target_seq: np.concatenate([b.target1_seq,b.target2_seq]),
        self._target_len: np.concatenate([b.target1_len,b.target2_len])
      }
    )
  
    # Grad here is (2*batch_size, 2*hidden_size)
    
    cond_log_prob1, cond_log_prob2 = np.split(
      cond_log_prob,2)
    xent1, xent2 = np.split(xent,2)
    xent = np.array([xent1,xent2]).T

    grad1,grad2 = np.split(grad,2)
    grad = np.array([grad1,grad2]).transpose([1,0,2])

    ps_given_c1 = self.score_sentence(
      cond_log_prob1,b.target1_seq[:,1:],b.target1_len-1)
    ps_given_c2 = self.score_sentence(
      cond_log_prob2,b.target2_seq[:,1:],b.target2_len-1)
    
    ps_given_c = np.array([ps_given_c1,ps_given_c2]).T
    
    # This is p(s|c)
    p_right = np.array([
      ps[e-1] for ps,e in zip(ps_given_c,b.rightending)])
    p_wrong = np.array([
      ps[1-(e-1)] for ps,e in zip(ps_given_c,b.rightending)])
    
    if reverse_prob:
      pc_given_s = ps_given_c - ps # logspace
      # pc_given_s = ps # LM score without context; works HORRIBLY, worse than p(s|c)
      pred = pc_given_s.argmax(axis=1) + 1
      # pred = ps_given_c.argmax(axis=1) + 1; print("Using ps_given_c")
      # pred = ps.argmax(axis=1) + 1; print("Using ps")
      # pred = self.logSymmKLdiffExp(ps_given_c,ps).argmax(axis=1)+1; print("Using KL-diff")
      # embed()
      # Updating the determining prob function
      # We're now using p(c|s)
      p_right = np.array([
        ps[e-1] for ps,e in zip(pc_given_s,b.rightending)])
      p_wrong = np.array([
        ps[1-(e-1)] for ps,e in zip(pc_given_s,b.rightending)])
      
    else:
      pred = ps_given_c.argmax(axis=1) + 1
      ps = None

    ### Cost computation
    xent_right = np.array([
        x[e-1] for x,e in zip(xent,b.rightending)])
    xent_wrong = np.array([
        x[1-(e-1)] for x,e in zip(xent,b.rightending)])
    
    grad_right = np.array([
      x[e-1] for x,e in zip(grad,b.rightending)])
    grad_wrong = np.array([
      x[1-(e-1)] for x,e in zip(grad,b.rightending)])
    
    enc_state = LSTMTupleSplit(enc_state,2)[0]
    
    # What to return as a metric for cost?
    # return xent_right.mean(), xent_wrong.mean(), pred
    return xent,ps,pc,ps_given_c,pred,grad,enc_state# _right,grad_wrong

class LangModel(EncDec):
  """Language model framework. Encodes the context
  sentences and then decodes the target training sentence.
  Backprop is done on all 5 sentences.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )
      
      cell = tf.contrib.rnn.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = cell.zero_state(tf.shape(
        self._context_len)[0],tf.float32)
      
      if is_training:
        inp_p = .6
        out_p = .6
        cell = tf.contrib.rnn.DropoutWrapper(
          cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("cell={}(inp_p={},out_p={})".format(
          type(cell).__name__,inp_p,out_p))
      
      # encoding the context sentences.
      context_logit, self._enc_state = self.rnn_with_embedding(
        cell,self._init_state,self._context_seq,
        self._context_len,is_training,scope="LM")
      
      # context_logit_list = tf.unstack(tf.transpose(context_logit,[1,0,2]))
      # context_list = tf.unstack(tf.transpose(self._context_seq,[1,0]))
      # context_list = context_list[1:]
       
      # Running RNN through target sentence
      logit, _ = self.rnn_with_embedding(
        cell,self._enc_state,self._target_seq,self._target_len,
        is_training=is_training,reuse=True,scope="LM")

      # logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      # target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      # target_list = target_list[1:]
      
      # self.p_s_given_c = self.sequence_prob(logit_list,target_list,
       #                                      self._target_len,max_targ_len)
      
      if is_training:
        self._lr = tf.Variable(self.learning_rate,
                               name="lr",trainable=False)
        # Loss computation
        
        context_xent = self.softmax_xent_loss_sequence(
          context_logit,self._context_seq[:,1:],self._context_len,max_cont_len)
        
        xent = self.softmax_xent_loss_sequence(
          logit,self._target_seq[:,1:],self._target_len,max_targ_len)
        
        self._cost = xent + context_xent
        
        self._softmaxes = tf.log(tf.nn.softmax(logit))
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        self._softmaxes = tf.log(tf.nn.softmax(logit))

        self._context_softmaxes = tf.log(tf.nn.softmax(context_logit))
        
        self._xent = self.softmax_xent_loss_sequence(
          logit,self._target_seq[:,1:],self._target_len,
          max_targ_len,False)
        context_xent = self.softmax_xent_loss_sequence(
          context_logit,self._context_seq[:,1:],
          self._context_len,max_cont_len,False)

        # 1st derivative saliency?
        # print("\033[93mWARNING: gradients not working... \033[0m")
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._xent,[self._enc_state.c, self._enc_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]

        self._cost = tf.reduce_mean(self._xent)#  + tf.reduce_mean(context_xent)

        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())

class LangModelBiRNN(LangModel):
  """Language model framework. Encodes the context
  sentences and then decodes the target training sentence.
  Backprop is done on all 5 sentences.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_story_len = max_story_len = reader.max_story_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))

      self._story_seq = tf.placeholder(
        tf.int64,
        [None,max_story_len],
        name="story_seq"
      )
      self._story_len = tf.placeholder(
        tf.int64,
        [None,],
        name="story_len"
      )
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )
      
      fwd_cell = tf.contrib.rnn.BasicLSTMCell(
        hidden_size,state_is_tuple=True)
      bwd_cell = tf.contrib.rnn.BasicLSTMCell(
        hidden_size,state_is_tuple=True)
      
      self._fwd_init_state = fwd_cell.zero_state(tf.shape(
        self._story_len)[0],tf.float32)
      self._bwd_init_state = bwd_cell.zero_state(tf.shape(
        self._story_len)[0],tf.float32)
      
      if is_training:
        inp_p = .8
        out_p = .8
        fwd_cell = tf.contrib.rnn.DropoutWrapper(
          fwd_cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("fwd_cell={}(inp_p={},out_p={})".format(
          type(fwd_cell).__name__,inp_p,out_p))
        
        bwd_cell = tf.contrib.rnn.DropoutWrapper(
          bwd_cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("bwd_cell={}(inp_p={},out_p={})".format(
          type(bwd_cell).__name__,inp_p,out_p))

      story_logit, self._fwd_state, _ = self.birnn_with_embedding(
        fwd_cell,bwd_cell,self._fwd_init_state,self._bwd_init_state,
        self._story_seq,self._story_len,scope="LM")
      
      # embed()#MAARTEN

      # logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      # target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      # target_list = target_list[1:]
      
      # self.p_s_given_c = self.sequence_prob(logit_list,target_list,
      #                                       self._target_len,max_targ_len)
      
      if is_training:
        self._lr = tf.Variable(self.learning_rate,
                               name="lr",trainable=False)
        # Loss computation        
        self._cost = xent = self.softmax_xent_loss_sequence(
          story_logit,self._story_seq[:,1:],self._story_len,max_story_len)
                
        self._softmaxes = tf.log(tf.nn.softmax(story_logit))
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        # encoding the context sentences.
        
        # context_logit, self._fwd_state,_ = self.birnn_with_embedding(
        #   fwd_cell,bwd_cell,self._fwd_init_state,self._bwd_init_state,
        #   self._context_seq,self._context_len,reuse=True,scope="LM")
        
        # # Running RNN through target sentence
        # logit, self._fwd_end_state, self._bwd_state = self.birnn_with_embedding(
        #   fwd_cell,bwd_cell,self._fwd_state,self._bwd_init_state,self._target_seq,
        #   self._target_len,reuse=True,scope="LM")
        
        # context_logit, _, self._bwd_end_state = self.birnn_with_embedding(
        #   fwd_cell,bwd_cell,self._fwd_init_state,self._bwd_state,
        #   self._context_seq,self._context_len,reuse=True,scope="LM")

        # self._softmaxes = tf.log(tf.nn.softmax(logit))

        # self._context_softmaxes = tf.log(tf.nn.softmax(context_logit))
        
        self._story_xent = self.softmax_xent_loss_sequence(
          story_logit,self._story_seq[:,1:],self._story_len,max_story_len,False)
        self._story_softmaxes = tf.log(tf.nn.softmax(story_logit))
        
        # self._xent = self.softmax_xent_loss_sequence(
        #   logit,self._target_seq[:,1:],self._target_len,
        #   max_targ_len,False)
        # self._context_xent = self.softmax_xent_loss_sequence(
        #   context_logit,self._context_seq[:,1:],
        #   self._context_len,max_cont_len,False)

        # # 1st derivative saliency?
        # # print("\033[93mWARNING: gradients not working... \033[0m")
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._story_xent,[self._fwd_init_state.c, self._fwd_init_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]

        self._cost = tf.reduce_mean(self._story_xent)

        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())
        
  def train_batch(self,b,cost_only=False):
    """ """
    #embed()
    fetch_vars = [self._cost, self._predicted_seq, self._fwd_state]
    feed_dict = {
      self._story_seq: b.story_seq,
      self._story_len: b.story_len,
    }
    
    if not cost_only:
      fetch_vars.insert(0,self._train_op)

    out = self.session.run(fetch_vars,feed_dict)

    if not cost_only: out.pop(0) # None from train_op
    cost = out[0]
    pred_seq = out[1]
    enc_state = np.concatenate(out[2],axis=1)
    return cost, pred_seq, enc_state


  def test_batch(self,b,reverse_prob=True):
    """Runs a batch through, scores the 2 endings and chooses 1"""

    # In any case, we need to run the BiRNN over the entire story
    # this gives us p(story) = p(c,e)
    # Get initial state
    init_state = self.session.run(
      [self._fwd_init_state,self._bwd_init_state],{
      self._story_len:np.concatenate([b.context_len,b.context_len])})

    init_state_ = self.session.run(
      [self._fwd_init_state,self._bwd_init_state],{
      self._story_len:b.context_len})

    #### p(story) = p(context,ending)
    p_words_story,xent = self.session.run(
      [self._story_softmaxes, self._story_xent],{
        self._fwd_init_state: init_state[0],
        self._bwd_init_state: init_state[1],
        self._story_seq: np.concatenate([b.story1_seq,b.story2_seq]),
        self._story_len: np.concatenate([b.story1_len,b.story2_len]),
      }
    )
    
    p_words_story1, p_words_story2 = np.split(
      p_words_story,2)
    xent1, xent2 = np.split(xent,2)
    xent = np.array([xent1,xent2]).T

    # grad1,grad2 = np.split(grad,2)
    # grad = np.array([grad1,grad2]).transpose([1,0,2])

    p_story1 = np.array(self.score_sentence(
      p_words_story1,b.story1_seq[:,1:],b.story1_len-1))
    p_story2 = np.array(self.score_sentence(
      p_words_story2,b.story2_seq[:,1:],b.story2_len-1))
    
    p_story = np.array([p_story1,p_story2]).T
    
    # This is p(story)
    p_story_right = np.array([
      ps[e-1] for ps,e in zip(p_story,b.rightending)])
    p_story_wrong = np.array([
      ps[1-(e-1)] for ps,e in zip(p_story,b.rightending)])

    ### p(context)
    p_words_context,context_xent,enc_state = self.session.run(
      [# self._context_softmaxes,self._context_xent,self._fwd_state
        self._story_softmaxes, self._story_xent,self._fwd_state
      ],{
        self._fwd_init_state: init_state_[0],
        # self._bwd_state: init_state_[1],
        self._bwd_init_state: init_state_[1],
        # self._context_seq: b.context_seq,
        # self._context_len: b.context_len,
        self._story_seq: b.context_seq,
        self._story_len: b.context_len,
        # self._story_len: b.story1_len,
      }
    )
    p_c = np.array(self.score_sentence(
      p_words_context,b.context_seq[:,1:],b.context_len-1))
    
    p_e_given_c = p_story - p_c[:,None]
    
    pred = p_story.argmax(axis=1) + 1
    

    ### p(ending)
    p_words_target,target_xent,grad = self.session.run(
      [self._story_softmaxes, self._story_xent, self._grad
       # self._softmaxes,self._xent,self._grad
      ],{
        # self._fwd_state: init_state[0],
        self._fwd_init_state: init_state[0],
        self._bwd_init_state: init_state[1],          
        self._story_seq: np.concatenate([b.target1_seq,b.target2_seq]),
        self._story_len: np.concatenate([b.target1_len,b.target2_len])
        # self._target_seq: np.concatenate([b.target1_seq,b.target2_seq]),
        # self._target_len: np.concatenate([b.target1_len,b.target2_len])
      }
    )
    grad1,grad2 = np.split(grad,2)
    grad = np.array([grad1,grad2]).transpose([1,0,2])

    
    ### p(ending)
    p_target1,p_target2 = np.split(np.array(self.score_sentence(
      p_words_target,np.concatenate([b.target1_seq,b.target2_seq])[:,1:],
      np.concatenate([b.target1_len,b.target2_len])+1)),2)
    
    p_e = np.array([p_target1,p_target2]).T
    if reverse_prob:
      pred = (p_e - p_e_given_c).argmax(axis=1) + 1
      
    # embed() # Maarten
    return xent,p_e,p_c,p_e_given_c,pred,grad,enc_state

  
class LangModelAttn(EncDec):
  """Language model framework with attention.
  Encodes the context sentences and then decodes
  the target training sentence, with attention over the context words.
  Backprop is done on all 5 sentences.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    
    self.session = session = tf.Session(
      graph=self.graph,config=config) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [args.batch_size,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [args.batch_size,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [args.batch_size,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [args.batch_size,],
        name="context_len"
      )

      cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = cell.zero_state(
        args.batch_size,tf.float32)
      
      if is_training:
        inp_p = .6
        out_p = .6
        cell = tf.nn.rnn_cell.DropoutWrapper(
          cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("cell={}(inp_p={},out_p={})".format(
          type(cell).__name__,inp_p,out_p))
      
      context_list = tf.unstack(tf.transpose(self._context_seq,[1,0]))
      target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))

      # Weird Google RNN with attention
      with tf.device('/cpu:0'), tf.variable_scope("dim_reduc"):
        un_emb = tf.get_variable("unemb",
                                 [self.hidden_size*4,self.vocab_size],
                                 tf.float32)
        # We need a bias
        un_emb_b = tf.get_variable("unemb_b",
                                   [self.vocab_size],
                                   # initializer=tf.constant_initializer(1.0),
                                   dtype=tf.float32)
      
      logit_list, final_state = \
        tf.nn.seq2seq.embedding_attention_seq2seq_lens(
          context_list,target_list,self._context_len,
          self._target_len,cell,vocab_size,hidden_size,
          output_projection=(un_emb,un_emb_b),
          scope="LM")
      # embed()
      self._final_state = final_state
      logit_list = logit_list[:-1]
      context_list = context_list[1:]
      target_list = target_list[1:]
      
      self._logits = tf.transpose(tf.stack(logit_list),[1,0,2])
      
      if is_training:
        self._lr = tf.Variable(self.learning_rate,
                               name="lr",trainable=False)
        # Loss computation
        xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,max_targ_len)
        
        self._cost = xent
        
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
          self._train_op = optimizer.minimize(
            self._cost,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
      else: #not training
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))

        self._xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,
          max_targ_len,reduce_mean=False)

        # 1st derivative saliency?
  
        self._grad = tf.zeros([args.batch_size,self.hidden_size*2],tf.float32)
        
        assert self._grad.get_shape().as_list() == [args.batch_size,self.hidden_size*2]
        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())
    
    
  def test_batch(self,b,reverse_prob=False):
    """For LangModelAttn. Runs a batch through, scores the 2 endings and chooses 1.
    Doesn't support reverse probability yet"""

    if reverse_prob:
      raise NotImplementedError("LangModelAttn.test_batch doesn't support rev_prob")
      """# Get initial state
      init_state = self.session.run(self._init_state,{
        self._context_len:np.concatenate([b.context_len,b.context_len])})
      
      #### p(s)
      cond_log_prob,xent = self.session.run(
        [self._softmaxes, self._xent],{
          self._enc_state: init_state,
          self._target_seq: np.concatenate([b.target1_seq,b.target2_seq]),
          self._target_len: np.concatenate([b.target1_len,b.target2_len]),
        }
      )
      
      cond_log_prob1, cond_log_prob2 = np.split(
        cond_log_prob,2)
      
      xent1, xent2 = np.split(xent,2)
      ps1 = self.score_sentence(
        cond_log_prob1,b.target1_seq[:,1:],b.target1_len-1)
      ps2 = self.score_sentence(
        cond_log_prob2,b.target2_seq[:,1:],b.target2_len-1)
      ps = np.array([ps1,ps2]).T"""
      
    #### p(sentence|context) ####
    # Step 1: run RNN over context
    embed()
    feed_dict = {
      self._context_seq: b.context_seq,
      self._context_len: b.context_len,
      
    }
    enc_state = self.session.run(
      self._enc_state,feed_dict)
    
    # Step 2: run RNN over both target sentences
    cond_log_prob,xent,grad = self.session.run(
      [self._softmaxes, self._xent, self._grad],{
        self._enc_state: enc_state,
        self._target_seq: np.concatenate([b.target1_seq,b.target2_seq]),
        self._target_len: np.concatenate([b.target1_len,b.target2_len])
      }
    )
    # Grad here is (2*batch_size, 2*hidden_size)
    
    cond_log_prob1, cond_log_prob2 = np.split(
      cond_log_prob,2)
    xent1, xent2 = np.split(xent,2)
    xent = np.array([xent1,xent2]).T

    grad1,grad2 = np.split(grad,2)
    grad = np.array([grad1,grad2]).transpose([1,0,2])

    ps_given_c1 = self.score_sentence(
      cond_log_prob1,b.target1_seq[:,1:],b.target1_len-1)
    ps_given_c2 = self.score_sentence(
      cond_log_prob2,b.target2_seq[:,1:],b.target2_len-1)
    
    ps_given_c = np.array([ps_given_c1,ps_given_c2]).T
    
    # This is p(s|c)
    p_right = np.array([
      ps[e-1] for ps,e in zip(ps_given_c,b.rightending)])
    p_wrong = np.array([
      ps[1-(e-1)] for ps,e in zip(ps_given_c,b.rightending)])
    
    if reverse_prob:
      pc_given_s = ps_given_c - ps # logspace
      # pc_given_s = ps # LM score without context; works HORRIBLY, worse than p(s|c)
      pred = pc_given_s.argmax(axis=1) + 1
      # pred = ps_given_c.argmax(axis=1) + 1; print("Using ps_given_c")
      # pred = ps.argmax(axis=1) + 1; print("Using ps")

      # Updating the determining prob function
      # We're now using p(c|s)
      p_right = np.array([
        ps[e-1] for ps,e in zip(pc_given_s,b.rightending)])
      p_wrong = np.array([
        ps[1-(e-1)] for ps,e in zip(pc_given_s,b.rightending)])
      
    else:
      pred = ps_given_c.argmax(axis=1) + 1
      ps = None

    ### Cost computation
    xent_right = np.array([
        x[e-1] for x,e in zip(xent,b.rightending)])
    xent_wrong = np.array([
        x[1-(e-1)] for x,e in zip(xent,b.rightending)])
    
    grad_right = np.array([
      x[e-1] for x,e in zip(grad,b.rightending)])
    grad_wrong = np.array([
      x[1-(e-1)] for x,e in zip(grad,b.rightending)])
  
    # What to return as a metric for cost?
    # return xent_right.mean(), xent_wrong.mean(), pred
    return xent,ps,ps_given_c,pred,grad# _right,grad_wrong

class EncDecLM(EncDec):
  """Encoder decoder framework with LM component. 
  Encodes the context into a compressed representation h_c
  then decodes target training sentence.
  Also backprops through the xent from the encoder as a LM.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )

      encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)
      decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = encoder_cell.zero_state(tf.shape(
        self._context_len)[0],tf.float32)
      
      if is_training:
        inp_p = .7
        out_p = .7
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
          decoder_cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("{}(inp_p={},out_p={})".format(
          type(decoder_cell).__name__,inp_p,out_p))
      
      # encoding the context sentences.
      c_logit, self._enc_state = self.rnn_with_embedding(
        encoder_cell,self._init_state,self._context_seq,
        self._context_len,is_training,scope="encoder")

      c_logit_list = tf.unstack(tf.transpose(c_logit,[1,0,2]))
      context_list = tf.unstack(tf.transpose(self._context_seq,[1,0]))
      context_list = context_list[1:]

      # Running RNN through target sentence
      logit, _ = self.rnn_with_embedding(
        decoder_cell,self._enc_state,self._target_seq,self._target_len,
        is_training=True,reuse=None,scope="decoder")
      
      logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      target_list = target_list[1:]      
      
      if is_training:
        # Loss computation
        xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,max_targ_len)
        c_xent = self.softmax_xent_loss_sequence(
          c_logit_list,context_list,self._context_len,max_cont_len)
        
        self._cost = xent  + c_xent
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
          self._train_op = optimizer.minimize(self._cost)
        
      else: #not training
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))
        
        self._xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,
          max_targ_len,reduce_mean=False)

        # 1st derivative saliency?
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._xent,[self._enc_state.c, self._enc_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]

        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())

        
class EncDecRec(EncDec):
  """Encoder decoder framework with reconstruction. 
  Encodes the context into a compressed representation h_c
  then decodes the target training sentence, and decodes to 
  reconstruct the context sentences.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )

      cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = cell.zero_state(tf.shape(
        self._context_len)[0],tf.float32)
      
      if is_training:
        inp_p = .5
        out_p = .5
        cell = tf.nn.rnn_cell.DropoutWrapper(
          cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("{}(inp_p={},out_p={})".format(
          type(cell).__name__,inp_p,out_p))
      
      # encoding the context sentences.
      _, self._enc_state = self.rnn_with_embedding(
        cell,self._init_state,self._context_seq,
        self._context_len,is_training,scope="encoder")

      # Running RNN through target sentence
      logit, _ = self.rnn_with_embedding(
        cell,self._enc_state,self._target_seq,self._target_len,
        is_training=True,reuse=False,scope="target_decoder")
      
      logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      target_list = target_list[1:]

      # Running RNN through context sentence
      c_logit, _ = self.rnn_with_embedding(
        cell,self._enc_state,self._context_seq,self._context_len,
        is_training=True,reuse=False,scope="context_decoder")
      
      c_logit_list = tf.unstack(tf.transpose(c_logit,[1,0,2]))
      context_list = tf.unstack(tf.transpose(self._context_seq,[1,0]))
      context_list = context_list[1:]

      if is_training:
        # Loss computation
        xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,max_targ_len)
        c_xent = self.softmax_xent_loss_sequence(
          c_logit_list,context_list,self._context_len,max_cont_len)
        
        self._cost = xent  + .01*c_xent
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))
        
        self._xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,
          max_targ_len,reduce_mean=False)

        # 1st derivative saliency?
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._xent,[self._enc_state.c, self._enc_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]

        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())


class LangModelOptimized(EncDec):
  """Language model framework. Encodes the context
  sentences and then decodes the target training sentence.
  Backprop is done on all 5 sentences, using p(c|s) as 
  objective function.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    raise NotImplementedError("This uses tf.gather_nd which doesn't"\
                              " have gradients yet.")
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )

      cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = cell.zero_state(tf.shape(
        self._context_len)[0],tf.float32)
      
      if is_training:
        inp_p = .5
        out_p = .5
        cell = tf.nn.rnn_cell.DropoutWrapper(
          cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("{}(inp_p={},out_p={})".format(
          type(cell).__name__,inp_p,out_p))
      
      # encoding the context sentences.
      context_logit, self._enc_state = self.rnn_with_embedding(
        cell,self._init_state,self._context_seq,
        self._context_len,is_training,scope="LM")
      
      context_logit_list = tf.unstack(tf.transpose(context_logit,[1,0,2]))
      context_list = tf.unstack(tf.transpose(self._context_seq,[1,0]))
      context_list = context_list[1:]
       
      # Running RNN through target sentence
      logit, _ = self.rnn_with_embedding(
        cell,self._enc_state,self._target_seq,self._target_len,
        is_training=True,reuse=True,scope="LM")
      
      logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      target_list = target_list[1:]

      # Running RNN through target sentence (sans context)
      logit_2, _ = self.rnn_with_embedding(
        cell,self._init_state,self._target_seq,self._target_len,
        is_training=True,reuse=True,scope="LM")
      
      logit_list_2 = tf.unstack(tf.transpose(logit_2,[1,0,2]))
      
      # all log probabilities
      self.p_s_given_c = self.sequence_prob(logit_list,target_list,
                                            self._target_len,max_targ_len)
      self.p_s = self.sequence_prob(logit_list_2,target_list,
                                    self._target_len,max_targ_len)
        
      if is_training:
        # Loss computation
        self._cost = tf.reduce_mean(self.p_s - self.p_s_given_c) #~ (-p(c|s))
        
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))
        
        self._xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,
          max_targ_len,reduce_mean=False)
        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())

class EncDecReversed(EncDec):
  """Encoder decoder framework, reversed: 
  Encodes the target into a compressed representation h_t
  then decodes the context training sentence.
  Note that for ease of computation, the context is
  concatenated into 1 sequence.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.max_targ_len = max_targ_len = reader.max_targ_len
    self.max_cont_len = max_cont_len = reader.max_cont_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries

    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_targ_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,max_cont_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,],
        name="context_len"
      )

      encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)
      
      decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      self._init_state = decoder_cell.zero_state(tf.shape(
        self._target_len)[0],tf.float32)
      
      if is_training:
        inp_p = .6
        out_p = .6
        decoder_cell = tf.nn.rnn_cell.DropoutWrapper(
          decoder_cell,input_keep_prob=inp_p,output_keep_prob=out_p)

        log("{}(inp_p={},out_p={})".format(
          type(decoder_cell).__name__,inp_p,out_p))
      
      # encoding the target sentence.
      _, self._enc_state = self.rnn_with_embedding(
        encoder_cell,self._init_state,self._target_seq,
        self._target_len,is_training,scope="target_encoder")

      # Running RNN through context sentence
      logit, _ = self.rnn_with_embedding(
        decoder_cell,self._enc_state,self._context_seq,self._context_len,
        is_training=True,reuse=False,scope="context_decoder")
      
      logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      context_list = tf.unstack(tf.transpose(self._context_seq,[1,0]))
      context_list = context_list[1:]

      if is_training:
        # Loss computation
        c_xent = self.softmax_xent_loss_sequence(
          logit_list,context_list,self._context_len,max_cont_len)
        
        self._cost = c_xent
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        # p(c|target)
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))
        
        self._xent = self.softmax_xent_loss_sequence(
          logit_list,context_list,self._context_len,
          max_cont_len,reduce_mean=False)
        
        # 1st derivative saliency?
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._xent,[self._enc_state.c, self._enc_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]
        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)

      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())

  def test_batch(self,b,reverse_prob=True):
    """Runs a batch through, scores the 2 endings and chooses 1"""
    
    if reverse_prob:
      raise NotImplementedError("Need a language model to do this")
      
    #### p(sentence|context) ####
    # Step 1: run RNN over targets
    feed_dict = {
      self._target_seq: np.concatenate([b.target1_seq,b.target2_seq]),
      self._target_len: np.concatenate([b.target1_len,b.target2_len]),
    }
    enc_state = self.session.run(
      self._enc_state,feed_dict)
    
    # Step 2: run RNN over context sentences
    cond_log_prob,xent,grad = self.session.run(
      [self._softmaxes, self._xent,self._grad],{
        self._enc_state: enc_state,
        self._context_seq: np.concatenate([b.context_seq]*2),
        self._context_len: np.concatenate([b.context_len]*2)
      }
    )
    cond_log_prob1,cond_log_prob2 = np.split(cond_log_prob,2)
    xent1,xent2 = np.split(xent,2)
    xent = xent.reshape([2,b.size]).T
    

    grad1,grad2 = np.split(grad,2)
    grad = np.array([grad1,grad2]).transpose([1,0,2])

    # p(c|s)
    pc_given_s1 = self.score_sentence(
      cond_log_prob1,b.context_seq[:,1:],b.context_len-1)
    pc_given_s2 = self.score_sentence(
      cond_log_prob2,b.context_seq[:,1:],b.context_len-1)
    
    pc_given_s = np.array([pc_given_s1,pc_given_s2]).T

    
    # This is p(c|s)
    p_right = np.array([
      ps[e-1] for ps,e in zip(pc_given_s,b.rightending)])
    p_wrong = np.array([
      ps[1-(e-1)] for ps,e in zip(pc_given_s,b.rightending)])
    
    pred = pc_given_s.argmax(axis=1) + 1
      
    ### Cost 
    xent_right = np.array([
      ps[e-1] for ps,e in zip(xent,b.rightending)])
    xent_wrong = np.array([
      ps[1-(e-1)] for ps,e in zip(xent,b.rightending)])

    grad_right = np.array([
      x[e-1] for x,e in zip(grad,b.rightending)])
    grad_wrong = np.array([
      x[1-(e-1)] for x,e in zip(grad,b.rightending)])

    
    # What to return as a metric for cost?
    # return xent_right.mean(), xent_wrong.mean(), pred
    return xent_right,xent_wrong,p_right,p_wrong,pred,grad# _right,grad_wrong


class EncDecSplit(EncDec):
  """Encoder Decoder framework. Encodes the context
  sentences and then decodes the target training sentence.
  Backprop is done on prediciton error of 5th sentence.
  Contrary to EncDec, this model encodes each of the 4 
  context sentences individually, and linearly merges the 
  4 hidden states.
  """
  def __init__(self,args,reader,is_training=True,init=None,
               scope=None,trained_model=None,summ=False):
    """Creates the graph and initializes if necessary.
    If a model is available, will share the graph with that.
    """
    self.reader = reader
    self.is_training = is_training
    self.reuse = reuse = not (is_training or trained_model is None)
    scope = type(self).__name__ if scope is None else scope
    self.graph = graph = tf.Graph() if trained_model is None else trained_model.graph
    self.session = session = tf.Session(
      graph=self.graph) if trained_model is None else trained_model.session
    
    self.hidden_size = hidden_size = args.hidden_size
    self.saver = None
    
    self.vocab_size = vocab_size = args.vocab_size
    self.context_size = context_size = reader.context_size
    self.max_seq_len = max_seq_len = reader.max_seq_len
    self.learning_rate = args.learning_rate
    
    init = is_training if init is None else init    
    self.summ = summ # whether or not we are making summaries
    
    with graph.as_default(), tf.variable_scope(scope,reuse=reuse) as vs:
      log("Building graph {}".format(vs.name))
      
      self._target_seq = tf.placeholder(
        tf.int64,
        [None,max_seq_len],
        name="target_seq"
      )
      self._target_len = tf.placeholder(
        tf.int64,
        [None,],
        name="target_len"
      )
      self._context_seq = tf.placeholder(
        tf.int64,
        [None,context_size,max_seq_len],
        name="context_seq"
      )
      self._context_len = tf.placeholder(
        tf.int64,
        [None,context_size],
        name="context_len"
      )

      # Context encoder cell (no dropout)
      encoder = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)

      decoder = tf.nn.rnn_cell.BasicLSTMCell(
        hidden_size,state_is_tuple=True)
      
      self._init_state = decoder.zero_state(tf.shape(
        self._context_len)[0],tf.float32)
      
      if is_training:
        inp_p = .8
        out_p = .8
        decoder = tf.nn.rnn_cell.DropoutWrapper(
          decoder,input_keep_prob=inp_p,output_keep_prob=out_p)
        log("decoder={}(inp_p={},out_p={})".format(
          type(decoder).__name__,inp_p,out_p))
      
      # encoding the context sentences.
      enc_state_list=[]
      
      for i in range(context_size):
        _, enc_state = self.rnn_with_embedding(
          encoder,self._init_state,self._context_seq[:,i,:],
          self._context_len[:,i],is_training,reuse=i!=0,
          scope="Encoder")
        enc_state_list.append(enc_state)
      enc_states_c = tf.concat(axis=1,values=[s.c for s in enc_state_list])
      enc_states_h = tf.concat(axis=1,values=[s.h for s in enc_state_list])
      enc_states = tf.concat(axis=1,values=[enc_states_c, enc_states_h])
      assert enc_states.get_shape()[1] == hidden_size * 2 * context_size
      
      # Dimension reduction from 4 context sentences to 1 story ending
      self._enc_state = tf.nn.rnn_cell._linear(
        enc_states,hidden_size*2,True,scope="DimReduction")
      self._enc_state = tf.nn.rnn_cell.LSTMStateTuple._make(
        tf.split(axis=1,num_or_size_splits=2,value=self._enc_state))
      
      # Running RNN through target sentence
      logit, _ = self.rnn_with_embedding(
        decoder,self._enc_state,self._target_seq,self._target_len,
        is_training=True,scope="Decoder")
      
      logit_list = tf.unstack(tf.transpose(logit,[1,0,2]))
      target_list = tf.unstack(tf.transpose(self._target_seq,[1,0]))
      target_list = target_list[1:]
      
      if is_training:
        self._lr = tf.Variable(self.learning_rate,
                               name="lr",trainable=False)
        # Loss computation
        xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,max_seq_len)
        
        self._cost = xent
        
        self._softmaxes = tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2])
        self._predicted_seq = tf.argmax(self._softmaxes,2)
        
        log(scope+"/Adding optimizer")
        with tf.variable_scope("AdamOptimizer"):
          optimizer = tf.train.AdamOptimizer(learning_rate=self._lr)
          self._train_op = optimizer.minimize(self._cost)
      else: #not training
        self._softmaxes = tf.log(tf.transpose(tf.stack(
          [tf.nn.softmax(t) for t in logit_list]),[1,0,2]))

        self._xent = self.softmax_xent_loss_sequence(
          logit_list,target_list,self._target_len,
          max_seq_len,reduce_mean=False)

        # 1st derivative saliency?
        self._grad = tf.abs(tf.concat(axis=1,values=tf.gradients(
          self._xent,[self._enc_state.c, self._enc_state.h])))
        assert self._grad.get_shape().as_list() == [None,self.hidden_size*2]
        
        self._cost = tf.reduce_mean(self._xent)
        if self.summ:
          # Summary
          log(scope+"/Adding summaries")
          self.add_summaries(args)
        
      if init:
        log(vs.name+"/Initializing variables")
        self.session.run(tf.global_variables_initializer())
