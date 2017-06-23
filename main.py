#!/usr/bin/env python3

import sys, os, re
import argparse

from reader import *
from model import *
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter # for smoothing?

p = argparse.ArgumentParser()
p.add_argument("--data_path", metavar="PATH",
               help="""directory with train.csv, val.csv, test.csv from the ROC story challenge""")
p.add_argument("--reader_path", metavar="PATH",
               help="""path where the Reader object is stored as a pickle""")
p.add_argument("--test", metavar="PATH",
               help="""tests a model loaded from a specified path (uses gold standard inputs)""")
p.add_argument("--train", metavar="PATH",
               help="trains a model and saves it to a specified path")

### Hyper parameters ish
p.add_argument("--reverse_prob",default=False,action="store_true",
               help="""Reverse the scoring prob from p(s|c) to p(c|s)""")
p.add_argument("--learning_rate", metavar="LR",type=float,default=0.001,
               help="""Learning rate for AdamOptimizer.""")
p.add_argument("--hidden_size", dest="hidden_size", type=int,default=256,
               help="""Size of tensors in hidden layer.""")
p.add_argument("-b","--batch_size", dest="batch_size", type=int,default=50,
               help="""Batch size to pass through the graph.""")
p.add_argument("--max_epoch", dest="max_epoch", type=int,default=150,
               help="""Determines how many passes throught the training data to do""")
p.add_argument("--vocab_cutoff", dest="vocab_cutoff", type=int,default=3,
               help="""Determines cutoff for word frequency in training""")

p.add_argument("--no_overfit_safety", default=True, action='store_false',
               dest='overfit_safety',help="stop when validation cost stops decreasing")


if len(sys.argv) < 2:
  p.print_help()
  exit(2)
  
args = p.parse_args()
log(args)

import time
from datetime import timedelta

def test_overfitting(cost_hist):
  if len(cost_hist) < 3: return False
  
  # window_size = 7
  # poly_order = 5
  # if len(cost_hist) < window_size:
  return cost_hist[-2] < cost_hist[-1] and \
      cost_hist[-3] < cost_hist[-2]
  # hist = np.array(cost_hist)
  # smooth = savgol_filter(hist,window_size,poly_order)
  # if len(cost_hist) > 20:
  #   embed()


def train(path, data_path, reader_path):
  log("Training!")
  if reader_path and os.path.isfile(reader_path):
    r = load_reader(reader_path)
  elif data_path:
    r = Reader(args.data_path,
               vocab_cutoff=args.vocab_cutoff)
    r.make_vocab()
    r.tokenize_docs()
    r.split_train_val()
    r.save(reader_path)
  else:
    log("Reader file does not exist",file=sys.stderr)
  log(r)

  args.vocab_size = r.vocab_size
  
  classs = os.path.basename(path).split("_")[0]
  classs = re.sub(r"^\d+","N",classs)
  
  batch_yielder = r.LMBatchYielder

  classs = eval(classs)

  # Has to be done in this order
  train_batches = [b for b in batch_yielder(
    args.batch_size,d="train_train")]#[:1]

  val_batches = [b for b in batch_yielder(
    args.batch_size,d="train_val")]#[:1]

  val_test_batches = [b for b in batch_yielder(
    args.batch_size,d="val")]#[:1]

  train_batches = [b for b in batch_yielder(
    args.batch_size,d="train_train")]#[:1]
  # print(r.max_targ_len,r.max_cont_len,r.max_story_len)
  ## exit()
  m = classs(args,r,is_training=True)

  val_m = classs(args,r,is_training=False,init=False,
                 trained_model=m,summ=True)
  
  prev_cost = prev_prev_cost = val_cost = 1e10
  cost_hist = []
  overfitting, overfit_start = False, 0
  
  for step in range(args.max_epoch):
    log("Epoch {:3d}/{:3d}".format(step+1,args.max_epoch))
    
    try:
      cost = m.train_epoch(train_batches,global_step=step)

      log("Epoch {:3d}/{:3d}: training cost {:.5f}".format(
        step+1,args.max_epoch,cost))
      
      prev_prev_cost = prev_cost
      prev_cost = val_cost

      # Xentropy
      val_cost = m.train_epoch(val_batches,cost_only=True)
      cost_hist.append(val_cost)
      
      # classification
      _, _,class_report = val_m.test_epoch(
        val_test_batches,
        reverse_prob=args.reverse_prob,
        global_step=step)
      print(class_report)
      
    except KeyboardInterrupt:
      print()
      log("Killing training task, but saving first")
      break

    if (step+1) % 5 == 0:
      save_model(path, m)
    
    log("Epoch {:3d}/{:3d}: validation cost {:.5f} (prev. val. cost {:.5f})".format(
      step+1,args.max_epoch,val_cost,prev_cost))

    if test_overfitting(cost_hist):
      overfitting = test_overfitting(cost_hist)
      
      if not overfitting:
        overfitting = True
        overfit_start = step
        overfit_msg = "Started overfitting at step {}".format(overfit_start)
        
      if args.overfit_safety:
        log("Started overfitting!!! Exiting now before it's too late")
        break
      else:
        log(overfit_msg+" not stopping though")
  log("Saving before exiting")
  save_model(path, m)
  log("Summary files saved to:\n{}".format(val_m.summaryWriter.get_logdir()))

# def trainLM(path, data_path, reader_path):
#   log("Training!")
#   if reader_path and os.path.isfile(reader_path):
#     r = load_reader(reader_path)
#   elif data_path:
#     r = LMReader("/home/msap/data/aclImdb/data_preprocessed.pkl",
#                  vocab_cutoff=args.vocab_cutoff)
#     r.make_vocab()
#     r.tokenize_docs()
#     r.save(reader_path)
#     log(r)
#     sys.exit(0)
#   else:
#     log("Reader file does not exist",file=sys.stderr)
#   log(r)
  
#   # reader determines vocab_size
#   args.vocab_size = r.vocab_size
  
#   classs = os.path.basename(path).split("_")[0]
#   classs = re.sub(r"^\d+","N",classs)
  
#   batch_yielder = r.EncDecBatchYielder if "EncDecSplit" == classs else r.LMBatchYielder
#   classs = eval(classs)

#   # Has to be done in this order
#   train_batches = [b for b in batch_yielder(
#     args.batch_size,d="train")]#[-2:]

#   val_batches = [b for b in batch_yielder(
#     args.batch_size,d="val")]#[-2:]

#   #val_test_batches = [b for b in batch_yielder(
#   # args.batch_size,d="val")]#[:1]
  
#   m = classs(args,r,is_training=True)
  
#   #val_m = classs(args,r,is_training=False,init=False,
#   #              trained_model=m,summ=True)
  
#   prev_cost = prev_prev_cost = val_cost = 1e10
#   cost_hist = []
#   overfitting, overfit_start = False, 0
  
#   for step in range(args.max_epoch):
#     log("Epoch {:3d}/{:3d}".format(step+1,args.max_epoch))
    
#     try:
#       cost = m.train_epoch(train_batches,global_step=step)

#       log("Epoch {:3d}/{:3d}: training cost {:.5f}".format(
#         step+1,args.max_epoch,cost))
      
#       prev_prev_cost = prev_cost
#       prev_cost = val_cost

#       # Xentropy
#       val_cost = m.train_epoch(val_batches,cost_only=True)
#       cost_hist.append(val_cost)
      
#       # classification
#       """_, class_report = val_m.test_epoch(
#         val_test_batches,
#         reverse_prob=args.reverse_prob,
#         global_step=step)
#       print(class_report)
#       """
      
#     except KeyboardInterrupt:
#       print()
#       log("Killing training task, but saving first")
#       break

#     if (step+1) % 5 == 0:
#       save_model(path, m)
    
#     log("Epoch {:3d}/{:3d}: validation cost {:.5f} (prev. val. cost {:.5f})".format(
#       step+1,args.max_epoch,val_cost,prev_cost))
    
#     if test_overfitting(cost_hist):
      
#       # Condition for overfitting
#       if not overfitting:
#         overfitting = True
#         overfit_start = step
#         overfit_msg = "Started overfitting at step {}".format(overfit_start)
        
#       if args.overfit_safety:
#         log("Started overfitting!!! Exiting now before it's too late")
#         break
#       else:
#         log(overfit_msg+" not stopping though")
#   log("Saving before exiting")
#   save_model(path, m)
#   log("Summary files saved to:\n{}".format(val_m.summaryWriter.get_logdir()))

  
def test(model_path,reader_path):
  r = load_reader(reader_path)
  # embed()
  log(r)
  # embed()
  # reader determines vocab_size
  args.vocab_size = r.vocab_size
  
  classs = re.sub(r"^\d+","N",os.path.basename(model_path).split("_")[0])
  if "EncDecSplit" == classs:
    batch_yielder = r.EncDecBatchYielder              
  elif "BiRNN" in classs:
    batch_yielder = r.BiRNNLMBatchYielder
  else:
    batch_yielder = r.LMBatchYielder

  # train_batches = [b for b in batch_yielder(
  # args.batch_size,d="train_train")]+[b for b in batch_yielder(
  # args.batch_size,d="train_val")]
  # # for exporting purposes
  # train_m = classs(args,r,is_training=True,init=False,summ=args.summ)
  # load_model(model_path,train_m)
  # train_m.train_epoch(train_batches,cost_only=True)

  classs = eval(classs) if classs!="LM" else EncDec
  
  test_batches = [b for b in batch_yielder(
    args.batch_size,d="test")]
  val_batches = [b for b in batch_yielder(
    args.batch_size,d="val")]#[:2]
  test_batches = [b for b in batch_yielder(
    args.batch_size,d="test")]#[:2]

  
  m = classs(args,r,is_training=False,init=False,summ=args.summ)
  load_model(model_path,m)
  
  xent_r, xent_w, class_report = m.test_epoch(val_batches,args.reverse_prob)
  mean_xent = (xent_r+xent_w)/2
  log("Total cost (val): right: {}, wrong: {:.5f}".format(xent_r,xent_w))
  print(class_report)
  xent_r, xent_w, class_report = m.test_epoch(test_batches,args.reverse_prob)
  mean_xent = (xent_r+xent_w)/2
  log("Total cost (test): right: {}, wrong: {:.5f}".format(xent_r,xent_w))
  print(class_report)


def testContextSim(model_path,reader_path):
  r = load_reader(reader_path)
  log(r)
  # embed()
  # reader determines vocab_size
  args.vocab_size = r.vocab_size
  
  classs = re.sub(r"^\d+","N",os.path.basename(model_path).split("_")[0])
  batch_yielder = r.EncDecBatchYielder if "EncDecSplit" == classs else r.LMBatchYielder
  classs = eval(classs) if classs!="LM" else EncDec
  """train_batches = [b for b in batch_yielder(
    args.batch_size,d="train_train")]+[b for b in batch_yielder(
    args.batch_size,d="train_val")]"""
  val_batches = [b for b in batch_yielder(
    args.batch_size,d="val")]
  m = classs(args,r,is_training=False,init=False,summ=args.summ)
  load_model(model_path,m)

  # for exporting purposes
  # train_m = classs(args,r,is_training=True,init=False,summ=args.summ)
  # load_model(model_path,train_m)
  # train_m.train_epoch(train_batches[:10],cost_only=True)
  
  mean_xent, class_report = m.test_epoch(val_batches,args.reverse_prob)
  log("Total cost: {}".format(mean_xent))
  print(class_report)
  
def plot(model_path,reader_path):
  r = load_reader(reader_path)
  log(r)
  
  # reader determines vocab_size
  args.vocab_size = r.vocab_size
  val_batches = [b for b in r.LMBatchYielder(
    1,d="val")]
  b = val_batches[np.random.randint(len(val_batches))]
  
  classs = re.sub(r"^\d+","N",os.path.basename(model_path).split("_")[0])
  classs = eval(classs) if classs!="LM" else EncDec
  
  m = classs(args,r,is_training=False,init=False)
  
  load_model(model_path,m)

  xent_right,xent_wrong,p_right,p_wrong,pred_ending,grad = \
    m.test_batch(b)
  
  while pred_ending == b.rightending:
    b = val_batches[np.random.randint(len(val_batches))]
    xent_right,xent_wrong,p_right,p_wrong,pred_ending,grad = \
      m.test_batch(b)
  grad_pred_r = grad[0][pred_ending[0]-1]
  grad_pred_w = grad[0][1 - (pred_ending[0]-1)]
  grad_true_r = grad[0][b.rightending[0]-1]
  grad_true_w = grad[0][1 - (b.rightending[0]-1)]
  
  if pred_ending == b.rightending:
    log("Model is right about this one.")
  else:
    log("Model is wrong about this one.")

  df = pd.DataFrame(np.concatenate([grad_true_r,grad_true_w]),
                    index=["real (avg grad: {:.4f})".format(grad_true_r.mean()),
                           "bad (avg grad: {:.4f})".format(grad_true_w.mean())])
  plt.figure(figsize=(18,2.5))
  plt.pcolor(df,cmap="Blues")
  plt.suptitle("Model {}correctly predicted this pair".format("in" if pred_ending != b.rightending else ""))
  plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
  plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
  plt.show()
  
def main(args):
  if args.train and (args.data_path or args.reader_path):
    train(args.train, args.data_path, args.reader_path)
  elif args.train_lm and (args.data_path or args.reader_path):
    trainLM(args.train_lm, args.data_path, args.reader_path)
  elif args.test and args.reader_path:
    test(args.test, args.reader_path)
  elif args.plot and args.reader_path:
    plot(args.plot, args.reader_path)
  else:
    print("Specify --train")
    p.print_help()
    exit(2)

  
if __name__ == '__main__':
  start_time = time.time()
  main(args)
  time_d = time.time() - start_time
  log("Done! [That took {}]".format(timedelta(seconds=time_d)))

