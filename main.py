#!/usr/bin/env python3

import sys, os, re
import argparse

from reader import *
from model import *

p = argparse.ArgumentParser()
p.add_argument("--data_path", metavar="PATH",
               help="""directory with train.csv, val.csv, test.csv from the ROC story challenge""")
p.add_argument("--reader_path", metavar="PATH",
               help="""path where the Reader object is stored as a pickle""")
p.add_argument("--test", metavar="PATH",
               help="""tests a model loaded from a specified path (uses gold standard inputs)""")
p.add_argument("--export", metavar="PATH",
               help="tests a model loaded from a specified path (uses gold standard inputs)"
               " and exports the log_probability scores (only use with --reverse_prob)")
p.add_argument("--train", metavar="PATH",
               help="trains a model and saves it to a specified path")

### Hyper parameters ish
p.add_argument("--reverse_prob",default=False,action="store_true",
               help="""Reverse the scoring prob from p(s|c) to p(c|s)""")
p.add_argument("--learning_rate", metavar="LR",type=float,default=0.001,
               help="""Learning rate for AdamOptimizer.""")
p.add_argument("--hidden_size", dest="hidden_size", type=int,default=512,
               help="""Size of tensors in hidden layer.""")
p.add_argument("-b","--batch_size", dest="batch_size", type=int,default=32,
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
  
  return cost_hist[-2] < cost_hist[-1] and \
      cost_hist[-3] < cost_hist[-2]



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
   
  batch_yielder = r.LMBatchYielder

  # Has to be done in this order
  train_batches = [b for b in batch_yielder(
    args.batch_size,d="train_train")]

  val_batches = [b for b in batch_yielder(
    args.batch_size,d="train_val")]

  val_test_batches = [b for b in batch_yielder(
    args.batch_size,d="val")]

  train_batches = [b for b in batch_yielder(
    args.batch_size,d="train_train")]

  m = LangModel(args,r,is_training=True)

  val_m = LangModel(args,r,is_training=False,init=False,
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
  log("Saving before exiting to", save_model(path, m) )
  
  log("Summary files saved to:\n{}".format(val_m.summaryWriter.get_logdir()))

  
def test(model_path,reader_path,export=False):
  r = load_reader(reader_path)
  
  log(r)
  
  # reader determines vocab_size
  args.vocab_size = r.vocab_size
  
  batch_yielder = r.LMBatchYielder
  
  test_batches = [b for b in batch_yielder(
    args.batch_size,d="test")]
  val_batches = [b for b in batch_yielder(
    args.batch_size,d="val")]
  test_batches = [b for b in batch_yielder(
    args.batch_size,d="test")]

  m = LangModel(args,r,is_training=False,init=False)
  load_model(model_path,m)
  
  out = m.test_epoch(val_batches,args.reverse_prob,export=export)
  xent_r, xent_w, class_report = out[:3]
  if export:
    val_df = out[3]
  mean_xent = (xent_r+xent_w)/2
  log("Total cost (val): right: {:.5f}, wrong: {:.5f}".format(xent_r,xent_w))
  print(class_report)
  
  out = m.test_epoch(test_batches,args.reverse_prob,export=export)
  xent_r, xent_w, class_report = out[:3]
  if export:
    test_df = out[3]
    
  mean_xent = (xent_r+xent_w)/2
  log("Total cost (test): right: {:.5f}, wrong: {:.5f}".format(xent_r,xent_w))
  print(class_report)

  if export:
    assert args.reverse_prob, "Can only export with --reverse_prob"
    
    val_df.to_csv("val_LMscores.csv")
    test_df.to_csv("test_LMscores.csv")

    log("Exported the Lang model scores to '{}', '{}'".format("val_LMscores.csv","test_LMscores.csv"))    
  
def main(args):
  if args.train and (args.data_path or args.reader_path):
    train(args.train, args.data_path, args.reader_path)
  elif args.test and args.reader_path:
    test(args.test, args.reader_path)
  elif args.export and args.reader_path:
    test(args.export, args.reader_path, export=True)
  else:
    print("Specify --train")
    p.print_help()
    exit(2)

  
if __name__ == '__main__':
  start_time = time.time()
  main(args)
  time_d = time.time() - start_time
  log("Done! [That took {}]".format(timedelta(seconds=time_d)))

