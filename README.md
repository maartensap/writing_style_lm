# Writing Style LM
Language model code for short stories.
This is the code used for the language modelling features in the following paper:

The Effect of Different Writing Tasks on Linguistic Style: A Case Study of the ROC Story Cloze Task 
Roy Schwartz, Maarten Sap, Yannis Konstas, Leila Zilles, Yejin Choi and Noah A. Smith, CoNLL 2017
[arXiv version](https://arxiv.org/abs/1702.01841)

Plug these features into the the main author's repository: https://github.com/roys174/writing_style.

## Dependencies
- Python 3.5
- Tensorflow 1.0.1, Pandas 0.18.1, NumPy 1.12.1, NLTK 3.2.1, Scikit-learn 0.17.1

## Training a model
The code first reads in the ROC story CSV files, which you should store in a directory (e.g. `ROCfiles/`) and name `train.csv`, `val.csv` and `test.csv`.
The following command will create a vocabulary, tokenize all ROC stories (with UNKing) and store the pre-processed data into `reader.pkl`:
`./main.py --train ROCLangModel --data_path ROCfiles --reader_path reader.pkl --vocab_cutoff 3 --hidden_size 512 --batch_size 32 --reverse_prob`
Subsequent runs will not re-process the data, it will simply work with the `reader.pkl` file:
`./main.py --train ROCLangModel --reader_path reader.pkl --hidden_size 512 --batch_size 32 --reverse_prob`

The training loop trains language model on the ROC story training data (after splitting those stories into train/val for early stopping). Convergence is tested on the validation portion of the training stories. After every epoch, we test on the story cloze task by classifying the two endings from the official validation set of the ROC stories. We either use $p(s_5|s_1,s_2,s_3,s_4)$ to select the ending or if `--reverse_prob` is set, we use $p(s_5|s_1,s_2,s_3,s_4)/p(s_5)$.

At convergence, it will save the model using the path specified by `--train`.

## Testing & exporting features
Once a model is trained, use the following commands to test your language model:
`./main.py --test ROCLangModel --reader_path reader.pkl --hidden_size 512 --batch_size 32 --reverse_prob`

For exporting purposes, use this:
`./main.py --train ROCLangModel --reader_path reader.pkl --hidden_size 512 --batch_size 32 --reverse_prob`
This will create two files containing $p(s_5)$ and $p(s_5|s_1,s_2,s_3,s_4)$ for the ROC validation and test sets.

## Misc
This code uses random initialization, so results will vary from the results in the paper.

## Contact
msap@cs.washington.edu
