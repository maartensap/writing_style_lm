# writing_style_lm
Language model code for short stories.

This is the code used for the style features classification in the following paper:

[The Effect of Different Writing Tasks on Linguistic Style: A Case Study of the ROC Story Cloze Task] (https://arxiv.org/abs/1702.01841)
Roy Schwartz, Maarten Sap, Yannis Konstas, Leila Zilles, Yejin Choi and Noah A. Smith, CoNLL 2017

Requirements:
- python3, tensorflow1.0, pandas, numpy, nltk

Running:
`./main.py ...`

## Misc
The pre-processing step generates different train/dev splits each time, so results will vary between runs (and specifically between runs and the results published in the paper)
This code does not generate the language model part described in the paper, just the style features.

## Contact
msap@cs.washington.edu
