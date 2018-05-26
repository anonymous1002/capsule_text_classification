# capsule_text_classification


Requirements: Code is written in Python (2.7) and requires Tensorflow (1.4.1).

The data folder consists of Movie Reviewer (MR), Stanford Sentiment Treebank (SST). Subject, TREC questions, Customer Reviews (CR), AGs news corpus. Note that Reuters-21578 dataset is included in NLTK.reuters package from http://www.nltk.org/nltk_data/.

The preprocess.py provides functions to clean the raw data and generate metadata for each dataset such as word vocabulary and their embedding representation initialised with pre-trained word2vec vectors from https://code.google.com/p/word2vec/ except Reuters corpus.

The reuters_process.py provides functions to clean the raw data and generate Reuters-Multilabel and Reuters-Full datasets.

The utils.py includes several wrapped and fundamental functions such as _conv2d_wrapper, _separable_conv2d_wrapper and _get_variable_wrapper etc. to make programming efficiently.

The layers.py implements capsule network including Primary Capsule Layer, Convolutional Capsule Layer, Capsule Flatten Layer, FC Capsule Layer. The proposed Routing Algorithm with leaky-softmax and coefficients and different version of squash function amendment are also included. Note that the orphan category is added in the main.py

The network.py provides the implementation of two different kinds of capsule network as well as KimCNN for comparison.

The loss.py provides the implementation of three different kinds of loss function: cross entropy, margin loss and spread loss.

The main.py provides arguments setting and main to run the code.
