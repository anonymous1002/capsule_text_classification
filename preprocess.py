import numpy as np
import h5py
import re
import sys
import operator
import argparse

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            #check if words are in the word2vec repos.
            if word in vocab: 
               word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')  
            else:
               f.read(binary_len)
    return word_vecs

def line_to_words(line, dataset):
  if dataset == 'SST1' or dataset == 'SST2':
    clean_line = clean_str_sst(line.strip())
  else:
    clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  words = words[1:]

  return words
        
def get_vocab(file_list, dataset=''):
  max_sent_len = 0
  word_to_idx = {}
  idx = 1
  for filename in file_list:
    f = open(filename, "r")
    for line in f:
        words = line_to_words(line, dataset)
        max_sent_len = max(max_sent_len, len(words))
        for word in words:
            if not word in word_to_idx:
                word_to_idx[word] = idx
                idx += 1
    f.close()
  return max_sent_len, word_to_idx


def load_data(dataset, train_name, test_name='', dev_name='', padding=0):
  """
  Load training data (dev/test optional).
  """
  f_names = [train_name]
  if not test_name == '': f_names.append(test_name)
  if not dev_name == '': f_names.append(dev_name)

  max_sent_len, word_to_idx = get_vocab(f_names, dataset)
  # that's for the purpose of shared vocabulary.
      
  print(max_sent_len)
  dev = []
  dev_label = []
  train = []
  train_label = []
  test = []
  test_label = []

  files = []
  data = []
  data_label = []

  f_train = open(train_name, 'r')
  files.append(f_train)
  data.append(train)
  data_label.append(train_label)
  if not test_name == '':
    f_test = open(test_name, 'r')
    files.append(f_test)
    data.append(test)
    data_label.append(test_label)
  if not dev_name == '':
    f_dev = open(dev_name, 'r')
    files.append(f_dev)
    data.append(dev)
    data_label.append(dev_label)

  for d, lbl, f in zip(data, data_label, files):
    for line in f:
      words = line_to_words(line, dataset)
      y = int(line.strip().split()[0])
      sent = [word_to_idx[word] for word in words if word in word_to_idx]
      if len(sent) > max_sent_len:
          sent = sent[:max_sent_len]
      else:    
          sent.extend([0] * (max_sent_len + padding - len(sent)))

      d.append(sent)
      lbl.append(y)

  f_train.close()
  if not test_name == '':
    f_test.close()
  if not dev_name == '':
    f_dev.close()

  return word_to_idx, np.array(train, dtype=np.int32), np.array(train_label, dtype=np.int32),\
         np.array(test, dtype=np.int32), np.array(test_label, dtype=np.int32), \
         np.array(dev, dtype=np.int32), np.array(dev_label, dtype=np.int32)

def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
  string = re.sub(r"\'s", " \'s", string) 
  string = re.sub(r"\'ve", " \'ve", string) 
  string = re.sub(r"n\'t", " n\'t", string) 
  string = re.sub(r"\'re", " \'re", string) 
  string = re.sub(r"\'d", " \'d", string) 
  string = re.sub(r"\'ll", " \'ll", string) 
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " ( ", string) 
  string = re.sub(r"\)", " ) ", string) 
  string = re.sub(r"\?", " ? ", string) 
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

def clean_str_sst(string):
  """
  Tokenization/string cleaning for the SST dataset
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()

FILE_PATHS = {"SST1": ("data/stsa.fine.train",
                  "data/stsa.fine.dev",
                  "data/stsa.fine.test"),
              "SST2": ("data/stsa.binary.train",
                  "data/stsa.binary.dev",
                  "data/stsa.binary.test"),
              "MR": ("data/rt-polarity.all", "", ""),
              "SUBJ": ("data/subj.all", "", ""),
              "CR": ("data/custrev.all", "", ""),
              "MPQA": ("data/mpqa.all", "", ""),
              "TREC": ("data/TREC.train.all", "", "data/TREC.test.all"),
              "AGNEWS": ("data/agnews.train.all", "", "data/agnews.test.all"),
              "DBP": ("data/dbp.train.all", "", "data/dbp.test.all"),
              }

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help="Data set", type=str, default="MR")
parser.add_argument('--w2v', help="word2vec file", type=str, default="GoogleNews-vectors-negative300.bin")
parser.add_argument('--train', help="custom train data", type=str, default="")
parser.add_argument('--test', help="custom test data", type=str, default="")
parser.add_argument('--dev', help="custom dev data", type=str, default="")
parser.add_argument('--padding', help="padding around each sentence", type=int, default=0)
args = parser.parse_args()

train_path, dev_path, test_path = FILE_PATHS[dataset]

word_to_idx, train, \
train_label, test, \
test_label, dev, dev_label = load_data(dataset, train_path, \
                                         test_name=test_path, \
                                         dev_name=dev_path, \
                                         padding=args.padding)

if dataset == "AGNEWS":
    train_label = train_label - 1
    test_label = test_label - 1

with open(dataset + '_word_mapping.txt', 'w+') as embeddings_f:
   embeddings_f.write("*PADDING* 0\n")
   for word, idx in sorted(word_to_idx.items(), key=operator.itemgetter(1)):
      embeddings_f.write("%s %d\n" % (word, idx))

w2v = load_bin_vec(args.w2v, word_to_idx)
V = len(word_to_idx) + 1 #plus padding
print 'Vocab size:', V 
  
def load_embedding(V,w2v):
    np.random.seed(1)
    # Not all words in word_to_idx are in w2v.
    # Word embeddings initialized to random Unif(-0.25, 0.25)
    embed = np.random.uniform(-0.25, 0.25, (V, len(w2v.values()[0])))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec # padding word is positioned at index 0
    return embed

w2v_embedding = load_embedding(V, w2v)

print 'train size:', train.shape
print ('dev size:', dev.shape)
print ('test size:', test.shape)

filename = dataset + '.hdf5'
with h5py.File(filename, "w") as f:
    f["w2v"] = np.array(w2v_embedding)
    f['train'] = train
    f['train_label'] = train_label
    f['test'] = test
    f['test_label'] = test_label
    f['dev'] = dev
    f['dev_label'] = dev_label    

