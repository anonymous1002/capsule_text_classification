from __future__ import division, print_function, unicode_literals
import argparse
import h5py
import numpy as np
import tensorflow as tf
from keras import utils
from keras import backend as K
from utils import _conv2d_wrapper
from network import capsule_model_v2, capsule_model_v0
from loss import margin_loss, spread_loss, cross_entropy
import tensorflow.contrib.slim as slim
from sklearn.utils import shuffle

tf.reset_default_graph()
np.random.seed(0)
tf.set_random_seed(0)

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='nonstatic',#nonstatic
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--dataset', type=str, default='MR',
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--loss_type', type=str, default='margin_loss',#nonstatic
                    help='Options: rand (randomly initialized word embeddings), static (pre-trained embeddings from word2vec, static during learning), nonstatic (pre-trained embeddings, tuned during learning), multichannel (two embedding channels, one static and one nonstatic)')

parser.add_argument('--folds', type=int, default=10,
                    help='number of folds to use. If test set provided, folds=1. max 10')
# preprocessed data
parser.add_argument('--has_test', type=int, default=1, help='If data has test, we use it. Otherwise, we use CV on folds')    
parser.add_argument('--has_dev', type=int, default=1, help='If data has dev, we use it, otherwise we split from train')    

# Training hyperparameters
parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for training')#50

# Model hyperparameters
parser.add_argument('--kernels', type=str, default='{3,4,5}', help='Kernel sizes of convolutions, table format')

parser.add_argument('--use_orphan', type=bool, default='False', help='Add orphan capsule or not')
parser.add_argument('--use_shared', type=bool, default='False', help='Use shared transformation matrix or not')
parser.add_argument('--use_leaky', type=bool, default='False', help='Use leaky-softmax or not')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for training')
parser.add_argument('--margin', type=float, default=0.2, help='the initial value for spread loss')

import json
args = parser.parse_args()

params = vars(args)
print('parsed input parameters:')
print(json.dumps(params, indent = 2))

fold_test_scores = {}
def load_data(dataset):
    train, train_label = [],[]
    dev, dev_label = [],[]
    test, test_label = [],[]
    
    f = h5py.File(dataset+'.hdf5', 'r') 
    print('loading data...')    
    print(dataset)
    print("Keys: %s" % f.keys())
  
    w2v = list(f['w2v'])
    train = list(f['train'])
    train_label = list(f['train_label'])
    args.num_classes = max(train_label) + 1
      
    if len(list(f['test'])) == 0:
        args.has_test = 0
    else:
        args.has_test = 1
        test = list(f['test'])
        test_label = list(f['test_label'])

    print('data loaded!')
    return train, train_label, test, test_label, dev, dev_label, w2v

class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, dataset,label, batch_size,input_size):
      self._dataset = dataset
      self._label = label
      self._batch_size = batch_size    
      self._cursor = 0      
      self._input_size = input_size      
      
      index = np.arange(len(self._dataset))
      np.random.shuffle(index)
      self._dataset = np.array(self._dataset)[index]
      self._label = np.array(self._label)[index]
      
    def next(self):
      if self._cursor + self._batch_size > len(self._dataset):
          self._cursor = 0     
      batch_x = self._dataset[self._cursor : self._cursor + self._batch_size,:]
      batch_y = self._label[self._cursor : self._cursor + self._batch_size]
      self._cursor += self._batch_size
      return batch_x, batch_y

train, train_label, test, test_label, dev, dev_label, w2v= load_data(args.dataset)    

args.vocab_size = len(w2v)
args.vec_size = w2v[0].shape[0]
args.max_sent = len(train[0])
print('max sent: ', args.max_sent)
print('vocab size: ', args.vocab_size)
print('vec size: ', args.vec_size)

train, train_label = shuffle(train, train_label)

with tf.device('/cpu:0'):
    global_step = tf.train.get_or_create_global_step()
    
if args.use_orphan:
    args.num_classes = args.num_classes + 1
    
X = tf.placeholder(tf.int32, [args.batch_size, args.max_sent], name="input_x")
y = tf.placeholder(tf.int64, [args.batch_size, args.num_classes], name="input_y")
learning_rate = tf.placeholder(dtype='float32') 
margin = tf.placeholder(shape=(),dtype='float32') 

if args.model_type == 'rand':
    W1 = tf.Variable(tf.random_uniform([args.vocab_size, args.vec_size], -0.25, 0.25),name="Wemb")
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[...,tf.newaxis]  
if args.model_type == 'static':
    W1 = tf.Variable(w2v, trainable = False) 
    X_embedding = tf.nn.embedding_lookup(W1, X)
    X_embedding = X_embedding[...,tf.newaxis]  
if args.model_type == 'nonstatic':
    W1 = tf.Variable(w2v, trainable = True)
    W2 = tf.Variable(w2v, trainable = False)
    X_1 = tf.nn.embedding_lookup(W1, X)
    X_2 = tf.nn.embedding_lookup(W2, X) 
    X_1 = X_1[...,tf.newaxis]
    X_2 = X_2[...,tf.newaxis]
    X_embedding = tf.concat([X_1,X_2],axis=-1)

tf.logging.info("input dimension:{}".format(X_embedding.get_shape()))

poses, activations = capsule_model_v0()  

if args.loss_type == 'spread_loss':
    loss = spread_loss(y, activations, margin)
if args.loss_type == 'margin_loss':    
    loss = margin_loss(y, activations)
if args.loss_type == 'cross_entropy':
    loss = cross_entropy(y, activations)

y_pred = tf.argmax(activations, axis=1, name="y_proba")    
correct = tf.equal(tf.argmax(y, axis=1), y_pred, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)   

gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 4.0)

grad_check = [tf.check_numerics(g, message='Gradient NaN Found!')
              for g in gradients if g is not None] + [tf.check_numerics(loss, message='Loss NaN Found')]
with tf.control_dependencies(grad_check):
    training_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)      

sess = tf.InteractiveSession()
        
n_iterations_per_epoch = len(train) // args.batch_size
n_iterations_test = len(test) // args.batch_size
n_iterations_dev = len(dev) // args.batch_size        
    
mr_train = BatchGenerator(train,train_label, args.batch_size, 0)
mr_test = BatchGenerator(test,test_label, args.batch_size, 0)
mr_dev = BatchGenerator(dev,dev_label, args.batch_size, 0)

best_model = None
best_epoch = 0
best_acc_val = 0.
best_loss_val = -np.inf

init = tf.global_variables_initializer()
sess.run(init)     
lr = args.learning_rate
m = args.margin
for epoch in range(args.num_epochs):
    for iteration in range(1, n_iterations_per_epoch + 1):            
        X_batch, y_batch = mr_train.next()     
        y_batch = utils.to_categorical(y_batch, args.num_classes)

        _, loss_train, probs, capsule_pose = sess.run(
            [training_op, loss, activations, poses],
            feed_dict={X: X_batch[:,:args.max_sent],
                       y: y_batch,
                       learning_rate:lr,
                       margin:m})

        print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                  iteration, n_iterations_per_epoch,
                  iteration * 100 / n_iterations_per_epoch,
                  loss_train),
              end="")            

    loss_vals = []
    acc_vals = []
    for iteration in range(1, n_iterations_test + 1):
        X_batch, y_batch = mr_test.next()            
        y_batch = utils.to_categorical(y_batch, args.num_classes)
                    
        loss_val, acc_val = sess.run(
                [loss, accuracy],
                feed_dict={X: X_batch[:,:args.max_sent],
                           y: y_batch,
                           margin:m})
        loss_vals.append(loss_val)
        acc_vals.append(acc_val)
        print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                  iteration, n_iterations_test,
                  iteration * 100 / n_iterations_test),
              end=" " * 10)
    loss_val = np.mean(loss_vals)
    acc_val = np.mean(acc_vals)
    print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}, {:.4f}, {:.4f}{}".format(
        epoch + 1, acc_val * 100, loss_val, 0, 0,
        " (improved)" if loss_val < best_loss_val else ""))
    if loss_val < best_loss_val:
        best_loss_val = loss_val 
    lr = max(1e-6, lr * 0.3)
    m = min(0.9, m + 0.2)
