import numpy as np

import theano
import theano.tensor as T

import lasagne
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import get_all_params, get_output
from lasagne.objectives import categorical_crossentropy
from lasagne.updates import adam

from dram import DRAM

from data import ClutteredMNIST


LEARNING_RATE = 1e-3
BATCH_SIZE = 128
N_ITER = 20000

img_dim = 60
N = 8
num_states = 256
num_glimpses = 16
fg_bias_init = 0.0

X = T.tensor3('input')
y = T.matrix('target')
ip = InputLayer((None, img_dim, img_dim), input_var=X)
dram = DRAM(ip, num_states=num_states, img_dim=img_dim, N=N, num_glimpses=num_glimpses, fg_bias_init=fg_bias_init)
y_hat = DenseLayer(dram, 10, nonlinearity=lasagne.nonlinearities.softmax)

prediction = get_output(y_hat)

loss = T.mean(categorical_crossentropy(prediction, y))
params = get_all_params(y_hat, trainable=True)
updates = adam(loss, params, learning_rate=LEARNING_RATE)

accuracy = T.mean(T.eq(T.argmax(prediction, axis=1), T.argmax(y, axis=1)), dtype=theano.config.floatX)

print "... begin compiling"

train_fn = theano.function([X, y], loss, updates=updates)
val_fn = theano.function([X, y], [loss, accuracy])

print "... done compiling"
print "... loading dataset"

data_iter = ClutteredMNIST(img_dim=img_dim)

print "... begin training"

smooth_train_loss = np.log(10)

for iter_n in xrange(1, N_ITER):
	X_train, y_train = data_iter.fetch_train(BATCH_SIZE)
	batch_train_loss = train_fn(X_train, y_train)
	smooth_train_loss = 0.95 * smooth_train_loss + 0.05 * batch_train_loss
	print 'iter: ', iter_n, "\t training loss:", smooth_train_loss
	if iter_n % 100 == 0:
		X_val, y_val = data_iter.fetch_validation()
		val_loss, val_acc = val_fn(X_val, y_val)
		print "====" * 20
		print "validation loss: \t", val_loss
		print "validation accuracy: \t", val_acc
		print "====" * 20

print "... training done"

print "... serializing model"

import cPickle

params = []
params.extend(dram.get_params())
params.extend(y_hat.get_params())

np_params = []
for param in params:
	np_params.append(param.get_value())

f = open('params.model', 'w')
cPickle.dump(np_params, f)
f.close()

print "... done serializing model"
print "... exiting ..."
