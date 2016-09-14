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

import cPickle


LEARNING_RATE = 1e-3
BATCH_SIZE = 128
N_ITER = 20000

img_dim = 60
N = 8
num_states = 256
num_glimpses = 8

print "... deserializing model"
f = open('params.model', 'r')
params = cPickle.load(f)
f.close()

def gp_from_hidden(H, W_att, img_dim, N):
	gp = T.tanh(T.dot(W_att, H.T).T)
	center_y = gp[:, 0].dimshuffle(0, 'x')
	center_x = gp[:, 1].dimshuffle(0, 'x')
	delta = T.exp(gp[:, 2]).dimshuffle(0, 'x')
	sigma = T.exp(gp[:, 3] / 2.0).dimshuffle(0, 'x')
	gamma = T.exp(gp[:, 4]).dimshuffle(0, 'x', 'x')
	center_y = (img_dim + 1.0) * (center_y + 1.0) / 2.0
	center_x = (img_dim + 1.0) * (center_x + 1.0) / 2.0
	delta = (img_dim - 1.0) / (N - 1.0) * delta
	return center_y, center_x, delta, sigma, gamma

X = T.tensor3('input')
ip = InputLayer((None, img_dim, img_dim), input_var=X)
dram = DRAM(ip, num_states=num_states, img_dim=img_dim, N=N, num_glimpses=num_glimpses, fg_bias_init=0.0, Ws=params[:2], final_state_only=False)
hiddens = get_output(dram)

GPS_ = []
for i in range(num_glimpses):	
	gp = gp_from_hidden(hiddens[i], params[1], img_dim, N)
	GPS_.extend(gp)

states = theano.function([X], GPS_)

data_iter = ClutteredMNIST(img_dim=img_dim)

X_val, y_val = data_iter.fetch_validation()

p = np.random.random_integers(X_val.shape[0])

GPS = states(X_val[p:p+1])
GPS = np.array(GPS).reshape(8, 5)

print GPS

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from time import sleep

fig, ax = plt.subplots(1)

ax.imshow(X_val[p], extent=[0.0, img_dim, 0.0, img_dim])

for i in range(num_glimpses):
	width = GPS[i, 2] * N
	x_coord = GPS[i, 1] - width / 2.0
	y_coord = img_dim - GPS[i, 0] - width / 2.0
	rect = patches.Rectangle((x_coord, y_coord), width, width, linewidth=2*GPS[i, 3], edgecolor='g', facecolor='none')
	ax.add_patch(rect)
	plt.pause(0.25)
	rect.remove()

plt.show()
