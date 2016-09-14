import numpy as np

import theano


class ClutteredMNIST(object):
	def __init__(self, path='data/mnist_cluttered_60x60_6distortions.npz', img_dim=60):
		data = np.load(path)
		X_train, y_train = data['x_train'], data['y_train']
		X_valid, y_valid = data['x_valid'], data['y_valid']
		X_test, y_test = data['x_test'], data['y_test']
				
		X_train = X_train.reshape((X_train.shape[0], img_dim, img_dim))
		X_valid = X_valid.reshape((X_valid.shape[0], img_dim, img_dim))
		X_test = X_test.reshape((X_test.shape[0], img_dim, img_dim))
		
		self.X_train, self.y_train = X_train, y_train
		self.X_valid, self.y_valid = X_valid, y_valid
		self.X_test, self.y_test = X_test, y_test

		self.img_dim = img_dim
		self.i = 0

	def fetch_train(self, batch_size):
		i = self.i
		X = self.X_train[i: i + batch_size]
		y = self.y_train[i: i + batch_size]
		i += batch_size
		if i >= self.X_train.shape[0]:
			p = range(self.X_train.shape[0])
			np.random.shuffle(p)
			self.X_train, self.y_train = self.X_train[p], self.y_train[p]
			i = 0
		self.i = i

		return X, y

	def fetch_validation(self):
		return self.X_valid, self.y_valid

	def fetch_test(self):
		return self.X_test, self.y_test
