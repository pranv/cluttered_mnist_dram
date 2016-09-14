import numpy as np

import theano
import theano.tensor as T

import lasagne


def batched_dot(A, B):
    C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])      
    return C.sum(axis=-2)


class Attender(object):
    def __init__(self, img_dim, N):
        self.img_dim = img_dim
        self.N = N

    def read(self, I, center_y, center_x, delta, sigma, gamma):
		N = self.N
		batch_size = I.shape[0]

		rng = T.arange(N, dtype=theano.config.floatX) - N / 2.0 + 0.5  

		muX = center_x + delta * rng
		muY = center_y + delta * rng

		a = T.arange(self.img_dim, dtype=theano.config.floatX)
		b = T.arange(self.img_dim, dtype=theano.config.floatX)

		FX = T.exp(-(a - muX.dimshuffle([0, 1, 'x'])) ** 2 / 2.0 / sigma.dimshuffle([0, 'x', 'x']) ** 2)
		FY = T.exp(-(b - muY.dimshuffle([0, 1, 'x'])) ** 2 / 2.0 / sigma.dimshuffle([0, 'x', 'x']) ** 2)
		FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + 1e-4)
		FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + 1e-4)

		G = gamma * batched_dot(batched_dot(FY, I), FX.transpose([0, 2, 1]))

		fx = FX.sum(axis=1).dimshuffle(0, 'x', 1)
		fy = FY.sum(axis=1).dimshuffle(0, 'x', 1)
		A = gamma * batched_dot(fy.transpose([0, 2, 1]), fx)
		
		return G, A


def orthogonal(shape):
	"""
	taken from: https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py#L327-L367
	"""
	a = np.random.normal(0.0, 1.0, shape)
	u, _, v = np.linalg.svd(a, full_matrices=False)
	q = u if u.shape == shape else v 	# pick the one with the correct shape
	return q.astype(theano.config.floatX)


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


class DRAM(lasagne.layers.Layer):
	def __init__(self, incoming, num_states, img_dim, N, 
					num_glimpses, fg_bias_init, Ws=None, final_state_only=True,**kwargs):
		super(DRAM, self).__init__(incoming, **kwargs)
		
		# orthogonal for input -> hidden
		# identity for hidden -> hidden
		# all biases except forget gate 0
		num_input = 2 * (N ** 2)
		if Ws is None:
			W = np.zeros((num_states * 4, num_input + num_states + 1), dtype=theano.config.floatX)
			for i in range(4):
				W[i*num_states:(i + 1) * num_states, :num_input] = orthogonal((num_states, num_input))
				W[i*num_states:(i + 1) * num_states, num_input:-1] = orthogonal((num_states, num_states))
			W[2 * num_states:3 * num_states, -1] = fg_bias_init

			Wg = np.random.randn(5, num_states) * 0.01
		else:
			W, Wg = Ws

		W = W.astype(theano.config.floatX)
		Wg = Wg.astype(theano.config.floatX)
		
		self.W = self.add_param(W, (num_states * 4, num_input + num_states + 1), name='W')
		self.Wg = self.add_param(Wg, (5, num_states), name='Wg')
		
		self.img_dim = img_dim
		self.N = N
		self.attender = Attender(img_dim, N)
		
		self.num_glimpses = num_glimpses
		self.num_states = num_states
		self.final_state_only = final_state_only

	def get_output_for(self, input, **kwargs):
		""" 
			takes in (batch_size, 2 * height, 2 * width)
			gives out (batch_size, num_states)
		"""
		batch_size = input.shape[0]
		num_states = self.num_states
		img_dim = self.img_dim
		N = self.N
		attender = self.attender
		theano.gradient.grad_clip

		def step(c_tm1, h_tm1, att_acc_tm1, input, W, Wg):
			center_y, center_x, delta, sigma, gamma = gp_from_hidden(h_tm1, Wg, img_dim, N)
			g, att = attender.read(input, center_y, center_x, delta, sigma, gamma) # (batch_size, N, N) and (batch_size, img_dim, img_dim)
		
			att_acc_t = T.clip(att_acc_tm1 + att, 0.0, 1.0)	# (batch_size, img_dim, img_dim)
			r = input[:, :, :img_dim] * (1.0 - att_acc_t) # (batch_size, img_dim, img_dim)
			R , _ = attender.read(r, *gp_from_hidden(T.zeros((batch_size, 5)), T.eye(5), img_dim, N)) # (batch_size, N, N)
			
			flat_g = g.reshape((batch_size, N * N)) # (batch_size, N * N)
			flat_R = R.reshape((batch_size, N * N)) # (batch_size, N * N)
			
			# concatenate gA, gB and h_tm1 to form a single matrix # (batch_size, N * N + N * N + num_states + 1)
			lstm_inp = T.concatenate([flat_g, flat_R, h_tm1, T.ones((batch_size, 1))], axis=1)

			# multiply by LSTM weights
			# (num_states * 4, num_input + num_states + 1) dot (batch_size, N * N + N * N + num_states + 1).T
			pre_act = T.dot(W, lstm_inp.T) 	# (4 * num_states, batch_size)

			# split up to get individual gates
			z = T.tanh(pre_act[0*num_states:1*num_states]) # (num_states, batch_size)
			i = T.nnet.sigmoid(pre_act[1*num_states:2*num_states])
			f = T.nnet.sigmoid(pre_act[2*num_states:3*num_states])
			o = T.nnet.sigmoid(pre_act[3*num_states:4*num_states])

			# do LSTM update
			c_t = f * c_tm1.T + i * z
			h_t = o * T.tanh(c_t)

			return c_t.T, h_t.T, att_acc_t	# 1, 2: (batch_size, num_states); 3, 4: (batch_size, img_dim, img_dim)

		c0 = T.zeros((batch_size, num_states))
		h0 = T.zeros((batch_size, num_states))
		att_acc0 = T.zeros((batch_size, img_dim, img_dim))
		
		cells, hiddens, att_acc_T = theano.scan(fn=step, non_sequences=[input, self.W, self.Wg], outputs_info=[c0, h0, att_acc0], 
										n_steps=self.num_glimpses, strict=True)[0]
		if self.final_state_only:
			return hiddens[-1]
		else:
			return hiddens

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.num_states)