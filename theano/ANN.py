#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16-12-25 上午9:33
# @Author  : xueyang
# @File    : ANN.py
# @Software: PyCharm Community Edition

import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data

class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, w=None, b=None,
                 activation=T.tanh):
        self.input = input
        if w is None:
            w_value = np.asarray(
                rng.uniform(
                    low=np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                w_value *= 4
            w = theano.shared(value=w_value, name='w', borrow=True)

        if b is None:
            b_value = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_value, name='b', borrow=True)

        self.w = w
        self.b = b
        lin_output=T.dot(input, self.w) + self.b
        self.output=(
            lin_output if activation is None
            else activation(lin_output)
        )
        self.params=[self.w, self.b]

class MLP(object):

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        self.hiddenLayer=HiddenLayer(rng=rng, input=input, n_in=n_in,
                                     n_out=n_hidden, activation=T.tanh)
        self.logRegressionLayer=LogisticRegression(input=self.hiddenLayer.output,
                                                   n_in=n_hidden, n_out=n_out)
        self.L1 = (abs(self.hiddenLayer.w).sum() + abs(self.logRegressionLayer.W).sum()+
                   abs(self.logRegressionLayer.W).sum())
        self.L2_sqr = ((self.hiddenLayer.w ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())
        self.negative_log_likelihood=self.logRegressionLayer.negative_log_likelihood
        self.errors = self.logRegressionLayer.negative_log_likelihood
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        self.input = input

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches =train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    b_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    rng = np.random.RandomState(1234)

    classifier = MLP(rng=rng, input=x, n_in=28 * 28, n_hidden=n_hidden, n_out=10)
    cost=(classifier.negative_log_likelihood(y) + L1_reg * classifier.L1 + L2_reg * classifier.L2_sqr)
    gparams = [T.grad(cost, param) for param in classifier.params]
    updatas = [(param, param - learning_rate * gparam) for param, gparam in zip(classifier.params, gparams)]
    train_model = theano.function(inputs=[index], outputs=cost, updates=updatas,
                                  givens={x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                          y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    epoch=0
    while(epoch < 10):
        cost = 0
        for minibatch_index in xrange(n_train_batches):
            cost += train_model(minibatch_index)
        print 'epoch:', epoch, '    error:', cost/n_train_batches
        epoch = epoch + 1

test_mlp()

