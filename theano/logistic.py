#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16-12-24 下午8:49
# @Author  : xueyang
# @File    : logistic.py
# @Software: PyCharm Community Edition

import numpy as np
import theano
import theano.tensor as T
rng = np.random

N = 10
feats = 3
D = (rng.randn(N, feats).astype(np.float32), rng.randint(size=N,
                low=0, high=2).astype(np.float32))
x = T.matrix('x')
y = T.vector('y')

w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0., name='b')

p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1)
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])

prediction = p_1 > 0.5

train = theano.function(inputs=[x, y], outputs=[prediction, xent],
                        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

training_steps = 1000
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print err.mean()
