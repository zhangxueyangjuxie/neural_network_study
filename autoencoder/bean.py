#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/07 10:33:36
#   Desc    :   learning python
#

import numpy as np
class nn():
    def __init__(self, nodes):
        self.layers = len(nodes)
        self.nodes = nodes
        self.u = 1.0
        self.W = list()
        self.B = list()
        self.P = list()
        self.values = list()
        self.error = 0
        self.loss = 0
        self.sparse = 0.2
        self.beta = 2.0
        self.denoise = 0.2
        for i in range(self.layers-1):
            self.W.append(np.random.random((self.nodes[i],
                self.nodes[i+1])) - 0.5)
            self.B.append(0)
            self.P.append(0)
        for j in range(self.layers):
            self.values.append(0)

class autoencoder():
    def __init__(self):
        self.encoders = list()
    def add_one(self,nn):
        self.encoders.append(nn)

