#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/07 10:46:26
#   Desc    :   learning python
#

import numpy as np
from bean import nn,autoencoder

def aebuilder(nodes):
    layers = len(nodes)
    ae = autoencoder()
    for i in range(layers-1):
        ae.add_one(nn([nodes[i], nodes[i+1], nodes[i]]))
    return ae

def aetrain(ae, x, interations):
    elayers = len(ae.encoders)
    for i in range(elayers):
        ae.encoders[i] = nntrain(ae.encoders[i],x, x, interations)
        nntemp = nnff(ae.encoders[i],x, x)
        x = nntemp.values[1]
    return ae

def nntrain(nn, x, y, iterations):
    for i in range(iterations):
        nnff(nn,x,y)
        nnbp(nn)
    return nn

'''
def nnff(nn, x, y):
    layers = nn.layers
    numbers = x.shape[0]
    nn.values[0] = x
    for i in range(1, layers):
        nn.values[i] = sigmod(np.dot(nn.values[i-1],nn.W[i-1])+nn.B[i-1])
    for j in range(1,layers-1):
        nn.P[j] = nn.values[j].sum(axis=0)/(nn.values[j].shape[0])
    sparsity = nn.sparse*np.log(nn.sparse/nn.P[layers-2])
    +(1-nn.sparse)*np.log((1-nn.sparse)/(1-nn.P[layers-2]))
    nn.error = y - nn.values[layers-1]
    nn.loss = 1.0/2.0*(nn.error**2).sum()/numbers + nn.B*sparsity.sum()
    return nn
'''
def nnff(nn, x, y):
    layers = nn.layers
    numbers = x.shape[0]
    nn.values[0] = x
    for i in range(1, layers):
        nn.values[i] = sigmod(np.dot(nn.values[i-1],nn.W[i-1])+nn.B[i-1])
    for j in range(1, layers-1):
        nn.values[j] = nn.values[j]*(np.random.random(nn.values[j].shape) > nn.denoise)
    nn.error = y - nn.values[layers-1]
    nn.loss = 1.0/2.0*(nn.error**2).sum()/numbers
    return nn

def sigmod(x):
    return 1.0 / (1.0 + np.exp(-x))

def nnbp(nn):
    layers = nn.layers
    deltas = list()
    for i in range(layers):
        deltas.append(0)
    deltas[layers-1] = -nn.error*nn.values[layers-1]*(1-nn.values[layers-1])
    for j in range(1,layers-1)[::-1]:
        pj = np.ones([nn.values[j].shape[0],1])*nn.P[j]
        sparsity = nn.beta*(-nn.sparse/pj + (1-nn.sparse)/(1-pj))
        deltas[j] = (np.dot(deltas[j+1],nn.W[j].T)+sparsity) * nn.values[j] * (1-nn.values[j])
    for k in range(layers-1):
        nn.W[k] -= nn.u*np.dot(nn.values[k].T, deltas[k+1]) / (deltas[k+1].shape[0])
        nn.B[k] -= nn.u*deltas[k+1]/(deltas[k+1].shape[0])
    return nn

