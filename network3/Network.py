#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/04 09:42:12
#   Desc    :   learning python
#
from Layer import Layer
from Connections import Connections

class Network(object):
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0;
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                    for upstream_node in self.layers[]]
