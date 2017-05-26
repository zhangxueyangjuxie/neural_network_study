#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/04 08:47:04
#   Desc    :   learning python
#

class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
                lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
                self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str = '\n\tdownstream' + downstream_str
