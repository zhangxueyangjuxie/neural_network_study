#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/04 09:20:23
#   Desc    :   learning python
#

class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        return self.gradient

    def update_weight(self, rete):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self)
	return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index, 
            self.upstream_node.node_index,
            self.downstream_node.layer_index, 
            self.downstream_node.node_index, 
            self.weight)
