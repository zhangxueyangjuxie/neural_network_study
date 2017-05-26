#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/04 13:55:04
#   Desc    :   learning python
#

from Network import Network 

def gradient_check(network, sample_feature, sample_label):
    network_error = lambda vec1, vec2: \
            0.5 * reduce(lambda a, b: a + b,
                    map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                        zip(vec1, vec2)))

    network.get_gradient(sample_feature, sample_label)
    for conn in network.connections.connections:
        actual_gradient = conn.get_gradient()

        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        conn.weight -=2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)

        expected_gradient = (error2 - error1) / (2 * epsilon)
        print 'expected gradient: \t%f\nactual gradient: \t%f' % ( \ 
                expected_gradient, actual_gradient)
