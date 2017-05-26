#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/07 13:57:31
#   Desc    :   learning python
#

import util
from bean import autoencoder, nn
import numpy as np

x = np.array([[0,0,1,0,0],
            [0,1,1,0,1],
            [1,0,0,0,1],
            [1,1,1,0,0],
            [0,1,0,1,0],
            [0,1,1,1,1],
            [0,1,0,0,1],
            [0,1,1,0,1],
            [1,1,1,1,0],
            [0,0,0,1,0]])
y = np.array([[0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [0],
            [1],
            [1],
            [0]])

nodes = [5,3,2]
ae = util.aebuilder(nodes)
ae = util.aetrain(ae, x, 6000)
nodescomplete = np.array([5,3,2,1])
aecomplete = nn(nodescomplete)
for i in range(len(nodescomplete) - 2):
    aecomplete.W[i] = ae.encoders[i].W[0]
aecomplete = util.nntrain(aecomplete, x, y, 3000)
print aecomplete.values[3]

