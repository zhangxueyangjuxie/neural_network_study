#!/usr/bin/env python
# -*- coding:utf-8 -*-
#   Author  :   xueyang
#   E-mail  :   1512935295@qq.com
#
#   Date    :   16/12/04 21:19:09
#   Desc    :   learning python
#

import numpy as np
import Filter

class ConvLayer(object):
    def __init__(self, input_width input_height,
            channel_number, filter_width,
            filter_height, filter_number,
            zero_padding, stride, activator,
            learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
                ConvLayer.calculate_output_size(
                        self.input_width, filter_width, zero_padding,
                        stride)
        self.output_height = \
                ConvLayer.calculate_output_size(
                        self.input_height, filter_height, zero_padding,
                        stride)
        self.output_array = np.zeros((self.filter_number,
            self.output_height, self.output_wight))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(filter_width,
                    filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

        @staticmethod
        def calculate_output_size(input_size,
                filter_size, zero_padding, stride):
            return (input_size - filter_size + 
                    2 * zero_padding) / stride + 1

        def forward(self, input_array):
            self.input_array = input_array
            self.padded_input_array = padding(input_array,
                    self.zero_padding)
            for f in range(self.filter_number):
                filter = self.filters[f]
                conv(self.padded_input_array,
                        filter.get_weights(), self.output_array[f],
                        self.stride, filter.get_bias())
                element_wise_op(self.output_array,
                        self.activator.forward)

        def bp_sensitivity_map(self, sensitivity_array, activator):
            expanded_array = self.expand_sensitivity_map(
                    sensitivity_array)
            expanded_width = expanded_array.shape[2]
            zp = (self.input_width + self.filters_width - 1
                    - expanded_width) / 2
            padded_array = padding(expanded_array, zp)
            self.delta_array = self.create_delta_array()
            for f in range(self.filter_number):
                filter = self.filters[f]
                flipped_weights = np.array(map(
                    lambda i: np.rot90(i,2),
                    filter.get_weights()))
                delta_array = self.create_delta_array()
                for d in range(delta_array.shape[0]):
                    conv(padded_array[f], flipped_weights[d],
                            delta_array[d], 1, 0)
                self.delta_array += delta_array
            derivative_array = np.array(self.input_array)
            element_wise_op(derivative_array,
                    activator.backward)
            self.delta_array *= derivative_array

        def expand_sensitivity_map(self, sensitivity_array):
            depth = sensitivity_array.shape[0]
            expanded_width = (self.input_width - 
                    self.filter_width + 2 * self.zero_padding + 1)
            expanded_height = (self.input_height - 
                    self.filter_height + 2 * self.zero_padding + 1)
            expand_array = np.zeros((depth, expanded_height,
                expanded_width))
            for i in range(self.output_height):
                for j in range(self.output_width):
                    i_pos = i * self.stride
                    j_pos = j * self.stride
                    expand_array[:,i_pos, j_pos] = \
                            sensitivity_array[:,i,j]
            return expand_array
        def create_delta_array(self):
            return np.zeros((self.channel_number,
                self.input_height, self.input_width))

        def bp_gradient(self, sensitivity_array):
            expanded_array = self.expand_sensitivity_map(
                    sensitivity_array)
            for f in range(self.filter_number):
                filter = self.filters[f]
                for d in range(filters.weights.shape[0]):
                    conv(self.padded_input_array[d],
                            expanded_array[f],
                            filter.weights_grad[d], 1, 0)
                filter.bias_grad = expanded_array[f].sum()

        def update(self):
            for filter in self.filters:
                filter.update(self.learning_rate)


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

def conv(input_array, kernel_array, output_array, stride, bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    if channel_number == 2:
        for i in range(output_height):
            for j in range(output_width):
                output_array[i][j] = (
                        get_patch(input_array, i, j, kernel_width,
                            kernel_height, stride) * kernel_array
                        ).sum() + bias
    elif channel_number == 3:
        depth = input_array.shape[0]
        for d in range(depth):
            for i in range(output_height):
                for j in range(output_width):
                    output_array[i][j] += (
                            get_patch(input_array[d], i, j, kernel_width,
                                kernel_height, stride) * kernel_array[d]
                            ).sum() + bias

def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[:,
                    zp : zp + input_height,
                    zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp : zp + input_height,
                    zp : zp + input_width] = input_array
            return padded_array












