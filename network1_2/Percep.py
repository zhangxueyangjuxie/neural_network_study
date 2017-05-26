#!/usr/bin/python   
#-*-coding:utf-8-*-
class Percep(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(
                reduce(lambda a,b: a+b,
                    map(lambda (x,w):x*w,
                        zip(input_vec,self.weights))
                ,0.0)+self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for (input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(output, input_vec, rate, label)

    def _update_weights(self, output, input_vec, rate, label):
        delta = label - output
        self.weights = map(
                lambda (x,w):w + rate * delta * x,
                zip(input_vec,self.weights))
        self.bias += rate * delta


def f(x):
    return 1 if x > 0 else 0

def get_train_dataset():
    input_vecs = [[0,0],[0,1],[1,0],[1,1]]
    labels = [0,0,0,1]
    return input_vecs,labels

def train_and_precision():
    p = Percep(2,f)
    input_vecs, labels = get_train_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__=='__main__':
    pre = train_and_precision()
    print pre

    print '1 and 1 = %d' % pre.predict([1,1])
    print '1 and 0 = %d' % pre.predict([1,0])
    print '0 and 1 = %d' % pre.predict([0,1])
    print '0 and 0 = %d' % pre.predict([0,0])








