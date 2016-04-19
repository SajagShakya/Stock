import time
import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import sigmoid
from utils import load_data
import cPickle as pickle


class simpleRNN(object):
    def __init__(self,n_input=5,n_hidden=10, n_output=1):
        numpy_rng = numpy.random.RandomState(123)
        self.n_input=n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        initial_Wx=numpy.asarray(
                numpy_rng.uniform(
                    low=-1*numpy.sqrt(6./(self.n_input+self.n_hidden)),
                    high=1*numpy.sqrt(6./(self.n_input+self.n_hidden)),
                    size=(self.n_input,self.n_hidden)),
                dtype=theano.config.floatX)
        self.Wx=theano.shared(value=initial_Wx,name='Wx',borrow=True)

        initial_Wh = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_hidden)),
                    size=(self.n_hidden, self.n_hidden)),
                dtype=theano.config.floatX)
        self.Wh = theano.shared(value=initial_Wh, name='Wh', borrow=True)

        initial_W = numpy.asarray(numpy_rng.uniform(
                    low=-1 * numpy.sqrt(6. / (self.n_hidden+self.n_output)),
                    high=1 * numpy.sqrt(6. / (self.n_hidden+self.n_output)),
                    size=(self.n_hidden, self.n_output)),
                dtype=theano.config.floatX)
        self.W = theano.shared(value=initial_W, name='W', borrow=True)

        initial_bh = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_hidden)),
                    high=4 * numpy.sqrt(6. / (self.n_hidden)),
                    size=(self.n_hidden,)),
                dtype=theano.config.floatX)
        self.bh = theano.shared(value=initial_bh, name='bh', borrow=True)

        initial_b = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (self.n_output)),
                    high=4 * numpy.sqrt(6. / (self.n_output)),
                    size=(self.n_output,)),
                dtype=theano.config.floatX)
        self.b = theano.shared(value=initial_b, name='b', borrow=True)

        self.params = [self.Wx,self.Wh, self.W,self.bh,self.b]

    def encode(self, x):
        h0 = theano.shared(value=np.zeros(self.n_hidden, dtype=theano.config.floatX))

        def _recurrence(x_t, h_tm1):
            h_t = sigmoid(T.dot(x_t , self.Wx ) + T.dot(h_tm1 , self.Wh) +self.bh)
            o_t = sigmoid(T.dot(h_t , self.W ) + self.b)
            return [o_t,h_t]

        [o,h], _ = theano.scan(fn = _recurrence,
                               sequences = x,
                               outputs_info = [None,h0])
        return o[-1]

    def build(self, lr,x,y):
        predict = self.encode(x)
        cost = T.mean(T.square(predict - y))

        # calculate the gradient
        gparams = T.grad(cost, self.params)
        updates=[(param, param - lr * gparam) for param, gparam in zip(self.params, gparams)]

        self.train_model = theano.function(
            inputs=[x,y],
            outputs=cost,
            updates=updates)

        self.predict_model=theano.function([x],predict)

def train_with_sgd(lr=0.1,n_epochs=150, print_every = 50,
            X_train=None,y_train=None,X_test=None,y_test=None):

    x = T.fmatrix('x')
    y = T.fvector('y')

    rnn = simpleRNN()
    rnn.build(lr=lr,x=x,y=y)

    # Training...
    print('Training...')
    start_time = time.clock()
    for epoch in range(n_epochs):
        cost_history = []
        for index in range(len(X_train)):
            cost= rnn.train_model(X_train[index],y_train[index])
            cost_history.append(cost)
            if index % print_every == 0:
                print 'Iteration %d,average cost %f' % (index, cost)
                cost_history = []

    training_time = (time.clock() - start_time)
    print 'Finished training %d epochs, took %d seconds' % (n_epochs, training_time)

    # Testing...
    print('Testing...')
    results=[]
    for index in range(len(X_test)):
        results.append(rnn.predict_model(X_test[index]))
    return results

if __name__ == '__main__':
    X_train,y_train=load_data('data/RNN_train.txt')
    X_test,y_test=load_data('data/RNN_test.txt')
    theano.compile.mode.Mode(linker='cvm', optimizer='fast_run')

    results=train_with_sgd(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,n_epochs=150,lr=0.1)
    with open('results.pkl','w')as f:
        pickle.dump(results,f)
