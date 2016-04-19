import numpy as np
import theano
import cPickle as pickle

def transfer(data):
    result=[]
    if len(data)>1:
        for item in data:
            result.append(float(item))
    return result



def load_data(filepath):
    X_train=[]
    y_train=[]

    lines=open(filepath).read().split('\n')
    for line in lines:
        data=transfer(line.split(' '))
        if len(data)>1:
            X_train.append([data[:5],data[5:10],data[10:15]])
            y_train.append([data[15]])
    '''
    for i in range(100):
        X_train.append(np.random.randn(3,5))
        y_train.append(np.random.randn(1))
    '''
    return np.asarray(X_train,dtype=theano.config.floatX),np.asarray(y_train,dtype=theano.config.floatX)


if __name__=="__main__":
    X_train,y_train=load_data('data/RNN_train.txt')
    X_test,y_test=load_data('data/RNN_test.txt')
    with open('train.pkl','w')as f:
        pickle.dump((X_train,y_train),f)
    with open('test.pkl','w')as f:
        pickle.dump((X_test,y_test),f)