#coding=utf-8

import numpy as np
import theano
#from sklearn import preprocessing

import gc #(garbage collector)
##del a
##gc.collect() 

def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, shared_y

def mean_std(x):
     std=x.std(axis=0)
     std=np.asarray([i if(i!=0) else 1 for i in std])
     x=np.asarray((x-x.mean(axis=0))/std,dtype='float32')
     return x

def min_max(x):
        Min=x.min(axis=0)
        Max=x.max(axis=0)
        diff=Max-Min
        diff=np.asarray([i if(i!=0) else 1 for i in diff])
        x=np.asarray((x-Min)/diff,dtype='float32')
        return x

def load_data():
        path="data/"
        x=np.loadtxt(path+"train_x.txt")
        x=min_max(x)
        y=np.loadtxt(path+"train_y.txt")
        y=min_max(y)
        train_set=(x[:50000],y[:50000])
        valid_set=(x[50001:60000],y[50001:60000])
        test_set=(x[60001:70000],y[60001:70000])
        del x,y
        gc.collect()
##        train_set=(np.loadtxt("E:\\nie\\cheng-work\\train_x.txt"),np.loadtxt("E:\\nie\\cheng-work\\train_y.txt"))
##	valid_set=(np.loadtxt("E:\\nie\\cheng-work\\valid_x.txt"),np.loadtxt("E:\\nie\\cheng-work\\valid_y.txt"))
##	test_set=(np.loadtxt("E:\\nie\\cheng-work\\test_x.txt"),np.loadtxt("E:\\nie\\cheng-work\\test_y.txt"))

	#test_set_x, test_set_y = shared_dataset(test_set)
	#valid_set_x, valid_set_y = shared_dataset(valid_set)
	#train_set_x, train_set_y = shared_dataset(train_set)
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y =train_set
	rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),(test_set_x, test_set_y)]
	#return rval
	return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y 
