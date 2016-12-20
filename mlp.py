#coding=utf-8

from __future__ import print_function

import sys
import os
import time
import matplotlib.pyplot as plt

import numpy as np
import theano
import theano.tensor as T
import theano.typed_list
import math

import lasagne
from mlp_util import load_data

def build_mlp(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None,  6),
                                     input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.001)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.001)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.001)

    l_hid3=lasagne.layers.DenseLayer(l_hid2_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal())
    l_hid3_drop = lasagne.layers.DropoutLayer(l_hid3, p=0.001)

    l_hid4=lasagne.layers.DenseLayer(l_hid3_drop, num_units=50,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal())
    l_hid4_drop = lasagne.layers.DropoutLayer(l_hid4, p=0.001)
    
    l_out = lasagne.layers.DenseLayer(
            l_hid4_drop, num_units=1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal())
			#  lasagne.nonlinearities.linear(x)

    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main(num_epochs=200):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')

    print("Building model and compiling functions...")
    network = build_mlp(input_var)

    prediction = lasagne.layers.get_output(network)
    #loss = lasagne.objectives.squared_error(prediction, target_var)
    #loss = loss.mean()
    loss= (abs(prediction-target_var)).mean()+ (lasagne.objectives.squared_error(prediction, target_var)).mean()
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.03, momentum=0.9)
    #updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.003, rho=0.05,epsilon=1e-6)
    #updates = lasagne.updates.adagrad(loss, params, learning_rate=0.05, epsilon=1e-6)
    #updates = lasagne.updates.adadelta(loss, params, learning_rate=0.05, rho=0.1,epsilon=1e-6)
    #updates = lasagne.updates.adam(loss, params, learning_rate=0.05, beta1=0.5, beta2=0.5, epsilon=1e-6)

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    #test_loss = lasagne.objectives.squared_error(test_prediction,target_var)
    #nominator = abs(test_prediction-target_var)
    #denominator = abs(target_var)

    test_loss=(abs(test_prediction-target_var)).mean()
    
    test_acc = T.mean(abs(test_prediction-target_var) > 0.1*abs(target_var), dtype = theano.config.floatX) 
   
    #outs= theano.typed_list.append(test_prediction, target_var) 
    
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc, test_prediction, target_var ])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    train_re=[]
    valid_re=[]
    valid_ac=[]
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        val_acc = 0
        predicted = []
        real = []
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=True):
            inputs, targets = batch
            err, acc, out, tar = val_fn(inputs, targets)
            predicted.append(out)
            real.append(tar)
            val_err += err
            val_acc += acc
            val_batches += 1
       
         
        # print(out)
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation error rate:\t\t{:.6f}".format(val_acc / val_batches))
        train_re.append(train_err / train_batches)
        valid_re.append(val_err / val_batches)
        valid_ac.append(val_acc / val_batches)
    #After training, we compute and print the test error:
    test_err = 0
    test_batches = 0
   
 
    print(predicted[0][0])
    print(real[0][0])

    e = 0.0
    absError = 0.0
    origLines = []
    nnLines = []
    with open('cmp.txt','w') as f:
      for i in xrange(len(predicted)):
        for j in xrange(len(predicted[0])):
          f.write("%f\t%f\n"%(predicted[i][j],real[i][j]))
          origLines.append(float(real[i][j]))
          nnLines.append(float(predicted[i][j]))

    cc = 0
    for i in range(len(origLines)):
      origPrice = float(origLines[i])
      nnPrice 	= float(nnLines[i])
      
      nominator   = abs(origPrice - nnPrice)
      denominator = abs(origPrice)
      if(denominator == 0):
        e = 1.0
      elif(math.isnan(nominator) or (math.isnan(denominator))):
        e = 1.0
      elif ((nominator / denominator > 1)):
        e = 1.0
      else:
        e = nominator / denominator
        cc += 1
      absError += e
    
    print("*** Error: %f" % (absError/float(len(origLines))))  
 
    print(len(origLines)) 
    print(cc)      
 
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=True):
        inputs, targets = batch
        err= val_fn(inputs, targets)
        test_err += err
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))

    


    #plot the loss with epoch
    '''
    plt.figure(1)
    line_up,=plt.plot(range(1,len(train_re)+1),train_re,'--*r',label='train loss')
    line_down,=plt.plot(range(1,len(valid_re)+1),valid_re, '--*g',label='valid loss')
    plt.xlabel("epoch")
    plt.ylabel("loss of train and valid")
    plt.ylim(0.03, 0.12)
    plt.title('loss with epoch')
    plt.legend(handles=[line_up,line_down])
    plt.show()
    '''
if __name__ == '__main__':
   main(num_epochs=200)
	
	
	
