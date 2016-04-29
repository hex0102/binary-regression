# Lasagne-for-Regression   

Deep Neural Networks for Regression Using Lasagne   

data:   
x: 1*44 vector   
y: 1*2  vector

using mlp fitting,    
the loss function  mean(abs(pred-target)/abs(target))   

the best result is 5% relative error

note:   
1. learning rate is import to this task, and we set learningRate=0.04 which is the best    
2. drop rate can't be too large, which may be harmful to the model, since the input dimension is only 44,in this task we using dropout rate=0.001    
3.activation functions of relu may not help to improve reduce the loss     
4. more data is good for this task
