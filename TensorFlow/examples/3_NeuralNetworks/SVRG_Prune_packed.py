#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 14:14:23 2016

@author: simon
"""

from __future__ import print_function
import time
import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import heapq
from multiprocessing.dummy import Pool as ThreadPool

mnist = input_data.read_data_sets("/tmp/data/MNIST", one_hot=True)

# Parameters
learning_rate = 0.001
training_iters = 400000
#training_iters = 2000
#retraining_iters = 2000
valid_size = 20000
batch_size = 128
display_step = 2

# Network Parameters
dropout = 1 # Dropout, probability to keep units
n_hidden_1 = 300  # 1st layer num features
n_hidden_2 = 100  # 2nd layer num features
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['fc1']), biases['fc1']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_1, dropout)
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['fc2']), biases['fc2']))  # Hidden layer with RELU activation
    tf.nn.dropout(layer_2, dropout)
    return tf.matmul(layer_2, weights['out']) + biases['out'],layer_1,layer_2

# Store layers weight & bias
weights = {
    'fc1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=0.01))
}

biases = {
    'fc1': tf.Variable(tf.random_normal([n_hidden_1], stddev=0.01)),
    'fc2': tf.Variable(tf.random_normal([n_hidden_2], stddev=0.01)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=0.01))
}


# Construct model
pred,layer_1,layer_2= conv_net(x, weights, biases, keep_prob)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
config = tf.ConfigProto(device_count = {'GPU': 0})

with tf.Session(config=config) as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Final Testing Accuracy:", \
    sess.run(accuracy, feed_dict={x: mnist.test.images[:valid_size],
                                  y: mnist.test.labels[:valid_size],
                                  keep_prob: 1.}))
    w_layer1,w_layer2,w_layer3 = sess.run([weights['fc1'], weights['fc2'], weights['out']])
    b_layer1,b_layer2,b_layer3 = sess.run([biases['fc1'], biases['fc2'], biases['out']])
    layer2_input_train, layer2_label_train=sess.run([layer_1, layer_2],feed_dict={x: mnist.train.images})
    
#build up trained model
weights_trained = {
    'fc1': tf.Variable(w_layer1),
    'fc2': tf.Variable(w_layer2),
    'out': tf.Variable(w_layer3)
}

biases_trained = {
    'fc1': tf.Variable(b_layer1),
    'fc2': tf.Variable(b_layer2),
    'out': tf.Variable(b_layer3)
}        

pred_trained, _ , _ = conv_net(x, weights_trained, biases_trained, keep_prob)
cost_trained = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_trained, y))
# Evaluate model
correct_pred_trained = tf.equal(tf.argmax(pred_trained, 1), tf.argmax(y, 1))
accuracy_trained = tf.reduce_mean(tf.cast(correct_pred_trained, tf.float32))

init2 = tf.initialize_all_variables()
# retraining........
def get_th(wd1_out1, wd1_out2, pp):
    wd1_out = np.vstack((wd1_out1,wd1_out2.T))
    wd1_size = (wd1_out.shape[0])*(wd1_out.shape[1])
#    print('wd1_size: ',wd1_size)
    nsmallestList = heapq.nsmallest(int(pp*wd1_size), wd1_out.reshape(wd1_size)) 
    th = nsmallestList[-1]
    wei=[]
    wei += [wd1_out1*(wd1_out1>th)]
    wei += [wd1_out2*(wd1_out2>th)]
    return wei   
    
def layer_wise_retain_model(weights_layer2,layer2_input,layer2_label):
        layer2_output = tf.nn.relu(tf.add(tf.matmul(layer2_input, weights_layer2['w']), weights_layer2['b']))
        cost_layer2 = tf.reduce_sum(tf.reduce_mean((layer2_label-layer2_output)*(layer2_label-layer2_output),0))*tf.constant(0.5,dtype='float32')
        without_mean = tf.nn.l2_loss(layer2_label-layer2_output)
        return cost_layer2, without_mean
        
def l1_loss(tensor, weight=1.0, scope=None):
  """Define a L1Loss, useful for regularize, i.e. lasso.
  Args:
    tensor: tensor to regularize.
    weight: scale the loss by this factor.
    scope: Optional scope for op_scope.
  Returns:
    the L1 loss op.
  """
  with tf.op_scope([tensor], scope, 'L1Loss'):
    weight = tf.convert_to_tensor(weight,
                                  dtype=tensor.dtype.base_dtype,
                                  name='loss_weight')
    loss = tf.mul(weight, tf.reduce_sum(tf.abs(tensor)), name='value')
#    tf.add_to_collection(LOSSES_COLLECTION, loss)
    return loss    
    
    
def layer_wise_transfer(w_layer2, b_layer2, layer2_input_train, layer2_label_train, retraining_iters=50, pp=0.9):   
    
    ######################################################
    retrain_batch_size = 550
    data_size=layer2_input_train.shape[0]
    m=80
    display_step=1
    ##########################################################
    n_hidden_1 = w_layer2.shape[0]
    n_hidden_2 = w_layer2.shape[1]
    weights_layer2_outer = {
                   'w': tf.Variable(w_layer2),
                   'b': tf.Variable(b_layer2)}
    weights_layer2_inner = {
                   'w': tf.Variable(w_layer2),
                   'b': tf.Variable(b_layer2)}
    
    layer2_input=tf.placeholder(tf.float32, [None, n_hidden_1])
    layer2_label=tf.placeholder(tf.float32, [None, n_hidden_2])
    full_grad = [tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])), tf.Variable(tf.random_normal([n_hidden_2]))]
    real_grad = [[tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),weights_layer2_inner['w']]
                  ,[tf.Variable(tf.random_normal([n_hidden_2])),weights_layer2_inner['b']]]
    w_grad_mode = tf.Variable(0.01)
    b_grad_mode = tf.Variable(0.01)
                    
    
    
    cost_layer2_outer,cost_layer2_outer_withoutmean = layer_wise_retain_model(weights_layer2_outer, layer2_input , layer2_label)
    cost_layer2_inner,cost_layer2_inner_withoutmean = layer_wise_retain_model(weights_layer2_inner, layer2_input , layer2_label)
    
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    
    part_grad_and_vars_inner = opt.compute_gradients(cost_layer2_inner, 
                                                     [weights_layer2_inner['w'],weights_layer2_inner['b']])
    part_grad_and_vars_outer = opt.compute_gradients(cost_layer2_outer, 
                                                     [weights_layer2_outer['w'],weights_layer2_outer['b']])
    real_grad[0][0] =  part_grad_and_vars_inner[0][0]-part_grad_and_vars_outer[0][0]+full_grad[0]
    real_grad[1][0] =  part_grad_and_vars_inner[1][0]-part_grad_and_vars_outer[1][0]+full_grad[1]
    
    w_grad_mode=l1_loss(real_grad[0][0])
    b_grad_mode=tf.nn.l2_loss(real_grad[1][0])
    
    optimizer_layer2 = opt.apply_gradients(real_grad)
    w_ass = tf.placeholder(tf.float32, [n_hidden_1, n_hidden_2])
    b_ass = tf.placeholder(tf.float32, [n_hidden_2])
    
    assign_w_inner = weights_layer2_inner['w'].assign(w_ass)
    assign_b_inner = weights_layer2_inner['b'].assign(b_ass)
    assign_w_outer = weights_layer2_outer['w'].assign(w_ass)
    assign_b_outer = weights_layer2_outer['b'].assign(b_ass)
    
    init1 = tf.initialize_all_variables()
    with tf.Session(config=config) as sess:
        sess.run(init1)
        iters = 0
        k=0
    #    a=sess.run(part_grad_and_vars_inner,feed_dict={layer2_input: layer2_input_train,
    #                                                 layer2_label: layer2_label_train})
#        b=sess.run(weights_layer2_inner['w'])
#        c=sess.run(cost_layer2_inner,feed_dict={layer2_input: layer2_input_train,
#                                                     layer2_label: layer2_label_train})
    #    print sess.run(part_grad_and_vars_inner,feed_dict=feed)
        while iters < retraining_iters:
#            a=sess.run(part_grad_and_vars_inner,feed_dict={layer2_input: layer2_input_train,
#                                                     layer2_label: layer2_label_train})
            full_grad_and_vars = sess.run(opt.compute_gradients(cost_layer2_outer, 
                                                                [weights_layer2_outer['w'],weights_layer2_outer['b']]),
                                          feed_dict={layer2_input: layer2_input_train,
                                                     layer2_label: layer2_label_train})
            sess.run(full_grad[0].assign(full_grad_and_vars[0][0]))
            sess.run(full_grad[1].assign(full_grad_and_vars[1][0]))
            for i in range(m):
                
    #            k =  random.randint(1,layer2_input_train.shape[0]/retrain_batch_size)
                k = k%(data_size/retrain_batch_size)
                k+=1
                feed = {layer2_input: layer2_input_train[(k-1)*retrain_batch_size:k*retrain_batch_size],
                        layer2_label: layer2_label_train[(k-1)*retrain_batch_size:k*retrain_batch_size]}
                
    #            part_grad_and_vars_inner = sess.run(opt.compute_gradients(cost_layer2_inner, 
    #                                                [weights_layer2_inner['w'],weights_layer2_inner['b']]),
    #                                                feed_dict = feed)
    #            part_grad_and_vars_outer = sess.run(opt.compute_gradients(cost_layer2_outer, 
    #                                                [weights_layer2_outer['w'],weights_layer2_outer['b']]),
    #                                                feed_dict = feed)
    #            real_grad = []
    #            for j in range(len(full_grad_and_vars)):
    #                real_grad[j] = 
    #            print(time.time())
                sess.run(optimizer_layer2, feed_dict = feed)
                print('\nw_grad:',sess.run(w_grad_mode, feed_dict = feed))
    #            print('\n before H cost:', sess.run(cost_layer2_inner,feed_dict={layer2_input: layer2_input_train,
    #                                                 layer2_label: layer2_label_train}))
    #            print('start preparation', time.time())
                
                new_w=get_th(sess.run(weights_layer2_inner['w']),sess.run(weights_layer2_inner['b']),pp)
    #            print('end preparation', time.time())
                sess.run([assign_b_inner, assign_w_inner],feed_dict={w_ass: new_w[0], b_ass: new_w[1]})
    #            sess.run(weights_layer2_inner['b'].assign(new_w[1]))
                
                print('Inner Loop:',i,'(',iters,')','after H cost:', 
                      sess.run(cost_layer2_inner,feed_dict={layer2_input: layer2_input_train,
                                                     layer2_label: layer2_label_train}))
    #        
    #            if i%1==0:
    #                print('Inner Done.',i,'(',iters,')')
            sess.run([assign_b_outer, assign_w_outer],feed_dict={w_ass: new_w[0], b_ass: new_w[1]})
            
            
#            if iters % display_step == 0:
#                # Calculate batch loss and accuracy
#                loss = sess.run(cost_layer2_outer, feed_dict={layer2_input: layer2_input_train,
#                                                 layer2_label: layer2_label_train})
#                print("Iter " + str(iters) + ", Loss= " + \
#                      "{:.6f}".format(loss))
    
            iters+=1
    #        d=sess.run(part_grad_and_vars_inner,feed_dict={layer2_input: layer2_input_train,
    #                                                 layer2_label: layer2_label_train})
    #        e=sess.run(cost_layer2_inner_withoutmean,feed_dict={layer2_input: layer2_input_train,
    #                                                 layer2_label: layer2_label_train})
    #        f=sess.run(cost_layer2_outer_withoutmean,feed_dict={layer2_input: layer2_input_train,
    #                                                 layer2_label: layer2_label_train})
    #        g=sess.run(cost_layer2_outer,feed_dict={layer2_input: layer2_input_train,
    #                                                 layer2_label: layer2_label_train})
    return new_w, loss
    
    
    
    
layer1_w_new,layer1_loss =  layer_wise_transfer(w_layer1, b_layer1, mnist.train.images, 
                                                layer2_input_train, 50, 0.9)   

layer2_w_new,layer2_loss =  layer_wise_transfer(w_layer2, b_layer2, layer2_input_train, 
                                                layer2_label_train, 50, 0.9)   
with tf.Session(config=config) as sess:
    sess.run(init2)
    print("Final Testing Accuracy111:", \
    sess.run(accuracy_trained, feed_dict={x: mnist.train.images,
                                  y: mnist.train.labels,
                                  keep_prob: 1.}))
    
    sess.run(weights_trained['fc2'].assign(layer2_w_new[0])) 
    sess.run(biases_trained['fc2'].assign(layer2_w_new[1])) 
    sess.run(weights_trained['fc1'].assign(layer1_w_new[0])) 
    sess.run(biases_trained['fc1'].assign(layer1_w_new[1])) 
    print("Final Testing Accuracy222:", \
    sess.run(accuracy_trained, feed_dict={x: mnist.train.images,
                                  y: mnist.train.labels,
                                  keep_prob: 1.}))
    
    
    
    
#pall
#pool = ThreadPool()


#def transfer_function_packed(dictionary):
#    layer1_w_new,layer1_loss  =  layer1_w_new,layer1_loss = layer_wise_transfer(dictionary['w'], 
#                                                   dictionary['b'], 
#                                                   dictionary['in'], 
#                                                   dictionary['out'], 
#                                                   dictionary['iters'], 
#                                                   dictionary['pp'])
#    return layer1_w_new,layer1_loss    
#dict_layer1 = {'w': w_layer1,
#               'b': b_layer1,
#               'in': mnist.train.images,
#               'out': layer2_input_train,
#               'iters': 50,
#               'pp':0.9
#    }
#    
#dict_layer2 = {'w': w_layer2,
#               'b': b_layer2,
#               'in':layer2_input_train ,
#               'out': layer2_label_train,
#               'iters': 50,
#               'pp':0.9
#    }
#    
#results = pool.map(transfer_function_packed,[dict_layer1,dict_layer2])
#pool.close()
#pool.join()


#with tf.Session(config=config) as sess:
#    sess.run(init2)
#    print("Final Testing Accuracy111:", \
#    sess.run(accuracy_trained, feed_dict={x: mnist.train.images,
#                                  y: mnist.train.labels,
#                                  keep_prob: 1.}))
#    
#    sess.run(weights_trained['fc2'].assign(results[0][0][0])) 
#    sess.run(biases_trained['fc2'].assign(results[0][0][1])) 
#    sess.run(weights_trained['fc1'].assign(results[1][0][0])) 
#    sess.run(biases_trained['fc1'].assign(results[1][0][1])) 
#    print("Final Testing Accuracy222:", \
#    sess.run(accuracy_trained, feed_dict={x: mnist.train.images,
#                                  y: mnist.train.labels,
#                                  keep_prob: 1.}))








