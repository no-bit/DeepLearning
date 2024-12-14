import inspect
import os

import numpy as np
import tensorflow as tf
import time

class earlyFusionNetwork:
    def __init__(self):
         print("class created")
         
    def build(self, im1, nclass, keep_prob):
	
        start_time = time.time()
        print("build model started")

        self.conv_1_1 = self.convBatchnormRelu(im1, 16, 3, name= 'conv_1_1')
        self.conv_1_2 = self.convBatchnormRelu(self.conv_1_1, 16, 3, name= 'conv_1_2')
        
        self.maxPool_1= self.max_pool(self.conv_1_2, name='max_pool_1')
        
        self.conv_2_1 = self.convBatchnormRelu(self.maxPool_1, 32, 3, name= 'conv_2_1')
        self.conv_2_2 = self.convBatchnormRelu(self.conv_2_1, 32, 3, name= 'conv_2_2')
        self.conv_2_3 = self.convBatchnormRelu(self.conv_2_2, 32, 3, name= 'conv_2_3')
        self.maxPool_2= self.max_pool(self.conv_2_3, name='max_pool_2')
        self.full_one_dropout_1 = self.dropout(self.maxPool_2, keep_prob=keep_prob)
        
        self.conv_3_1 = self.convBatchnormRelu(self.full_one_dropout_1, 64, 3, name= 'conv_3_1')
        self.conv_3_2 = self.convBatchnormRelu(self.conv_3_1, 64, 3, name= 'conv_3_2')
        self.maxPool_3= self.max_pool(self.conv_3_2, name='max_pool_3')
        
        self.conv_4_1 = self.convBatchnormRelu(self.maxPool_3, 128, 3, name= 'conv_4_1')
        self.conv_4_2 = self.convBatchnormRelu(self.conv_4_1, 128, 3, name= 'conv_4_2')
        self.maxPool_4= self.max_pool(self.conv_4_2, name='max_pool_4')
        
        self.convT_1 = self.convTranspose2d(self.maxPool_4, 128, 128, name='convT_1')
        self.concat_1 = self.concat(self.convT_1, self.conv_4_2, ax=-1)
        
        self.deconv_1_1 = self.convBatchnormRelu(self.concat_1, 128, 3, name= 'deconv_1_1')
        self.deconv_1_2 = self.convBatchnormRelu(self.deconv_1_1, 64, 3, name= 'deconv_1_2')
        self.convT_2= self.convTranspose2d(self.deconv_1_2, 64, 64, name='convT_2')
        self.concat_2 = self.concat(self.convT_2, self.conv_3_2, ax=-1)
        self.full_one_dropout_2 = self.dropout(self.concat_2, keep_prob=keep_prob)
        
        self.deconv_2_1 = self.convBatchnormRelu(self.full_one_dropout_2, 64, 3, name= 'deconv_2_1')
        self.deconv_2_2 = self.convBatchnormRelu(self.deconv_2_1, 32, 3, name= 'deconv_2_2')
        self.convT_3= self.convTranspose2d(self.deconv_2_2, 32, 32, name='convT_3')
        self.concat_3 = self.concat(self.convT_3, self.conv_2_2, ax=-1)
        
        self.deconv_3_1 = self.convBatchnormRelu(self.concat_3, 32, 3, name= 'deconv_3_1')
        self.deconv_3_2 = self.convBatchnormRelu(self.deconv_3_1, 16, 3, name= 'deconv_3_2')
        self.convT_4= self.convTranspose2d(self.deconv_3_2, 16, 16, name='convT_4')
        self.concat_4 = self.concat(self.convT_4, self.conv_1_2, ax=-1)
        self.full_one_dropout_3 = self.dropout(self.concat_4, keep_prob=keep_prob)
        
        self.deconv_4_1 = self.convBatchnormRelu(self.full_one_dropout_3, 16, 3, name= 'deconv_4_1')
        
        self.full_one_dropout_4 = self.dropout(self.deconv_4_1, keep_prob=keep_prob)
        
        self.deconv_4_2 = self.convBatchnormRelu(self.full_one_dropout_4, 8, 3, name= 'deconv_4_2')
        
        self.full_one_dropout_5 = self.dropout(self.deconv_4_2, keep_prob=keep_prob)
        
        self.deconv_4_3 = self.convBatchnormRelu(self.full_one_dropout_5, 2, 3, name= 'deconv_4_3')
        
        print(("build model finished: %ds" % (time.time() - start_time)))

    def concat(self, layer1, layer2, ax=-1):
        return tf.compat.v1.concat([layer1, layer2], ax)  
    
    def avg_pool(self, bottom, name):
        return tf.compat.v1.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.compat.v1.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	
    def conv2d(self, input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
            conv = tf.compat.v1.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

            if isbias == True:
                biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
                conv = tf.compat.v1.nn.bias_add(conv, biases)
			
        return conv
	
    def conv2dRelu(self, input_, output_dim, kernel=5, stride=2, stddev=0.02, name="conv2d", padding='SAME', isbias=True):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable('w', [kernel, kernel, input_.get_shape()[-1], output_dim], initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev))
            conv = tf.compat.v1.nn.conv2d(input_, w, strides=[1, stride, stride, 1], padding=padding)

            if isbias == True:
                biases = tf.compat.v1.get_variable('biases', [output_dim], initializer=tf.compat.v1.constant_initializer(0.0))
                conv = tf.compat.v1.nn.bias_add(conv, biases)
		
        conv = self.relu(conv, name=name + "relu")
        return conv	
        
    def convBatchnormRelu(self, input_, output_dim, kernel=3, name='convBatchnormRelu'):
        conv = self.conv2d(input_, output_dim, kernel, stride=1 ,name=name+'conv2d')
        conv = self.batch_norm(conv, name=name+'batchNorm')
        conv = self.relu(conv, name=name+'Relu')
        return conv
    
    def convTranspose2d(self, x,  middle_channel, out_channel, name):
        x = self.conv2dRelu(x, output_dim=middle_channel, kernel=3, stride=1, name=name + '_conv2d_relu')
        x = self.deconv2d(x, [x.get_shape().as_list()[0], x.get_shape().as_list()[1]*2, x.get_shape().as_list()[2]*2, out_channel], kernel = 3, stride=2, name=name + '_deconv2d')
        x = self.relu(x)
        return x
	
    def deconv2d(self, input_, output_shape, kernel=5, stride=2, stddev=0.02, name="deconv2d"):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable('w', [kernel, kernel, output_shape[-1], input_.get_shape()[-1]], initializer=tf.compat.v1.random_normal_initializer(stddev=stddev))
    
            deconv = tf.compat.v1.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride, stride, 1])

            biases = tf.compat.v1.get_variable('biases', [output_shape[-1]], initializer=tf.compat.v1.constant_initializer(0.0))
            deconv = tf.compat.v1.reshape(tf.compat.v1.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

    def batch_norm(self, x, epsilon=1e-5, momentum = 0.9, name="batch_norm", training=True):
        return tf.compat.v1.layers.batch_normalization(x, training=training)
					  
    def prelu(self, x, name='prelu'):
        with tf.compat.v1.variable_scope(name):
            beta = tf.compat.v1.get_variable('beta', [x.get_shape()[-1]], tf.compat.v1.float32, 
									initializer=tf.compat.v1.constant_initializer(0.01))
		
        beta = tf.compat.v1.minimum(0.2, tf.compat.v1.maximum(beta, 0.01))
			
        return tf.compat.v1.maximum(x, beta*x)
		
		
    def instance_norm(self,x, name='const_norm'):
        mean, var = tf.compat.v1.nn.moments(x, [1, 2], keep_dims=True)
        return tf.compat.v1.div(tf.compat.v1.subtract(x, mean), tf.compat.v1.sqrt(tf.compat.v1.add(var, 1e-9)))

    def channel_norm(self,x, name='channel_norm'):
        mean, var = tf.compat.v1.nn.moments(x, [1, 2, 3], keep_dims=True)
        return tf.compat.v1.div(tf.compat.v1.subtract(x, mean), tf.compat.v1.sqrt(tf.compat.v1.add(var, 1e-9)))

    def dropout(self,x, keep_prob=0.5, training=True):
        return tf.compat.v1.nn.dropout(x, keep_prob=keep_prob)

    def lrelu(self,x, leak=0.2, name='lrelu'):
        return tf.compat.v1.maximum(x, leak*x)

    def relu(self,x, name='relu'):
        return tf.compat.v1.nn.relu(x)