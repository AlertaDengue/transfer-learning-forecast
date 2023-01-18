from lstm import custom_loss_msle
import tensorflow as tf
import numpy as np

def test_increasing_data():
    yt = tf.Variable(np.arange(5).reshape(1,5)) # increasing data, to apply p
    yp = tf.Variable(np.arange(1,6).reshape(1,5)) 
    loss1 = custom_loss_msle(p=1)
    loss10 = custom_loss_msle(p=10)
    assert loss1(yt,yp) < loss10(yt,yp)

def test_stable_data():
    yt = tf.Variable(np.ones(5).reshape(1,5)) # constant data, to ignore p
    yp = tf.Variable(np.ones(5).reshape(1,5)+1) 
    loss1 = custom_loss_msle(p=1)
    loss10 = custom_loss_msle(p=10)
    assert loss1(yt,yp) == loss10(yt,yp)